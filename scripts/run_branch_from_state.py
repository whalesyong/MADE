"""Stage 2 branching rollout: continue (or counterfactually diverge from) a
Stage 1 episode starting from a saved per-step checkpoint.

Why this script exists
----------------------
Stage 2 of the sign-agreement experiment needs to launch two rollouts from
the same intermediate state of a completed Stage 1 trajectory and compare
their returns. ``scripts/run_benchmark.py`` only supports resuming an
interrupted episode from *its own* latest checkpoint and deletes checkpoints
on successful completion — neither is enough for branching.

This script:

1. Loads the per-step checkpoint written when a Stage 1 run was launched with
   ``experiment.preserve_checkpoints=true``.
2. Instantiates the environment and agent from the checkpoint's embedded
   config.
3. Replays the trajectory up to step ``k`` so the environment is in the right
   state, then loads the saved agent state so the policy is at the right
   state too.
4. Optionally forces the first proposed structure (the "counterfactual
   action") from a JSON dict.
5. Runs the remaining budget under a fresh output / rollout directory tagged
   with ``--branch-id`` so branches never collide with each other or with
   Stage 1.

The checkpoint layout assumed here matches what ``save_checkpoint`` writes
when ``preserve_per_step=True``::

    <checkpoint-root>/<run_name>/<system_id>/episode_NNN/
        checkpoint.json                  # rolling (latest)
        checkpoint_step_0000.json
        checkpoint_step_0001.json
        ...

Usage
-----
    uv run scripts/run_branch_from_state.py \\
        --checkpoint-dir ./checkpoints/<run_name>/<system_id>/episode_003 \\
        --step 12 \\
        --branch-id branch_A \\
        --output-dir ./results/stage2/<run_name>/<system_id>/ep003/step012/branch_A \\
        [--override-action-json ./path/to/action.json] \\
        [--sampling-seed 42]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import dotenv
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pymatgen.core.structure import Structure

logger = logging.getLogger(__name__)


def _load_checkpoint(checkpoint_dir: Path, step: int) -> tuple[Path, dict[str, Any]]:
    per_step = checkpoint_dir / f"checkpoint_step_{step:04d}.json"
    if not per_step.exists():
        raise FileNotFoundError(
            f"No per-step checkpoint at {per_step}. "
            "Did the Stage 1 run use experiment.preserve_checkpoints=true?"
        )
    with per_step.open() as f:
        return per_step, json.load(f)


def _replay_env(env, trajectory: list[dict[str, Any]]) -> None:
    env.reset()
    for obs in trajectory:
        proposal_dict = obs["proposal"]
        proposal = (
            Structure.from_dict(proposal_dict)
            if isinstance(proposal_dict, dict)
            else proposal_dict
        )
        env.step(proposal)


def run_branch(
    checkpoint_dir: Path,
    step: int,
    branch_id: str,
    output_dir: Path,
    override_action_json: Path | None = None,
    sampling_seed: int | None = None,
) -> dict[str, Any]:
    _, checkpoint = _load_checkpoint(checkpoint_dir, step)

    if "config" not in checkpoint:
        raise ValueError(
            "Checkpoint is missing an embedded 'config' field; cannot rebuild env/agent. "
            "Re-run Stage 1 with a version of run_benchmark.py that saves config into the checkpoint."
        )

    config: DictConfig = OmegaConf.create(checkpoint["config"])

    # Route logs / rollout data under a branch-specific directory so multiple
    # counterfactual branches from the same (episode, step) never collide.
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.update(config, "experiment.output_dir", str(output_dir), force_add=True)

    rollout_dir = output_dir / "rollouts"
    OmegaConf.update(
        config, "experiment.rollout_save_dir", str(rollout_dir), force_add=True
    )
    OmegaConf.update(
        config, "experiment.rollout_save_full_state", True, force_add=True
    )
    # Stage 2 branches do not need to be themselves-branchable; keep their
    # own per-step checkpointing off to save disk.
    OmegaConf.update(
        config, "experiment.preserve_checkpoints", False, force_add=True
    )

    if sampling_seed is not None:
        random.seed(sampling_seed)
        np.random.seed(sampling_seed)

    system_id = os.environ.get("MADE_SYSTEM_ID") or checkpoint_dir.parent.name
    episode_id_raw = checkpoint_dir.name.replace("episode_", "")
    try:
        episode_id = int(episode_id_raw)
    except ValueError:
        episode_id = 0

    os.environ["MADE_RUN_NAME"] = f"branch_{branch_id}"
    os.environ["MADE_EPISODE_ID"] = str(episode_id)
    os.environ["MADE_SYSTEM_ID"] = system_id
    os.environ["MADE_EXPERIMENT_OUTPUT_DIR"] = str(output_dir)
    trace_file = output_dir / "llm_traces" / system_id / f"episode_{episode_id:03d}.jsonl"
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    os.environ["MADE_LLM_TRACE_PATH"] = str(trace_file)

    dataset = instantiate(config.dataset)
    oracle = instantiate(config.oracle)
    env = instantiate(config.environment, dataset=dataset, oracle=oracle)
    agent = instantiate(config.agent)

    trajectory_prefix = checkpoint["trajectory"][:step]
    logger.info(
        "Branching from step %d (trajectory prefix length=%d, target budget=%s)",
        step,
        len(trajectory_prefix),
        OmegaConf.select(config, "environment.budget", default="<unset>"),
    )
    _replay_env(env, trajectory_prefix)

    agent.load_state(checkpoint["agent_state"])

    # Load optional override action.
    override_action: Structure | None = None
    if override_action_json is not None:
        with override_action_json.open() as f:
            override_action = Structure.from_dict(json.load(f))

    trajectory: list[dict[str, Any]] = list(trajectory_prefix)
    query_count = step

    rollout_file = rollout_dir / system_id / f"episode_{episode_id:03d}.jsonl"
    rollout_file.parent.mkdir(parents=True, exist_ok=True)

    capture_logprobs = bool(
        OmegaConf.select(config, "experiment.capture_action_logprobs", default=False)
    )
    if capture_logprobs:
        os.environ["MADE_CAPTURE_LOGPROBS"] = "1"
        from made.utils.logprob_capture import drain as _drain
        from made.utils.logprob_capture import enable as _enable

        _enable()

    branch_tag = {"branch_id": branch_id, "branch_from_step": step}

    forced_first = override_action is not None
    while not env.is_done():
        state = env.get_state()
        pre_step_state = env.get_state(include_counterfactual_state=True)
        pre_step_agent_state = agent.get_state()

        if forced_first:
            # Still drive the agent so its internal state updates, but discard
            # its proposal in favour of the override.
            agent.update_state(state)
            struct = override_action  # type: ignore[assignment]
            forced_first = False
        else:
            _, struct = agent(state)

        action_logprobs = _drain() if capture_logprobs else None
        obs, _ = env.step(struct)
        obs_serializable = {
            k: (v.as_dict() if isinstance(v, Structure) else v) for k, v in obs.items()
        }
        record = {
            "step": query_count,
            "episode_id": episode_id,
            "system_id": system_id,
            "pre_step_env_state": pre_step_state,
            "pre_step_agent_state": pre_step_agent_state,
            "action": struct.as_dict(),
            "post_step_obs": obs_serializable,
            "metrics": env.get_latest_metrics(),
            **branch_tag,
        }
        if action_logprobs is not None:
            record["action_logprobs"] = action_logprobs
        with rollout_file.open("a", encoding="utf-8") as rf:
            rf.write(json.dumps(record, ensure_ascii=False) + "\n")

        proposal = obs.get("proposal")
        if isinstance(proposal, Structure):
            obs["proposal"] = proposal.as_dict()
        trajectory.append(obs)
        query_count += 1

    final_metrics = env.get_latest_metrics()
    summary = {
        "branch_id": branch_id,
        "branched_from_step": step,
        "episode_id": episode_id,
        "system_id": system_id,
        "checkpoint_dir": str(checkpoint_dir),
        "override_action": override_action_json and str(override_action_json),
        "final_metrics": {
            k: v for k, v in final_metrics.items() if isinstance(v, (int, float, bool))
        },
        "query_count": query_count,
    }
    with (output_dir / "branch_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Branch complete: %s", summary)
    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint-dir",
        required=True,
        type=Path,
        help="Episode directory containing checkpoint_step_NNNN.json files.",
    )
    p.add_argument("--step", required=True, type=int, help="Step index to branch from.")
    p.add_argument(
        "--branch-id",
        required=True,
        help="Unique id for this branch (e.g. 'branch_A', 'branch_B').",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write branch rollouts + summary. Defaults to "
        "./results/stage2_branches/<timestamp>/<branch_id>.",
    )
    p.add_argument(
        "--override-action-json",
        type=Path,
        default=None,
        help="Optional path to a pymatgen Structure .as_dict() JSON to use "
        "as the first action instead of the agent's proposal.",
    )
    p.add_argument(
        "--sampling-seed",
        type=int,
        default=None,
        help="Seed for Python/numpy RNGs (vLLM sampling seed is controlled via its own config).",
    )
    return p.parse_args()


def main() -> None:
    dotenv.load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _parse_args()
    output_dir = args.output_dir or Path(
        f"./results/stage2_branches/{datetime.now():%Y%m%d-%H%M%S}/{args.branch_id}"
    )
    run_branch(
        checkpoint_dir=args.checkpoint_dir,
        step=args.step,
        branch_id=args.branch_id,
        output_dir=output_dir,
        override_action_json=args.override_action_json,
        sampling_seed=args.sampling_seed,
    )


if __name__ == "__main__":
    main()
