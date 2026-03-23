"""
Multi-system benchmark runner.

Reads a list of chemical systems from config.experiment.systems (list of lists of elements),
runs the existing benchmark episode loop per system via the run_benchmark.run_episode function
on local or Modal infra, saves per-system outputs, and aggregates metrics across systems.
"""

import csv
import importlib.util
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import dotenv
import hydra
import modal
import numpy as np
import tqdm.auto as tqdm
from omegaconf import DictConfig, OmegaConf
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram

import wandb

logger = logging.getLogger(__name__)


def _import_run_benchmark():
    """Dynamically import run_benchmark.py so script can be invoked directly."""
    script_path = Path(__file__).resolve().parent / "run_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_benchmark", str(script_path))
    if spec is None or spec.loader is None:
        raise ImportError("Could not load run_benchmark.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def aggregate_metrics(per_episode: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    def collect(name: str) -> list[float]:
        vals: list[float] = []
        for m in per_episode:
            if name in m and isinstance(m[name], (int, float)):
                vals.append(float(m[name]))
        return vals

    all_final_keys = sorted(
        {k for m in per_episode for k in m.keys() if k.startswith("final/")}
    )
    summary: dict[str, dict[str, float]] = {}
    for k in all_final_keys:
        values = collect(k)
        if values:
            arr = np.array(values, dtype=float)
            n = len(arr)
            summary[k] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "sem": float(np.std(arr) / np.sqrt(n)) if n > 0 else 0.0,
            }
    return summary


def _save_episode_outputs(
    system_id: str, ep: int, result: dict, trajectories_dir: Path
) -> None:
    """Write trajectory JSON and phase diagram image for a completed episode."""
    with open(trajectories_dir / f"episode_{ep:03d}.json", "w") as f:
        json.dump(result, f, indent=2)
    num_elements = len(result["final_env_state"].get("elements", []))
    if num_elements <= 4:
        phase_diagram = PhaseDiagram(
            [
                PDEntry.from_dict(e)
                for e in result["final_env_state"]["phase_diagram_all_entries"]
            ]
        )
        fig = phase_diagram.get_plot(backend="plotly", show_unstable=1.0)
        fig.write_image(trajectories_dir / f"phase_diagram_episode_{ep:03d}.png")


def _run_system_local(
    system_id: str,
    elements: list[str],
    cfg_container: dict,
    trajectories_dir_str: str,
    num_episodes: int,
    wandb_run_name: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Run all episodes for one chemical system sequentially.

    Module-level (not nested) so it is picklable by ProcessPoolExecutor.
    Episodes run sequentially to preserve the reflexion invariant: episode N+1
    reads the reflection file written by episode N before it starts.
    """
    dotenv.load_dotenv()

    rb = _import_run_benchmark()
    cfg_sys = OmegaConf.create(cfg_container)
    cfg_sys.dataset.elements = list(elements)
    trajectories_dir = Path(trajectories_dir_str)

    per_episode_metrics: list[dict[str, Any]] = []
    try:
        for ep in range(num_episodes):
            result = rb.run_episode_local(
                cfg_sys, ep, wandb_run_name=wandb_run_name, system_id=system_id
            )
            per_episode_metrics.append(result.get("metrics", {}))
            _save_episode_outputs(system_id, ep, result, trajectories_dir)
    except KeyboardInterrupt:
        pass  # return whatever was collected before the interrupt
    return system_id, per_episode_metrics


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def run_multi_systems(config: DictConfig) -> None:
    dotenv.load_dotenv()

    # Configure logging level from config
    log_level = config.experiment.get("logging_level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # clear elements from config
    config.dataset.elements = []

    rb = _import_run_benchmark()

    if config.logger.get("use_wandb", False):
        wandb.init(
            project=config.logger.get("wandb_project", "made-benchmark"),
            entity=config.logger.get("wandb_entity", None),
            tags=config.logger.get("wandb_tags", ["benchmark", "multi-system"]),
        )
        wandb.config.update(flatten_dict(OmegaConf.to_container(config, resolve=False)))
        wandb_run_name = wandb.run.name
        wandb_run_id = wandb.run.id
    else:
        wandb_run_id = None
        # Use timestamp as run name for checkpoint path when wandb is disabled
        wandb_run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_dir = Path(config.experiment.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Load systems either directly from config or from systems_file (JSON)
    systems: list[list[str]] = []
    if config.experiment.get("systems_file", None):
        systems_file = Path(config.experiment.systems_file)
        with open(systems_file) as f:
            systems = [list(s) for s in json.load(f)]
    else:
        systems = [list(s) for s in config.experiment.get("systems", [])]

    # Optional cap on number of systems
    max_systems = config.experiment.get("max_systems", None)
    if max_systems is not None:
        systems = systems[: int(max_systems)]
    if not systems:
        raise ValueError(
            "experiment.systems must be a non-empty list of element lists, e.g., [[Co, Nb, Sn], [Li, O]]"
        )

    systems_dir = out_dir / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)

    num_episodes = int(config.experiment.get("num_episodes", 1))
    all_episodes_across_systems: list[dict[str, Any]] = []
    per_system_summaries: dict[str, dict[str, dict[str, float]]] = {}

    # Pre-build per-system configs and directories
    system_setups: list[tuple[str, DictConfig, Path, Path]] = []
    for elements in systems:
        system_id = "-".join(elements)
        cfg_sys = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
        cfg_sys.dataset.elements = list(elements)
        system_dir = systems_dir / system_id
        trajectories_dir = system_dir / "trajectories"
        summary_dir = system_dir / "summary"
        trajectories_dir.mkdir(parents=True, exist_ok=True)
        summary_dir.mkdir(parents=True, exist_ok=True)
        system_setups.append((system_id, cfg_sys, trajectories_dir, summary_dir))

    # Accumulate per-system metrics (full results are written to disk immediately)
    per_system_metrics: dict[str, list[dict[str, Any]]] = {
        sid: [] for sid, *_ in system_setups
    }

    try:
        if config.experiment.infra == "modal":
            # Wave-based parallel dispatch across systems:
            #   Wave 0: episode 0 for all systems in parallel
            #   Wave 1: episode 1 for all systems in parallel (reflexion files from wave 0 are
            #           already on disk, so each system's agent loads its own prior reflection)
            #   ...
            # This preserves the reflexion dependency (ep N+1 sees ep N's reflection) while
            # parallelising across independent systems within each wave.
            _, app, checkpoint_volume, model_checkpoints_volume = rb.build_modal_app()
            run_episode_fn = rb.build_modal_episode_fn(
                app, checkpoint_volume, model_checkpoints_volume
            )

            with modal.enable_output():
                with app.run():
                    for ep in range(num_episodes):
                        logger.info(
                            f"Wave {ep + 1}/{num_episodes}: dispatching "
                            f"{len(system_setups)} systems in parallel"
                        )
                        wave_args = [
                            (cfg_sys, ep, wandb_run_name, system_id)
                            for system_id, cfg_sys, _, _ in system_setups
                        ]
                        for (system_id, _, trajectories_dir, _), result in zip(
                            system_setups, run_episode_fn.starmap(wave_args)
                        ):
                            per_system_metrics[system_id].append(
                                result.get("metrics", {})
                            )
                            _save_episode_outputs(
                                system_id, ep, result, trajectories_dir
                            )
        else:
            # Local path.
            # num_parallel_systems > 1: run that many systems concurrently, each in its
            # own worker process. Episodes within a system stay sequential (reflexion).
            # num_parallel_systems == 1 (default): fully sequential, original behaviour.
            num_parallel = int(config.experiment.get("num_parallel_systems", 1))

            if num_parallel > 1:
                logger.info(
                    f"Running {len(system_setups)} systems with "
                    f"{num_parallel} parallel worker(s)"
                )
                worker_args_list = [
                    (
                        system_id,
                        list(cfg_sys.dataset.elements),
                        OmegaConf.to_container(cfg_sys, resolve=False),
                        str(trajectories_dir),
                        num_episodes,
                        wandb_run_name,
                    )
                    for system_id, cfg_sys, trajectories_dir, _ in system_setups
                ]
                with ProcessPoolExecutor(max_workers=num_parallel) as executor:
                    futures = {
                        executor.submit(_run_system_local, *args): args[0]
                        for args in worker_args_list
                    }
                    for future in as_completed(futures):
                        sid = futures[future]
                        try:
                            _, metrics = future.result()
                            per_system_metrics[sid] = metrics
                            logger.info(
                                f"System {sid} completed "
                                f"({len(metrics)} episode(s))"
                            )
                        except Exception as e:
                            logger.error(
                                f"System {sid} failed: {e}", exc_info=True
                            )
            else:
                # Sequential: systems one at a time, episodes one at a time.
                for sys_idx, (system_id, cfg_sys, trajectories_dir, _) in enumerate(
                    system_setups
                ):
                    logger.info(
                        f"Running system {sys_idx + 1}/{len(system_setups)}: {system_id}"
                    )
                    try:
                        for ep in tqdm.trange(
                            num_episodes, desc=f"Episodes ({system_id})"
                        ):
                            result = rb.run_episode_local(
                                cfg_sys,
                                ep,
                                wandb_run_name=wandb_run_name,
                                system_id=system_id,
                            )
                            per_system_metrics[system_id].append(
                                result.get("metrics", {})
                            )
                            _save_episode_outputs(
                                system_id, ep, result, trajectories_dir
                            )
                    except KeyboardInterrupt:
                        import traceback

                        logger.warning(
                            f"Keyboard interrupt during system {system_id}, stopping its "
                            f"episodes. Stack trace: {traceback.format_exc()}"
                        )
    except KeyboardInterrupt:
        import traceback

        logger.warning(
            f"Keyboard interrupt, stopping benchmark. Stack trace: {traceback.format_exc()}"
        )

    # --- Per-system post-processing (runs on whatever data was collected) ---
    for system_id, cfg_sys, trajectories_dir, summary_dir in system_setups:
        elements = list(cfg_sys.dataset.elements)
        per_episode = per_system_metrics[system_id]
        if not per_episode:
            continue

        # Save ground truth phase diagram (read from first-episode trajectory on disk)
        ep0_path = trajectories_dir / "episode_000.json"
        if len(elements) <= 4 and ep0_path.exists():
            with open(ep0_path) as f_first:
                first_result = json.load(f_first)
            phase_diagram_gt = PhaseDiagram(
                [PDEntry.from_dict(e) for e in first_result["phase_diagram_gt"]]
            )
            fig_gt = phase_diagram_gt.get_plot(backend="plotly", show_unstable=1.0)
            fig_gt.write_image(summary_dir / "phase_diagram_gt.png")

        # Write per-system episodes metrics
        with open(summary_dir / "episodes.json", "w") as f_json:
            json.dump(per_episode, f_json, indent=2)
        all_keys: list[str] = sorted({k for m in per_episode for k in m.keys()})
        with open(summary_dir / "episodes.csv", "w", newline="") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=all_keys)
            writer.writeheader()
            for row in per_episode:
                writer.writerow(row)

        # Per-system summary
        system_summary = aggregate_metrics(per_episode)
        with open(summary_dir / "summary.json", "w") as f_sum:
            json.dump(
                {"episodes": len(per_episode), "summary": system_summary},
                f_sum,
                indent=2,
            )

        # Track for overall aggregation
        all_episodes_across_systems.extend(per_episode)
        per_system_summaries[system_id] = system_summary

        # Minimal wandb logging
        if config.logger.get("use_wandb", False):
            if os.path.exists(summary_dir / "phase_diagram_gt.png"):
                wandb.log(
                    {
                        f"{system_id}/phase_diagram/ground_truth": wandb.Image(
                            str(summary_dir / "phase_diagram_gt.png")
                        )
                    }
                )

    # Overall aggregation across systems
    overall_summary = aggregate_metrics(all_episodes_across_systems)
    overall_dir = out_dir / "overall_summary"
    overall_dir.mkdir(parents=True, exist_ok=True)
    with open(overall_dir / "summary.json", "w") as f_overall:
        json.dump(
            {
                "systems": ["-".join(s) for s in systems],
                "episodes_total": len(all_episodes_across_systems),
                "summary": overall_summary,
            },
            f_overall,
            indent=2,
        )

    # Per-system summary CSV
    per_system_csv = overall_dir / "per_system_summary.csv"
    metric_keys = sorted({k for s in per_system_summaries.values() for k in s.keys()})
    with open(per_system_csv, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["system", "metric", "mean", "std", "sem"])
        for sys_id, metrics in per_system_summaries.items():
            for k in metric_keys:
                if k in metrics:
                    v = metrics[k]
                    writer.writerow(
                        [
                            sys_id,
                            k,
                            v.get("mean", ""),
                            v.get("std", ""),
                            v.get("sem", ""),
                        ]
                    )

    if config.logger.get("use_wandb", False):
        # resume overall run to add an overall table
        wandb.init(
            project=config.logger.get("wandb_project", "made-benchmark"),
            entity=config.logger.get("wandb_entity", None),
            resume="must",
            id=wandb_run_id,
        )
        overall_table = wandb.Table(
            columns=["metric", "mean", "std", "sem"],
            data=[
                [k, v["mean"], v["std"], v["sem"]] for k, v in overall_summary.items()
            ],
        )
        wandb.log({"overall_summary_table": overall_table})
        # save recursively
        wandb.save(f"{str(systems_dir)}/*/*/*", base_path=str(out_dir), policy="live")
        wandb.save(f"{str(overall_dir)}/*", base_path=str(out_dir), policy="live")
        wandb.finish()

    logger.info(f"Overall summary: {json.dumps(overall_summary, indent=2)}")
    logger.info(f"Saved per-system and overall results under {str(out_dir)}")


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    run_multi_systems()
