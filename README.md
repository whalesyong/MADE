# MADE: MAterials Discovery Environments
(Forked by Wei Yong)

> **MADE: Benchmark Environments for Closed-Loop Materials Discovery**
> [Arxiv Preprint](https://arxiv.org/abs/2601.20996) · [NeurIPS AI4Mat 2025 Workshop Paper](https://openreview.net/pdf?id=Cfj7uBu5dy)

MADE provides dynamic benchmark environments for evaluating end-to-end autonomous materials discovery pipelines. Agents iteratively propose crystal structures, an oracle evaluates formation energies, and the environment tracks discovery progress on a phase diagram (convex hull).

---

## Installation

```bash
uv sync --all-extras
```

Set API keys (or add to `.env`):

```bash
export MATERIALS_PROJECT_API_KEY=your_key
export WANDB_API_KEY=your_key          # optional
export ANTHROPIC_API_KEY=your_key      # for Anthropic agents
export OPENAI_API_KEY=your_key         # for OpenAI agents
```

---

## Running LLM-based Experiments with a vLLM Server

LLM-based agents (planner, scorer, orchestrator, reflexion) require an OpenAI-compatible server. The recommended setup uses [vLLM](https://docs.vllm.ai/).

### Step 1 — Start the vLLM server

**Single server (actor only, or shared actor + reflector):**

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key local-token \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.93 \
  --max-model-len 120000
```

**Two servers (separate actor + reflector, e.g., on different GPUs):**

```bash
# Actor
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --host 0.0.0.0 --port 8000 --api-key local-token \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.93 --max-model-len 120000 &

# Reflector (can be a different model/GPU)
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --host 0.0.0.0 --port 8001 --api-key local-token \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.93 --max-model-len 120000 &
```

### Step 2 — Export environment variables

```bash
# API key set in --api-key above
export VLLM_API_KEY=local-token

# Model string for the actor / orchestrator (openai/ prefix required for DSPy/LiteLLM)
export MODEL_STR=openai/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8

# Model string for the reflector (can differ from actor; defaults to MODEL_STR if unset)
export REFLECTOR_MODEL_STR=openai/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8

# Port for the actor / orchestrator vLLM server (default: 8000)
export VLLM_ACTPORT=8000

# Port for the reflector vLLM server (default: 8000, i.e. shared with actor)
export VLLM_REFPORT=8001
```

> **Note**: `VLLM_ACTPORT` and `VLLM_REFPORT` are picked up by
> `configs/agent/llm_react_orchestrator.yaml` via Hydra's `${oc.env:...}` interpolation.
> If both ports are the same, the actor and reflector share one server.

### Step 3 — Run an experiment

```bash
# Single run (5 episodes, Li-O system)
uv run scripts/run_benchmark.py agent=llm_react_orchestrator experiment.num_episodes=5

# Baseline sweep over ternary systems
uv run scripts/run_baseline_experiments.py \
    --agent-configs llm_react_orchestrator \
    --systems-file ./data/systems_10_mp_20/systems_ternary_n10_maxatoms20_intermetallic_smact.json \
    --num-episodes 5 \
    --budget 50

# With Reflexion (inter-episode verbal reflection)
uv run scripts/run_baseline_experiments.py \
    --agent-configs llm_react_orchestrator \
    --systems-file ./data/systems_10_mp_20/systems_ternary_n10_maxatoms20_intermetallic_smact.json \
    --num-episodes 5 \
    --budget 50 \
    --reflexion
```

**Parallel sweep shortcut** (`run_table1_llm_orch_only.sh`):

```bash
export MODEL_STR=openai/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
export VLLM_ACTPORT=8000
export VLLM_REFPORT=8001
export VLLM_API_KEY=local-token

CUDA_VISIBLE_DEVICES=0 CHECK_LLM_BACKEND=0 REFLEXION=1 \
  RUN_PROFILE=custom PARALLEL_SYSTEM_RUNS=1 SYSTEM_SCOPE=ternary \
  OUTPUT_DIR=results/baselines/custom/qwen3-30b-instr_reflx/ \
  CUSTOM_MAX_SYSTEMS=2 CUSTOM_BUDGET=50 CUSTOM_NUM_EPISODES=5 \
  bash scripts/run_table1_llm_orch_only.sh
```

---

## Stage 1: Actor Rollout Collection (preference data)

Stage 1 of the sign-agreement experiment collects actor rollouts that Stage 2
will branch from to produce counterfactual pairs. Three switches on top of the
normal benchmark call make a run "Stage 1 ready":

| `experiment.*` flag             | Purpose                                                                                                                  |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `rollout_save_dir`              | Directory for per-step rollout JSONL. Required.                                                                          |
| `rollout_save_full_state`       | Include proposed/newly-discovered entries in `pre_step_env_state` **and** snapshot `agent.get_state()` before each step. Required for Stage 2 branching. |
| `preserve_checkpoints`          | Write an immutable `checkpoint_step_NNNN.json` per step and keep the rolling `checkpoint.json` after the episode ends. Required for Stage 2 branching. |
| `capture_action_logprobs`       | Force `logprobs=True` on actor LM calls and record per-completion log-prob sums in the rollout record. Recommended for Stage 3 (IPO/DPO) to guard against actor drift. |

### Launching a Stage 1 sweep

```bash
uv run scripts/run_baseline_experiments.py \
  --agent-configs llm_react_orchestrator \
  --systems-file ./data/systems_stage1/systems_ternary_n40.json \
  --num-episodes 5 \
  --budget 50 \
  --rollout-save-dir ./results/stage1/rollouts \
  --rollout-save-full-state \
  --preserve-checkpoints \
  --capture-action-logprobs \
  --vary-starting-seed-per-episode \
  --starting-seed-base 0
```

Or a single run (one system):

```bash
uv run scripts/run_benchmark.py \
  agent=llm_react_orchestrator \
  dataset.elements='[Co,Nb,Sn]' \
  experiment.num_episodes=5 \
  experiment.rollout_save_dir=./results/stage1/rollouts \
  experiment.rollout_save_full_state=true \
  experiment.preserve_checkpoints=true \
  experiment.capture_action_logprobs=true \
  experiment.vary_starting_seed_per_episode=true
```

### Stage 1 output layout

```
results/stage1/rollouts/<system_id>/episode_NNN.jsonl   # one line per step
checkpoints/<run_name>/<system_id>/episode_NNN/
    checkpoint.json                   # rolling (latest)
    checkpoint_step_0000.json         # per-step snapshots — required for branching
    checkpoint_step_0001.json
    ...
<hydra-output-dir>/llm_traces/<system_id>/episode_NNN.jsonl
<hydra-output-dir>/reflections/<system_id>/reflections.json  # if Reflexion is on
```

Each rollout JSONL record includes, at minimum: `step`, `episode_id`,
`system_id`, `initial_compound_seed`, `pre_step_env_state`, `action`,
`post_step_obs`, `metrics`. With the Stage 1 flags on, it additionally
carries `pre_step_agent_state` and `action_logprobs` so Stage 2 can
re-hydrate the policy at the branching point and Stage 3 can verify the
on-policy assumption.

### Launching Stage 2 branches from a Stage 1 state

```bash
uv run scripts/run_branch_from_state.py \
  --checkpoint-dir ./checkpoints/<run_name>/<system_id>/episode_003 \
  --step 12 \
  --branch-id branch_A \
  --output-dir ./results/stage2/<run_name>/<system_id>/ep003/step012/branch_A \
  --sampling-seed 42

# Counterfactual branch B from the same (episode, step) — optionally force the
# first action to a specific structure.
uv run scripts/run_branch_from_state.py \
  --checkpoint-dir ./checkpoints/<run_name>/<system_id>/episode_003 \
  --step 12 \
  --branch-id branch_B \
  --output-dir ./results/stage2/<run_name>/<system_id>/ep003/step012/branch_B \
  --override-action-json ./pairs/ep003_step012_alt.json \
  --sampling-seed 43
```

The branch script: loads the per-step checkpoint, rebuilds env/agent from the
embedded config, replays the trajectory prefix so env state matches step `k`,
loads the saved `agent_state` so the policy matches step `k`, and then runs
to the episode budget end — writing a fresh rollout JSONL tagged with
`branch_id` under its own output directory.

### vLLM log-prob capture

`capture_action_logprobs=true` injects `logprobs=True` into every call the
actor makes through `build_dspy_lm`, and wraps the LM with
`made.utils.logprob_capture.LogProbCapturingLM`. For this to produce non-null
values, the vLLM server must be started with log-prob support (default
behavior for `vllm serve`; no extra flag needed). If the response does not
expose log-probs, the rollout record stores `action_logprobs.summary = null`
rather than crashing the run.

---

## Quick Start (non-LLM agents)

```bash
# Default: random agent, ORB oracle, Li-O system, 3 episodes
uv run scripts/run_benchmark.py

# Chemeleon generative baseline, MACE oracle, 5 episodes
uv run scripts/run_benchmark.py agent=chemeleon_generative_baseline oracle=mace experiment.num_episodes=5

# Custom elements
uv run scripts/run_benchmark.py dataset.elements='[Fe,O]' experiment.num_episodes=5

# Run on Modal (parallel episodes)
uv run scripts/run_benchmark.py experiment.infra=modal
```

---

## Architecture

- **Environment** (`ConvexHullEnvironment`): tracks phase diagram discovery task
- **Oracle** (ORB, MACE, Analytic): evaluates formation energy of proposed structures
- **Agent**: pipeline or LLM orchestrator that proposes structures each step

All components are configured via Hydra configs in `configs/`. See `README_original.md` for full architecture details.

---

## Running Baseline Experiments

```bash
# Local (sequential)
uv run scripts/run_baseline_experiments.py \
    --agent-configs "random_generator_baseline chemeleon_generative_baseline" \
    --systems-file ./data/systems_10_mp_20/systems_ternary_n10_maxatoms20_intermetallic_smact.json

# Modal (parallel)
uv run modal run --detach scripts/run_baseline_experiments_modal.py \
    --agent-configs "random_generator_baseline chemeleon_generative_baseline" \
    --systems-file ./data/systems_10_mp_20/systems_ternary_n10_maxatoms20_intermetallic_smact.json
```

---

## Results & Analysis

Results are saved to `results/<timestamp>-<oracle>-<agent>/` (single run) or `results/baselines_<date>/` (baseline sweep). See `README_original.md` for the full directory layout.

Notebooks:
- `notebooks/basic_analysis.ipynb` — load and analyze a single benchmark run
- `notebooks/results_analysis_utils.py` — utilities for comparing baseline experiments

---

## Extending MADE

- **Oracles**: subclass `Oracle` from `made.oracles.base`, implement `evaluate(structure) -> dict`
- **Agents/components**: see `src/made/agents/README.md`

---

## License

MIT License — see [LICENSE](LICENSE)

## Citation

```bibtex
@misc{malik2026made,
      title={MADE: Benchmark Environments for Closed-Loop Materials Discovery}, 
      author={Shreshth A Malik and Tiarnan Doherty and Panagiotis Tigas and Muhammed Razzak and Stephen J. Roberts and Aron Walsh and Yarin Gal},
      year={2026},
      eprint={2601.20996},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.20996}, 
}
```
