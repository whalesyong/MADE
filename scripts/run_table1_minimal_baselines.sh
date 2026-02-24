#!/usr/bin/env bash

set -euo pipefail

# Minimal subset of Table 1 reproduction commands:
# - Random baseline row (random_generator_baseline)
# - LLM Orch. row (llm_react_orchestrator)
# Across ternary/quaternary/quinary intermetallic system sets.
#
# Defaults match the paper setup for Table 1:
# - 10 systems per size
# - 50 query budget
# - 5 episodes per system
# - stability tolerance 0.1 eV
#
# Usage:
#   bash scripts/run_table1_minimal_baselines.sh
#
# Optional overrides:
#   INFRA=modal NUM_EPISODES=5 MAX_SYSTEMS=10 BUDGET=50 bash scripts/run_table1_minimal_baselines.sh
#   RESUME=1 bash scripts/run_table1_minimal_baselines.sh
#   AGENTS="random_generator_baseline" bash scripts/run_table1_minimal_baselines.sh
#
# Notes:
# - `llm_react_orchestrator` requires your LLM backend to be running/configured
#   (see README and configs/agent/llm_react_orchestrator.yaml).

INFRA="${INFRA:-local}"
MAX_SYSTEMS="${MAX_SYSTEMS:-10}"
BUDGET="${BUDGET:-50}"
NUM_EPISODES="${NUM_EPISODES:-5}"
STABILITY_TOLERANCE="${STABILITY_TOLERANCE:-0.1}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/baselines}"
MAX_STOICHIOMETRY="${MAX_STOICHIOMETRY:-20}"

# RESUME=1 -> --resume, otherwise --no-resume (default)
if [[ "${RESUME:-0}" == "1" ]]; then
  RESUME_FLAG="--resume"
else
  RESUME_FLAG="--no-resume"
fi

# Space-separated list override is supported.
AGENTS_STR="${AGENTS:-random_generator_baseline llm_react_orchestrator}"
read -r -a AGENT_CONFIGS <<< "${AGENTS_STR}"

SYSTEM_FILES=(
  "./data/systems_10_mp_20/systems_ternary_n10_maxatoms20_intermetallic_smact.json"
  "./data/systems_10_mp_20/systems_quaternary_n10_maxatoms20_intermetallic_smact.json"
  "./data/systems_10_mp_20/systems_quinary_n10_maxatoms20_intermetallic_smact.json"
)

echo "Running minimal Table 1 baseline subset"
echo "  Agents: ${AGENT_CONFIGS[*]}"
echo "  Infra: ${INFRA}"
echo "  Max systems: ${MAX_SYSTEMS}"
echo "  Budget: ${BUDGET}"
echo "  Episodes: ${NUM_EPISODES}"
echo "  Stability tolerance (eV): ${STABILITY_TOLERANCE}"
echo "  Output base dir: ${OUTPUT_DIR}"
echo "  Resume flag: ${RESUME_FLAG}"
echo

for systems_file in "${SYSTEM_FILES[@]}"; do
  echo "=================================================================="
  echo "Systems file: ${systems_file}"
  echo "=================================================================="

  uv run scripts/run_baseline_experiments.py \
    --agent-configs "${AGENT_CONFIGS[@]}" \
    --systems-file "${systems_file}" \
    --max-systems "${MAX_SYSTEMS}" \
    --budget "${BUDGET}" \
    --num-episodes "${NUM_EPISODES}" \
    --stability-tolerance "${STABILITY_TOLERANCE}" \
    --max-stoichiometry "${MAX_STOICHIOMETRY}" \
    --infra "${INFRA}" \
    --output-dir "${OUTPUT_DIR}" \
    "${RESUME_FLAG}"
done

echo
echo "Completed minimal Table 1 baseline subset runs."
