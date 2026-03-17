#!/usr/bin/env bash

set -euo pipefail

# Run only the Random baseline row from the Table 1 setup.
# Systems: ternary / quaternary / quinary intermetallic sets.
#
# Profiles:
# - fast:      5 systems, 25 budget, 2 episodes
# - balanced:  8 systems, 35 budget, 3 episodes
# - fidelity: 10 systems, 50 budget, 5 episodes
#
# Defaults (balanced profile):
# - 8 systems per size
# - 35 query budget
# - 3 episodes per system
# - stability tolerance 0.1 eV
#
# Usage:
#   bash scripts/run_table1_random_only.sh
#
# Useful overrides:
#   RESUME=1 bash scripts/run_table1_random_only.sh
#   INFRA=modal bash scripts/run_table1_random_only.sh
#   OUTPUT_DIR=./results/baselines_random bash scripts/run_table1_random_only.sh
#   RUN_PROFILE=fast bash scripts/run_table1_random_only.sh
#   PARALLEL_SYSTEM_RUNS=3 bash scripts/run_table1_random_only.sh
#   SYSTEM_SCOPE=ternary bash scripts/run_table1_random_only.sh

INFRA="${INFRA:-local}"
RUN_PROFILE="${RUN_PROFILE:-balanced}"   # fast | balanced | fidelity

case "${RUN_PROFILE}" in
  fast)
    PROFILE_MAX_SYSTEMS=5
    PROFILE_BUDGET=25
    PROFILE_NUM_EPISODES=2
    ;;
  balanced)
    PROFILE_MAX_SYSTEMS=8
    PROFILE_BUDGET=35
    PROFILE_NUM_EPISODES=3
    ;;
  fidelity)
    PROFILE_MAX_SYSTEMS=10
    PROFILE_BUDGET=50
    PROFILE_NUM_EPISODES=5
    ;;
  *)
    echo "ERROR: RUN_PROFILE must be one of: fast, balanced, fidelity"
    exit 1
    ;;
esac

MAX_SYSTEMS="${MAX_SYSTEMS:-$PROFILE_MAX_SYSTEMS}"
BUDGET="${BUDGET:-$PROFILE_BUDGET}"
NUM_EPISODES="${NUM_EPISODES:-$PROFILE_NUM_EPISODES}"
STABILITY_TOLERANCE="${STABILITY_TOLERANCE:-0.1}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/baselines_random}"

MAX_STOICHIOMETRY="${MAX_STOICHIOMETRY:-20}"
PARALLEL_SYSTEM_RUNS="${PARALLEL_SYSTEM_RUNS:-1}"   # 1 = sequential, 2/3 = concurrent

if ! [[ "${PARALLEL_SYSTEM_RUNS}" =~ ^[0-9]+$ ]] || (( PARALLEL_SYSTEM_RUNS < 1 || PARALLEL_SYSTEM_RUNS > 3 )); then
  echo "ERROR: PARALLEL_SYSTEM_RUNS must be an integer in [1, 3]"
  exit 1
fi

# Fail fast by default if one system run errors.
if [[ "${STOP_ON_ERROR:-1}" == "1" ]]; then
  STOP_ON_ERROR_FLAG="--stop-on-error"
else
  STOP_ON_ERROR_FLAG=""
fi

# RESUME=1 -> --resume, otherwise --no-resume (default)
if [[ "${RESUME:-0}" == "1" ]]; then
  RESUME_FLAG="--resume"
else
  RESUME_FLAG="--no-resume"
fi

# SYSTEM_SCOPE controls which system files to run:
# - all (default), ternary, quaternary, quinary
SYSTEM_SCOPE="${SYSTEM_SCOPE:-all}"

SYSTEM_FILES=(
  #"./data/systems_10_mp_20/systems_ternary_n10_maxatoms20_intermetallic_smact.json"
  "./data/systems_10_mp_20/systems_quaternary_n10_maxatoms20_intermetallic_smact.json"
  "./data/systems_10_mp_20/systems_quinary_n10_maxatoms20_intermetallic_smact.json"
)

case "${SYSTEM_SCOPE}" in
  all) ;;
  ternary)
    SYSTEM_FILES=("./data/systems_10_mp_20/systems_ternary_n10_maxatoms20_intermetallic_smact.json")
    ;;
  quaternary)
    SYSTEM_FILES=("./data/systems_10_mp_20/systems_quaternary_n10_maxatoms20_intermetallic_smact.json")
    ;;
  quinary)
    SYSTEM_FILES=("./data/systems_10_mp_20/systems_quinary_n10_maxatoms20_intermetallic_smact.json")
    ;;
  *)
    echo "ERROR: SYSTEM_SCOPE must be one of: all, ternary, quaternary, quinary"
    exit 1
    ;;
esac

echo "Running Table 1 Random baseline only"
echo "  Agent: random_generator_baseline"
echo "  Profile: ${RUN_PROFILE}"
echo "  Infra: ${INFRA}"
echo "  Max systems: ${MAX_SYSTEMS}"
echo "  Budget: ${BUDGET}"
echo "  Episodes: ${NUM_EPISODES}"
echo "  Stability tolerance (eV): ${STABILITY_TOLERANCE}"
echo "  Output base dir: ${OUTPUT_DIR}"
echo "  System scope: ${SYSTEM_SCOPE}"
echo "  Parallel system runs: ${PARALLEL_SYSTEM_RUNS}"
echo "  Resume flag: ${RESUME_FLAG}"
echo "  Stop-on-error: ${STOP_ON_ERROR_FLAG:-off}"
echo

run_one_system_file() {
  local systems_file="$1"
  local systems_tag
  systems_tag="$(basename "${systems_file}" .json)"
  local system_output_dir="${OUTPUT_DIR}/${systems_tag}"

  echo "=================================================================="
  echo "Systems file: ${systems_file}"
  echo "Output root: ${system_output_dir}"
  echo "=================================================================="

  local cmd=(
    uv run scripts/run_baseline_experiments.py
    --agent-configs random_generator_baseline
    --systems-file "${systems_file}"
    --max-systems "${MAX_SYSTEMS}"
    --budget "${BUDGET}"
    --num-episodes "${NUM_EPISODES}"
    --stability-tolerance "${STABILITY_TOLERANCE}"
    --max-stoichiometry "${MAX_STOICHIOMETRY}"
    --infra "${INFRA}"
    --output-dir "${system_output_dir}"
    "${RESUME_FLAG}"
  )

  if [[ -n "${STOP_ON_ERROR_FLAG}" ]]; then
    cmd+=("${STOP_ON_ERROR_FLAG}")
  fi

  "${cmd[@]}"
}

if (( PARALLEL_SYSTEM_RUNS == 1 )); then
  for systems_file in "${SYSTEM_FILES[@]}"; do
    run_one_system_file "${systems_file}"
  done
else
  declare -a pids=()
  declare -a files=("${SYSTEM_FILES[@]}")
  failures=0
  idx=0
  total=${#files[@]}

  # Portable batching (works on macOS bash 3.x; avoids wait -n).
  while (( idx < total )); do
    pids=()
    batch_end=$((idx + PARALLEL_SYSTEM_RUNS))
    if (( batch_end > total )); then
      batch_end=$total
    fi

    while (( idx < batch_end )); do
      run_one_system_file "${files[idx]}" &
      pids+=("$!")
      idx=$((idx + 1))
    done

    for pid in "${pids[@]}"; do
      if ! wait "${pid}"; then
        failures=$((failures + 1))
      fi
    done
  done

  if (( failures > 0 )); then
    echo "ERROR: ${failures} parallel run(s) failed."
    exit 1
  fi
fi

echo
echo "Completed Table 1 Random baseline-only runs."
