#!/usr/bin/env bash

set -euo pipefail

# Run only the LLM Orchestrator row from the Table 1 minimal baseline setup.
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
#   bash scripts/run_table1_llm_orch_only.sh
#
# Useful overrides:
#   RESUME=1 bash scripts/run_table1_llm_orch_only.sh
#   INFRA=modal bash scripts/run_table1_llm_orch_only.sh
#   OUTPUT_DIR=./results/baselines_llm_orch bash scripts/run_table1_llm_orch_only.sh
#   REFLEXION=1 bash scripts/run_table1_llm_orch_only.sh
#   INITIAL_FAMILY_MODE=hard bash scripts/run_table1_llm_orch_only.sh
#   INITIAL_FAMILY_MODE=easy bash scripts/run_table1_llm_orch_only.sh
#   RUN_PROFILE=fast bash scripts/run_table1_llm_orch_only.sh
#   PARALLEL_SYSTEM_RUNS=3 bash scripts/run_table1_llm_orch_only.sh
#   NUM_PARALLEL_SYSTEMS=4 bash scripts/run_table1_llm_orch_only.sh
#
# Optional preflight controls:
#   CHECK_LLM_BACKEND=0                  # skip endpoint check
#   REQUIRE_API_KEY=0                    # do not require VLLM_API_KEY env var
#   LLM_BASE_URL=http://127.0.0.1:8000/v1
#   LLM_HEALTH_PATH=/models

INFRA="${INFRA:-local}"
RUN_PROFILE="${RUN_PROFILE:-balanced}"   # fast | balanced | fidelity | custom

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
  custom)
    PROFILE_MAX_SYSTEMS="${CUSTOM_MAX_SYSTEMS:?ERROR: CUSTOM_MAX_SYSTEMS must be set when RUN_PROFILE=custom}"
    PROFILE_BUDGET="${CUSTOM_BUDGET:?ERROR: CUSTOM_BUDGET must be set when RUN_PROFILE=custom}"
    PROFILE_NUM_EPISODES="${CUSTOM_NUM_EPISODES:?ERROR: CUSTOM_NUM_EPISODES must be set when RUN_PROFILE=custom}"
    ;;
  *)
    echo "ERROR: RUN_PROFILE must be one of: fast, balanced, fidelity, custom"
    exit 1
    ;;
esac

MAX_SYSTEMS="${MAX_SYSTEMS:-$PROFILE_MAX_SYSTEMS}"
BUDGET="${BUDGET:-$PROFILE_BUDGET}"
NUM_EPISODES="${NUM_EPISODES:-$PROFILE_NUM_EPISODES}"
STABILITY_TOLERANCE="${STABILITY_TOLERANCE:-0.1}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/baselines}"

MAX_STOICHIOMETRY="${MAX_STOICHIOMETRY:-20}"
PARALLEL_SYSTEM_RUNS="${PARALLEL_SYSTEM_RUNS:-1}"   # 1 = sequential, 2/3 = concurrent (across system files)
NUM_PARALLEL_SYSTEMS="${NUM_PARALLEL_SYSTEMS:-1}"   # systems to run in parallel within each file

if ! [[ "${PARALLEL_SYSTEM_RUNS}" =~ ^[0-9]+$ ]] || (( PARALLEL_SYSTEM_RUNS < 1 || PARALLEL_SYSTEM_RUNS > 3 )); then
  echo "ERROR: PARALLEL_SYSTEM_RUNS must be an integer in [1, 3]"
  exit 1
fi

# REFLEXION=1 -> --reflexion, otherwise omitted (default)
if [[ "${REFLEXION:-0}" == "1" ]]; then
  REFLEXION_FLAG="--reflexion"
else
  REFLEXION_FLAG=""
fi

# INITIAL_FAMILY_MODE=hard|easy -> --initial-family-mode hard|easy
INITIAL_FAMILY_MODE="${INITIAL_FAMILY_MODE:-}"
INITIAL_FAMILY_FLAGS=""
if [[ -n "${INITIAL_FAMILY_MODE}" ]]; then
  if [[ "${INITIAL_FAMILY_MODE}" != "hard" && "${INITIAL_FAMILY_MODE}" != "easy" ]]; then
    echo "ERROR: INITIAL_FAMILY_MODE must be 'hard' or 'easy'"
    exit 1
  fi
  INITIAL_FAMILY_FLAGS="--initial-family-mode ${INITIAL_FAMILY_MODE}"
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

# LLM backend preflight checks.
CHECK_LLM_BACKEND="${CHECK_LLM_BACKEND:-1}"
REQUIRE_API_KEY="${REQUIRE_API_KEY:-1}"

LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8004/v1}"
LLM_HEALTH_PATH="${LLM_HEALTH_PATH:-/models}"
LLM_CHECK_TIMEOUT="${LLM_CHECK_TIMEOUT:-5}"

# SYSTEM_SCOPE controls which system files to run:
# - all (default), ternary, quaternary, quinary
SYSTEM_SCOPE="${SYSTEM_SCOPE:-all}"

SYSTEM_FILES=(
  "./data/systems_10_mp_20/systems_ternary_n10_maxatoms20_intermetallic_smact.json"
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

if [[ "${REQUIRE_API_KEY}" == "1" ]] && [[ -z "${VLLM_API_KEY:-}" ]]; then
  echo "ERROR: VLLM_API_KEY is not set."
  echo "Set VLLM_API_KEY before running (or set REQUIRE_API_KEY=0 to bypass)."
  exit 1
fi

if [[ "${CHECK_LLM_BACKEND}" == "1" ]]; then
  if command -v curl >/dev/null 2>&1; then
    CHECK_URL="${LLM_BASE_URL%/}${LLM_HEALTH_PATH}"
    echo "Checking LLM backend: ${CHECK_URL}"

    CURL_ARGS=(
      -sS
      --max-time "${LLM_CHECK_TIMEOUT}"
      -o /dev/null
      -w "%{http_code}"
    )
    if [[ -n "${VLLM_API_KEY:-}" ]]; then
      CURL_ARGS+=(-H "Authorization: Bearer ${VLLM_API_KEY}")
    fi

    HTTP_CODE="$(curl "${CURL_ARGS[@]}" "${CHECK_URL}" || true)"

    if [[ "${HTTP_CODE}" =~ ^2[0-9][0-9]$ ]]; then
      :
    elif [[ "${HTTP_CODE}" == "401" || "${HTTP_CODE}" == "403" ]]; then
      if [[ -n "${VLLM_API_KEY:-}" ]]; then
        echo "ERROR: LLM backend reached but auth was rejected (HTTP ${HTTP_CODE})."
        echo "Check VLLM_API_KEY matches your server --api-key value."
        exit 1
      fi
      echo "WARN: LLM backend reachable (HTTP ${HTTP_CODE}) but auth is required."
      echo "Set VLLM_API_KEY to your server --api-key value."
      exit 1
    else
      echo "ERROR: LLM backend check failed at ${CHECK_URL} (HTTP ${HTTP_CODE})."

      echo "Start/fix your backend, then rerun (or set CHECK_LLM_BACKEND=0 to bypass)."
      exit 1
    fi
  else
    echo "WARN: curl not found; skipping LLM backend reachability check."
  fi
fi

echo "Running Table 1 LLM Orch only"
echo "  Agent: llm_react_orchestrator"
echo "  Profile: ${RUN_PROFILE}"
echo "  Infra: ${INFRA}"
echo "  Max systems: ${MAX_SYSTEMS}"
echo "  Budget: ${BUDGET}"
echo "  Episodes: ${NUM_EPISODES}"
echo "  Stability tolerance (eV): ${STABILITY_TOLERANCE}"
echo "  Output base dir: ${OUTPUT_DIR}"
echo "  System scope: ${SYSTEM_SCOPE}"
echo "  Parallel system runs (across files): ${PARALLEL_SYSTEM_RUNS}"
echo "  Parallel systems (within file):     ${NUM_PARALLEL_SYSTEMS}"
echo "  Resume flag: ${RESUME_FLAG}"
echo "  Stop-on-error: ${STOP_ON_ERROR_FLAG:-off}"
echo "  Reflexion: ${REFLEXION_FLAG:-off}"
echo "  Initial family mode: ${INITIAL_FAMILY_MODE:-off}"
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
    --agent-configs llm_react_orchestrator \
    --systems-file "${systems_file}"
    --max-systems "${MAX_SYSTEMS}"
    --budget "${BUDGET}"
    --num-episodes "${NUM_EPISODES}"
    --stability-tolerance "${STABILITY_TOLERANCE}"
    --max-stoichiometry "${MAX_STOICHIOMETRY}"
    --infra "${INFRA}"
    --output-dir "${system_output_dir}"
    --parallel-systems "${NUM_PARALLEL_SYSTEMS}"
    "${RESUME_FLAG}"
  )

  if [[ -n "${STOP_ON_ERROR_FLAG}" ]]; then
    cmd+=("${STOP_ON_ERROR_FLAG}")
  fi
  if [[ -n "${REFLEXION_FLAG}" ]]; then
    cmd+=("${REFLEXION_FLAG}")
  fi
  if [[ -n "${INITIAL_FAMILY_FLAGS}" ]]; then
    cmd+=(${INITIAL_FAMILY_FLAGS})
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
echo "Completed LLM Orchestrator-only baseline runs."
