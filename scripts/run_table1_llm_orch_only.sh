#!/usr/bin/env bash

set -euo pipefail

# Run only the LLM Orchestrator row from the Table 1 minimal baseline setup.
# Systems: ternary / quaternary / quinary intermetallic sets.
#
# Defaults:
# - 10 systems per size
# - 50 query budget
# - 5 episodes per system
# - stability tolerance 0.1 eV
#
# Usage:
#   bash scripts/run_table1_llm_orch_only.sh
#
# Useful overrides:
#   RESUME=1 bash scripts/run_table1_llm_orch_only.sh
#   INFRA=modal bash scripts/run_table1_llm_orch_only.sh
#   OUTPUT_DIR=./results/baselines_llm_orch bash scripts/run_table1_llm_orch_only.sh
#
# Optional preflight controls:
#   CHECK_LLM_BACKEND=0                  # skip endpoint check
#   REQUIRE_API_KEY=0                    # do not require VLLM_API_KEY env var
#   LLM_BASE_URL=http://127.0.0.1:8000/v1
#   LLM_HEALTH_PATH=/models

INFRA="${INFRA:-local}"
MAX_SYSTEMS="${MAX_SYSTEMS:-10}"
BUDGET="${BUDGET:-50}"
NUM_EPISODES="${NUM_EPISODES:-5}"
STABILITY_TOLERANCE="${STABILITY_TOLERANCE:-0.1}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/baselines}"
MAX_STOICHIOMETRY="${MAX_STOICHIOMETRY:-20}"

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
LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8000/v1}"
LLM_HEALTH_PATH="${LLM_HEALTH_PATH:-/models}"
LLM_CHECK_TIMEOUT="${LLM_CHECK_TIMEOUT:-5}"

SYSTEM_FILES=(
  "./data/systems_10_mp_20/systems_ternary_n10_maxatoms20_intermetallic_smact.json"
  "./data/systems_10_mp_20/systems_quaternary_n10_maxatoms20_intermetallic_smact.json"
  "./data/systems_10_mp_20/systems_quinary_n10_maxatoms20_intermetallic_smact.json"
)

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
echo "  Infra: ${INFRA}"
echo "  Max systems: ${MAX_SYSTEMS}"
echo "  Budget: ${BUDGET}"
echo "  Episodes: ${NUM_EPISODES}"
echo "  Stability tolerance (eV): ${STABILITY_TOLERANCE}"
echo "  Output base dir: ${OUTPUT_DIR}"
echo "  Resume flag: ${RESUME_FLAG}"
echo "  Stop-on-error: ${STOP_ON_ERROR_FLAG:-off}"
echo

for systems_file in "${SYSTEM_FILES[@]}"; do
  echo "=================================================================="
  echo "Systems file: ${systems_file}"
  echo "=================================================================="

  uv run scripts/run_baseline_experiments.py \
    --agent-configs llm_react_orchestrator \
    --systems-file "${systems_file}" \
    --max-systems "${MAX_SYSTEMS}" \
    --budget "${BUDGET}" \
    --num-episodes "${NUM_EPISODES}" \
    --stability-tolerance "${STABILITY_TOLERANCE}" \
    --max-stoichiometry "${MAX_STOICHIOMETRY}" \
    --infra "${INFRA}" \
    --output-dir "${OUTPUT_DIR}" \
    "${RESUME_FLAG}" \
    ${STOP_ON_ERROR_FLAG}
done

echo
echo "Completed LLM Orchestrator-only baseline runs."
