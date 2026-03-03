#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/check_baseline_run.sh <experiment_dir> [expected_systems] [expected_episodes_per_system]

Examples:
  bash scripts/check_baseline_run.sh ./results/baselines/20260224-120000/llm_react_orchestrator_systems_ternary_n10_maxatoms20_intermetallic_smact_10systems_50queries_100stabilitymeV
  bash scripts/check_baseline_run.sh <experiment_dir> 10 5

Notes:
  - expected_systems defaults to 10
  - expected_episodes_per_system defaults to 5
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

EXP_DIR="$1"
EXPECTED_SYSTEMS="${2:-10}"
EXPECTED_EPISODES_PER_SYSTEM="${3:-5}"
EXPECTED_TOTAL=$((EXPECTED_SYSTEMS * EXPECTED_EPISODES_PER_SYSTEM))

python3 - "$EXP_DIR" "$EXPECTED_SYSTEMS" "$EXPECTED_EPISODES_PER_SYSTEM" <<'PY'
import json
import pathlib
import sys

exp_dir = pathlib.Path(sys.argv[1]).expanduser().resolve()
expected_systems = int(sys.argv[2])
expected_eps_per_system = int(sys.argv[3])
expected_total = expected_systems * expected_eps_per_system

progress_path = exp_dir / "progress.json"
summary_path = exp_dir / "overall_summary" / "summary.json"
metadata_path = exp_dir / "experiment_metadata.json"
error_log_path = exp_dir / "error.log"

failed = False

print(f"Experiment dir: {exp_dir}")
print(f"Expected systems: {expected_systems}")
print(f"Expected episodes/system: {expected_eps_per_system}")
print(f"Expected episodes_total: {expected_total}")
print()

if not exp_dir.exists():
    print("FAIL: experiment directory does not exist.")
    sys.exit(1)

# Metadata
if metadata_path.exists():
    try:
        metadata = json.loads(metadata_path.read_text())
        print("Metadata:")
        print(f"  agent_config: {metadata.get('agent_config')}")
        print(f"  systems_file: {metadata.get('systems_file')}")
        print(f"  max_systems: {metadata.get('max_systems')}")
        print(f"  num_episodes: {metadata.get('num_episodes')}")
        print(f"  budget: {metadata.get('budget')}")
    except Exception as e:
        failed = True
        print(f"FAIL: could not parse experiment_metadata.json ({e})")
else:
    print("WARN: experiment_metadata.json missing")

print()

# Progress
progress = None
if not progress_path.exists():
    failed = True
    print("FAIL: progress.json missing")
else:
    try:
        progress = json.loads(progress_path.read_text())
        status = progress.get("status")
        completed = progress.get("completed_systems", [])
        total = progress.get("total_systems")
        print("Progress:")
        print(f"  status: {status}")
        print(f"  completed_systems: {len(completed)}/{total}")
        if status != "completed":
            failed = True
            print("  FAIL: status is not 'completed'")
        if total is not None and int(total) != expected_systems:
            failed = True
            print(f"  FAIL: total_systems ({total}) != expected_systems ({expected_systems})")
        if len(completed) != expected_systems:
            failed = True
            print(f"  FAIL: completed_systems count ({len(completed)}) != expected_systems ({expected_systems})")
    except Exception as e:
        failed = True
        print(f"FAIL: could not parse progress.json ({e})")

print()

# Overall summary
if not summary_path.exists():
    failed = True
    print("FAIL: overall_summary/summary.json missing")
else:
    try:
        summary_blob = json.loads(summary_path.read_text())
        episodes_total = summary_blob.get("episodes_total")
        metrics = summary_blob.get("summary", {})
        print("Overall summary:")
        print(f"  episodes_total: {episodes_total}")
        print(f"  metrics_count: {len(metrics)}")
        if episodes_total != expected_total:
            failed = True
            print(f"  FAIL: episodes_total ({episodes_total}) != expected ({expected_total})")
        if not metrics:
            failed = True
            print("  FAIL: no metrics in overall summary")
    except Exception as e:
        failed = True
        print(f"FAIL: could not parse overall_summary/summary.json ({e})")

print()

if error_log_path.exists():
    failed = True
    print(f"FAIL: error.log exists at {error_log_path}")

if failed:
    print("RESULT: NOT VALID / INCOMPLETE")
    sys.exit(1)
else:
    print("RESULT: VALID")
    sys.exit(0)
PY
