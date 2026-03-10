"""
JSONL tracing for LLM component outputs (planner / scorer / orchestrator).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _json_safe(value: Any, _seen: set[int] | None = None) -> Any:
    """Recursively convert values to JSON-safe objects.

    This preserves nested dict/list structure where possible and only falls back to
    string conversion at unsupported leaves.
    """
    if _seen is None:
        _seen = set()

    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Path):
        return str(value)

    obj_id = id(value)
    if obj_id in _seen:
        return "<recursive-ref>"
    _seen.add(obj_id)

    if isinstance(value, dict):
        return {str(k): _json_safe(v, _seen) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [_json_safe(v, _seen) for v in value]

    # Common object->dict serializers.
    for attr in ("toDict", "as_dict", "model_dump", "dict"):
        fn = getattr(value, attr, None)
        if callable(fn):
            try:
                return _json_safe(fn(), _seen)
            except Exception:
                pass

    # Shallow object fallback.
    if hasattr(value, "__dict__"):
        try:
            return _json_safe(vars(value), _seen)
        except Exception:
            pass

    return str(value)


def append_llm_trace(
    component: str,
    llm_config: Any,
    output: Any,
    inputs: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """
    Append one JSON line with LLM call output metadata.

    Enabled by llm_config.trace_outputs=true.
    Path resolution order:
      1) llm_config.trace_file
      2) $MADE_LLM_TRACE_PATH
      3) ./llm_traces.jsonl (current run dir)
    """
    if not _cfg_get(llm_config, "trace_outputs", False):
        return

    trace_file = (
        _cfg_get(llm_config, "trace_file")
        or os.getenv("MADE_LLM_TRACE_PATH")
        or "./llm_traces.jsonl"
    )

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "component": component,
        "model": _cfg_get(llm_config, "model"),
        "api_base": _cfg_get(llm_config, "base_url") or _cfg_get(llm_config, "api_base"),
        "pid": os.getpid(),
        "output": _json_safe(output),
    }
    run_meta = {
        "experiment_output_dir": os.getenv("MADE_EXPERIMENT_OUTPUT_DIR"),
        "run_name": os.getenv("MADE_RUN_NAME"),
        "system_id": os.getenv("MADE_SYSTEM_ID"),
        "episode_id": os.getenv("MADE_EPISODE_ID"),
    }
    run_meta = {k: v for k, v in run_meta.items() if v is not None}
    if run_meta:
        record["run"] = run_meta

    if _cfg_get(llm_config, "trace_inputs", False) and inputs is not None:
        record["inputs"] = _json_safe(inputs)

    if extra:
        record["extra"] = _json_safe(extra)

    path = Path(trace_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    # JSONL append (simple + server friendly)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
