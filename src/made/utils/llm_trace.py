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


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except Exception:
        if hasattr(value, "toDict"):
            try:
                return value.toDict()
            except Exception:
                pass
        if hasattr(value, "as_dict"):
            try:
                return value.as_dict()
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

    if _cfg_get(llm_config, "trace_inputs", False) and inputs is not None:
        record["inputs"] = _json_safe(inputs)

    if extra:
        record["extra"] = _json_safe(extra)

    path = Path(trace_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    # JSONL append (simple + server friendly)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
