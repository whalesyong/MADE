#!/usr/bin/env python3
"""
Inspect LLM payloads captured in MADE JSONL trace files.

Works with:
- New traces where `extra` is structured JSON.
- Legacy traces where `extra` was stringified Python repr.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSONL line: {e}") from e
    return records


def _parse_legacy_extra(extra_text: str) -> dict[str, Any]:
    """Best-effort parser for old `extra` strings."""
    try:
        parsed = ast.literal_eval(extra_text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    result: dict[str, Any] = {}
    m = re.search(r"'num_lm_calls':\s*(\d+)", extra_text)
    if m:
        result["num_lm_calls"] = int(m.group(1))

    calls: list[dict[str, Any]] = []
    for mm in re.finditer(
        r"'messages':\s*(\[[\s\S]*?\])\s*,\s*'kwargs':", extra_text
    ):
        msg_text = mm.group(1)
        try:
            messages = ast.literal_eval(msg_text)
        except Exception:
            messages = []
        calls.append({"messages": messages})
    if calls:
        result["lm_calls"] = calls
    return result


def _normalize_extra(extra: Any) -> dict[str, Any]:
    if isinstance(extra, dict):
        return extra
    if isinstance(extra, str):
        return _parse_legacy_extra(extra)
    return {}


def _messages_from_record(record: dict[str, Any]) -> list[list[dict[str, Any]]]:
    extra = _normalize_extra(record.get("extra"))
    calls = extra.get("lm_calls", [])
    out: list[list[dict[str, Any]]] = []
    if not isinstance(calls, list):
        return out
    for c in calls:
        if isinstance(c, dict) and isinstance(c.get("messages"), list):
            out.append(c["messages"])
    return out


def _render_text(text: str, max_chars: int | None) -> str:
    if max_chars is None:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... [truncated, total={len(text)} chars]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect LLM payloads in MADE traces.")
    parser.add_argument("trace_file", type=Path, help="Path to episode_XXX.jsonl trace file")
    parser.add_argument(
        "--record-index",
        type=int,
        default=0,
        help="0-based JSONL record index to inspect (default: 0)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1200,
        help="Max chars to print per message content; use -1 for full content",
    )
    parser.add_argument(
        "--dump-json",
        type=Path,
        default=None,
        help="Optional path to dump extracted messages as JSON",
    )
    args = parser.parse_args()

    records = _load_records(args.trace_file)
    if not records:
        raise ValueError(f"No records in {args.trace_file}")
    if args.record_index < 0 or args.record_index >= len(records):
        raise IndexError(
            f"record-index {args.record_index} out of range [0, {len(records)-1}]"
        )

    record = records[args.record_index]
    message_batches = _messages_from_record(record)
    max_chars = None if args.max_chars == -1 else args.max_chars

    print(f"trace_file: {args.trace_file}")
    print(f"record_index: {args.record_index}")
    print(f"component: {record.get('component')}")
    print(f"timestamp: {record.get('ts')}")
    print(f"model: {record.get('model')}")
    print(f"lm_calls: {len(message_batches)}")

    extracted: dict[str, Any] = {
        "record_meta": {
            "component": record.get("component"),
            "ts": record.get("ts"),
            "model": record.get("model"),
        },
        "lm_calls": [],
    }

    for call_idx, messages in enumerate(message_batches):
        print(f"\n=== lm_call[{call_idx}] messages={len(messages)} ===")
        out_msgs: list[dict[str, Any]] = []
        for msg_idx, m in enumerate(messages):
            role = m.get("role", "<missing-role>") if isinstance(m, dict) else "<invalid>"
            content = ""
            if isinstance(m, dict):
                raw_content = m.get("content")
                content = raw_content if isinstance(raw_content, str) else str(raw_content)
            rendered = _render_text(content, max_chars=max_chars)
            print(f"[{msg_idx}] role={role} chars={len(content)}")
            print(rendered)
            print("-" * 80)
            out_msgs.append({"role": role, "content": content})
        extracted["lm_calls"].append({"messages": out_msgs})

    if args.dump_json is not None:
        args.dump_json.parent.mkdir(parents=True, exist_ok=True)
        with args.dump_json.open("w", encoding="utf-8") as f:
            json.dump(extracted, f, indent=2, ensure_ascii=False)
        print(f"\nWrote extracted payload JSON: {args.dump_json}")


if __name__ == "__main__":
    main()
