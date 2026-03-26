#!/usr/bin/env python3
"""
Post-processing script to generate per-step causal summaries of LLM orchestrator traces.

For each orchestrator step in a completed episode, produces a structured explanation:
  "Given [context], the agent chose to explore [families] / avoid [families] because [reason]"

Self-reflection entries at episode end are both:
  - preserved verbatim, and
  - summarized into structured causal "because -> do/avoid" action guidance.

Usage:
    # Single experiment directory (recursively finds all episode_*.jsonl files)
    uv run scripts/summarize_causal_traces.py \\
        --trace-dir results/my_experiment/llm_traces \\
        --output-dir results/my_experiment/causal_summaries \\
        --model openai/Qwen/Qwen3.5-122B-A10B-FP8 \\
        --base-url http://127.0.0.1:8001/v1

    # Multiple experiments at once
    uv run scripts/summarize_causal_traces.py \\
        --trace-dir results/ \\
        --output-dir results/causal_summaries/ \\
        --batch-size 16 --overwrite

Output per episode:
    {
      "system_id": "Mg-Sn-Sr",
      "episode_id": "0",
      "steps": [
        {
          "step": 1,
          "evaluation_history_len": 0,
          "context_summary": "...",
          "decision": "...",
          "causal_reasoning": "...",
          "families_pursued": [...],
          "families_deprioritized": [...]
        }, ...
      ],
      "episode_reflection": "<verbatim self-reflection text>",
      "reflection_causal_summaries": [
        {
          "reflection_index": 1,
          "context_summary": "...",
          "reflection_summary": "...",
          "causal_reasoning": "...",
          "actions_recommended": [...],
          "actions_avoided": [...],
          "families_pursued": [...],
          "families_deprioritized": [...]
        }
      ]
    }
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

try:
    from openai import AsyncOpenAI
except ImportError:
    print("ERROR: openai package not found. Install with: uv add openai", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Global limits
# ---------------------------------------------------------------------------

# Set any value to None to disable that limit.
TRAJECTORY_TOOL_ARGS_MAX_CHARS: int | None = None
TRAJECTORY_OBSERVATION_MAX_CHARS: int | None = None

STEP_CONTEXT_FIELD_MAX_CHARS: int | None = None
STEP_REASONING_MAX_CHARS: int | None = None
STEP_ANSWER_MAX_CHARS: int | None = None
STEP_TRAJECTORY_MAX_CHARS: int | None = None
STEP_SUMMARY_MAX_TOKENS: int | None = None

REFLECTION_CONTEXT_OTHER_FIELD_MAX_CHARS: int | None = None
REFLECTION_CONTEXT_EPISODE_TRAJECTORY_MAX_CHARS: int | None = None
REFLECTION_PROMPT_CONTEXT_MAX_CHARS: int | None = None
REFLECTION_PROMPT_TEXT_MAX_CHARS: int | None = None
REFLECTION_SUMMARY_MAX_TOKENS: int | None = None


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

CAUSAL_SUMMARY_SYSTEM = """\
You are analyzing the reasoning of a materials discovery AI agent.
The agent iteratively proposes crystal structures to find stable phases on a phase diagram (convex hull).
It uses a ReAct loop: it examines its evaluation history, generates/scores structures, then selects which to evaluate with an oracle.

Your task: produce a concise CAUSAL summary of ONE decision step — why did the agent choose to explore certain composition families and avoid others?

Respond ONLY with a JSON object (no markdown fences) with these exact fields:
  "context_summary"      – Key information available at this step (evaluation history highlights, prior reflections). 2-3 sentences max.
  "decision"             – What the agent actually did: which compositions were generated, scored, selected for evaluation.
  "causal_reasoning"     – The causal link: WHY that decision followed from the context. 2-4 sentences.
  "families_pursued"     – JSON array of composition families (strings) the agent chose to prioritize/generate more of.
  "families_deprioritized" – JSON array of composition families the agent chose to reduce/skip.
"""

CAUSAL_SUMMARY_USER = """\
=== CONTEXT AT THIS STEP ===
Evaluations completed so far: {eval_count}
{context_fields}

=== AGENT TRAJECTORY ===
{trajectory}

=== AGENT FINAL REASONING ===
{reasoning}

=== AGENT ANSWER ===
{answer}
"""

REFLECTION_CAUSAL_SYSTEM = """\
You are analyzing the self-reflection of a materials discovery AI agent at the end of an episode.
The reflection should encode strategy updates for the next episode based on what worked or failed.

Your task: produce a concise CAUSAL summary of ONE self-reflection entry:
"Because of [episode evidence], the reflection suggests doing [actions] and avoiding [actions]."

Respond ONLY with a JSON object (no markdown fences) with these exact fields:
  "context_summary"      – Key episode evidence or outcomes that motivated this reflection. 2-3 sentences max.
  "reflection_summary"   – What the reflection is advising overall.
  "causal_reasoning"     – WHY these recommendations follow from the evidence. 2-4 sentences.
  "actions_recommended"  – JSON array of concrete next-episode actions to prioritize.
  "actions_avoided"      – JSON array of concrete actions/family directions to reduce or avoid.
  "families_pursued"     – JSON array of composition families (strings) to prioritize next episode.
  "families_deprioritized" – JSON array of composition families (strings) to reduce/avoid next episode.
"""

REFLECTION_CAUSAL_USER = """\
=== EPISODE CONTEXT ===
{context_fields}

=== RAW SELF-REFLECTION ===
{reflection}
"""


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_dspy_fields(content: str) -> dict[str, str]:
    """Extract [[ ## field_name ## ]] blocks from a DSPy-formatted prompt string."""
    pattern = r"\[\[\s*##\s*(\w+)\s*##\s*\]\]\s*(.*?)(?=\[\[\s*##|\Z)"
    matches = re.findall(pattern, content, re.DOTALL)
    return {k: v.strip() for k, v in matches}


def truncate_text(text: str, max_chars: int | None) -> str:
    """Truncate text with ellipsis; None disables truncation."""
    if max_chars is None:
        return text
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3] + "..."


def format_trajectory(trajectory: dict) -> str:
    """Convert trajectory dict (thought_N / tool_name_N / tool_args_N / observation_N) to readable text."""
    lines: list[str] = []
    i = 0
    while f"thought_{i}" in trajectory or f"tool_name_{i}" in trajectory:
        thought = trajectory.get(f"thought_{i}", "")
        tool = trajectory.get(f"tool_name_{i}", "")
        args = trajectory.get(f"tool_args_{i}", {})
        obs = trajectory.get(f"observation_{i}", "")

        if thought:
            lines.append(f"[Thought {i + 1}] {thought}")
        if tool:
            args_str = json.dumps(args, default=str)
            args_str = truncate_text(args_str, TRAJECTORY_TOOL_ARGS_MAX_CHARS)
            lines.append(f"[Action {i + 1}] {tool}({args_str})")
        if obs:
            obs_str = str(obs)
            obs_str = truncate_text(obs_str, TRAJECTORY_OBSERVATION_MAX_CHARS)
            lines.append(f"[Observation {i + 1}] {obs_str}")
        i += 1

    return "\n".join(lines)


def build_context_fields(entry: dict) -> str:
    """Extract DSPy input fields from lm_calls if available, else fall back to extra fields."""
    lm_calls = entry.get("extra", {}).get("lm_calls", [])
    if lm_calls:
        messages = lm_calls[0].get("messages", [])
        # The user message (index 1) contains the DSPy field values
        user_content = next(
            (m.get("content", "") for m in messages if m.get("role") == "user"),
            "",
        )
        if user_content:
            fields = parse_dspy_fields(user_content)
            sections: list[str] = []
            for key in ("evaluation_history", "buffer_summary", "prior_reflections", "known_stable_materials"):
                val = fields.get(key, "")
                if val and val.lower() not in ("none", "no evaluations yet.", ""):
                    val = truncate_text(val, STEP_CONTEXT_FIELD_MAX_CHARS)
                    sections.append(f"[{key}]\n{val}")
            return "\n\n".join(sections)

    # Fallback: use extra metadata
    extra = entry.get("extra", {})
    parts = []
    if extra.get("evaluation_history_len", 0) == 0:
        parts.append("evaluation_history: No evaluations yet.")
    else:
        parts.append(f"evaluation_history_len: {extra['evaluation_history_len']}")
    if "buffer_compositions" in extra:
        parts.append(f"buffer_compositions: {extra['buffer_compositions']}")
    return "\n".join(parts)


def build_reflection_context_fields(entry: dict) -> str:
    """Extract reflection-time DSPy fields from lm_call if available, else fallback metadata."""
    lm_call = entry.get("extra", {}).get("lm_call", {})
    messages = lm_call.get("messages", [])
    user_content = next(
        (m.get("content", "") for m in messages if m.get("role") == "user"),
        "",
    )
    if user_content:
        fields = parse_dspy_fields(user_content)
        sections: list[str] = []
        for key in ("chemical_system", "episode_outcome", "prior_reflections", "episode_trajectory"):
            val = fields.get(key, "")
            if val and val.lower() not in ("none", "none (this is the first episode).", ""):
                # episode_trajectory can be very long; trim aggressively
                max_len = (
                    REFLECTION_CONTEXT_EPISODE_TRAJECTORY_MAX_CHARS
                    if key == "episode_trajectory"
                    else REFLECTION_CONTEXT_OTHER_FIELD_MAX_CHARS
                )
                val = truncate_text(val, max_len)
                sections.append(f"[{key}]\n{val}")
        if sections:
            return "\n\n".join(sections)

    run = entry.get("run", {})
    fallback = []
    if run.get("system_id"):
        fallback.append(f"system_id: {run['system_id']}")
    if run.get("episode_id"):
        fallback.append(f"episode_id: {run['episode_id']}")
    reflection = str(entry.get("output", {}).get("reflection", ""))
    if reflection:
        fallback.append(f"raw_reflection_len: {len(reflection)} chars")
    return "\n".join(fallback)


def parse_json_response(raw: str) -> dict:
    """Parse model response as JSON, with fence-stripping and fallback."""
    text = raw.strip()

    # Some models emit hidden reasoning wrapped in <think>...</think>.
    # Keep only the content after the closing tag before JSON parsing.
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    # Strip markdown fences if present
    if text.startswith("```"):
        text = re.sub(r"```(?:json)?\n?", "", text).strip("` \n")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: try to recover a JSON object embedded in extra text.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        return {"raw_text": text, "parse_error": True}


def ensure_list_field(obj: dict, key: str) -> None:
    """Normalize a possibly-missing/malformed list field to a list of strings."""
    val = obj.get(key, [])
    if isinstance(val, list):
        obj[key] = [str(v) for v in val]
        return
    if isinstance(val, str):
        obj[key] = [val] if val.strip() else []
        return
    obj[key] = []


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

async def summarize_step(
    client: AsyncOpenAI,
    model: str,
    entry: dict,
    step_number: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Call the LLM to produce a causal summary for one orchestrator step."""
    output = entry.get("output", {})
    trajectory_text = format_trajectory(output.get("trajectory", {}))
    reasoning = truncate_text(str(output.get("reasoning", "")), STEP_REASONING_MAX_CHARS)
    answer = truncate_text(str(output.get("answer", "")), STEP_ANSWER_MAX_CHARS)
    eval_count = entry.get("extra", {}).get("evaluation_history_len", 0)
    context_fields = build_context_fields(entry)

    user_msg = CAUSAL_SUMMARY_USER.format(
        eval_count=eval_count,
        context_fields=context_fields,
        trajectory=truncate_text(trajectory_text, STEP_TRAJECTORY_MAX_CHARS),
        reasoning=reasoning,
        answer=answer,
    )

    async with semaphore:
        kwargs = {
            "model": model.removeprefix("openai/"),
            "messages": [
                {"role": "system", "content": CAUSAL_SUMMARY_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.1,
        }
        if STEP_SUMMARY_MAX_TOKENS is not None:
            kwargs["max_tokens"] = STEP_SUMMARY_MAX_TOKENS
        response = await client.chat.completions.create(**kwargs)

    raw = response.choices[0].message.content or ""
    summary = parse_json_response(raw)

    return {
        "step": step_number,
        "ts": entry.get("ts", ""),
        "evaluation_history_len": eval_count,
        **summary,
    }


async def summarize_reflection(
    client: AsyncOpenAI,
    model: str,
    entry: dict,
    reflection_index: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Call the LLM to produce a causal summary for one self-reflection entry."""
    reflection_text = str(entry.get("output", {}).get("reflection", ""))
    context_fields = build_reflection_context_fields(entry)

    user_msg = REFLECTION_CAUSAL_USER.format(
        context_fields=truncate_text(context_fields, REFLECTION_PROMPT_CONTEXT_MAX_CHARS),
        reflection=truncate_text(reflection_text, REFLECTION_PROMPT_TEXT_MAX_CHARS),
    )

    async with semaphore:
        kwargs = {
            "model": model.removeprefix("openai/"),
            "messages": [
                {"role": "system", "content": REFLECTION_CAUSAL_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.1,
        }
        if REFLECTION_SUMMARY_MAX_TOKENS is not None:
            kwargs["max_tokens"] = REFLECTION_SUMMARY_MAX_TOKENS
        response = await client.chat.completions.create(**kwargs)

    raw = response.choices[0].message.content or ""
    summary = parse_json_response(raw)
    if not summary.get("parse_error"):
        for key in (
            "actions_recommended",
            "actions_avoided",
            "families_pursued",
            "families_deprioritized",
        ):
            ensure_list_field(summary, key)

    return {
        "reflection_index": reflection_index,
        "ts": entry.get("ts", ""),
        **summary,
    }


# ---------------------------------------------------------------------------
# Episode processing
# ---------------------------------------------------------------------------

async def process_episode(
    client: AsyncOpenAI,
    model: str,
    trace_file: Path,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Process all orchestrator steps in one episode trace file."""
    entries: list[dict] = []
    with open(trace_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    orchestrator_entries = [e for e in entries if e.get("component") == "orchestrator"]
    reflection_entries = [e for e in entries if e.get("component") == "self_reflection"]

    # Summarize all orchestrator steps concurrently (bounded by semaphore)
    step_tasks = [
        summarize_step(client, model, entry, i + 1, semaphore)
        for i, entry in enumerate(orchestrator_entries)
    ]
    reflection_tasks = [
        summarize_reflection(client, model, entry, i + 1, semaphore)
        for i, entry in enumerate(reflection_entries)
    ]

    step_summaries = await asyncio.gather(*step_tasks) if step_tasks else []
    reflection_summaries = await asyncio.gather(*reflection_tasks) if reflection_tasks else []

    # Keep original self-reflection text for backward compatibility.
    episode_reflection: str | None = None
    if reflection_entries:
        episode_reflection = reflection_entries[0].get("output", {}).get("reflection")

    # Infer system/episode id from run metadata or file path
    if orchestrator_entries:
        run_meta = orchestrator_entries[0].get("run", {})
    elif reflection_entries:
        run_meta = reflection_entries[0].get("run", {})
    else:
        run_meta = {}
    system_id = run_meta.get("system_id") or trace_file.parent.name
    episode_id = run_meta.get("episode_id") or trace_file.stem.replace("episode_", "")
    if orchestrator_entries:
        model_used = orchestrator_entries[0].get("model", "")
    elif reflection_entries:
        model_used = reflection_entries[0].get("model", "")
    else:
        model_used = ""

    return {
        "system_id": system_id,
        "episode_id": episode_id,
        "model": model_used,
        "trace_file": str(trace_file),
        "num_steps": len(step_summaries),
        "num_reflections": len(reflection_summaries),
        "steps": step_summaries,
        "episode_reflection": episode_reflection,
        "reflection_causal_summaries": reflection_summaries,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> None:
    trace_dir = Path(args.trace_dir)
    output_dir = Path(args.output_dir)

    api_key = os.environ.get(args.api_key_env, "EMPTY")
    client = AsyncOpenAI(base_url=args.base_url, api_key=api_key)
    semaphore = asyncio.Semaphore(args.batch_size)

    # Find all episode trace files (skip Jupyter checkpoint copies)
    trace_files = sorted(
        p for p in trace_dir.rglob("episode_*.jsonl")
        if ".ipynb_checkpoints" not in str(p)
    )
    if not trace_files:
        print(f"No trace files found under {trace_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(trace_files)} episode trace file(s)")

    for trace_file in trace_files:
        # Mirror the input directory structure under output_dir
        relative = trace_file.relative_to(trace_dir)
        out_file = output_dir / relative.parent / (trace_file.stem + "_causal_summary.json")
        out_file.parent.mkdir(parents=True, exist_ok=True)

        if out_file.exists() and not args.overwrite:
            print(f"  skip  {relative}  (already processed, use --overwrite)")
            continue

        print(f"  processing  {relative} ...", end=" ", flush=True)
        try:
            summary = await process_episode(client, args.model, trace_file, semaphore)
        except Exception as exc:
            print(f"ERROR: {exc}")
            continue

        with open(out_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"done  ({summary['num_steps']} steps → {out_file})")

    print("\nAll done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate per-step causal summaries of MADE orchestrator traces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--trace-dir",
        required=True,
        help="Root directory containing episode_*.jsonl files (searched recursively)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for causal summary JSON files",
    )
    parser.add_argument(
        "--model",
        default="openai/Qwen/Qwen3.5-122B-A10B-FP8",
        help="Model name (with openai/ prefix for DSPy compatibility). Default: %(default)s",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8001/v1",
        help="vLLM / OpenAI-compatible API base URL. Default: %(default)s",
    )
    parser.add_argument(
        "--api-key-env",
        default="VLLM_API_KEY",
        metavar="ENV_VAR",
        help="Environment variable holding the API key. Default: %(default)s",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Max concurrent LLM requests. Default: %(default)s",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-process episodes that already have a summary file",
    )
    asyncio.run(main(parser.parse_args()))
