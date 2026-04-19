"""Per-action log-prob capture for Stage 1 actor rollouts.

Usage pattern (driven by experiment.capture_action_logprobs):

1. Enable capture at process startup via ``enable()`` (run_benchmark.py does
   this when the config flag is set).
2. ``build_dspy_lm`` wraps the LM with :class:`LogProbCapturingLM`, which
   forces ``logprobs=True`` on every underlying call and pushes per-completion
   summaries into a thread-local buffer.
3. The benchmark loop calls :func:`drain` immediately after each
   ``agent(state)`` call to attach whatever log-probs accumulated during that
   step into the rollout JSONL.

We intentionally do not raise on missing log-probs: vLLM may or may not return
them depending on server flags and DSPy's response adapter. A ``None`` entry
means "capture attempted but the response did not expose usable log-probs"
— which is strictly more informative than silently dropping them.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


_STATE = threading.local()
_ENABLED = False


def enable() -> None:
    """Globally enable log-prob capture for this process."""
    global _ENABLED
    _ENABLED = True


def is_enabled() -> bool:
    return _ENABLED


def _buffer() -> list[dict[str, Any]]:
    buf = getattr(_STATE, "buffer", None)
    if buf is None:
        buf = []
        _STATE.buffer = buf
    return buf


def push(entry: dict[str, Any]) -> None:
    """Append one capture entry (one per LM completion)."""
    _buffer().append(entry)


def drain() -> list[dict[str, Any]]:
    """Pop and return everything captured since the last drain."""
    buf = _buffer()
    out = list(buf)
    buf.clear()
    return out


def _extract_logprob_sum(response: Any) -> dict[str, Any] | None:
    """Extract per-completion log-prob summaries from a raw chat-completion
    response (LiteLLM / OpenAI schema).

    Returns ``None`` if the response does not carry usable log-probs.
    """
    try:
        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, dict):
            choices = response.get("choices")
        if not choices:
            return None

        summaries: list[dict[str, Any]] = []
        for choice in choices:
            logprobs_obj = (
                getattr(choice, "logprobs", None)
                if not isinstance(choice, dict)
                else choice.get("logprobs")
            )
            if logprobs_obj is None:
                summaries.append({"sum_logprob": None, "num_tokens": 0})
                continue
            content = (
                getattr(logprobs_obj, "content", None)
                if not isinstance(logprobs_obj, dict)
                else logprobs_obj.get("content")
            )
            if not content:
                summaries.append({"sum_logprob": None, "num_tokens": 0})
                continue
            lp_sum = 0.0
            n = 0
            for token in content:
                lp = (
                    getattr(token, "logprob", None)
                    if not isinstance(token, dict)
                    else token.get("logprob")
                )
                if lp is None:
                    continue
                lp_sum += float(lp)
                n += 1
            summaries.append({"sum_logprob": lp_sum if n else None, "num_tokens": n})
        return {"completions": summaries}
    except Exception as exc:  # defensive: never let logging break the run
        logger.debug(f"logprob extraction failed: {exc}")
        return None


class LogProbCapturingLM:
    """Thin wrapper around a ``dspy.LM``-compatible object that forces
    ``logprobs=True`` on the underlying request and captures per-call
    log-prob sums.

    Implemented as a subclass of ``dspy.BaseLM`` so it plugs in wherever the
    wrapped LM was used (see :class:`_ThinkStrippedLM` for the sibling
    pattern).
    """

    def __new__(cls, inner):  # noqa: D401 — factory so we can subclass BaseLM lazily
        # Import here to avoid importing dspy at module import time for
        # non-LLM agents.
        import dspy

        class _Impl(dspy.BaseLM):
            def __init__(self, lm):
                self._lm = lm

            def _inject(self, kwargs: dict[str, Any]) -> dict[str, Any]:
                # vLLM / OpenAI-compatible parameter.
                kwargs.setdefault("logprobs", True)
                return kwargs

            def _capture(self) -> None:
                try:
                    history = getattr(self._lm, "history", None)
                    if not history:
                        return
                    last = history[-1]
                    # LiteLLM stores the raw response under a couple of names
                    # depending on version — try both.
                    response = (
                        last.get("response")
                        if isinstance(last, dict)
                        else None
                    )
                    if response is None and isinstance(last, dict):
                        response = last.get("raw_response") or last.get("litellm_response")
                    summary = _extract_logprob_sum(response) if response else None
                    push(
                        {
                            "model": getattr(self._lm, "model", None),
                            "summary": summary,
                        }
                    )
                except Exception as exc:
                    logger.debug(f"logprob capture failed: {exc}")

            def __call__(self, *args, **kwargs):
                kwargs = self._inject(kwargs)
                out = self._lm(*args, **kwargs)
                self._capture()
                return out

            async def acall(self, *args, **kwargs):
                kwargs = self._inject(kwargs)
                out = await self._lm.acall(*args, **kwargs)
                self._capture()
                return out

            def copy(self, **kwargs):
                return LogProbCapturingLM(self._lm.copy(**kwargs))

            def __getattr__(self, name: str):
                return getattr(self._lm, name)

        return _Impl(inner)
