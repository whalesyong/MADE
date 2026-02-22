"""
Helpers for constructing DSPy LMs from Hydra/OmegaConf-friendly config objects.
"""

import os
from typing import Any

import dspy


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read a key from dict-like or attribute-style config objects."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def build_dspy_lm(llm_config: Any) -> dspy.LM:
    """Build a DSPy LM with support for OpenAI-compatible local endpoints (e.g. vLLM)."""
    model = _cfg_get(llm_config, "model", "openai/gpt-4o-mini")
    cache = _cfg_get(llm_config, "cache", True)
    max_tokens = _cfg_get(llm_config, "max_output_tokens")
    temperature = _cfg_get(llm_config, "temperature")

    # Support either `base_url` (user-facing) or `api_base` (LiteLLM-style).
    api_base = _cfg_get(llm_config, "api_base") or _cfg_get(llm_config, "base_url")

    # Allow explicit api_key or env-driven key lookup.
    api_key = _cfg_get(llm_config, "api_key")
    api_key_env = _cfg_get(llm_config, "api_key_env")
    if api_key is None and api_key_env:
        api_key = os.getenv(api_key_env)

    kwargs: dict[str, Any] = {"cache": cache}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature
    if api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key

    extra_kwargs = _cfg_get(llm_config, "extra_kwargs", {}) or {}
    kwargs.update(dict(extra_kwargs))

    return dspy.LM(model, **kwargs)
