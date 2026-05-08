"""LLM client routing — model name → provider dispatch.

Per V10_DESIGN.md §11.2 (model substitution): V10 is model-agnostic at
the call site. This module is the ONLY place provider-specific SDKs
get imported. Business logic always calls ``complete_chat(...)`` with
a role label; the role's model is looked up from
``harness/config/models.yaml`` (with env-var overrides).

Routing rules (extend ``_route_model`` to add providers):
  - model name starts with "claude"   → Anthropic SDK
  - model name starts with "deepseek" → OpenAI-compatible endpoint at
                                        https://api.deepseek.com/v1
  - other openai/openrouter/etc. variants: future work, not in V10
"""

from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass
from typing import Any, Optional

import yaml


_CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent / "config" / "models.yaml"


@dataclass
class ChatResult:
    """Picklable response shape from ``complete_chat``."""
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    raw: Any = None  # provider-native response, useful for debugging


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_config() -> dict:
    """Read harness/config/models.yaml. Cached on first call."""
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"models.yaml not found at {_CONFIG_PATH}. "
            "V10 cannot dispatch LLM calls without role->model mapping."
        )
    return yaml.safe_load(_CONFIG_PATH.read_text())


_CACHED_CONFIG: dict | None = None


def get_config() -> dict:
    global _CACHED_CONFIG
    if _CACHED_CONFIG is None:
        _CACHED_CONFIG = _load_config()
    return _CACHED_CONFIG


_OPUS_PATCH_GEN_ROLES: tuple[str, ...] = (
    "patch_generator_pipeline",
    "patch_generator_agent",
)
_OPUS_PATCH_GEN_MODEL: str = "claude-opus-4-7"

# Models for which the Anthropic API rejects the `temperature` kwarg
# (extended-thinking-capable Opus 4.x). Match by prefix so future
# 4.x variants are auto-covered.
_ANTHROPIC_NO_TEMPERATURE_PREFIXES: tuple[str, ...] = (
    "claude-opus-4-",
)


def _temperature_deprecated_for(model: str) -> bool:
    return any(model.startswith(p) for p in _ANTHROPIC_NO_TEMPERATURE_PREFIXES)


def model_for_role(role: str) -> str:
    """Map a role label to a model name.

    Lookup order:
      1. Env var ``V10_<ROLE_UPPER>_MODEL`` (e.g. ``V10_RERANKER_MODEL``)
      2. Meta-flag ``V10_USE_OPUS_PATCH_GEN=1`` — when set, the two
         patch-gen roles (patch_generator_pipeline,
         patch_generator_agent) resolve to ``claude-opus-4-7`` unless
         their explicit per-role env override is already set. All
         other roles are unaffected. Used by the P3f Opus ablation
         (V10_DESIGN.md §11 stepped rollout).
      3. ``roles.<role>`` in models.yaml
      4. KeyError if neither is set.
    """
    env_key = f"V10_{role.upper()}_MODEL"
    env_val = os.environ.get(env_key)
    if env_val:
        return env_val
    if role in _OPUS_PATCH_GEN_ROLES and os.environ.get("V10_USE_OPUS_PATCH_GEN"):
        return _OPUS_PATCH_GEN_MODEL
    cfg = get_config()
    if role in cfg.get("roles", {}):
        return cfg["roles"][role]
    raise KeyError(
        f"role {role!r} not in V10 model config; check {_CONFIG_PATH} "
        f"or set {env_key} in env."
    )


def price_for_model(model: str) -> dict:
    """Return {'input': $/1M, 'output': $/1M} for the given model.
    Falls back to a generic estimate if unknown."""
    prices = get_config().get("prices", {})
    if model in prices:
        return prices[model]
    return {"input": 1.0, "output": 3.0}  # generic


# ---------------------------------------------------------------------------
# Routing + dispatch
# ---------------------------------------------------------------------------


def _route_model(model: str) -> str:
    """Return the provider key for a given model name."""
    name = model.lower()
    if name.startswith("claude"):
        return "anthropic"
    if name.startswith("deepseek"):
        return "deepseek"
    raise ValueError(
        f"unknown model provider for {model!r}; extend "
        f"harness.llm.clients._route_model"
    )


def complete_chat(
    *,
    messages: list[dict],
    role: str | None = None,
    model: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    response_format_json: bool = False,
) -> ChatResult:
    """Send a chat completion request. Caller must supply EITHER
    ``role`` (looked up via config) OR ``model`` (explicit name).

    ``messages`` is the standard chat shape:
      [{"role": "system", "content": "..."},
       {"role": "user", "content": "..."}]

    ``response_format_json=True`` requests JSON-only output where the
    provider supports it (deepseek-chat does; Claude sonnet uses
    a system-prompt-instructed convention).
    """
    if role is None and model is None:
        raise ValueError("complete_chat requires role= or model=")
    if model is None:
        model = model_for_role(role)
    provider = _route_model(model)
    if provider == "deepseek":
        return _call_deepseek(
            messages=messages, model=model, max_tokens=max_tokens,
            temperature=temperature, response_format_json=response_format_json,
        )
    if provider == "anthropic":
        return _call_anthropic(
            messages=messages, model=model, max_tokens=max_tokens,
            temperature=temperature, response_format_json=response_format_json,
        )
    raise AssertionError(f"unreachable provider: {provider}")


def _call_deepseek(
    *, messages: list[dict], model: str, max_tokens: int, temperature: float,
    response_format_json: bool,
) -> ChatResult:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY not set. Either export it in env or "
            "switch the role's model to a Claude variant via "
            "harness/config/models.yaml or V10_<ROLE>_MODEL env override."
        )
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    kwargs: dict = dict(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if response_format_json:
        kwargs["response_format"] = {"type": "json_object"}
    res = client.chat.completions.create(**kwargs)
    text = res.choices[0].message.content or ""
    usage = res.usage
    return ChatResult(
        text=text,
        model=model,
        input_tokens=getattr(usage, "prompt_tokens", 0),
        output_tokens=getattr(usage, "completion_tokens", 0),
        raw=res,
    )


def _call_anthropic(
    *, messages: list[dict], model: str, max_tokens: int, temperature: float,
    response_format_json: bool,
) -> ChatResult:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Either export it in env or "
            "switch the role's model to a deepseek variant via "
            "harness/config/models.yaml or V10_<ROLE>_MODEL env override."
        )
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    # Anthropic separates system prompts; pull them out.
    system_msg: Optional[str] = None
    chat_msgs: list[dict] = []
    for m in messages:
        if m["role"] == "system":
            system_msg = m["content"]
        else:
            chat_msgs.append({"role": m["role"], "content": m["content"]})
    # JSON-only: prepend instruction to the system prompt.
    if response_format_json:
        instr = "Respond with valid JSON only. No prose, no markdown fences."
        system_msg = f"{system_msg}\n\n{instr}" if system_msg else instr
    kwargs: dict = dict(
        model=model,
        max_tokens=max_tokens,
        messages=chat_msgs,
    )
    # Anthropic deprecated the `temperature` parameter on the Opus 4.x
    # extended-thinking-capable models. The API returns 400
    # "temperature is deprecated for this model" if we pass it. Detect
    # via prefix and omit. All older Claude models still accept it.
    if not _temperature_deprecated_for(model):
        kwargs["temperature"] = temperature
    if system_msg:
        kwargs["system"] = system_msg
    res = client.messages.create(**kwargs)
    text = "".join(b.text for b in res.content if hasattr(b, "text"))
    return ChatResult(
        text=text,
        model=model,
        input_tokens=res.usage.input_tokens,
        output_tokens=res.usage.output_tokens,
        raw=res,
    )


__all__ = [
    "ChatResult",
    "complete_chat",
    "model_for_role",
    "price_for_model",
    "get_config",
]
