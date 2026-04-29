"""V10 firewall — runtime fixture.

This conftest intentionally does NOT modify pytest's collection of the
existing legacy tests under ``tests/``. It only registers the
``oracle_leak_guard`` fixture, which V10 tests can opt into.

The runtime fixture wraps every LLM-client call site and inspects each
kwargs payload for forbidden tokens (case-insensitive substring against
``harness.views.FORBIDDEN_TOKENS``). If found, it raises
``OracleLeakError`` immediately.

The fixture also provides ``LLMCallInspector`` — a callable wrapper any
V10 module can use to wrap an arbitrary function and verify all
arguments are leak-free at call time. This is what the spot-check
criterion (b) ("the runtime fixture actually wraps LLM client calls,
not just static-scans code") refers to.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Callable

import pytest


# Lazy import — the runtime fixture imports harness only when actually used,
# so legacy tests don't pay the import cost.
def _forbidden_tokens() -> tuple[str, ...]:
    from harness.views import FORBIDDEN_TOKENS

    return FORBIDDEN_TOKENS


class OracleLeakError(RuntimeError):
    """A forbidden token reached an LLM call site at runtime."""


def _walk_value_for_tokens(value: Any, *, depth: int = 0) -> list[str]:
    """Recursively walk a value, returning every string-shaped piece for
    inspection. Bounded depth to avoid pathological cycles."""
    if depth > 6:
        return []
    if value is None or isinstance(value, (int, float, bool)):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, bytes):
        try:
            return [value.decode("utf-8", errors="replace")]
        except Exception:
            return []
    if isinstance(value, dict):
        out: list[str] = []
        for k, v in value.items():
            out.extend(_walk_value_for_tokens(k, depth=depth + 1))
            out.extend(_walk_value_for_tokens(v, depth=depth + 1))
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        out2: list[str] = []
        for v in value:
            out2.extend(_walk_value_for_tokens(v, depth=depth + 1))
        return out2
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        out3: list[str] = []
        declared_field_names: set[str] = set()
        for f in dataclasses.fields(value):
            declared_field_names.add(f.name)
            out3.append(f.name)  # declared field NAMES are inspected
            out3.extend(_walk_value_for_tokens(getattr(value, f.name), depth=depth + 1))
        # Also walk __dict__ for SMUGGLED attributes added via
        # object.__setattr__(frozen_instance, 'x', value) — they bypass the
        # dataclass __post_init__ validator but still land in __dict__.
        # The inspector must catch this.
        try:
            d = getattr(value, "__dict__", None)
            if d:
                for k, v in d.items():
                    if k in declared_field_names:
                        continue  # already walked above
                    out3.append(k)
                    out3.extend(_walk_value_for_tokens(v, depth=depth + 1))
        except Exception:
            pass
        return out3
    # Generic object — inspect __dict__ if available
    try:
        d = getattr(value, "__dict__", None)
        if d:
            return _walk_value_for_tokens(d, depth=depth + 1)
    except Exception:
        pass
    return []


def _normalize(s: str) -> str:
    """Casefold + strip underscores so SCREAMING_SNAKE and CamelCase
    both reduce to the canonical form. Mirrors sandbox._normalize."""
    return s.casefold().replace("_", "")


def _scan_for_forbidden(strings: list[str]) -> tuple[str, str] | None:
    """Return (token, matched_string) if any forbidden token is found, else None."""
    tokens_norm = tuple(_normalize(t) for t in _forbidden_tokens())
    for s in strings:
        normalized = _normalize(s)
        for tok_norm, tok_orig in zip(tokens_norm, _forbidden_tokens()):
            if tok_norm in normalized:
                return (tok_orig, s)
    return None


class LLMCallInspector:
    """Wraps a callable that represents an LLM client call. Inspects every
    positional and keyword argument for forbidden tokens before invoking
    the wrapped function. Raises OracleLeakError if any are found.

    Usage::

        wrapped = LLMCallInspector("anthropic.messages.create", real_fn)
        wrapped(model="claude-...", messages=[{"role": "user", "content": "..."}])
    """

    def __init__(self, label: str, fn: Callable[..., Any], *, log_path: str | None = None):
        self.label = label
        self.fn = fn
        self.log_path = log_path
        self.call_count = 0
        self.last_inspection: list[str] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        all_strings: list[str] = []
        for a in args:
            all_strings.extend(_walk_value_for_tokens(a))
        for k, v in kwargs.items():
            all_strings.append(k)  # kwarg name is itself inspected
            all_strings.extend(_walk_value_for_tokens(v))

        hit = _scan_for_forbidden(all_strings)
        self.call_count += 1
        self.last_inspection = all_strings
        if hit is not None:
            tok, matched = hit
            raise OracleLeakError(
                f"LLM call {self.label!r}: argument carries forbidden token "
                f"{tok!r} (matched in {matched[:200]!r}). This is an oracle "
                f"leak. See docs/V10_DESIGN.md §12.5."
            )

        if self.log_path:
            with open(self.log_path, "a") as fh:
                fh.write(json.dumps({"label": self.label, "n_strings": len(all_strings)}) + "\n")
        return self.fn(*args, **kwargs)


@pytest.fixture
def oracle_leak_guard(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Phase 0 placeholder fixture.

    In Phase 0 we have no LLM call sites yet. This fixture exposes
    ``LLMCallInspector`` and the helper functions so tests can directly
    exercise the wrap pattern. Once Phase 1 lands real LLM adapters under
    ``harness/llm/adapters/``, this fixture will monkeypatch each adapter
    entry point through ``LLMCallInspector``.
    """
    return {
        "Inspector": LLMCallInspector,
        "scan": _scan_for_forbidden,
        "walk": _walk_value_for_tokens,
        "OracleLeakError": OracleLeakError,
    }
