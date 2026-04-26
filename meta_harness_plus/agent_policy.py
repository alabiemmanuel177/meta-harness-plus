"""LLM-backed agent policies.

Wraps a chat-style LLM client into the ``AgentPolicy`` callable shape
``(goal, history, last_observation) -> (action, est_tokens)``.

Two implementations:

- ``llm_agent_policy``: builds a single chat prompt with goal + history
  and asks for the next shell command. Uses few-shot examples + a
  short system prompt that constrains the response to a single command.
- ``RuleBasedPolicy``: deterministic offline policy keyed on goal-text
  prefixes. Useful for unit-test smoke-runs without API calls.

The LLM-backed policy is intentionally minimal — its purpose is to plug
into MH++'s search machinery, not to be a state-of-the-art agent. The
real research contribution is the *search over harnesses* containing
this policy + retrievers + reflectors + voters.
"""

from __future__ import annotations

import re
from typing import Callable

from .agent import Turn


# ----------------- prompt builder -----------------

_SYSTEM = """You are a careful shell-using agent. Each turn, output exactly ONE shell command to run, and nothing else — no markdown, no commentary.
When the task is complete, output exactly the literal string: <DONE>
Use only standard POSIX tools and python3. Stay inside the current working directory.
Never run destructive commands (rm -rf /, shutdown, mkfs)."""


def _format_history(history: list[Turn], max_turns: int = 4) -> str:
    """Render the last ``max_turns`` turns as a compact prompt block."""
    if not history:
        return ""
    recent = history[-max_turns:]
    out: list[str] = []
    for t in recent:
        out.append(f"Turn {t.turn_idx}:")
        out.append(f"  $ {t.action}")
        if t.stdout:
            out.append(f"  stdout: {t.stdout.rstrip()[:300]}")
        if t.stderr:
            out.append(f"  stderr: {t.stderr.rstrip()[:300]}")
        out.append(f"  exit: {t.exit_code}")
    return "\n".join(out)


def _build_prompt(goal: str, history: list[Turn], last_obs: str) -> str:
    parts = [
        f"Goal: {goal.strip()}",
        "",
        "Recent history:",
        _format_history(history) or "  (no turns yet)",
        "",
    ]
    if last_obs and (not history or last_obs != history[-1].stdout):
        parts.extend(["Last observation:", last_obs.rstrip()[:600], ""])
    parts.append("Next shell command (or <DONE>):")
    return "\n".join(parts)


def _extract_command(text: str) -> str:
    """Pull a single command out of an LLM completion.

    Strategy:
    - Strip leading/trailing whitespace.
    - If response is wrapped in ``` fences, take the inside.
    - If the literal token ``<DONE>`` appears (anywhere on its own
      line), treat as DONE.
    - Otherwise take the first non-empty line.
    """
    s = text.strip()
    if not s:
        return "<DONE>"
    # Strip common code fence patterns.
    fence = re.match(r"^```(?:bash|sh|shell)?\s*\n(.*?)\n```\s*$", s, flags=re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    # Token-style DONE.
    for line in s.splitlines():
        if line.strip() == "<DONE>":
            return "<DONE>"
    # First non-empty line is the command.
    for line in s.splitlines():
        line = line.strip()
        if line:
            # Strip a leading "$ " prompt if present.
            if line.startswith("$ "):
                line = line[2:].strip()
            return line
    return "<DONE>"


# ----------------- LLM-backed policy -----------------

# A "chat fn" is anything that takes (system, user) -> (response_text, tokens).
ChatFn = Callable[[str, str], tuple[str, int]]


def llm_agent_policy(chat_fn: ChatFn):
    """Return an AgentPolicy that calls ``chat_fn(system, user)`` per turn."""

    def policy(goal: str, history: list[Turn], last_obs: str) -> tuple[str, int]:
        user = _build_prompt(goal, history, last_obs)
        resp, tokens = chat_fn(_SYSTEM, user)
        action = _extract_command(resp)
        return (action, tokens)

    return policy


# ----------------- rule-based offline policy -----------------

class RuleBasedPolicy:
    """Deterministic policy keyed on substrings in the goal.

    ``rules`` is a list of (substring_pattern, [step1, step2, ...])
    pairs. The first matching rule's steps are emitted in order, then
    ``<DONE>``. Useful for end-to-end smoke tests without API costs.
    """

    def __init__(self, rules: list[tuple[str, list[str]]]):
        self.rules = list(rules)

    def __call__(self, goal: str, history: list[Turn], last_obs: str) -> tuple[str, int]:
        for needle, steps in self.rules:
            if needle in goal:
                i = len(history)
                if i < len(steps):
                    return (steps[i], 10)
                return ("<DONE>", 1)
        return ("<DONE>", 1)
