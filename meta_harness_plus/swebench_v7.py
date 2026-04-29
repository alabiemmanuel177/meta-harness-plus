"""V7 SWE-bench harness pieces: reviewer-actor verification and audit.

The v5 path is a single actor loop. V7 adds an independent reviewer pass
that critiques the actor patch, proposes edge-case reproductions, and can
flag benchmark/test-suite defects as first-class artifacts.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from .agent_swebench_loop import Trajectory
from .swebench_adapter import SWEBenchInstance


V7_ACTOR_SYSTEM_PROMPT = (
    "You are an expert Python developer fixing a real-world SWE-bench bug "
    "inside /testbed. Your goal is a small production-code patch that passes "
    "the task without side effects.\n\n"
    "V7 operating rules:\n"
    "1. Start with a focused reproduction whenever the hidden FAIL_TO_PASS "
    "selector is absent, noisy, or not collectable. Use run_reproduction for "
    "a durable script or run_python for a very small assertion.\n"
    "2. Use structural tools before broad text search when tracing code: "
    "go_to_definition and find_references are available for Python symbols.\n"
    "3. Prefer replace_text over write_file. If state-space recovery restores "
    "a checkpoint, switch hypothesis; do not repeat the same edit.\n"
    "4. Run FAIL_TO_PASS when collectable. If not collectable, run the "
    "reproduction and nearby existing tests.\n"
    "5. If you have strong evidence the benchmark test is wrong, call "
    "propose_test_fix with a precise reason and test patch. Still keep the "
    "submitted production patch clean.\n\n"
    "Be concise and surgical. Call done only after verification."
)


V7_REVIEWER_SYSTEM_PROMPT = (
    "You are an independent SWE-bench patch reviewer. You did not write the "
    "patch. Your job is to find edge cases, missing verification, regression "
    "risk, and benchmark-test defects before final evaluation. Return only "
    "strict JSON."
)


@dataclass
class ReviewerVerdict:
    approved: bool
    needs_revision: bool
    risk_summary: str = ""
    edge_cases: list[str] = field(default_factory=list)
    reproduction_code: str = ""
    test_commands: list[str] = field(default_factory=list)
    suspect_benchmark_test: bool = False
    test_fix_reason: str = ""
    proposed_test_patch: str = ""
    regression_risk: str = ""
    potentially_broken_tests: list[str] = field(default_factory=list)
    raw_text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _content_text(resp: dict) -> str:
    parts: list[str] = []
    for block in resp.get("content", []) or []:
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
    if not parts and resp.get("text"):
        parts.append(str(resp["text"]))
    return "\n".join(parts).strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from model text."""
    if not text:
        return {}
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return json.loads(stripped[start:end + 1])
    except json.JSONDecodeError:
        return {}


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def trajectory_summary(traj: Trajectory, *, max_turns: int = 8) -> str:
    """Compact trajectory summary for reviewer context."""
    lines = [
        f"stop_reason={traj.stop_reason}",
        f"turns={len(traj.turns)}",
        f"reproduction_attempted={traj.reproduction_attempted}",
        f"reproduction_passed={traj.reproduction_passed}",
        f"recoveries={len(traj.recovery_events)}",
    ]
    for turn in traj.turns[-max_turns:]:
        calls = ", ".join(c.get("name", "") for c in turn.tool_calls)
        obs = ""
        if turn.tool_results:
            first = turn.tool_results[0]
            obs = str(first.get("content", "") or first.get("text", ""))
            obs = " ".join(obs.split())[:400]
        lines.append(f"turn {turn.turn_idx}: tools=[{calls}] obs={obs}")
    return "\n".join(lines)


def build_reviewer_prompt(
    inst: SWEBenchInstance,
    *,
    patch: str,
    actor_summary: str = "",
) -> str:
    fail_to_pass = "\n".join(f"- {t}" for t in inst.fail_to_pass[:12])
    return (
        f"Repository: {inst.repo}\n"
        f"Instance: {inst.instance_id}\n"
        f"Base commit: {inst.base_commit}\n\n"
        f"## Problem statement\n{inst.problem_statement}\n\n"
        f"## FAIL_TO_PASS selectors\n{fail_to_pass}\n\n"
        f"## Actor trajectory summary\n{actor_summary or '(none)'}\n\n"
        f"## Proposed production patch\n{patch or '(empty patch)'}\n\n"
        "Return a JSON object with exactly these keys:\n"
        "{\n"
        '  "approved": boolean,\n'
        '  "needs_revision": boolean,\n'
        '  "risk_summary": string,\n'
        '  "edge_cases": string[],\n'
        '  "reproduction_code": string,\n'
        '  "test_commands": string[],\n'
        '  "suspect_benchmark_test": boolean,\n'
        '  "test_fix_reason": string,\n'
        '  "proposed_test_patch": string,\n'
        '  "regression_risk": string,\n'
        '  "potentially_broken_tests": string[]\n'
        "}\n\n"
        "Reviewer rubric: approve only if the patch directly addresses the "
        "issue, is small, has a plausible verification story, and avoids "
        "obvious regressions. If the benchmark test appears wrong, explain "
        "why and provide a proposed test diff, but do not reject correct "
        "production code solely because a hidden selector is absent.\n\n"
        "REGRESSION CHECK (V8 — important): trace what *other* code paths use "
        "the function/class touched by this patch. Identify nearby existing "
        "tests in the repo that exercise the modified surface. List any "
        "PASS_TO_PASS-style tests that this patch could break in "
        "`potentially_broken_tests`. If the patch changes a public API's "
        "signature, return-type, or error class, that's a regression risk — "
        "set `regression_risk` to a one-paragraph description and "
        "`needs_revision: true`. Be especially wary of changes that touch "
        "shared utility functions, base classes, or hot paths."
    )


def run_patch_reviewer(
    *,
    inst: SWEBenchInstance,
    patch: str,
    actor_trajectory: Trajectory | None,
    reviewer_chat_fn: Callable[..., dict],
    model: str,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    price_per_mtok: dict[str, float] | None = None,
) -> ReviewerVerdict:
    """Ask an independent reviewer model to audit the actor patch."""
    price = price_per_mtok or {"in": 3.0, "out": 15.0}
    summary = trajectory_summary(actor_trajectory) if actor_trajectory else ""
    prompt = build_reviewer_prompt(inst, patch=patch, actor_summary=summary)
    t0 = time.perf_counter()
    resp = reviewer_chat_fn(
        system=V7_REVIEWER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        tools=[],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    usage = resp.get("usage", {}) or {}
    in_tok = int(usage.get("input_tokens", 0) or 0)
    out_tok = int(usage.get("output_tokens", 0) or 0)
    usd = (in_tok * price["in"] + out_tok * price["out"]) / 1_000_000.0
    text = _content_text(resp)
    data = _extract_json_object(text)
    approved = bool(data.get("approved", False))
    needs_revision = bool(data.get("needs_revision", not approved))
    return ReviewerVerdict(
        approved=approved,
        needs_revision=needs_revision,
        risk_summary=str(data.get("risk_summary", "")),
        edge_cases=_coerce_str_list(data.get("edge_cases")),
        reproduction_code=str(data.get("reproduction_code", "")),
        test_commands=_coerce_str_list(data.get("test_commands")),
        suspect_benchmark_test=bool(data.get("suspect_benchmark_test", False)),
        test_fix_reason=str(data.get("test_fix_reason", "")),
        proposed_test_patch=str(data.get("proposed_test_patch", "")),
        regression_risk=str(data.get("regression_risk", "")),
        potentially_broken_tests=_coerce_str_list(data.get("potentially_broken_tests")),
        raw_text=text,
        input_tokens=in_tok,
        output_tokens=out_tok,
        latency_ms=latency_ms,
        usd=usd,
    )


def build_revision_problem_statement(
    inst: SWEBenchInstance,
    verdict: ReviewerVerdict,
) -> str:
    """Append reviewer feedback for a short second actor pass."""
    edge_cases = "\n".join(f"- {e}" for e in verdict.edge_cases[:8])
    commands = "\n".join(f"- {c}" for c in verdict.test_commands[:8])
    return (
        f"{inst.problem_statement}\n\n"
        "## Independent reviewer feedback\n"
        f"Risk summary: {verdict.risk_summary}\n\n"
        f"Edge cases to address:\n{edge_cases or '- (none)'}\n\n"
        f"Suggested verification commands:\n{commands or '- (none)'}\n\n"
        "Revise the existing worktree only if the reviewer found a real "
        "production-code issue. Keep the patch small. If the reviewer is "
        "wrong, verify and call done."
    )
