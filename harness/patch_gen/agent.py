"""Phase 3 agent path — multi-turn ACI loop for patch generation.

Per docs/V10_DESIGN_PHASE3.md §3.3:

  - Five tools: read_file, search_text, list_dir, apply_patch, submit.
  - NO run_tests, NO run_shell. The agent never names a test selector.
  - T_max=20 in dev, 30/50 by difficulty in production.
  - Per-instance cost cap $0.50 dev / $5 production.
  - Action-chunked: each turn produces ONE tool call as JSON.
  - apply_patch validates with `git apply --check` (worktree never
    mutated). On success the diff is stored as the current
    candidate. submit finalizes the most-recently-validated diff.

The agent inputs are the same firewall-clean fields that flow into
the pipeline path: ``InstanceView`` + ``RankedFile[]``. NO repro
test contents reach the agent (per V10_DESIGN_PHASE2.md §2 + the
cross-phase firewall test).

To satisfy the §2.2 superset-assertion firewall, the agent module
calls ``build_patch_gen_context_with_superset_check`` even though
it doesn't put file CONTENT in the prompt — it uses the resulting
``PatchGenContext`` for path discovery (the agent reads files
itself via the read_file tool). The single-entrypoint contract
holds.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass

from harness.cost import CostTracker
from harness.localization_signals import RankedFile
from harness.patch_gen.context import (
    build_patch_gen_context_with_superset_check,
)
from harness.patch_gen.views import (
    PatchCandidate,
    PatchGenError,
)
from harness.views import InstanceView


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------


DEFAULT_AGENT_T_MAX: int = 20
DEFAULT_AGENT_COST_CAP_USD: float = 0.50
DEFAULT_AGENT_MAX_OUTPUT_TOKENS: int = 1024  # one tool call worth
DEFAULT_READ_FILE_LIMIT_LINES: int = 200
DEFAULT_SEARCH_RESULTS_CAP: int = 60
DEFAULT_LIST_DIR_CAP: int = 80
DEFAULT_TOOL_OBSERVATION_CAP_CHARS: int = 8_000


# ---------------------------------------------------------------------------
# Tool schema (the system prompt enumerates these)
# ---------------------------------------------------------------------------


_TOOL_SCHEMA = """\
Tools available (one call per turn — emit a single JSON object):

  read_file:
    args: {"path": "<repo-relative path>", "offset": <int line>, "limit": <int max lines>}
    Reads a file; pass offset (default 0) and limit (default 200, max 800) to
    page through large files. Returns lines with leading 1-based numbers.

  search_text:
    args: {"query": "<regex>", "glob": "<optional path glob>"}
    Runs ripgrep across the repo. Returns up to 60 matches in
    "path:line:context" format. Use --glob like "*.py" to scope.

  list_dir:
    args: {"path": "<repo-relative dir, '.' for repo root>"}
    Lists entries with a trailing slash on directories. Up to 80 entries.

  apply_patch:
    args: {"diff": "<unified diff text>"}
    Validates the diff with `git apply --check` (no worktree mutation).
    On success, stores the diff as the current candidate. On failure,
    returns the git error so you can revise. You can call apply_patch
    repeatedly — each successful call replaces the prior candidate.

  submit:
    args: {"rationale": "<1-3 sentence explanation>"}
    Finalizes the most-recently-validated candidate as your submission
    and ends the loop. submit fails if no apply_patch has succeeded.

Output format — JSON only, no prose, no markdown fences:

  {"tool": "<name>", "args": {...}}
"""


_SYSTEM_PROMPT = """\
You are a software engineer fixing a bug in a Python codebase. You
have a 5-tool toolbox to explore the repo and produce a unified diff.

Your goal: emit a unified diff that, when applied to the repo at its
current state, fixes the bug described in the issue. The diff must:

  1. Apply cleanly with `git apply` (verify via the apply_patch tool).
  2. Edit ONLY the files needed to fix the bug.
  3. Do NOT modify test files. Tests are evaluated separately.
  4. Be MINIMAL — the smallest change that resolves the issue.
  5. Match the repo's existing style.

Workflow:
  1. Read the issue and the localizer's top file candidates.
  2. Use read_file / search_text / list_dir to navigate the code.
  3. Construct a diff and call apply_patch to validate it.
  4. If apply_patch fails, fix the diff (line numbers, context lines)
     and try again. Repeat until apply_patch succeeds.
  5. Once apply_patch succeeds, call submit with a short rationale.

You have a hard turn budget. Don't waste turns — keep reads focused
on the suspect files; submit as soon as you have a valid diff.

""" + _TOOL_SCHEMA


# ---------------------------------------------------------------------------
# Tool implementations (all read-only against the worktree)
# ---------------------------------------------------------------------------


def _ripgrep_command(query: str, glob: str | None) -> str:
    """Build the ripgrep command. ``query`` is treated as a regex
    (rg's default). ``glob`` is forwarded as ``--glob`` if given."""
    parts = [
        "rg", "--line-number", "--no-heading", "--smart-case",
        "--max-count", "20", "--max-columns", "200",
    ]
    if glob:
        parts.extend(["--glob", _shell_quote(glob)])
    parts.append(_shell_quote(query))
    parts.append(".")
    return " ".join(parts)


def _shell_quote(s: str) -> str:
    """Single-quote a shell argument; escape internal single quotes."""
    return "'" + s.replace("'", "'\\''") + "'"


def _tool_read_file(sandbox, args: dict) -> str:
    path = str(args.get("path", "")).strip()
    if not path:
        return "[error] read_file: missing 'path' arg"
    offset = int(args.get("offset", 0))
    limit = int(args.get("limit", DEFAULT_READ_FILE_LIMIT_LINES))
    limit = max(1, min(limit, 800))  # hard cap

    # Read raw, then slice by line. Read up to ~50K chars to keep it cheap.
    res = sandbox.read_file(path, max_chars=200_000)
    if res.exit_code != 0:
        return f"[error] read_file: {path}: not found or unreadable"
    lines = res.stdout.splitlines()
    total = len(lines)
    end = min(offset + limit, total)
    sliced = lines[offset:end]
    width = len(str(max(end, 1)))
    body_lines = [f"{i + offset + 1:>{width}}\t{line}" for i, line in enumerate(sliced)]
    body = "\n".join(body_lines)
    if len(body) > DEFAULT_TOOL_OBSERVATION_CAP_CHARS:
        body = body[:DEFAULT_TOOL_OBSERVATION_CAP_CHARS] + "\n…(observation truncated)"
    header = f"[read_file: {path} lines {offset+1}-{end} of {total}]"
    return f"{header}\n{body}"


def _tool_search_text(sandbox, args: dict) -> str:
    query = str(args.get("query", "")).strip()
    if not query:
        return "[error] search_text: missing 'query' arg"
    glob = args.get("glob")
    glob = str(glob).strip() if glob else None
    cmd = "cd /testbed && " + _ripgrep_command(query, glob) + f" | head -{DEFAULT_SEARCH_RESULTS_CAP}"
    try:
        res = sandbox.run_shell(cmd, timeout_s=20.0)
    except Exception as exc:  # noqa: BLE001
        return f"[error] search_text: {type(exc).__name__}: {exc}"
    if res.exit_code != 0 and not res.stdout:
        return f"[search_text: {query!r} glob={glob!r}] no matches (rg exit={res.exit_code})"
    out = res.stdout
    if len(out) > DEFAULT_TOOL_OBSERVATION_CAP_CHARS:
        out = out[:DEFAULT_TOOL_OBSERVATION_CAP_CHARS] + "\n…(observation truncated)"
    glob_str = f" glob={glob!r}" if glob else ""
    return f"[search_text: {query!r}{glob_str}]\n{out.rstrip()}"


def _tool_list_dir(sandbox, args: dict) -> str:
    path = str(args.get("path", ".")).strip() or "."
    cmd = (
        f"cd /testbed && ls -1F {_shell_quote(path)} 2>&1 | head -{DEFAULT_LIST_DIR_CAP}"
    )
    try:
        res = sandbox.run_shell(cmd, timeout_s=10.0)
    except Exception as exc:  # noqa: BLE001
        return f"[error] list_dir: {type(exc).__name__}: {exc}"
    if res.exit_code != 0:
        return f"[error] list_dir: {path}: {(res.stdout or '').strip()[:200]}"
    body = (res.stdout or "").rstrip()
    if len(body) > DEFAULT_TOOL_OBSERVATION_CAP_CHARS:
        body = body[:DEFAULT_TOOL_OBSERVATION_CAP_CHARS] + "\n…(observation truncated)"
    return f"[list_dir: {path}]\n{body}"


@dataclass
class _AgentState:
    """In-loop mutable state carried alongside the conversation."""

    candidate_diff: str | None = None     # last successful apply_patch
    candidate_apply_count: int = 0        # how many times we successfully validated
    apply_attempts: int = 0               # all apply_patch calls (success + fail)
    submitted_rationale: str | None = None
    submit_called: bool = False


def _tool_apply_patch(sandbox, args: dict, state: _AgentState) -> str:
    diff = args.get("diff", "")
    if not isinstance(diff, str) or not diff.strip():
        return "[error] apply_patch: 'diff' must be a non-empty string"
    state.apply_attempts += 1

    # Use --check semantics: run on a fresh worktree, but DON'T mutate it.
    # Strategy: write diff to /tmp/v10_agent_check.patch, run `git apply --check`,
    # then if it succeeds, leave the patch on disk + remember the diff string;
    # we do NOT actually apply it (the grader runs apply later on the
    # final submission).
    path_inside = "/tmp/v10_agent_check.patch"
    cmd = (
        f"cat > {path_inside} <<'V10_AGENT_EOF'\n{diff}\nV10_AGENT_EOF\n"
        f"cd /testbed && git apply --check --whitespace=nowarn {path_inside}"
    )
    try:
        res = sandbox.run_shell(cmd, timeout_s=60.0)
    except Exception as exc:  # noqa: BLE001
        return f"[error] apply_patch: {type(exc).__name__}: {exc}"
    if res.exit_code == 0:
        state.candidate_diff = diff
        state.candidate_apply_count += 1
        return (
            "[apply_patch: ok]\n"
            "Diff validates with `git apply --check` and is stored as your "
            "current candidate. Call submit to finalize, or call "
            "apply_patch again with a revision to overwrite."
        )
    err = (res.stderr or res.stdout or "").strip()
    if len(err) > DEFAULT_TOOL_OBSERVATION_CAP_CHARS:
        err = err[:DEFAULT_TOOL_OBSERVATION_CAP_CHARS] + "\n…(error truncated)"
    return f"[apply_patch: FAILED]\n{err}"


def _tool_submit(args: dict, state: _AgentState) -> str:
    rationale = str(args.get("rationale", "")).strip()
    if state.candidate_diff is None:
        return (
            "[error] submit: no candidate to finalize. Call apply_patch "
            "with a valid diff first."
        )
    state.submitted_rationale = rationale or "(no rationale supplied)"
    state.submit_called = True
    return "[submit: candidate finalized — agent loop ends]"


# ---------------------------------------------------------------------------
# Tool dispatch + JSON parsing
# ---------------------------------------------------------------------------


_VALID_TOOLS = ("read_file", "search_text", "list_dir", "apply_patch", "submit")


def _parse_tool_call(text: str) -> dict:
    """Parse one tool-call JSON object from the LLM. Tolerates fenced
    output. Raises ``PatchGenError`` on any structural problem."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PatchGenError(f"agent output is not valid JSON: {exc}")
    if not isinstance(data, dict):
        raise PatchGenError(f"agent output must be a JSON object, got {type(data).__name__}")
    tool = data.get("tool")
    if tool not in _VALID_TOOLS:
        raise PatchGenError(
            f"agent output 'tool' must be one of {_VALID_TOOLS}, got {tool!r}"
        )
    args = data.get("args", {})
    if not isinstance(args, dict):
        raise PatchGenError(f"agent output 'args' must be a dict, got {type(args).__name__}")
    return {"tool": tool, "args": args}


def _dispatch_tool(sandbox, call: dict, state: _AgentState) -> str:
    tool = call["tool"]
    args = call["args"]
    try:
        if tool == "read_file":
            return _tool_read_file(sandbox, args)
        if tool == "search_text":
            return _tool_search_text(sandbox, args)
        if tool == "list_dir":
            return _tool_list_dir(sandbox, args)
        if tool == "apply_patch":
            return _tool_apply_patch(sandbox, args, state)
        if tool == "submit":
            return _tool_submit(args, state)
    except Exception as exc:  # noqa: BLE001 — last-ditch guard
        return f"[tool error] {tool}: {type(exc).__name__}: {exc}"
    return f"[error] unknown tool: {tool!r}"


# ---------------------------------------------------------------------------
# User prompt (path-only context, no file contents)
# ---------------------------------------------------------------------------


def _build_user_prompt(
    view: InstanceView,
    ranked_files: list[RankedFile],
    seed_diff: str | None = None,
) -> str:
    """The agent gets the issue + localizer's top candidates as paths.
    File CONTENTS are read by the agent itself (read_file tool); this
    avoids the truncation problem that hit the pipeline path.

    When ``seed_diff`` is supplied (P3c-v2 Fix C — pipeline-bootstrapped
    agent), the prompt explicitly tells the agent to apply that diff
    via apply_patch as the FIRST action, and to fix any apply errors
    before exploring. This narrows the agent's task from "explore +
    write a diff from scratch" (which P3c data showed it loses on,
    median apply_attempts=0 in 28/29 failures) to "fix the apply
    errors in this diff or replace it" — playing to the agent's
    38% applied-correctness strength.
    """
    parts: list[str] = []
    parts.append("# Issue\n\n")
    parts.append(view.problem_statement.strip())
    parts.append("\n\n# Test directories (DO NOT modify test files)\n\n")
    for d in view.test_directives.dirs:
        parts.append(f"  - {d}\n")
    parts.append("\n# Localizer top-K candidate files\n\n")
    for rf in ranked_files[:10]:
        parts.append(f"  - `{rf.file_path}` (score={rf.final_score:.2f}): {rf.rationale}\n")

    if seed_diff and seed_diff.strip():
        parts.append("\n# Seed diff (from a single-shot pre-pass)\n\n")
        parts.append(
            "A pipeline-stage call produced the following candidate diff. "
            "**Your FIRST action MUST be to call apply_patch on this diff.** "
            "If apply_patch succeeds, refine OR submit. If apply_patch "
            "fails, read the relevant files (per the localizer) to "
            "understand the actual line numbers and context, then call "
            "apply_patch again with a corrected diff. The seed is a "
            "starting point — feel free to replace it entirely if it's "
            "wrong, but do not waste turns exploring before trying it.\n\n"
        )
        parts.append("```diff\n")
        parts.append(seed_diff)
        if not seed_diff.endswith("\n"):
            parts.append("\n")
        parts.append("```\n")
        parts.append("\n# Begin\n\nEmit your first tool call now (apply_patch on the seed).\n")
    else:
        parts.append("\n# Begin\n\nEmit your first tool call now.\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
# T_max - 5 force-finalize nudge (P3c-v2 Fix A)
# ---------------------------------------------------------------------------


def _build_force_finalize_nudge(turns_remaining: int, has_candidate: bool) -> str:
    """Inject as a user message when turn count reaches T_max - 5.

    Per P3c-v2 Fix A: the diagnostic showed 28/29 agent-no-submit
    failures had median apply_attempts=0 — the agent burned all 20
    turns exploring without ever committing. The nudge breaks that
    loop by hard-pressuring the agent toward apply_patch + submit.
    """
    if has_candidate:
        return (
            f"[harness] {turns_remaining} turns remaining. You have a "
            f"validated candidate diff. Call submit on the next turn — "
            f"do not keep exploring or revising unless the candidate is "
            f"clearly wrong. The empty submission is worse than this "
            f"candidate."
        )
    return (
        f"[harness] {turns_remaining} turns remaining. STOP exploring. "
        f"Based on what you have already read, write your best-guess "
        f"unified diff and call apply_patch. If apply_patch fails, fix "
        f"the line numbers and context lines and try again. If it "
        f"succeeds, call submit. The empty submission (no apply_patch "
        f"success in N=20 turns) is the worst possible outcome — a "
        f"wrong patch is strictly better than no patch."
    )


# ---------------------------------------------------------------------------
# Cost helper
# ---------------------------------------------------------------------------


def _cost_for_chat(model: str, input_tokens: int, output_tokens: int) -> float:
    from harness.llm.clients import price_for_model
    p = price_for_model(model)
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000.0


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentGenerationResult:
    """Outcome of one agent run on one instance."""

    candidate: PatchCandidate | None
    turns_used: int
    apply_attempts: int
    apply_successes: int
    submitted: bool
    cost_cap_hit: bool
    total_cost_usd: float
    final_status: str    # "submitted" | "no_apply" | "tmax" | "cost_cap" | "parse_failures_exhausted"


# ---------------------------------------------------------------------------
# The agent loop
# ---------------------------------------------------------------------------


def generate_agent(
    *,
    view: InstanceView,
    ranked_files: list[RankedFile],
    sandbox,
    t_max: int = DEFAULT_AGENT_T_MAX,
    cost_cap_usd: float = DEFAULT_AGENT_COST_CAP_USD,
    max_output_tokens: int = DEFAULT_AGENT_MAX_OUTPUT_TOKENS,
    repro_context_files: tuple[str, ...] | None = None,
    parse_failure_budget: int = 3,
    seed_diff: str | None = None,
    force_finalize_at_turns_remaining: int = 5,
) -> AgentGenerationResult:
    """Run the agent path on one instance.

    Returns ``AgentGenerationResult`` with ``candidate=PatchCandidate``
    on submit, else ``candidate=None`` with a ``final_status`` describing
    why.

    Hard rules:
      - The §2.2 superset assertion runs via
        ``build_patch_gen_context_with_superset_check`` — even though the
        agent reads files via the tool, the firewall test requires every
        prompt-build path call this function. ``ContextOversizeError``
        from the builder propagates (cap-hitter instances are still
        agent-route candidates because their context budget here is the
        TOOL observations, not a single prompt; but the firewall test
        AST scan ensures the call is present).
      - Cost cap is checked BETWEEN turns. An in-flight turn's spend
        is counted but not aborted.
      - T_max is a HARD ceiling regardless of progress.
      - parse_failure_budget bounds invalid-JSON responses; each
        invalid response decrements t_max by 1 (per design §3.3).
    """
    from harness.llm.clients import complete_chat, model_for_role

    # §2.2 firewall — call the single entrypoint even though we'll read
    # files via tool. Use a tiny per_file_char_cap to keep memory low;
    # we don't actually use the FileSnippet contents in the prompt.
    try:
        build_patch_gen_context_with_superset_check(
            view=view,
            ranked_files=ranked_files,
            sandbox=sandbox,
            per_file_char_cap=1024,         # we don't put content in prompt
            projected_token_limit=10_000_000,  # don't oversize-gate the agent path
            repro_context_files=repro_context_files,
        )
    except Exception:
        # If the superset assertion fires, propagate. Other errors
        # (e.g., empty ranked_files) propagate too.
        raise

    model = model_for_role("patch_generator_agent")

    state = _AgentState()
    tracker = CostTracker(instance_id=view.instance_id, cap_usd=cost_cap_usd)

    user_msg_initial = _build_user_prompt(view, ranked_files, seed_diff=seed_diff)
    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_msg_initial},
    ]

    parse_failures = 0
    turns_used = 0
    cost_cap_hit = False
    final_status = "tmax"
    force_finalize_injected = False

    t_start_global = time.perf_counter()

    while turns_used < t_max:
        if tracker.hard_cap_reached:
            cost_cap_hit = True
            final_status = "cost_cap"
            break

        # Fix A — T_max-5 force-finalize nudge. Inject ONCE when the
        # agent crosses into the last `force_finalize_at_turns_remaining`
        # turns. Past that point every observation already implicitly
        # carries the time pressure (the conversation is long), so we
        # don't inject repeatedly.
        turns_remaining = t_max - turns_used
        if (
            not force_finalize_injected
            and turns_remaining <= force_finalize_at_turns_remaining
            and not state.submit_called
        ):
            messages.append({
                "role": "user",
                "content": _build_force_finalize_nudge(
                    turns_remaining=turns_remaining,
                    has_candidate=state.candidate_diff is not None,
                ),
            })
            force_finalize_injected = True

        turns_used += 1
        try:
            chat = complete_chat(
                messages=messages,
                model=model,
                max_tokens=max_output_tokens,
                temperature=0.0,
                response_format_json=True,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "[agent] %s: complete_chat failed at turn %d: %s",
                view.instance_id, turns_used, exc,
            )
            final_status = "llm_error"
            break

        tracker.record(
            stage="patch_gen_agent",
            model=chat.model,
            input_tokens=chat.input_tokens,
            output_tokens=chat.output_tokens,
        )

        # Parse the tool call. Invalid JSON → re-prompt with the error.
        try:
            call = _parse_tool_call(chat.text)
        except PatchGenError as exc:
            parse_failures += 1
            messages.append({"role": "assistant", "content": chat.text})
            messages.append({
                "role": "user",
                "content": f"[parse error] {exc}\n\nYour previous response was not a valid tool call. "
                           f"Emit a single JSON object: {{\"tool\": \"<name>\", \"args\": {{...}}}}",
            })
            if parse_failures >= parse_failure_budget:
                final_status = "parse_failures_exhausted"
                break
            continue

        # Dispatch + observation
        observation = _dispatch_tool(sandbox, call, state)

        messages.append({"role": "assistant", "content": chat.text})
        messages.append({"role": "user", "content": observation})

        if state.submit_called:
            final_status = "submitted"
            break

    duration_s = time.perf_counter() - t_start_global

    if not state.submit_called:
        if final_status == "tmax" and turns_used >= t_max and state.candidate_diff is not None:
            # We hit T_max with a validated candidate but no submit. Auto-finalize.
            log.info(
                "[agent] %s: T_max reached with validated candidate; auto-finalizing",
                view.instance_id,
            )
            state.submitted_rationale = "(auto-finalized at T_max — agent had a validated candidate)"
            state.submit_called = True
            final_status = "tmax_autosubmit"
        else:
            # Status refinement: if we hit T_max with no candidate, name
            # that explicitly so the audit can split "tmax with progress"
            # from "tmax with nothing to show".
            status_out = final_status
            if final_status == "tmax" and state.candidate_diff is None:
                status_out = "tmax_no_apply"
            return AgentGenerationResult(
                candidate=None,
                turns_used=turns_used,
                apply_attempts=state.apply_attempts,
                apply_successes=state.candidate_apply_count,
                submitted=False,
                cost_cap_hit=cost_cap_hit,
                total_cost_usd=tracker.total_usd,
                final_status=status_out,
            )

    # Sum tokens across all turns for the candidate's audit fields.
    total_input = sum(e.input_tokens for e in tracker.entries)
    total_output = sum(e.output_tokens for e in tracker.entries)
    candidate = PatchCandidate(
        instance_id=view.instance_id,
        candidate_id="agent_a0_final",
        diff=state.candidate_diff or "",
        source_route="agent",
        source_temperature=None,
        source_attempt_index=0,
        generator_model=model,
        generator_input_tokens=total_input,
        generator_output_tokens=total_output,
        generation_cost_usd=tracker.total_usd,
        duration_s=duration_s,
    )

    return AgentGenerationResult(
        candidate=candidate,
        turns_used=turns_used,
        apply_attempts=state.apply_attempts,
        apply_successes=state.candidate_apply_count,
        submitted=True,
        cost_cap_hit=cost_cap_hit,
        total_cost_usd=tracker.total_usd,
        final_status=final_status,
    )


__all__ = [
    "AgentGenerationResult",
    "DEFAULT_AGENT_COST_CAP_USD",
    "DEFAULT_AGENT_MAX_OUTPUT_TOKENS",
    "DEFAULT_AGENT_T_MAX",
    "generate_agent",
]
