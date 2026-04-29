"""Multi-turn ReAct-style agent loop for SWE-bench tasks.

Architecture:

- Drives Anthropic's tool-use API (Sonnet 4.6 / Haiku 4.5 / etc.)
  with a fixed tool schema: ``read_file``, ``read_file_range``,
  ``search_text``, ``replace_text``, ``write_file``, ``run_python``,
  ``run_tests``, ``list_files``, ``bash``, ``done``.
- On each turn, the model emits one tool call. The executor runs it,
  and the next turn passes the observation back via a
  ``tool_result`` content block.
- Loop exits when:
    1. Model emits ``done`` tool call, OR
    2. ``max_turns`` reached, OR
    3. Per-task budget guard tripped (token / wall / USD), OR
    4. An unrecoverable error.
- After the loop, the harness extracts the patch from
  ``shell.get_diff()`` (i.e., ``git diff`` of the working tree).

Why one-shot patch generation isn't enough:
- Real SWE-bench bugs need: read context (1+ files), figure out the
  fix, edit, run tests, iterate. Single-call patch generation hits a
  ~30% ceiling because the model can't see actual test failures.
- Multi-turn unlocks running tests after each edit and converging
  on a fix that genuinely passes — which is what "wow" requires.

This module is provider-agnostic for the LLM; the constructor takes
an Anthropic-compatible chat callable. We default to a Sonnet 4.6
client built on the existing ``HTTPClient``.
"""

from __future__ import annotations

import json
import hashlib
import os
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .agent_docker import DockerShellExecutor, ExecResult


# ----------------- Tool schema (Anthropic format) -----------------

TOOLS = [
    {
        "name": "read_file",
        "description": (
            "Read the contents of a file in the repository. "
            "The file path is relative to the repo root (/testbed). "
            "For large files, prefer read_file_range."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to file."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "read_file_range",
        "description": (
            "Read a bounded line range from a file, with line numbers. "
            "Use this instead of read_file for large files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to file."},
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1},
            },
            "required": ["path", "start_line", "end_line"],
        },
    },
    {
        "name": "search_text",
        "description": (
            "Fixed-string recursive search with file paths and line numbers. "
            "Use this to find symbols or error text before reading large files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Exact text to search for."},
                "path": {"type": "string", "default": "."},
                "max_matches": {"type": "integer", "default": 80},
            },
            "required": ["query"],
        },
    },
    {
        "name": "go_to_definition",
        "description": (
            "LSP-style structural lookup for a Python symbol. Uses an AST "
            "index to find class, function, and assignment definitions without "
            "reading whole files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Symbol name."},
                "path": {"type": "string", "default": "."},
                "max_results": {"type": "integer", "default": 40},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "find_references",
        "description": (
            "LSP-style token reference search for a Python symbol. Prefer this "
            "over broad grep when tracing a function/class/variable."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Symbol name."},
                "path": {"type": "string", "default": "."},
                "max_matches": {"type": "integer", "default": 80},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "replace_text",
        "description": (
            "Replace exact text in one file. Prefer this for small edits, "
            "especially in large files, because write_file requires the full "
            "new file body. Test-file edits are blocked."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old": {"type": "string", "description": "Exact existing text."},
                "new": {"type": "string", "description": "Replacement text."},
                "expected_count": {
                    "type": "integer",
                    "default": 1,
                    "description": "Expected number of occurrences; 0 means replace all.",
                },
            },
            "required": ["path", "old", "new"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Overwrite a file with new content. The file path is relative "
            "to the repo root. The full content of the file is written; "
            "you cannot do partial edits — pass the entire new file body. "
            "For small edits to existing files, prefer replace_text. "
            "Test-file edits are blocked."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string",
                            "description": "The full new content of the file."},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "run_reproduction",
        "description": (
            "Write and run a focused Python reproduction script outside the "
            "git checkout. Use this before the first production edit whenever "
            "the official hidden pytest selector is unavailable or noisy."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python reproduction/assertion code.",
                },
                "expected_behavior": {
                    "type": "string",
                    "description": "What this script proves about the bug.",
                },
                "timeout_s": {"type": "number", "default": 60},
            },
            "required": ["code", "expected_behavior"],
        },
    },
    {
        "name": "run_python",
        "description": (
            "Run a short Python reproduction or assertion in the repository "
            "environment. Use this when hidden SWE-bench pytest selectors are "
            "not present in the checkout."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Keep it short and focused.",
                },
                "timeout_s": {"type": "number", "default": 60},
            },
            "required": ["code"],
        },
    },
    {
        "name": "run_tests",
        "description": (
            "Run the FAIL_TO_PASS test(s) for this task with pytest. "
            "Returns the test output. Use this to verify your fix."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "test_targets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Pytest selectors (e.g. 'tests/test_foo.py' or "
                        "'tests/test_foo.py::test_bar'). Empty means run all."
                    ),
                },
            },
            "required": ["test_targets"],
        },
    },
    {
        "name": "propose_test_fix",
        "description": (
            "Record a suspected benchmark/test-suite defect and a proposed "
            "test patch. This does not edit the submitted production patch; "
            "it creates an audit artifact for impossible/noisy tasks."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "test_path": {"type": "string"},
                "reason": {
                    "type": "string",
                    "description": "Why the benchmark test is wrong or noisy.",
                },
                "patch": {
                    "type": "string",
                    "description": "Unified diff for the test-suite fix.",
                },
                "confidence": {"type": "number", "default": 0.5},
            },
            "required": ["test_path", "reason", "patch"],
        },
    },
    {
        "name": "list_files",
        "description": (
            "List files under a directory. Useful for exploring repo "
            "structure. With ``pattern``, does a recursive find."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "default": "."},
                "pattern": {"type": "string",
                            "description": "Optional glob (e.g. '*.py')."},
            },
        },
    },
    {
        "name": "bash",
        "description": (
            "Run an arbitrary shell command. Use sparingly — prefer the "
            "specialized tools. Useful for grep, sed, git log, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "done",
        "description": (
            "Signal that you have finished fixing the bug. The current "
            "state of the working tree will be diff'd and submitted as "
            "your patch. Only call this once your tests pass."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string",
                            "description": "Brief summary of the fix."},
            },
        },
    },
]

TOOL_NAMES = {t["name"] for t in TOOLS}


def select_tools(tool_set: str) -> list[dict]:
    """Filter TOOLS by named subset.

    - ``"minimal"``: search/read/LSP/repro/edit/test/done, without arbitrary bash.
    - ``"full"``: + list_files, bash.
    """
    if tool_set == "minimal":
        keep = {
            "read_file",
            "read_file_range",
            "search_text",
            "go_to_definition",
            "find_references",
            "replace_text",
            "write_file",
            "run_reproduction",
            "run_python",
            "run_tests",
            "propose_test_fix",
            "list_files",
            "done",
        }
    else:  # full
        keep = TOOL_NAMES
    return [t for t in TOOLS if t["name"] in keep]


def _tool_signature(tool_calls: list[dict]) -> str:
    """Compact, stable signature for repeated-action detection."""
    compact_calls = []
    for call in tool_calls:
        compact_input = {}
        for key, value in (call.get("input") or {}).items():
            if isinstance(value, str) and len(value) > 240:
                digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
                compact_input[key] = f"<str len={len(value)} sha1={digest}>"
            else:
                compact_input[key] = value
        compact_calls.append({
            "name": call.get("name"),
            "input": compact_input,
        })
    return json.dumps(compact_calls, sort_keys=True, separators=(",", ":"))


def _is_test_path(path: str) -> bool:
    rel = Path(path.lstrip("/"))
    return "tests" in rel.parts or rel.name.startswith("test_")


# ----------------- Budget guard -----------------

@dataclass
class BudgetGuard:
    max_turns: int = 20
    max_tokens_in: int = 300_000     # cumulative across all turns for one task
    max_tokens_out: int = 50_000
    max_wall_s: float = 600.0        # 10-min wall cap per task
    max_usd: float = 1.50            # per-task $ cap (Sonnet 4.6 ≈ $0.15-0.50/task typically)

    @staticmethod
    def of_model(model: str) -> "BudgetGuard":
        """Pricing-aware budget defaults."""
        return BudgetGuard()


@dataclass
class BudgetTracker:
    guard: BudgetGuard
    turns: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    wall_s: float = 0.0
    usd: float = 0.0

    def add_turn(self, in_tok: int, out_tok: int, wall_s: float, usd: float) -> None:
        self.turns += 1
        self.tokens_in += in_tok
        self.tokens_out += out_tok
        self.wall_s += wall_s
        self.usd += usd

    def should_stop(self) -> tuple[bool, str]:
        g = self.guard
        if self.turns >= g.max_turns:
            return True, f"max_turns ({g.max_turns}) reached"
        if self.tokens_in >= g.max_tokens_in:
            return True, f"tokens_in cap ({g.max_tokens_in}) reached"
        if self.tokens_out >= g.max_tokens_out:
            return True, f"tokens_out cap ({g.max_tokens_out}) reached"
        if self.wall_s >= g.max_wall_s:
            return True, f"wall ({g.max_wall_s}s) reached"
        if self.usd >= g.max_usd:
            return True, f"usd cap (${g.max_usd}) reached"
        return False, ""


# ----------------- Agent loop -----------------

# Pricing in USD per million tokens, by model name.
# Keys are normalized via lowercase + substring match (the run gets the
# closest fit). When a model isn't matched, we fall back to Sonnet rates.
_PRICING = {
    # Anthropic Claude family — per published Anthropic pricing.
    "claude-sonnet": {"in": 3.0, "in_cached": 0.30, "out": 15.0},
    "claude-opus":   {"in": 15.0, "in_cached": 1.50, "out": 75.0},
    "claude-haiku":  {"in": 1.0,  "in_cached": 0.10, "out": 5.0},
    # DeepSeek — V4-Pro promo prices through 2026-05-31; cached input is 1/120 of full.
    "deepseek-v4-pro":   {"in": 0.435, "in_cached": 0.0036, "out": 0.87},
    "deepseek-reasoner": {"in": 0.435, "in_cached": 0.0036, "out": 0.87},
    "deepseek-v4-flash": {"in": 0.14,  "in_cached": 0.0028, "out": 0.28},
    "deepseek-chat":     {"in": 0.14,  "in_cached": 0.0028, "out": 0.28},
}
_DEFAULT_PRICE = {"in": 3.0, "in_cached": 0.30, "out": 15.0}  # Sonnet fallback


def price_for_model(model: str | None) -> dict:
    """Look up per-Mtok pricing for a model name (best-effort substring match)."""
    if not model:
        return dict(_DEFAULT_PRICE)
    m = model.lower()
    for key, table in _PRICING.items():
        if key in m or m in key:
            return dict(table)
    return dict(_DEFAULT_PRICE)


@dataclass
class RecoveryPolicy:
    """State-space recovery knobs for v7-style runs.

    When enabled, the loop keeps a stable worktree checkpoint and restores it
    if the actor repeatedly edits the same file without any successful
    verification signal.
    """

    enabled: bool = False
    same_file_edit_threshold: int = 3
    max_recoveries: int = 2
    checkpoint_after_successful_reproduction: bool = True
    checkpoint_after_successful_tests: bool = True


@dataclass
class RecoveryEvent:
    turn_idx: int
    reason: str
    edited_path: str
    edit_count: int
    checkpoint_path: str
    restore_exit_code: int
    restore_stderr: str = ""


@dataclass
class TurnRecord:
    turn_idx: int
    text: str
    tool_calls: list[dict]
    tool_results: list[dict]
    in_tokens: int
    out_tokens: int
    wall_ms: float
    stop_reason: str | None = None


@dataclass
class Trajectory:
    instance_id: str
    turns: list[TurnRecord] = field(default_factory=list)
    final_patch: str = ""
    done_emitted: bool = False
    stop_reason: str = ""
    total_in_tokens: int = 0
    total_out_tokens: int = 0
    total_wall_s: float = 0.0
    total_usd: float = 0.0
    reproduction_attempted: bool = False
    reproduction_passed: bool = False
    recovery_events: list[RecoveryEvent] = field(default_factory=list)
    test_fix_proposals: list[dict[str, Any]] = field(default_factory=list)


def _execute_tool(
    sh: DockerShellExecutor,
    name: str,
    args: dict,
    *,
    fail_to_pass: list[str],
) -> str:
    """Dispatch one tool call to the DockerShellExecutor and return
    a human-readable observation string."""
    try:
        if name == "read_file":
            r = sh.read_file(args["path"])
            if r.exit_code != 0:
                return f"Error reading {args['path']!r}: {r.stderr[:300]}"
            return f"```\n{r.stdout}\n```"
        if name == "read_file_range":
            path = args.get("path")
            if path is None:
                return "Error: read_file_range requires 'path'."
            try:
                start_line = int(args["start_line"])
                end_line = int(args["end_line"])
            except (KeyError, TypeError, ValueError):
                return "Error: read_file_range requires integer start_line and end_line."
            r = sh.read_file_range(path, start_line, end_line)
            if r.exit_code != 0:
                return f"Error reading range from {path!r}: {r.stderr[:300]}"
            return f"```\n{r.stdout}\n```"
        if name == "search_text":
            query = args.get("query")
            if not query:
                return "Error: search_text requires non-empty 'query'."
            try:
                max_matches = int(args.get("max_matches", 80))
            except (TypeError, ValueError):
                max_matches = 80
            r = sh.search_text(
                query,
                args.get("path", "."),
                max_matches=max_matches,
            )
            if r.exit_code != 0:
                return f"Error searching for {query!r}: {r.stderr[:300]}"
            return r.stdout or "No matches."
        if name == "go_to_definition":
            symbol = args.get("symbol")
            if not symbol:
                return "Error: go_to_definition requires non-empty 'symbol'."
            try:
                max_results = int(args.get("max_results", 40))
            except (TypeError, ValueError):
                max_results = 40
            r = sh.go_to_definition(
                symbol,
                args.get("path", "."),
                max_results=max_results,
            )
            if r.exit_code != 0:
                return f"Error finding definition for {symbol!r}: {r.stderr[:300]}"
            return r.stdout or "No definitions found."
        if name == "find_references":
            symbol = args.get("symbol")
            if not symbol:
                return "Error: find_references requires non-empty 'symbol'."
            try:
                max_matches = int(args.get("max_matches", 80))
            except (TypeError, ValueError):
                max_matches = 80
            r = sh.find_references(
                symbol,
                args.get("path", "."),
                max_matches=max_matches,
            )
            if r.exit_code != 0:
                return f"Error finding references for {symbol!r}: {r.stderr[:300]}"
            return r.stdout or "No references found."
        if name == "replace_text":
            path = args.get("path")
            old = args.get("old")
            new = args.get("new")
            if path is None or old is None or new is None:
                return "Error: replace_text requires 'path', 'old', and 'new'."
            if _is_test_path(path):
                return (
                    "Error: editing test files is disabled for this SWE-bench "
                    "agent. The hidden official tests are applied during eval; "
                    "fix production code and use run_python or nearby existing "
                    "tests for verification."
                )
            try:
                expected_count = int(args.get("expected_count", 1))
            except (TypeError, ValueError):
                expected_count = 1
            r = sh.replace_text(
                path,
                old,
                new,
                expected_count=expected_count,
            )
            if r.exit_code != 0:
                return f"Error replacing text in {path!r}: {r.stderr[:500]}"
            base_msg = r.stdout or f"Replaced text in {path}."
            # V8 linter pre-check: catch syntax errors before model wastes a
            # full pytest cycle on a broken file.
            if path.endswith(".py"):
                lint = sh.run(f"python -m py_compile {shlex.quote(path)} 2>&1",
                              timeout_s=10)
                if lint.exit_code != 0:
                    base_msg += (
                        "\n\nLINT WARNING: file has Python syntax errors after "
                        "this edit:\n" + (lint.stdout + lint.stderr)[:400]
                        + "\nFix syntax before running tests."
                    )
            return base_msg
        if name == "write_file":
            path = args.get("path")
            content = args.get("content")
            if path is None or content is None:
                return (
                    "Error: write_file requires both 'path' and 'content'. "
                    "If your previous response was cut off mid-call by max_tokens, "
                    "try a smaller / more surgical edit."
                )
            if _is_test_path(path):
                return (
                    "Error: editing test files is disabled for this SWE-bench "
                    "agent. The submitted patch should fix production code."
                )
            r = sh.write_file(path, content)
            if r.exit_code != 0:
                return f"Error writing {path!r}: {r.stderr[:300]}"
            base_msg = f"Wrote {len(content)} chars to {path}."
            # V8 linter pre-check.
            if path.endswith(".py"):
                lint = sh.run(f"python -m py_compile {shlex.quote(path)} 2>&1",
                              timeout_s=10)
                if lint.exit_code != 0:
                    base_msg += (
                        "\n\nLINT WARNING: file has Python syntax errors after "
                        "this edit:\n" + (lint.stdout + lint.stderr)[:400]
                        + "\nFix syntax before running tests."
                    )
            return base_msg
        if name == "run_reproduction":
            code = args.get("code")
            if not code:
                return "Error: run_reproduction requires non-empty 'code'."
            try:
                timeout_s = float(args.get("timeout_s", 60.0))
            except (TypeError, ValueError):
                timeout_s = 60.0
            timeout_s = min(max(timeout_s, 1.0), 120.0)
            r = sh.run_reproduction(code, timeout_s=timeout_s)
            expected = args.get("expected_behavior", "")
            return (f"Reproduction exit_code={r.exit_code}\n"
                    f"Expected behavior: {expected}\n"
                    f"--- stdout ---\n{r.stdout}\n"
                    f"--- stderr ---\n{r.stderr[:500]}")
        if name == "run_python":
            code = args.get("code")
            if not code:
                return "Error: run_python requires non-empty 'code'."
            try:
                timeout_s = float(args.get("timeout_s", 60.0))
            except (TypeError, ValueError):
                timeout_s = 60.0
            timeout_s = min(max(timeout_s, 1.0), 120.0)
            r = sh.run_python(code, timeout_s=timeout_s)
            return (f"Python exit_code={r.exit_code}\n"
                    f"--- stdout ---\n{r.stdout}\n"
                    f"--- stderr ---\n{r.stderr[:500]}")
        if name == "run_tests":
            targets = args.get("test_targets", [])
            if not targets:
                # Default to FAIL_TO_PASS (the agent should know what tests to run).
                targets = list(fail_to_pass)
            r = sh.run_tests(targets, timeout_s=210)
            hint = ""
            combined = f"{r.stdout}\n{r.stderr}".lower()
            if r.exit_code == 4 and (
                "not found" in combined
                or "no tests ran" in combined
                or "not match" in combined
            ):
                hint = (
                    "\nNote: pytest could not collect one or more requested "
                    "selectors. In SWE-bench this often means the hidden "
                    "test patch is not applied in the worktree. Do not add "
                    "or edit tests just to create the missing selector; fix "
                    "the production code and run a nearby existing test or "
                    "a small reproduction instead."
                )
            return (f"Test exit_code={r.exit_code}\n"
                    f"--- stdout ---\n{r.stdout}\n"
                    f"--- stderr ---\n{r.stderr[:500]}"
                    f"{hint}")
        if name == "propose_test_fix":
            test_path = args.get("test_path", "")
            reason = args.get("reason", "")
            patch = args.get("patch", "")
            if not test_path or not reason or not patch:
                return (
                    "Error: propose_test_fix requires test_path, reason, "
                    "and patch."
                )
            return (
                "Test-fix proposal recorded for audit. This does not modify "
                f"the submitted production patch.\nPath: {test_path}\n"
                f"Reason: {reason[:800]}"
            )
        if name == "list_files":
            r = sh.list_files(args.get("path", "."), pattern=args.get("pattern"))
            return r.stdout if r.exit_code == 0 else f"Error: {r.stderr[:300]}"
        if name == "bash":
            r = sh.run(args["command"], timeout_s=60)
            return (f"exit_code={r.exit_code}\n"
                    f"stdout: {r.stdout}\n"
                    f"stderr: {r.stderr[:300]}")
        if name == "done":
            return f"Done acknowledged. Summary: {args.get('summary', '')}"
        return f"Unknown tool: {name!r}"
    except Exception as e:
        return f"Tool execution error: {type(e).__name__}: {e}"


def run_agent_loop(
    *,
    instance_id: str,
    problem_statement: str,
    fail_to_pass: list[str],
    sh: DockerShellExecutor,
    chat_fn: Callable[..., dict],
    system_prompt: str,
    tool_set: str = "minimal",
    temperature: float = 0.0,
    max_tokens_per_turn: int = 4096,
    budget_guard: BudgetGuard | None = None,
    price_per_mtok: dict | None = None,
    recovery_policy: RecoveryPolicy | None = None,
    require_reproduction_before_edit: bool = False,
) -> Trajectory:
    """Run one multi-turn agent loop for one SWE-bench task.

    ``chat_fn`` is a callable that takes Anthropic Messages-API-style
    arguments (``system, messages, tools, max_tokens, temperature``)
    and returns the response dict (with ``content``, ``usage``,
    ``stop_reason``).

    Returns a ``Trajectory`` with the final ``git diff`` as
    ``final_patch``.
    """
    guard = budget_guard or BudgetGuard()
    price = price_per_mtok or _DEFAULT_PRICE
    tracker = BudgetTracker(guard=guard)
    tools = select_tools(tool_set)

    user_msg = (
        f"You are working on bug `{instance_id}`. The repository is "
        f"checked out at /testbed (your shell's CWD).\n\n"
        f"## Problem statement\n{problem_statement}\n\n"
        f"## Tests that must pass after your fix\n"
        + "\n".join(f"- {t}" for t in fail_to_pass[:6])
        + "\n\nUse the tools to explore, edit, and verify. When tests "
        + "pass, call the `done` tool."
    )
    messages: list[dict] = [{"role": "user", "content": user_msg}]

    traj = Trajectory(instance_id=instance_id)
    recent_action_signatures: list[str] = []
    recent_text_signatures: list[str] = []
    no_tool_retries = 0
    edit_made = False
    paralysis_nudge_sent = False
    EDITING_TOOLS = {"write_file", "replace_text"}
    PARALYSIS_TURN_THRESHOLD = 3
    fail_to_pass_passed = False
    end_game_nudge_sent = False
    done_redirect_sent = False
    fail_to_pass_set = {t.strip() for t in fail_to_pass}
    recovery = recovery_policy or RecoveryPolicy()
    stable_checkpoint_path = ""
    edit_counts_since_checkpoint: dict[str, int] = {}
    recoveries_used = 0

    def checkpoint_stable(reason: str) -> None:
        nonlocal stable_checkpoint_path
        if not recovery.enabled:
            return
        try:
            r = sh.create_checkpoint(f"v7_{reason}_{len(traj.turns)}")
        except Exception:
            return
        if r.exit_code == 0 and r.stdout.strip():
            stable_checkpoint_path = r.stdout.strip().splitlines()[-1]
            edit_counts_since_checkpoint.clear()

    checkpoint_stable("start")

    while True:
        stop, reason = tracker.should_stop()
        if stop:
            traj.stop_reason = reason
            break

        t0 = time.perf_counter()
        try:
            resp = chat_fn(
                system=system_prompt,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens_per_turn,
                temperature=temperature,
            )
        except Exception as e:
            traj.stop_reason = f"chat_fn error: {type(e).__name__}: {e}"
            break
        wall_ms = (time.perf_counter() - t0) * 1000

        usage = resp.get("usage", {})
        in_tok = usage.get("input_tokens", 0)
        out_tok = usage.get("output_tokens", 0)
        # Cache-aware billing. Providers expose hit/miss splits differently:
        # - DeepSeek: prompt_cache_hit_tokens / prompt_cache_miss_tokens
        # - Anthropic: cache_read_input_tokens / cache_creation_input_tokens
        # If hit-token info is present, charge at the cached rate; remainder
        # is full-price input. Otherwise treat all as full-price input.
        cache_hit = (
            usage.get("cache_hit_tokens", 0)
            or usage.get("prompt_cache_hit_tokens", 0)
            or usage.get("cache_read_input_tokens", 0)
            or 0
        )
        cache_hit = min(cache_hit, in_tok)
        cache_miss = max(0, in_tok - cache_hit)
        rate_in_cached = price.get("in_cached", price.get("in", 0.0))
        usd = (
            cache_miss * price["in"]
            + cache_hit * rate_in_cached
            + out_tok * price["out"]
        ) / 1_000_000.0

        # Parse content blocks: text + tool_use.
        content_blocks = resp.get("content", [])
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        for block in content_blocks:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append({
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "input": block.get("input", {}),
                })

        text = "\n".join(text_parts)
        loop_warning = ""
        if tool_calls:
            sig = _tool_signature(tool_calls)
            recent_action_signatures.append(sig)
            if (
                len(recent_action_signatures) >= 3
                and len(set(recent_action_signatures[-3:])) == 1
            ):
                loop_warning = (
                    "Loop guard: you have repeated the same tool action three "
                    "times. Do not repeat that action again. Pick one concrete "
                    "alternative now: inspect a smaller range, use search_text, "
                    "make a different exact replace_text edit, run_tests, or "
                    "call done if the fix is already complete."
                )
        if text.strip():
            text_sig = " ".join(text.lower().split())[:500]
            recent_text_signatures.append(text_sig)
            if (
                not loop_warning
                and len(recent_text_signatures) >= 3
                and len(set(recent_text_signatures[-3:])) == 1
            ):
                loop_warning = (
                    "Loop guard: your last three reasoning messages were "
                    "effectively identical. Stop restating the same plan and "
                    "take a new concrete tool action."
                )
        tracker.add_turn(in_tok, out_tok, wall_ms / 1000.0, usd)
        traj.total_in_tokens += in_tok
        traj.total_out_tokens += out_tok
        traj.total_wall_s += wall_ms / 1000.0
        traj.total_usd += usd

        # Append the assistant message for next-turn context.
        messages.append({"role": "assistant", "content": content_blocks})

        # Execute tool calls (typically 1 per turn for Sonnet/Anthropic).
        tool_results = []
        done_seen = False
        passed_tests_seen = False
        for call in tool_calls:
            name = call["name"]
            call_input = call["input"]
            if (
                name in EDITING_TOOLS
                and require_reproduction_before_edit
                and not traj.reproduction_attempted
            ):
                obs = (
                    "Speculative test gate: before the first production-code "
                    "edit, write and run a focused reproduction with "
                    "run_reproduction (preferred) or run_python. Show the bug "
                    "or the desired invariant first; then edit."
                )
            else:
                obs = _execute_tool(sh, name, call_input,
                                    fail_to_pass=fail_to_pass)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": call["id"],
                "content": obs,
            })

            if name == "propose_test_fix" and not obs.startswith("Error:"):
                traj.test_fix_proposals.append({
                    "turn_idx": tracker.turns - 1,
                    "test_path": call_input.get("test_path", ""),
                    "reason": call_input.get("reason", ""),
                    "patch": call_input.get("patch", ""),
                    "confidence": call_input.get("confidence", 0.5),
                })

            if name in {"run_reproduction", "run_python"}:
                traj.reproduction_attempted = True
                if (
                    obs.startswith("Reproduction exit_code=0")
                    or obs.startswith("Python exit_code=0")
                ):
                    traj.reproduction_passed = True
                    if recovery.checkpoint_after_successful_reproduction:
                        checkpoint_stable("repro_pass")

            if name in EDITING_TOOLS and not obs.startswith("Error:") and not obs.startswith("Speculative"):
                edit_made = True
                edited_path = str(call_input.get("path", ""))
                edit_counts_since_checkpoint[edited_path] = (
                    edit_counts_since_checkpoint.get(edited_path, 0) + 1
                )
                edit_count = edit_counts_since_checkpoint[edited_path]
                if (
                    recovery.enabled
                    and edited_path
                    and edit_count >= recovery.same_file_edit_threshold
                    and recoveries_used < recovery.max_recoveries
                    and not passed_tests_seen
                    and not fail_to_pass_passed
                ):
                    reason = (
                        f"same file edited {edit_count} times without a "
                        "successful verification signal"
                    )
                    try:
                        if stable_checkpoint_path:
                            restore = sh.restore_checkpoint(stable_checkpoint_path)
                            checkpoint_label = stable_checkpoint_path
                        else:
                            restore = sh.reset_to_clean()
                            checkpoint_label = "HEAD"
                    except Exception as exc:  # pragma: no cover - defensive
                        restore = ExecResult(
                            stdout="",
                            stderr=f"{type(exc).__name__}: {exc}",
                            exit_code=1,
                            elapsed_s=0.0,
                        )
                        checkpoint_label = stable_checkpoint_path or "HEAD"
                    recoveries_used += 1
                    edit_counts_since_checkpoint.clear()
                    event = RecoveryEvent(
                        turn_idx=tracker.turns - 1,
                        reason=reason,
                        edited_path=edited_path,
                        edit_count=edit_count,
                        checkpoint_path=checkpoint_label,
                        restore_exit_code=restore.exit_code,
                        restore_stderr=restore.stderr[:500],
                    )
                    traj.recovery_events.append(event)
                    tool_results.append({
                        "type": "text",
                        "text": (
                            "STATE-SPACE RECOVERY: restored the worktree to "
                            f"{checkpoint_label} because {reason}. Do not "
                            "repeat the same edit. Switch hypothesis, inspect "
                            "definitions/references, or write a smaller "
                            "reproduction before editing again."
                        ),
                    })

            if name == "done":
                done_seen = True
            if name == "run_tests" and obs.startswith("Test exit_code=0"):
                passed_tests_seen = True
                # STRICT auto-stop: only treat tests as "passed" for the
                # purposes of stopping if the agent actually ran the
                # FAIL_TO_PASS targets (exact selectors). Otherwise it may
                # be running unrelated tests that pass with no fix.
                requested = call_input.get("test_targets", [])
                if not requested:
                    requested = list(fail_to_pass)
                requested_set = {t.strip() for t in requested}
                if fail_to_pass_set and fail_to_pass_set.issubset(requested_set):
                    fail_to_pass_passed = True
                if recovery.checkpoint_after_successful_tests:
                    checkpoint_stable("tests_pass")

        # Record the turn.
        traj.turns.append(TurnRecord(
            turn_idx=tracker.turns - 1,
            text=text,
            tool_calls=tool_calls,
            tool_results=tool_results,
            in_tokens=in_tok, out_tokens=out_tok, wall_ms=wall_ms,
            stop_reason=resp.get("stop_reason"),
        ))

        if done_seen:
            # Verification gate: don't accept `done` unless the agent has
            # actually verified its fix by passing FAIL_TO_PASS at least once.
            # If not, redirect once and let the loop continue.
            if fail_to_pass_passed or done_redirect_sent or not fail_to_pass_set:
                traj.done_emitted = True
                traj.stop_reason = "done"
                break
            done_redirect_sent = True
            tool_results.append({
                "type": "text",
                "text": (
                    "DONE rejected: you have not yet successfully run the "
                    "FAIL_TO_PASS tests for this task. Before calling done, "
                    "call run_tests with these exact targets and confirm "
                    "exit_code=0:\n"
                    + "\n".join(f"  - {t}" for t in list(fail_to_pass)[:6])
                ),
            })
            # fall through to append tool_results and continue

        if passed_tests_seen and fail_to_pass_passed:
            production_patch = sh.get_diff(exclude_tests=True)
            if production_patch.strip():
                traj.final_patch = production_patch
                traj.done_emitted = True
                traj.stop_reason = "tests_passed"
                break

        if not tool_calls:
            if resp.get("stop_reason") == "max_tokens" and no_tool_retries < 2:
                no_tool_retries += 1
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": (
                            "Your previous response hit max_tokens before a "
                            "tool call. On the next turn, do not explain. Make "
                            "exactly one small tool call, preferably search_text, "
                            "go_to_definition, find_references, read_file_range, "
                            "run_reproduction, replace_text, run_tests, or done."
                        ),
                    }],
                })
                continue
            # Model returned only text and no tool call — nothing to do.
            traj.stop_reason = "no_tool_call"
            break

        no_tool_retries = 0
        if loop_warning:
            tool_results.append({
                "type": "text",
                "text": loop_warning,
            })

        # Anti-paralysis: if the agent has explored for too long without
        # making any edit, fire an escalating nudge every turn until an
        # edit is made. One-shot nudges get ignored; persistent ones don't.
        if (
            not edit_made
            and tracker.turns >= PARALYSIS_TURN_THRESHOLD
        ):
            paralysis_nudge_sent = True
            if require_reproduction_before_edit and not traj.reproduction_attempted:
                nudge_text = (
                    f"ANTI-PARALYSIS (turn {tracker.turns}, no reproduction "
                    "or edits yet): stop broad exploration. Your next tool "
                    "call MUST be run_reproduction with a small Python script "
                    "that captures the bug or expected invariant. After that, "
                    "make one surgical production-code edit."
                )
            else:
                nudge_text = (
                    f"ANTI-PARALYSIS (turn {tracker.turns}, no edits yet): "
                    "STOP exploring. Your next tool call MUST be replace_text "
                    "or write_file. Read the issue body again, pick the most "
                    "likely production file to change, and make a best-guess "
                    "edit. Wrong edits are recoverable; no edit is not. "
                    "Do not call list_files, search_text, read_file, or "
                    "read_file_range on this next turn."
                )
            tool_results.append({
                "type": "text",
                "text": nudge_text,
            })

        # End-game nudge: when the turn budget is almost spent, push the
        # agent to verify and finalize rather than keep exploring.
        turns_left = max(0, guard.max_turns - tracker.turns)
        if (
            not end_game_nudge_sent
            and turns_left <= 5
            and edit_made
            and not fail_to_pass_passed
        ):
            end_game_nudge_sent = True
            ftp_str = "\n".join(f"  - {t}" for t in list(fail_to_pass)[:6])
            tool_results.append({
                "type": "text",
                "text": (
                    f"END-GAME (only {turns_left} turns left): stop exploring "
                    f"and make this trajectory count. Run the FAIL_TO_PASS "
                    f"tests now with exactly these targets:\n{ftp_str}\n"
                    f"If they pass, call done. If they fail, make ONE more "
                    f"surgical replace_text fix based on the failure output, "
                    f"then run again. Do not read more files."
                ),
            })

        # Append tool results for next turn.
        messages.append({"role": "user", "content": tool_results})

    # Extract the final patch from the working tree.
    if not traj.final_patch:
        traj.final_patch = sh.get_diff(exclude_tests=True)
    return traj
