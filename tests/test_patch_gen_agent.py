"""Phase 3 P3c — agent-path unit tests.

Covers:

  - Tool-call JSON parsing: clean JSON, fenced JSON, missing tool,
    unknown tool, non-dict args.
  - read_file tool: offset/limit, missing file, line-numbered output.
  - search_text tool: regex query, optional glob, no-matches case.
  - list_dir tool: relative paths, missing dir.
  - apply_patch tool: success path stores diff, failure returns git
    error, repeated calls overwrite the candidate.
  - submit tool: requires a prior successful apply_patch; finalizes
    rationale.
  - Agent loop:
      * scripted multi-turn: read_file → apply_patch (fail) →
        apply_patch (ok) → submit → done.
      * T_max enforced even with a half-finished trajectory.
      * cost cap enforced between turns.
      * parse_failure_budget bounds invalid-JSON loops.
      * T_max with validated candidate auto-finalizes.
      * T_max with no candidate returns final_status="tmax_no_apply".
      * superset assertion runs (firewall test still green).
      * NO repro module imported (firewall test still green).

No real LLM calls; no real Docker. ``_FakeSandbox`` emulates the
read_file / run_shell surface used by the tool implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import patch

import pytest

from harness.localization_signals import RankedFile
from harness.patch_gen import (
    AgentGenerationResult,
    PatchCandidate,
    PatchGenError,
    generate_agent,
)
from harness.patch_gen.agent import (
    _AgentState,
    _dispatch_tool,
    _parse_tool_call,
    _tool_apply_patch,
    _tool_list_dir,
    _tool_read_file,
    _tool_search_text,
    _tool_submit,
)
from harness.views import (
    InstanceView,
    RepoSkeleton,
    TestDirectives,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class _FakeExecResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""


@dataclass
class _FakeSandbox:
    """Just enough of Sandbox to run the agent's 5 tools.

    `read_file` returns from the in-memory file map.
    `run_shell` is scripted: each invocation pops a result from the
    `shell_outputs` queue, falling back to a default exit=1 result.
    """

    files: dict[str, str] = field(default_factory=dict)
    shell_outputs: list[_FakeExecResult] = field(default_factory=list)
    shell_calls: list[str] = field(default_factory=list)

    def read_file(self, path: str, max_chars: int | None = None) -> _FakeExecResult:
        if path not in self.files:
            return _FakeExecResult(exit_code=1)
        content = self.files[path]
        if max_chars is not None and len(content) > max_chars:
            content = content[:max_chars]
        return _FakeExecResult(exit_code=0, stdout=content)

    def run_shell(self, cmd: str, timeout_s: float = 30.0) -> _FakeExecResult:
        self.shell_calls.append(cmd)
        if self.shell_outputs:
            return self.shell_outputs.pop(0)
        return _FakeExecResult(exit_code=1, stdout="(no scripted output)")


def _make_view() -> InstanceView:
    return InstanceView(
        instance_id="example__example-1",
        repo="example/example",
        base_commit="abcdef0123",
        problem_statement="Bug: foo() returns None instead of raising",
        repo_skeleton=RepoSkeleton(repo="example/example", base_commit="abcdef0123"),
        test_directives=TestDirectives(dirs=("tests/",), source="discovery:1"),
    )


def _make_ranked_files(paths: list[str]) -> list[RankedFile]:
    return [
        RankedFile(
            file_path=p, final_score=1.0 - i * 0.05,
            rationale=f"r{i}", upstream_signals=("bm25",),
            upstream_best_rank={"bm25": i + 1},
        )
        for i, p in enumerate(paths)
    ]


@dataclass
class _FakeChat:
    text: str
    model: str = "deepseek-chat"
    input_tokens: int = 100
    output_tokens: int = 50


# ---------------------------------------------------------------------------
# 1. JSON parser
# ---------------------------------------------------------------------------


def test_parse_tool_call_clean():
    out = _parse_tool_call('{"tool": "read_file", "args": {"path": "src/foo.py"}}')
    assert out["tool"] == "read_file"
    assert out["args"]["path"] == "src/foo.py"


def test_parse_tool_call_fenced():
    out = _parse_tool_call('```json\n{"tool": "list_dir", "args": {"path": "."}}\n```')
    assert out["tool"] == "list_dir"


def test_parse_tool_call_invalid_json():
    with pytest.raises(PatchGenError, match="not valid JSON"):
        _parse_tool_call("not JSON at all")


def test_parse_tool_call_unknown_tool():
    with pytest.raises(PatchGenError, match="must be one of"):
        _parse_tool_call('{"tool": "exec_python", "args": {}}')


def test_parse_tool_call_args_must_be_dict():
    with pytest.raises(PatchGenError, match="args"):
        _parse_tool_call('{"tool": "read_file", "args": "src/foo.py"}')


def test_parse_tool_call_top_level_must_be_object():
    with pytest.raises(PatchGenError, match="must be a JSON object"):
        _parse_tool_call('["read_file", {}]')


# ---------------------------------------------------------------------------
# 2. read_file tool
# ---------------------------------------------------------------------------


def test_read_file_returns_line_numbered_slice():
    sb = _FakeSandbox(files={"src/foo.py": "a\nb\nc\nd\ne\n"})
    out = _tool_read_file(sb, {"path": "src/foo.py"})
    assert "1\ta" in out
    assert "5\te" in out


def test_read_file_offset_and_limit():
    sb = _FakeSandbox(files={"src/foo.py": "\n".join(f"line{i}" for i in range(50))})
    out = _tool_read_file(sb, {"path": "src/foo.py", "offset": 10, "limit": 5})
    assert "11\tline10" in out
    assert "15\tline14" in out
    assert "16\tline15" not in out


def test_read_file_missing_returns_error():
    sb = _FakeSandbox()
    out = _tool_read_file(sb, {"path": "nope.py"})
    assert out.startswith("[error]") and "not found" in out


def test_read_file_missing_path_arg():
    sb = _FakeSandbox()
    out = _tool_read_file(sb, {})
    assert out.startswith("[error]") and "missing" in out


# ---------------------------------------------------------------------------
# 3. search_text tool
# ---------------------------------------------------------------------------


def test_search_text_with_results():
    sb = _FakeSandbox(shell_outputs=[
        _FakeExecResult(exit_code=0, stdout="src/foo.py:42:def foo(x):\nsrc/bar.py:13:def foo_helper():\n"),
    ])
    out = _tool_search_text(sb, {"query": "def foo"})
    assert "src/foo.py:42:def foo" in out
    assert "rg" in sb.shell_calls[-1]


def test_search_text_with_glob():
    sb = _FakeSandbox(shell_outputs=[_FakeExecResult(exit_code=0, stdout="x")])
    _tool_search_text(sb, {"query": "foo", "glob": "*.py"})
    assert "--glob" in sb.shell_calls[-1]
    assert "'*.py'" in sb.shell_calls[-1]


def test_search_text_no_matches():
    sb = _FakeSandbox(shell_outputs=[_FakeExecResult(exit_code=1, stdout="")])
    out = _tool_search_text(sb, {"query": "definitely_not_present"})
    assert "no matches" in out


def test_search_text_missing_query():
    sb = _FakeSandbox()
    out = _tool_search_text(sb, {})
    assert out.startswith("[error]")


# ---------------------------------------------------------------------------
# 4. list_dir tool
# ---------------------------------------------------------------------------


def test_list_dir_lists_entries():
    sb = _FakeSandbox(shell_outputs=[
        _FakeExecResult(exit_code=0, stdout="foo.py\nbar.py\nsubdir/\n"),
    ])
    out = _tool_list_dir(sb, {"path": "src"})
    assert "foo.py" in out
    assert "subdir/" in out


def test_list_dir_default_path_is_repo_root():
    sb = _FakeSandbox(shell_outputs=[_FakeExecResult(exit_code=0, stdout="x")])
    _tool_list_dir(sb, {})
    assert "ls -1F '.'" in sb.shell_calls[-1]


# ---------------------------------------------------------------------------
# 5. apply_patch tool
# ---------------------------------------------------------------------------


def test_apply_patch_success_stores_candidate():
    sb = _FakeSandbox(shell_outputs=[_FakeExecResult(exit_code=0)])
    state = _AgentState()
    out = _tool_apply_patch(sb, {"diff": "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n"}, state)
    assert out.startswith("[apply_patch: ok]")
    assert state.candidate_diff is not None
    assert state.candidate_apply_count == 1
    assert state.apply_attempts == 1


def test_apply_patch_failure_returns_git_error():
    sb = _FakeSandbox(shell_outputs=[
        _FakeExecResult(exit_code=1, stderr="error: patch failed: src/x.py:42\n"),
    ])
    state = _AgentState()
    out = _tool_apply_patch(sb, {"diff": "diff data"}, state)
    assert "FAILED" in out
    assert "src/x.py:42" in out
    assert state.candidate_diff is None
    assert state.apply_attempts == 1


def test_apply_patch_replaces_prior_candidate():
    sb = _FakeSandbox(shell_outputs=[
        _FakeExecResult(exit_code=0),
        _FakeExecResult(exit_code=0),
    ])
    state = _AgentState()
    _tool_apply_patch(sb, {"diff": "diff1"}, state)
    _tool_apply_patch(sb, {"diff": "diff2"}, state)
    assert state.candidate_diff == "diff2"
    assert state.candidate_apply_count == 2


def test_apply_patch_empty_diff_rejected():
    sb = _FakeSandbox()
    state = _AgentState()
    out = _tool_apply_patch(sb, {"diff": ""}, state)
    assert out.startswith("[error]")
    assert state.apply_attempts == 0


# ---------------------------------------------------------------------------
# 6. submit tool
# ---------------------------------------------------------------------------


def test_submit_requires_prior_apply():
    state = _AgentState()
    out = _tool_submit({"rationale": "fixed it"}, state)
    assert out.startswith("[error]")
    assert not state.submit_called


def test_submit_finalizes_after_apply():
    state = _AgentState(candidate_diff="diff")
    out = _tool_submit({"rationale": "fixed it"}, state)
    assert "finalized" in out
    assert state.submit_called is True
    assert state.submitted_rationale == "fixed it"


def test_submit_default_rationale_when_blank():
    state = _AgentState(candidate_diff="diff")
    _tool_submit({"rationale": ""}, state)
    assert state.submitted_rationale and "no rationale" in state.submitted_rationale


# ---------------------------------------------------------------------------
# 7. Agent loop with scripted LLM
# ---------------------------------------------------------------------------


def _scripted_chat(scripts: list[str]):
    """Yield one _FakeChat per call by popping from the script list.
    Raises StopIteration after exhaustion (which becomes a parse-fail
    or T_max in the loop)."""
    it = iter(scripts)

    def _next(*args, **kwargs):
        return _FakeChat(text=next(it))

    return _next


def test_agent_loop_happy_path_submits():
    """read_file → apply_patch (ok) → submit."""
    view = _make_view()
    rfs = _make_ranked_files(["src/foo.py"])
    sb = _FakeSandbox(
        files={"src/foo.py": "def foo(): return None\n"},
        shell_outputs=[_FakeExecResult(exit_code=0)],  # apply_patch success
    )
    scripts = [
        '{"tool": "read_file", "args": {"path": "src/foo.py"}}',
        '{"tool": "apply_patch", "args": {"diff": "diff --git a/x b/x\\n--- a/x\\n+++ b/x\\n@@ -1 +1 @@\\n-return None\\n+raise\\n"}}',
        '{"tool": "submit", "args": {"rationale": "raise instead of return None"}}',
    ]
    with patch("harness.llm.clients.complete_chat", side_effect=_scripted_chat(scripts)):
        result = generate_agent(view=view, ranked_files=rfs, sandbox=sb)
    assert result.submitted is True
    assert result.candidate is not None
    assert result.candidate.source_route == "agent"
    assert result.candidate.candidate_id == "agent_a0_final"
    assert result.final_status == "submitted"
    assert result.turns_used == 3
    assert result.apply_successes == 1


def test_agent_loop_apply_fail_then_succeed():
    """apply_patch fails once; agent retries with fixed diff; then submits."""
    view = _make_view()
    rfs = _make_ranked_files(["src/foo.py"])
    sb = _FakeSandbox(
        files={"src/foo.py": "def foo(): pass\n"},
        shell_outputs=[
            _FakeExecResult(exit_code=1, stderr="error: patch failed: src/foo.py:1"),
            _FakeExecResult(exit_code=0),
        ],
    )
    scripts = [
        '{"tool": "apply_patch", "args": {"diff": "broken_diff_v1"}}',
        '{"tool": "apply_patch", "args": {"diff": "fixed_diff_v2"}}',
        '{"tool": "submit", "args": {"rationale": "fixed line numbers"}}',
    ]
    with patch("harness.llm.clients.complete_chat", side_effect=_scripted_chat(scripts)):
        result = generate_agent(view=view, ranked_files=rfs, sandbox=sb)
    assert result.submitted is True
    assert result.candidate.diff == "fixed_diff_v2"
    assert result.apply_attempts == 2
    assert result.apply_successes == 1


def test_agent_loop_tmax_with_validated_candidate_autosubmits():
    """Agent validates a patch but never calls submit before T_max.
    The loop auto-finalizes."""
    view = _make_view()
    rfs = _make_ranked_files(["src/foo.py"])
    sb = _FakeSandbox(
        files={"src/foo.py": "x"},
        shell_outputs=[_FakeExecResult(exit_code=0)] + [_FakeExecResult(exit_code=0, stdout="ok")] * 10,
    )
    scripts = (
        ['{"tool": "apply_patch", "args": {"diff": "diff data"}}']
        + ['{"tool": "list_dir", "args": {"path": "."}}'] * 9
    )
    with patch("harness.llm.clients.complete_chat", side_effect=_scripted_chat(scripts)):
        result = generate_agent(view=view, ranked_files=rfs, sandbox=sb, t_max=10)
    assert result.submitted is True
    assert result.final_status == "tmax_autosubmit"
    assert result.candidate.diff == "diff data"


def test_agent_loop_tmax_with_no_validated_candidate_returns_no_apply():
    view = _make_view()
    rfs = _make_ranked_files(["src/foo.py"])
    sb = _FakeSandbox(
        files={"src/foo.py": "x"},
        shell_outputs=[_FakeExecResult(exit_code=0, stdout="ok")] * 5,
    )
    scripts = ['{"tool": "list_dir", "args": {"path": "."}}'] * 5
    with patch("harness.llm.clients.complete_chat", side_effect=_scripted_chat(scripts)):
        result = generate_agent(view=view, ranked_files=rfs, sandbox=sb, t_max=5)
    assert result.submitted is False
    assert result.candidate is None
    assert result.final_status == "tmax_no_apply"


def test_agent_loop_parse_failures_exhausted():
    view = _make_view()
    rfs = _make_ranked_files(["src/foo.py"])
    sb = _FakeSandbox(files={"src/foo.py": "x"})
    scripts = ["not JSON"] * 5
    with patch("harness.llm.clients.complete_chat", side_effect=_scripted_chat(scripts)):
        result = generate_agent(
            view=view, ranked_files=rfs, sandbox=sb,
            t_max=10, parse_failure_budget=3,
        )
    assert result.submitted is False
    assert result.final_status == "parse_failures_exhausted"


def test_agent_loop_cost_cap_stops_loop():
    view = _make_view()
    rfs = _make_ranked_files(["src/foo.py"])
    sb = _FakeSandbox(files={"src/foo.py": "x"}, shell_outputs=[_FakeExecResult(exit_code=0, stdout="ok")] * 100)
    # Each call records 10M output tokens × $1.10/1M = $11. Cap at $0.50;
    # one call exceeds the cap immediately. The next-turn check stops.
    expensive = _FakeChat(text='{"tool": "list_dir", "args": {"path": "."}}', input_tokens=0, output_tokens=10_000_000)
    with patch("harness.llm.clients.complete_chat", return_value=expensive):
        result = generate_agent(view=view, ranked_files=rfs, sandbox=sb, t_max=20, cost_cap_usd=0.50)
    assert result.cost_cap_hit is True
    assert result.final_status == "cost_cap"
    assert result.turns_used == 1   # first turn fires; second turn check stops


# ---------------------------------------------------------------------------
# 8. Defense in depth — superset assertion runs from the agent path
# ---------------------------------------------------------------------------


def test_agent_loop_propagates_superset_assertion_failure():
    """If repro_context_files isn't a subset of patch context, the
    superset assertion should fire from inside generate_agent."""
    from harness.patch_gen.views import ContextSupersetError

    view = _make_view()
    rfs = _make_ranked_files(["src/foo.py"])
    sb = _FakeSandbox(files={"src/foo.py": "x"})
    with pytest.raises(ContextSupersetError):
        generate_agent(
            view=view, ranked_files=rfs, sandbox=sb,
            repro_context_files=("src/foo.py", "src/never_seen.py"),
        )


def test_dispatch_unknown_tool_via_dispatch_returns_error():
    """Defense in depth: even if the parser missed a tool, _dispatch_tool
    is the last line."""
    state = _AgentState()
    out = _dispatch_tool(_FakeSandbox(), {"tool": "wat", "args": {}}, state)
    assert "unknown tool" in out


# ---------------------------------------------------------------------------
# 9. Field-firewall on PatchCandidate from agent
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 10. P3c-v2 Fix C — pipeline-bootstrapped agent (seed_diff)
# ---------------------------------------------------------------------------


def test_seed_diff_appears_in_initial_user_prompt():
    """When seed_diff is provided, the initial user prompt must include
    the seed and a directive to apply_patch it as the first action."""
    from harness.patch_gen.agent import _build_user_prompt

    view = _make_view()
    rfs = _make_ranked_files(["src/foo.py"])
    seed = "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n"
    prompt = _build_user_prompt(view, rfs, seed_diff=seed)
    assert "Seed diff" in prompt or "seed diff" in prompt.lower()
    assert seed in prompt
    assert "apply_patch" in prompt
    assert "FIRST action" in prompt or "first action" in prompt.lower()


def test_seed_diff_none_produces_unbootstrapped_prompt():
    """No seed → original prompt with 'Begin' but no seed-diff section."""
    from harness.patch_gen.agent import _build_user_prompt

    view = _make_view()
    rfs = _make_ranked_files(["src/foo.py"])
    prompt = _build_user_prompt(view, rfs, seed_diff=None)
    assert "Seed diff" not in prompt
    assert "# Begin" in prompt


def test_seed_diff_empty_string_treated_as_no_seed():
    from harness.patch_gen.agent import _build_user_prompt

    view = _make_view()
    rfs = _make_ranked_files(["src/foo.py"])
    prompt = _build_user_prompt(view, rfs, seed_diff="   ")
    assert "Seed diff" not in prompt


def test_agent_loop_with_seed_passes_seed_into_initial_message():
    """End-to-end: when generate_agent is called with seed_diff, the
    initial user message contains the seed."""
    view = _make_view()
    rfs = _make_ranked_files(["src/foo.py"])
    sb = _FakeSandbox(
        files={"src/foo.py": "x"},
        shell_outputs=[_FakeExecResult(exit_code=0)],
    )
    seed = "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n"
    captured: list[list] = []

    def _capture_chat(messages, **kwargs):
        # First call: capture the initial message stack.
        if not captured:
            captured.append(list(messages))
        return _FakeChat(text='{"tool": "apply_patch", "args": {"diff": "d"}}')

    scripts = [
        '{"tool": "apply_patch", "args": {"diff": "d"}}',
        '{"tool": "submit", "args": {"rationale": "r"}}',
    ]
    with patch("harness.llm.clients.complete_chat", side_effect=_scripted_chat(scripts)) as mock_cc:
        result = generate_agent(view=view, ranked_files=rfs, sandbox=sb, seed_diff=seed)
    assert result.submitted is True
    # Inspect what was passed in messages on the first call.
    first_call_messages = mock_cc.call_args_list[0].kwargs.get("messages") or mock_cc.call_args_list[0].args[0]
    user_msg = next(m for m in first_call_messages if m["role"] == "user")
    assert seed in user_msg["content"]


# ---------------------------------------------------------------------------
# 11. P3c-v2 Fix A — T_max-5 force-finalize nudge
# ---------------------------------------------------------------------------


def test_force_finalize_nudge_with_candidate_says_submit():
    from harness.patch_gen.agent import _build_force_finalize_nudge

    msg = _build_force_finalize_nudge(turns_remaining=5, has_candidate=True)
    assert "5 turns remaining" in msg
    assert "submit" in msg.lower()
    assert "validated candidate" in msg.lower()


def test_force_finalize_nudge_without_candidate_says_apply_patch():
    from harness.patch_gen.agent import _build_force_finalize_nudge

    msg = _build_force_finalize_nudge(turns_remaining=5, has_candidate=False)
    assert "5 turns remaining" in msg
    assert "STOP exploring" in msg or "stop exploring" in msg.lower()
    assert "apply_patch" in msg
    assert "wrong patch" in msg.lower() or "best-guess" in msg.lower()


def test_agent_loop_injects_nudge_at_t_max_minus_5():
    """With t_max=20 and force_finalize_at_turns_remaining=5, the
    nudge fires when turns_used reaches 15 (turns_remaining=5)."""
    view = _make_view()
    rfs = _make_ranked_files(["src/foo.py"])
    sb = _FakeSandbox(
        files={"src/foo.py": "x"},
        shell_outputs=[_FakeExecResult(exit_code=0, stdout="ok")] * 30,
    )
    # Script: 14 list_dir calls (no apply attempts), then a 15th turn
    # (the harness injects the nudge BEFORE this turn), then another
    # list_dir call. We don't care about the result; we want to inspect
    # the messages for the nudge.
    scripts = ['{"tool": "list_dir", "args": {"path": "."}}'] * 20
    captured_messages: list = []

    def _capture(*args, **kwargs):
        msgs = kwargs.get("messages") or args[0]
        captured_messages.append([dict(m) for m in msgs])
        return _FakeChat(text=next(it_scripts))
    it_scripts = iter(scripts)
    with patch("harness.llm.clients.complete_chat", side_effect=_capture):
        generate_agent(view=view, ranked_files=rfs, sandbox=sb, t_max=20)

    # Find the nudge — should appear as a user message starting with [harness].
    found_nudge_at = None
    for turn_n, msgs in enumerate(captured_messages, start=1):
        for m in msgs:
            if m["role"] == "user" and m["content"].startswith("[harness]"):
                found_nudge_at = turn_n
                break
        if found_nudge_at:
            break
    # Nudge fires at t_max - 5 = 15; the next LLM call (turn 16) is the
    # first to see it in its message stack. Our injection happens
    # BEFORE the chat call when turns_remaining <= 5.
    assert found_nudge_at is not None
    assert 14 <= found_nudge_at <= 16, f"nudge appeared at turn {found_nudge_at}, expected ~15-16"


def test_agent_loop_injects_nudge_only_once():
    """Even when the nudge condition stays true (agent keeps not
    submitting), the nudge is injected only once."""
    view = _make_view()
    rfs = _make_ranked_files(["src/foo.py"])
    sb = _FakeSandbox(
        files={"src/foo.py": "x"},
        shell_outputs=[_FakeExecResult(exit_code=0, stdout="ok")] * 30,
    )
    scripts = ['{"tool": "list_dir", "args": {"path": "."}}'] * 25
    captured_messages: list = []

    def _capture(*args, **kwargs):
        msgs = kwargs.get("messages") or args[0]
        captured_messages.append([dict(m) for m in msgs])
        return _FakeChat(text=next(it_scripts))
    it_scripts = iter(scripts)
    with patch("harness.llm.clients.complete_chat", side_effect=_capture):
        generate_agent(view=view, ranked_files=rfs, sandbox=sb, t_max=20)

    # Inspect the LAST message stack — count how many nudges are present.
    if captured_messages:
        last = captured_messages[-1]
        n_nudges = sum(1 for m in last if m["role"] == "user" and m["content"].startswith("[harness]"))
        assert n_nudges <= 1, f"nudge injected {n_nudges} times; should be 1"


def test_agent_loop_skips_nudge_after_submit():
    """If the agent submits BEFORE T_max - 5, no nudge is needed."""
    view = _make_view()
    rfs = _make_ranked_files(["src/foo.py"])
    sb = _FakeSandbox(
        files={"src/foo.py": "x"},
        shell_outputs=[_FakeExecResult(exit_code=0)],
    )
    scripts = [
        '{"tool": "apply_patch", "args": {"diff": "d"}}',
        '{"tool": "submit", "args": {"rationale": "fast"}}',
    ]
    captured_messages: list = []

    def _capture(*args, **kwargs):
        msgs = kwargs.get("messages") or args[0]
        captured_messages.append([dict(m) for m in msgs])
        return _FakeChat(text=next(it_scripts))
    it_scripts = iter(scripts)
    with patch("harness.llm.clients.complete_chat", side_effect=_capture):
        result = generate_agent(view=view, ranked_files=rfs, sandbox=sb, t_max=20)
    assert result.submitted is True
    # No nudge should have been injected (submit happened on turn 2 of 20).
    if captured_messages:
        last = captured_messages[-1]
        n_nudges = sum(1 for m in last if m["role"] == "user" and m["content"].startswith("[harness]"))
        assert n_nudges == 0


# ---------------------------------------------------------------------------
# 12. Field firewall on the new agent path
# ---------------------------------------------------------------------------


def test_agent_candidate_passes_field_firewall():
    """Make sure the agent's PatchCandidate passes the same firewall
    check the pipeline candidates do."""
    view = _make_view()
    rfs = _make_ranked_files(["src/foo.py"])
    sb = _FakeSandbox(
        files={"src/foo.py": "x"},
        shell_outputs=[_FakeExecResult(exit_code=0)],
    )
    scripts = [
        '{"tool": "apply_patch", "args": {"diff": "d"}}',
        '{"tool": "submit", "args": {"rationale": "r"}}',
    ]
    with patch("harness.llm.clients.complete_chat", side_effect=_scripted_chat(scripts)):
        result = generate_agent(view=view, ranked_files=rfs, sandbox=sb)
    assert isinstance(result.candidate, PatchCandidate)
    assert result.candidate.source_route == "agent"
    assert result.candidate.source_temperature is None
