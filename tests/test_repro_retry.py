"""Phase 2 commit-17b tests: must-fail-at-base + retry + instrumentation.

Covers the four retry-loop guardrails from the 17b spec:

  1. Cost-cap interaction: checked BETWEEN attempts only, never
     mid-API-call. Mid-call kills waste partial spend and corrupt
     the trace.
  2. JSONL trace requirements: one row per attempt actually made;
     post_patch_pass_status MUST be 'unknown' in V0 (V0 cannot see
     the gold patch per §8.5); instance_id mismatch raises.
  3. ReproRejectReason controlled vocabulary: structured reasons,
     not free-form strings.
  4. Widen-flag triggering: deterministic on prior reject reason.

All LLM + sandbox interactions are mocked. Zero real LLM spend.
"""

from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from harness.localization_signals import RankedFile
from harness.repro import (
    GenerateWithRetryResult,
    ReproAttemptTraceRow,
    ReproGeneratorError,
    ReproRejectReason,
    ReproStatus,
    ReproTestCase,
    V0_POST_PATCH_PASS_STATUS,
    _classify_verify_result,
    generate_with_retry,
    widen_for_next_attempt,
    write_trace_row,
)
from harness.views import (
    InstanceView,
    RepoSkeleton,
    TestDirectives,
)


# ---------------------------------------------------------------------------
# Fixtures + fakes
# ---------------------------------------------------------------------------


def _make_view(iid: str = "example__example-1") -> InstanceView:
    return InstanceView(
        instance_id=iid,
        repo="example/example",
        base_commit="abcdef0123",
        problem_statement="Bug: foo() returns wrong value when bar is None",
        repo_skeleton=RepoSkeleton(
            repo="example/example",
            base_commit="abcdef0123",
            files=(),
        ),
        test_directives=TestDirectives(dirs=("tests",), source="discovery:1"),
    )


def _make_ranked_files(paths: list[str]) -> list[RankedFile]:
    return [
        RankedFile(
            file_path=p, final_score=1.0 - i * 0.05,
            rationale="r", upstream_signals=("bm25",),
            upstream_best_rank={"bm25": i + 1},
        )
        for i, p in enumerate(paths)
    ]


@dataclass
class _FakeExecResult:
    exit_code: int
    stdout: str
    stderr: str = ""
    elapsed_s: float = 0.1
    truncated: bool = False


class _FakeSandbox:
    """Sandbox stub that scripts ordered verify outcomes for the retry loop.

    write_file: always succeeds.
    read_file: returns canned snippets.
    run_repro_test: pops from the verify_outcomes queue.
    view: returns a fixed InstanceView.
    """

    def __init__(self, *, view: InstanceView, file_contents: dict[str, str], verify_outcomes: list[tuple[int, str]]):
        self._view = view
        self._files = file_contents
        self._verify_outcomes = list(verify_outcomes)
        self.write_calls: list[tuple] = []
        self.run_calls: list[tuple] = []

    @property
    def view(self):
        return self._view

    def read_file(self, rel_path: str, max_chars: int | None = None):
        text = self._files.get(rel_path, "")
        if max_chars is not None and len(text) > max_chars:
            text = text[:max_chars]
        return _FakeExecResult(
            exit_code=0 if text else 2,
            stdout=text,
        )

    def write_file(self, rel_path: str, content: str):
        self.write_calls.append((rel_path, content[:80]))
        return _FakeExecResult(exit_code=0, stdout="")

    def run_repro_test(self, *, test_id: str, test_code: str | None = None, timeout_s: float = 60.0):
        if not self._verify_outcomes:
            return _FakeExecResult(exit_code=0, stdout="all passed")
        exit_code, stdout = self._verify_outcomes.pop(0)
        self.run_calls.append((test_id, exit_code))
        return _FakeExecResult(exit_code=exit_code, stdout=stdout)


def _mock_chat(text: str, *, input_tokens: int = 100, output_tokens: int = 50, model: str = "deepseek-chat"):
    return MagicMock(
        text=text, model=model,
        input_tokens=input_tokens, output_tokens=output_tokens,
    )


def _valid_repro_json(test_id_suffix: str = "test_x", filename: str = "test_repro.py") -> str:
    return json.dumps({
        "test_filename": filename,
        "test_code": f"def {test_id_suffix}(): pass",
        "target_test_id": f"tests/{filename}::{test_id_suffix}",
        "rationale": "reproduces the bug",
    })


# ---------------------------------------------------------------------------
# 1. ReproRejectReason enum + widen_for_next_attempt determinism
# ---------------------------------------------------------------------------


def test_reject_reason_values():
    # The exact enum values are an external contract for the JSONL audit.
    assert ReproRejectReason.ACCEPTED.value == "accepted"
    assert ReproRejectReason.PASSES_AT_BASE.value == "passes_at_base"
    assert ReproRejectReason.IMPORT_ERROR.value == "import_error"
    assert ReproRejectReason.SYNTAX_ERROR.value == "syntax_error"
    assert ReproRejectReason.TIMEOUT.value == "timeout"
    assert ReproRejectReason.OUTPUT_TOKEN_OVERFLOW.value == "output_token_overflow"
    assert ReproRejectReason.FIREWALL_VIOLATION.value == "firewall_violation"
    assert ReproRejectReason.OTHER.value == "other"


def test_widen_for_next_attempt_widens_on_passes_at_base():
    assert widen_for_next_attempt(ReproRejectReason.PASSES_AT_BASE) is True


def test_widen_for_next_attempt_widens_on_import_error():
    assert widen_for_next_attempt(ReproRejectReason.IMPORT_ERROR) is True


def test_widen_for_next_attempt_does_not_widen_on_syntax_error():
    """Generator-quality issues don't benefit from more context."""
    assert widen_for_next_attempt(ReproRejectReason.SYNTAX_ERROR) is False


def test_widen_for_next_attempt_does_not_widen_on_timeout():
    assert widen_for_next_attempt(ReproRejectReason.TIMEOUT) is False


def test_widen_for_next_attempt_does_not_widen_on_token_overflow():
    assert widen_for_next_attempt(ReproRejectReason.OUTPUT_TOKEN_OVERFLOW) is False


def test_widen_for_next_attempt_does_not_widen_on_firewall_violation():
    """Firewall violation should be fatal anyway, but if it ever does
    cycle through, widening can't help — same context shape."""
    assert widen_for_next_attempt(ReproRejectReason.FIREWALL_VIOLATION) is False


def test_widen_for_next_attempt_does_not_widen_on_other():
    assert widen_for_next_attempt(ReproRejectReason.OTHER) is False


# ---------------------------------------------------------------------------
# 2. ReproAttemptTraceRow schema + V0 enforcement
# ---------------------------------------------------------------------------


def _make_row(**overrides) -> ReproAttemptTraceRow:
    base = dict(
        instance_id="example__example-1",
        attempt_index=0,
        time_to_result_s=1.2,
        base_commit_fail_status="fails-at-base",
        post_patch_pass_status=V0_POST_PATCH_PASS_STATUS,
        generator_stated_reason="r",
        reject_reason=ReproRejectReason.ACCEPTED.value,
        reject_reason_detail="",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        widen_flag=False,
    )
    base.update(overrides)
    return ReproAttemptTraceRow(**base)


def test_trace_row_v0_rejects_real_post_patch_pass():
    """V0 MUST NOT contain real post-patch verification — that would
    breach the gold-patch firewall (§8.5)."""
    with pytest.raises(ValueError, match="post_patch_pass_status"):
        _make_row(post_patch_pass_status="passes")


def test_trace_row_v0_rejects_post_patch_fail():
    with pytest.raises(ValueError, match="post_patch_pass_status"):
        _make_row(post_patch_pass_status="fails")


def test_trace_row_v0_accepts_unknown():
    row = _make_row(post_patch_pass_status="unknown")
    assert row.post_patch_pass_status == "unknown"


def test_trace_row_rejects_invalid_reject_reason():
    with pytest.raises(ValueError, match="reject_reason"):
        _make_row(reject_reason="lol_bogus_reason")


def test_trace_row_rejects_invalid_base_status():
    with pytest.raises(ValueError, match="base_commit_fail_status"):
        _make_row(base_commit_fail_status="kinda-failing-maybe")


def test_trace_row_rejects_negative_attempt():
    with pytest.raises(ValueError, match="attempt_index"):
        _make_row(attempt_index=-1)


# ---------------------------------------------------------------------------
# 3. Trace writer: instance_id mismatch, file path, append-mode
# ---------------------------------------------------------------------------


def test_write_trace_row_lands_at_correct_path(tmp_path):
    row = _make_row()
    p = write_trace_row(tmp_path, "example__example-1", row)
    expected = tmp_path / "repro_trace" / "example__example-1.jsonl"
    assert p == expected
    assert p.exists()
    contents = [json.loads(line) for line in p.read_text().splitlines() if line]
    assert len(contents) == 1
    assert contents[0]["instance_id"] == "example__example-1"
    assert contents[0]["post_patch_pass_status"] == "unknown"


def test_write_trace_row_appends_multiple(tmp_path):
    row0 = _make_row(attempt_index=0)
    row1 = _make_row(attempt_index=1, reject_reason=ReproRejectReason.PASSES_AT_BASE.value)
    write_trace_row(tmp_path, "example__example-1", row0)
    write_trace_row(tmp_path, "example__example-1", row1)
    p = tmp_path / "repro_trace" / "example__example-1.jsonl"
    rows = [json.loads(line) for line in p.read_text().splitlines() if line]
    assert len(rows) == 2
    assert rows[0]["attempt_index"] == 0
    assert rows[1]["attempt_index"] == 1


def test_write_trace_row_rejects_instance_id_mismatch(tmp_path):
    """Cross-contamination defense: a row tagged with a different
    instance_id than the run can NEVER land in that run's JSONL."""
    row = _make_row(instance_id="OTHER_INSTANCE")
    with pytest.raises(ValueError, match="(?i)cross-instance"):
        write_trace_row(tmp_path, "example__example-1", row)


# ---------------------------------------------------------------------------
# 4. Verify-result classifier
# ---------------------------------------------------------------------------


def test_classify_verify_accepts_exit_1():
    """pytest exit 1 = some tests failed = bug reproduced. ACCEPTED."""
    reason, detail = _classify_verify_result(1, "1 failed")
    assert reason is ReproRejectReason.ACCEPTED


def test_classify_verify_passes_at_base_on_exit_0():
    reason, _ = _classify_verify_result(0, "1 passed")
    assert reason is ReproRejectReason.PASSES_AT_BASE


def test_classify_verify_import_error():
    reason, detail = _classify_verify_result(2, "ImportError: no module 'foo'")
    assert reason is ReproRejectReason.IMPORT_ERROR


def test_classify_verify_module_not_found():
    reason, _ = _classify_verify_result(2, "ModuleNotFoundError: No module named 'bar'")
    assert reason is ReproRejectReason.IMPORT_ERROR


def test_classify_verify_syntax_error():
    reason, _ = _classify_verify_result(2, "SyntaxError: invalid syntax")
    assert reason is ReproRejectReason.SYNTAX_ERROR


def test_classify_verify_timeout_exit_124():
    reason, _ = _classify_verify_result(124, "")
    assert reason is ReproRejectReason.TIMEOUT


def test_classify_verify_other_fallthrough():
    reason, _ = _classify_verify_result(5, "weird pytest internal")
    assert reason is ReproRejectReason.OTHER


# ---------------------------------------------------------------------------
# 5. End-to-end retry loop with mocked LLM + sandbox
# ---------------------------------------------------------------------------


def test_retry_loop_stops_on_first_accept(tmp_path):
    """First attempt's test fails at base (exit_code=1, our success).
    Loop should stop after attempt 0; trace has 1 row."""
    view = _make_view()
    rfs = _make_ranked_files(["src/foo.py"])
    sandbox = _FakeSandbox(
        view=view,
        file_contents={"src/foo.py": "def foo(): pass"},
        verify_outcomes=[(1, "1 failed - bug reproduced")],
    )

    chat = _mock_chat(_valid_repro_json())
    with patch("harness.llm.clients.complete_chat", return_value=chat), \
         patch("harness.llm.clients.model_for_role", return_value="deepseek-chat"):
        result = generate_with_retry(
            view=view, ranked_files=rfs, sandbox=sandbox,
            run_dir=tmp_path,
        )

    assert result.status == ReproStatus.USABLE
    assert result.case is not None
    assert result.attempts_made == 1
    p = tmp_path / "repro_trace" / view.instance_id + ".jsonl" if False else tmp_path / "repro_trace" / f"{view.instance_id}.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines() if l]
    assert len(rows) == 1
    assert rows[0]["reject_reason"] == "accepted"
    assert rows[0]["base_commit_fail_status"] == "fails-at-base"
    assert rows[0]["post_patch_pass_status"] == "unknown"


def test_retry_loop_runs_all_three_when_each_fails(tmp_path):
    view = _make_view()
    rfs = _make_ranked_files(["src/foo.py"])
    # Three rejections: passes-at-base, then import error, then passes-at-base.
    sandbox = _FakeSandbox(
        view=view,
        file_contents={"src/foo.py": "pass"},
        verify_outcomes=[
            (0, "1 passed"),                          # passes_at_base — would widen
            (2, "ImportError: no foo"),               # import_error — would widen
            (0, "1 passed"),                          # passes_at_base
        ],
    )
    chat = _mock_chat(_valid_repro_json())
    with patch("harness.llm.clients.complete_chat", return_value=chat), \
         patch("harness.llm.clients.model_for_role", return_value="deepseek-chat"):
        result = generate_with_retry(
            view=view, ranked_files=rfs, sandbox=sandbox,
            run_dir=tmp_path,
        )

    assert result.status == ReproStatus.NO_REPRO
    assert result.attempts_made == 3
    p = tmp_path / "repro_trace" / f"{view.instance_id}.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines() if l]
    assert len(rows) == 3
    assert [r["attempt_index"] for r in rows] == [0, 1, 2]
    # widen_flag deterministic: attempt 0 never widens; attempt 1 widens
    # because attempt 0 was passes_at_base; attempt 2 widens because
    # attempt 1 was import_error.
    assert rows[0]["widen_flag"] is False
    assert rows[1]["widen_flag"] is True
    assert rows[2]["widen_flag"] is True


def test_retry_loop_does_not_widen_on_syntax_error(tmp_path):
    view = _make_view()
    rfs = _make_ranked_files(["src/foo.py"])
    sandbox = _FakeSandbox(
        view=view,
        file_contents={"src/foo.py": "pass"},
        verify_outcomes=[
            (2, "SyntaxError: invalid syntax"),   # syntax_error — should NOT widen
            (0, "1 passed"),                       # passes_at_base
        ],
    )
    chat = _mock_chat(_valid_repro_json())
    with patch("harness.llm.clients.complete_chat", return_value=chat), \
         patch("harness.llm.clients.model_for_role", return_value="deepseek-chat"):
        result = generate_with_retry(
            view=view, ranked_files=rfs, sandbox=sandbox,
            run_dir=tmp_path, n_attempts=2,
        )

    assert result.status == ReproStatus.NO_REPRO
    p = tmp_path / "repro_trace" / f"{view.instance_id}.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines() if l]
    assert rows[1]["widen_flag"] is False  # syntax error doesn't trigger widening


# ---------------------------------------------------------------------------
# 6. Cost cap: between attempts only
# ---------------------------------------------------------------------------


def test_cost_cap_fires_between_attempts(tmp_path):
    """First attempt uses heavy tokens that push spend over $0.30.
    The attempt completes (we don't kill mid-call); the next attempt
    sees hard_cap_reached and emits NO_REPRO_BUDGET. Trace contains
    the row for the completed first attempt."""
    view = _make_view()
    rfs = _make_ranked_files(["src/foo.py"])
    # First attempt gives passes-at-base so the loop wants to retry.
    sandbox = _FakeSandbox(
        view=view,
        file_contents={"src/foo.py": "pass"},
        verify_outcomes=[(0, "1 passed")],
    )
    # Heavy token usage to blow past the $0.30 cap on a single call.
    # claude-opus-4-7 input is $15/M; 25,000 tokens = $0.375.
    heavy_chat = _mock_chat(
        _valid_repro_json(),
        input_tokens=25_000,
        output_tokens=200,
        model="claude-opus-4-7",
    )
    with patch("harness.llm.clients.complete_chat", return_value=heavy_chat), \
         patch("harness.llm.clients.model_for_role", return_value="claude-opus-4-7"):
        result = generate_with_retry(
            view=view, ranked_files=rfs, sandbox=sandbox,
            run_dir=tmp_path, cost_cap_usd=0.30, n_attempts=3,
        )

    assert result.status == ReproStatus.NO_REPRO_BUDGET
    assert result.attempts_made == 1     # exactly one completed before cap fired
    assert result.total_cost_usd > 0.30  # the completed attempt did pay
    p = tmp_path / "repro_trace" / f"{view.instance_id}.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines() if l]
    assert len(rows) == 1                # the completed attempt's trace row
    assert rows[0]["reject_reason"] == "passes_at_base"


def test_cost_cap_emits_no_repro_budget_status(tmp_path):
    """Confirm the explicit status string produced when cap fires."""
    view = _make_view()
    rfs = _make_ranked_files(["src/foo.py"])
    sandbox = _FakeSandbox(
        view=view,
        file_contents={"src/foo.py": "pass"},
        verify_outcomes=[(0, "1 passed")],
    )
    heavy_chat = _mock_chat(
        _valid_repro_json(), input_tokens=25_000, output_tokens=200,
        model="claude-opus-4-7",
    )
    with patch("harness.llm.clients.complete_chat", return_value=heavy_chat), \
         patch("harness.llm.clients.model_for_role", return_value="claude-opus-4-7"):
        result = generate_with_retry(
            view=view, ranked_files=rfs, sandbox=sandbox,
            run_dir=tmp_path, cost_cap_usd=0.30, n_attempts=3,
        )
    assert result.status == "no_repro_budget"


# ---------------------------------------------------------------------------
# 7. Firewall violation is fatal (no retry)
# ---------------------------------------------------------------------------


def test_firewall_violation_does_not_retry(tmp_path):
    """If the input firewall fires on attempt 0, retrying with the
    same view + ranked_files will fire again. Bail with NO_REPRO."""
    bad_view = InstanceView(
        instance_id="example__example-2",
        repo="example/example",
        base_commit="abcdef0123",
        problem_statement="see fail_to_pass field",  # planted token
        repo_skeleton=RepoSkeleton(repo="example/example", base_commit="abcdef0123"),
        test_directives=TestDirectives(dirs=("tests",), source="discovery:1"),
    )
    rfs = _make_ranked_files(["src/foo.py"])
    sandbox = _FakeSandbox(
        view=bad_view,
        file_contents={},
        verify_outcomes=[],
    )
    # No need to mock complete_chat — firewall fires before the call.
    with patch("harness.llm.clients.model_for_role", return_value="deepseek-chat"):
        result = generate_with_retry(
            view=bad_view, ranked_files=rfs, sandbox=sandbox,
            run_dir=tmp_path,
        )

    assert result.status == ReproStatus.NO_REPRO
    assert result.attempts_made == 1
    assert result.final_reject_reason is ReproRejectReason.FIREWALL_VIOLATION
    p = tmp_path / "repro_trace" / f"{bad_view.instance_id}.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines() if l]
    assert len(rows) == 1
    assert rows[0]["reject_reason"] == "firewall_violation"
    assert rows[0]["base_commit_fail_status"] == "errors"
    assert rows[0]["post_patch_pass_status"] == "unknown"
