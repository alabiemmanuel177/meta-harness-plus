"""Phase 4 — validator orchestrator unit tests.

Mocked sandbox; no real Docker. The orchestrator's flow is tested
end-to-end with scripted apply + suite + static outcomes."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from harness.patch_gen.views import PatchCandidate
from harness.validation import ValidationResult, validate_candidate
from harness.views import (
    DiffStats,
    InstanceView,
    PublicSuiteSignal,
    ReproSignal,
    RepoSkeleton,
    StaticSignal,
    TestDirectives,
)


@dataclass
class _FakeExecResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    log_excerpt: str = ""
    duration_s: float = 0.0
    truncated: bool = False
    workdir_state: str = "base"
    test_dirs: tuple = ()


@dataclass
class _FakeSandbox:
    apply_outcomes: list = field(default_factory=list)
    suite_outcomes: list = field(default_factory=list)
    shell_outcomes: list = field(default_factory=list)
    apply_calls: list = field(default_factory=list)
    reset_calls: int = 0

    def apply_patch(self, diff):
        self.apply_calls.append(diff)
        if self.apply_outcomes:
            return self.apply_outcomes.pop(0)
        return _FakeExecResult(exit_code=0)

    def reset_worktree(self):
        self.reset_calls += 1
        return _FakeExecResult(exit_code=0)

    def run_public_suite(self, *, state_label="base", timeout_s=480, collect_only=False):
        if self.suite_outcomes:
            return self.suite_outcomes.pop(0)
        # default — empty success
        return _FakeExecResult(
            exit_code=0,
            log_excerpt="===== 10 passed in 1.2s =====",
            workdir_state=state_label,
        )

    def run_shell(self, cmd, timeout_s=30.0):
        if self.shell_outcomes:
            return self.shell_outcomes.pop(0)
        return _FakeExecResult(exit_code=1)


def _make_view() -> InstanceView:
    return InstanceView(
        instance_id="example__example-1",
        repo="example/example",
        base_commit="abcdef0123456789",
        problem_statement="Bug: foo() returns None",
        repo_skeleton=RepoSkeleton(repo="example/example", base_commit="abcdef0123456789"),
        test_directives=TestDirectives(dirs=("tests/",), source="discovery:1"),
    )


def _make_candidate(diff: str = "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n") -> PatchCandidate:
    return PatchCandidate(
        instance_id="example__example-1",
        candidate_id="pipe_t0",
        diff=diff,
        source_route="pipeline",
        source_temperature=0.0,
        source_attempt_index=0,
        generator_model="deepseek-chat",
        generator_input_tokens=100,
        generator_output_tokens=50,
        generation_cost_usd=0.001,
        duration_s=1.0,
    )


# ---------------------------------------------------------------------------
# 1. Apply failure path
# ---------------------------------------------------------------------------


def test_validator_apply_failure_emits_apply_failed_marker():
    sb = _FakeSandbox(apply_outcomes=[
        _FakeExecResult(exit_code=1, stderr="error: patch does not apply"),
    ])
    cand = _make_candidate()
    result = validate_candidate(view=_make_view(), sandbox=sb, candidate=cand,
                                skip_static=True, run_dir=None)
    assert isinstance(result, ValidationResult)
    assert result.apply_result.applied is False
    assert "apply_failed" in result.candidate_view.notes


def test_validator_apply_failure_skips_suite_and_static():
    sb = _FakeSandbox(apply_outcomes=[_FakeExecResult(exit_code=1)])
    cand = _make_candidate()
    result = validate_candidate(view=_make_view(), sandbox=sb, candidate=cand,
                                skip_static=True, run_dir=None)
    # public_suite_signal is the empty sentinel; static is empty.
    assert result.candidate_view.public_suite_signal.suite_ran_at_base is False
    assert result.candidate_view.static_signal.ruff_clean is None
    assert result.candidate_view.static_signal.mypy_clean is None


# ---------------------------------------------------------------------------
# 2. Happy path
# ---------------------------------------------------------------------------


def test_validator_happy_path_assembles_candidate_view():
    sb = _FakeSandbox(
        apply_outcomes=[
            _FakeExecResult(exit_code=0),  # initial apply
            _FakeExecResult(exit_code=0),  # re-apply after baseline
        ],
        suite_outcomes=[
            _FakeExecResult(  # baseline
                exit_code=0,
                log_excerpt="===== 10 passed in 1.0s =====",
                workdir_state="base",
            ),
            _FakeExecResult(  # patched
                exit_code=0,
                log_excerpt="===== 10 passed in 1.0s =====",
                workdir_state="patched",
            ),
        ],
    )
    cand = _make_candidate()
    result = validate_candidate(view=_make_view(), sandbox=sb, candidate=cand,
                                skip_static=True, run_dir=None)
    assert result.apply_result.applied is True
    assert result.candidate_view.candidate_id == "pipe_t0"
    assert result.candidate_view.public_suite_signal.new_failures_count == 0
    assert result.candidate_view.cluster_id != ""


def test_validator_propagates_repro_signal():
    sb = _FakeSandbox(
        apply_outcomes=[_FakeExecResult(exit_code=0)] * 3,
        suite_outcomes=[
            _FakeExecResult(exit_code=0, log_excerpt="===== 10 passed ====="),
            _FakeExecResult(exit_code=0, log_excerpt="===== 10 passed ====="),
        ],
    )
    cand = _make_candidate()
    repro = ReproSignal(status="pass", duration_s=1.5, log_excerpt="ok")
    result = validate_candidate(view=_make_view(), sandbox=sb, candidate=cand,
                                repro_signal=repro,
                                skip_static=True, run_dir=None)
    assert result.candidate_view.repro_signal == repro


def test_validator_repro_signal_can_be_none():
    sb = _FakeSandbox(
        apply_outcomes=[_FakeExecResult(exit_code=0)] * 3,
        suite_outcomes=[
            _FakeExecResult(exit_code=0, log_excerpt="===== 10 passed ====="),
            _FakeExecResult(exit_code=0, log_excerpt="===== 10 passed ====="),
        ],
    )
    cand = _make_candidate()
    result = validate_candidate(view=_make_view(), sandbox=sb, candidate=cand,
                                repro_signal=None,
                                skip_static=True, run_dir=None)
    assert result.candidate_view.repro_signal is None


# ---------------------------------------------------------------------------
# 3. Regression detection
# ---------------------------------------------------------------------------


def test_validator_detects_regression():
    sb = _FakeSandbox(
        apply_outcomes=[_FakeExecResult(exit_code=0)] * 3,
        suite_outcomes=[
            _FakeExecResult(  # baseline: clean
                exit_code=0, log_excerpt="===== 10 passed =====",
            ),
            _FakeExecResult(  # patched: 1 regression
                exit_code=1,
                log_excerpt=(
                    "FAILED tests/test_a.py::test_regressed - AssertionError\n"
                    "===== 1 failed, 9 passed in 1.2s ====="
                ),
            ),
        ],
    )
    cand = _make_candidate()
    result = validate_candidate(view=_make_view(), sandbox=sb, candidate=cand,
                                skip_static=True, run_dir=None)
    assert result.candidate_view.public_suite_signal.new_failures_count == 1


# ---------------------------------------------------------------------------
# 4. Baseline cache
# ---------------------------------------------------------------------------


def test_validator_cache_hit_skips_baseline_run(tmp_path):
    """Pre-seed a baseline snapshot, then run validate_candidate;
    only the patched suite should run (not the baseline)."""
    from harness.validation.public_suite import (
        BaselineSnapshot,
        save_baseline_snapshot,
    )
    snap = BaselineSnapshot(
        instance_id="example__example-1",
        base_commit="abcdef0123456789",
        failing_ids=(),
        passing_count=10,
        duration_s=1.0,
    )
    save_baseline_snapshot(tmp_path, snap)

    sb = _FakeSandbox(
        apply_outcomes=[_FakeExecResult(exit_code=0)],
        suite_outcomes=[
            # Only ONE suite run expected (patched only).
            _FakeExecResult(exit_code=0, log_excerpt="===== 10 passed ====="),
        ],
    )
    cand = _make_candidate()
    result = validate_candidate(view=_make_view(), sandbox=sb, candidate=cand,
                                skip_static=True, run_dir=tmp_path)
    assert result.baseline_cache_hit is True


def test_validator_cache_miss_writes_snapshot(tmp_path):
    sb = _FakeSandbox(
        apply_outcomes=[_FakeExecResult(exit_code=0)] * 2,
        suite_outcomes=[
            _FakeExecResult(exit_code=0, log_excerpt="===== 10 passed ====="),
            _FakeExecResult(exit_code=0, log_excerpt="===== 10 passed ====="),
        ],
    )
    cand = _make_candidate()
    result = validate_candidate(view=_make_view(), sandbox=sb, candidate=cand,
                                skip_static=True, run_dir=tmp_path)
    assert result.baseline_cache_hit is False
    # Re-run should cache hit.
    sb2 = _FakeSandbox(
        apply_outcomes=[_FakeExecResult(exit_code=0)],
        suite_outcomes=[_FakeExecResult(exit_code=0, log_excerpt="===== 10 passed =====")],
    )
    r2 = validate_candidate(view=_make_view(), sandbox=sb2, candidate=cand,
                            skip_static=True, run_dir=tmp_path)
    assert r2.baseline_cache_hit is True
