"""Phase 4 — apply_clean unit tests."""

from __future__ import annotations

from dataclasses import dataclass

from harness.validation.apply import ApplyResult, apply_clean


@dataclass
class _FakeExecResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""


@dataclass
class _FakeSandbox:
    apply_outcomes: list

    def apply_patch(self, diff):
        if self.apply_outcomes:
            return self.apply_outcomes.pop(0)
        return _FakeExecResult(exit_code=1, stderr="(no scripted outcome)")


def test_apply_clean_returns_applied_true_on_exit_zero():
    sb = _FakeSandbox([_FakeExecResult(exit_code=0)])
    res = apply_clean(sb, "diff --git a/x b/x\n@@ -1 +1 @@\n-a\n+b\n")
    assert isinstance(res, ApplyResult)
    assert res.applied is True


def test_apply_clean_returns_applied_false_on_nonzero():
    sb = _FakeSandbox([_FakeExecResult(exit_code=1, stderr="error: patch failed")])
    res = apply_clean(sb, "broken_diff")
    assert res.applied is False
    assert "patch failed" in res.stderr_excerpt


def test_apply_clean_rejects_empty_diff():
    sb = _FakeSandbox([])
    res = apply_clean(sb, "")
    assert res.applied is False
    assert "empty" in res.stderr_excerpt.lower()


def test_apply_clean_handles_sandbox_exception():
    class _RaisingSandbox:
        def apply_patch(self, diff):
            raise RuntimeError("docker died")
    res = apply_clean(_RaisingSandbox(), "diff data")
    assert res.applied is False
    assert "RuntimeError" in res.stderr_excerpt


def test_apply_clean_truncates_long_stderr():
    long_err = "x" * 10_000
    sb = _FakeSandbox([_FakeExecResult(exit_code=1, stderr=long_err)])
    res = apply_clean(sb, "diff data")
    assert len(res.stderr_excerpt) <= 1100
