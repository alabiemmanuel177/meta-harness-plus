"""Phase 4 — public_suite unit tests.

Mocked sandbox; no real pytest invocations."""

from __future__ import annotations

import pathlib
import tempfile
from dataclasses import dataclass

import pytest

from harness.validation.public_suite import (
    BaselineSnapshot,
    PublicSuiteRunResult,
    diff_against_baseline,
    load_baseline_snapshot,
    run_baseline,
    run_patched,
    save_baseline_snapshot,
    _parse_pytest_output,
    _parse_summary_counts,
)
from harness.views import PublicSuiteSignal


# ---------------------------------------------------------------------------
# Pytest-output parsing
# ---------------------------------------------------------------------------


def test_parse_pytest_output_extracts_failed_ids():
    out = """\
============================= FAILURES =========================
FAILED tests/test_a.py::test_foo - AssertionError
FAILED tests/test_b.py::test_bar - TypeError
PASSED tests/test_a.py::test_baz
"""
    _, failing = _parse_pytest_output(out)
    assert failing == {"tests/test_a.py::test_foo", "tests/test_b.py::test_bar"}


def test_parse_pytest_output_handles_error_lines():
    out = """\
ERROR tests/test_a.py::test_foo - ImportError
FAILED tests/test_a.py::test_bar - X
"""
    _, failing = _parse_pytest_output(out)
    assert "tests/test_a.py::test_foo" in failing
    assert "tests/test_a.py::test_bar" in failing


def test_parse_summary_counts_parses_counts():
    out = "===== 5 failed, 23 passed, 1 error in 1.2s ====="
    counts = _parse_summary_counts(out)
    assert counts["failed"] == 5
    assert counts["passed"] == 23
    assert counts["errors"] == 1


# ---------------------------------------------------------------------------
# Baseline snapshot caching
# ---------------------------------------------------------------------------


def test_save_and_load_baseline_snapshot(tmp_path):
    snap = BaselineSnapshot(
        instance_id="foo__bar-1",
        base_commit="abcdef0123456789",
        failing_ids=("tests/test_a.py::test_x",),
        passing_count=42,
        duration_s=12.5,
        log_excerpt="some output",
    )
    p = save_baseline_snapshot(tmp_path, snap)
    assert p.exists()
    loaded = load_baseline_snapshot(tmp_path, "foo__bar-1", "abcdef0123456789")
    assert loaded is not None
    assert loaded.instance_id == "foo__bar-1"
    assert loaded.failing_ids == ("tests/test_a.py::test_x",)
    assert loaded.passing_count == 42


def test_load_baseline_snapshot_returns_none_on_miss(tmp_path):
    assert load_baseline_snapshot(tmp_path, "absent__instance-9", "abc123") is None


# ---------------------------------------------------------------------------
# diff_against_baseline
# ---------------------------------------------------------------------------


def test_diff_against_baseline_no_change():
    base = BaselineSnapshot(
        instance_id="x", base_commit="c",
        failing_ids=("a::test_1",),
        passing_count=10,
        duration_s=1.0,
    )
    patched = PublicSuiteRunResult(
        failing_ids={"a::test_1"},
        passing_count=10, duration_s=2.0, suite_ran=True,
    )
    sig = diff_against_baseline(base, patched)
    assert isinstance(sig, PublicSuiteSignal)
    assert sig.new_failures_count == 0
    assert sig.new_passes_count == 0
    assert sig.suite_ran_at_base is True


def test_diff_against_baseline_one_regression():
    base = BaselineSnapshot(
        instance_id="x", base_commit="c",
        failing_ids=(), passing_count=10, duration_s=1.0,
    )
    patched = PublicSuiteRunResult(
        failing_ids={"new_test::regressed"},
        passing_count=9, duration_s=1.0, suite_ran=True,
    )
    sig = diff_against_baseline(base, patched)
    assert sig.new_failures_count == 1
    assert sig.new_passes_count == 0


def test_diff_against_baseline_one_gain():
    base = BaselineSnapshot(
        instance_id="x", base_commit="c",
        failing_ids=("flaky::test_1",), passing_count=9, duration_s=1.0,
    )
    patched = PublicSuiteRunResult(
        failing_ids=set(),
        passing_count=10, duration_s=1.0, suite_ran=True,
    )
    sig = diff_against_baseline(base, patched)
    assert sig.new_failures_count == 0
    assert sig.new_passes_count == 1


def test_diff_against_baseline_records_flake_retries():
    base = BaselineSnapshot(
        instance_id="x", base_commit="c",
        failing_ids=(), passing_count=1, duration_s=1.0,
    )
    patched = PublicSuiteRunResult(
        failing_ids=set(), passing_count=1, duration_s=1.0, suite_ran=True,
    )
    sig = diff_against_baseline(base, patched, flake_retries=5)
    assert sig.flake_retries == 5
