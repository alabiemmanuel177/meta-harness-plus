"""Unit tests for the V10 sandbox and cache hygiene.

The sandbox class wraps DockerShellExecutor for primitives but
deliberately does NOT expose ``run_tests``. This is enforced two ways:

  1. The Sandbox class doesn't define a ``run_tests`` method.
  2. The runtime guard fires if someone tries to smuggle a forbidden
     token through any of the exposed methods.

These tests exercise both invariants without requiring Docker.
"""

from __future__ import annotations

import pathlib

import pytest

from harness import views as V
from harness.cache import (
    CacheLeakError,
    V10_CACHE_DIRS,
    V10_TAG_PREFIX,
    assert_clean_cache_at_startup,
)
from harness.cost import CostTracker
from harness.sandbox import (
    OracleLeakError,
    Sandbox,
    _assert_no_forbidden_token,
)


# ---------------------------------------------------------------------------
# Sandbox API surface
# ---------------------------------------------------------------------------


def test_sandbox_does_not_expose_run_tests() -> None:
    """``Sandbox`` must not have a ``run_tests`` method on its public
    surface. Callers must use ``run_public_suite()`` instead, which is
    leak-free by construction."""
    public = [n for n in dir(Sandbox) if not n.startswith("_")]
    assert "run_tests" not in public, (
        "Sandbox.run_tests would expose the legacy leak-shaped API "
        "(see V10_DESIGN.md §12.4(b))"
    )
    assert "run_public_suite" in public


def test_runtime_guard_blocks_forbidden_token_in_command() -> None:
    """``run_shell`` checks for forbidden tokens in the command string."""
    with pytest.raises(OracleLeakError):
        _assert_no_forbidden_token(
            "run pytest -k 'fail" + "_to_pass'",
            label="test.fake_command",
        )


def test_runtime_guard_passes_clean_command() -> None:
    """A clean command does not raise."""
    _assert_no_forbidden_token("python -m pytest tests/", label="test.fake_command")
    _assert_no_forbidden_token(["tests/", "lib/foo/tests/"], label="test.fake_dirs")


def test_runtime_guard_substring_matches_screaming_snake() -> None:
    """Tightening 3: case-insensitive substring match on FAIL_TO_PASS."""
    with pytest.raises(OracleLeakError):
        _assert_no_forbidden_token("FAIL" + "_TO_PASS=1", label="test.env")


def test_runtime_guard_substring_matches_camelcase() -> None:
    """CamelCase variants also match (folded substring)."""
    with pytest.raises(OracleLeakError):
        # "FailToPass" -> casefolded -> "failtopass" — does NOT contain
        # "fail_to_pass" (with underscore). This is an honest expectation:
        # CamelCase without underscore separators won't trip the guard
        # purely via substring; that's why we ALSO have the static AST
        # scan for forbidden names. Verify the documented behavior:
        _assert_no_forbidden_token("FailToPass", label="test")


def test_runtime_guard_blocks_fail_to_pass_token() -> None:
    """The classic leak: a kwarg or string value mentioning a SWE-bench
    dataset field name. The substring guard fires at the sandbox
    boundary. (Replaces the prior 'resolved'-token test; per commit
    17c the bare-word 'resolved' was dropped from FORBIDDEN_TOKENS
    because it's not a real dataset field and over-blocked legitimate
    English usage.)"""
    with pytest.raises(OracleLeakError):
        _assert_no_forbidden_token(
            "candidate.fail_to_pass == True",
            label="test.selector",
        )


# ---------------------------------------------------------------------------
# Cost tracker
# ---------------------------------------------------------------------------


def test_cost_tracker_records_and_rolls_up() -> None:
    t = CostTracker(instance_id="x__y-1", cap_usd=20.0)
    t.record("localization", model="claude-sonnet-4-6", input_tokens=4_000, output_tokens=200)
    t.record("repro_gen", model="claude-sonnet-4-6", input_tokens=2_000, output_tokens=100)
    assert t.total_usd > 0
    assert "localization" in t.by_stage()
    assert not t.hard_cap_reached
    assert not t.soft_cap_reached


def test_cost_tracker_soft_cap_fires() -> None:
    t = CostTracker(instance_id="x__y-2", cap_usd=1.0, soft_guard_fraction=0.5)
    # 50k input + 10k output at Opus pricing ($15/$75 per M tokens)
    # = $0.75 + $0.75 = $1.50 — above hard cap.
    t.record("patch_gen_agent", model="claude-opus-4-7",
             input_tokens=50_000, output_tokens=10_000)
    assert t.hard_cap_reached
    assert t.soft_cap_reached


# ---------------------------------------------------------------------------
# Cache hygiene
# ---------------------------------------------------------------------------


def test_clean_cache_passes_when_dirs_absent(tmp_path: pathlib.Path) -> None:
    assert_clean_cache_at_startup(cwd=tmp_path)


def test_clean_cache_passes_with_only_v10_entries(tmp_path: pathlib.Path) -> None:
    (tmp_path / "runs" / "v10_test_run").mkdir(parents=True)
    (tmp_path / "trajectories" / "v10_001").mkdir(parents=True)
    assert_clean_cache_at_startup(cwd=tmp_path)


def test_clean_cache_raises_on_legacy_entry_in_exclusive_dir(tmp_path: pathlib.Path) -> None:
    """Default check (V10_EXCLUSIVE_DIRS only) must raise on a legacy
    entry in eval_outputs/ — that dir is V10-owned by the grader."""
    (tmp_path / "eval_outputs" / "swebench_500_v7_run").mkdir(parents=True)
    with pytest.raises(CacheLeakError):
        assert_clean_cache_at_startup(cwd=tmp_path)


def test_clean_cache_tolerates_legacy_in_namespaced_parent(tmp_path: pathlib.Path) -> None:
    """Legacy entries in V10_NAMESPACED_PARENTS (runs/, trajectories/)
    must NOT raise the default check — those dirs are shared with V7
    baselines per V10_DESIGN.md §13.3."""
    (tmp_path / "runs" / "swebench_500_v7").mkdir(parents=True)
    (tmp_path / "trajectories" / "trajectories_v7").mkdir(parents=True)
    # Should NOT raise — runs/ and trajectories/ are namespaced parents.
    assert_clean_cache_at_startup(cwd=tmp_path)


def test_v10_tag_prefix_is_canonical() -> None:
    assert V10_TAG_PREFIX == "v10_"
    # V10_CACHE_DIRS is the union of exclusive + namespaced parents.
    assert pathlib.Path("eval_outputs") in V10_CACHE_DIRS
    assert pathlib.Path("runs") in V10_CACHE_DIRS
    assert pathlib.Path("trajectories") in V10_CACHE_DIRS
