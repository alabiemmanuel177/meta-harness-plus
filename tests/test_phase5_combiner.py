"""Phase 5 — combiner + selectors B/C unit tests."""

from __future__ import annotations

import pytest

from harness.selection import (
    DEFAULT_WEIGHTS,
    SelectionResult,
    score_by_cluster,
    score_by_heuristic,
    select,
)
from harness.views import (
    CandidateView,
    DiffStats,
    PublicSuiteSignal,
    ReproSignal,
    StaticSignal,
)


def _cv(
    *,
    cid: str,
    cluster_id: str = "cluster_a",
    diff: str = "diff data",
    additions: int = 1,
    deletions: int = 0,
    repro_status: str | None = "pass",
    new_failures: int = 0,
    suite_ran: bool = True,
    ruff_clean: bool | None = True,
    mypy_clean: bool | None = None,
) -> CandidateView:
    return CandidateView(
        candidate_id=cid,
        diff=diff,
        diff_stats=DiffStats(
            files_touched=("src/foo.py",),
            additions=additions,
            deletions=deletions,
            ast_normalized_hash=cluster_id + "_full_hash_xyz",
        ),
        repro_signal=(
            None if repro_status is None
            else ReproSignal(status=repro_status, duration_s=1.0)
        ),
        public_suite_signal=PublicSuiteSignal(
            suite_ran_at_base=suite_ran,
            new_failures_count=new_failures,
            new_passes_count=0,
            flake_retries=0,
            duration_s=1.0,
        ),
        static_signal=StaticSignal(
            ruff_clean=ruff_clean,
            mypy_clean=mypy_clean,
            duration_s=0.5,
        ),
        cluster_id=cluster_id,
        notes="",
    )


# ---------------------------------------------------------------------------
# Selector B (cluster vote)
# ---------------------------------------------------------------------------


def test_score_by_cluster_largest_cluster_wins():
    cands = [
        _cv(cid="a", cluster_id="X"),
        _cv(cid="b", cluster_id="X"),
        _cv(cid="c", cluster_id="Y"),
    ]
    s = score_by_cluster(cands)
    # X has 2/3 = 0.667, Y has 1/3 = 0.333
    assert s["a"] == s["b"]
    assert s["a"] > s["c"]
    assert abs(s["a"] - 2/3) < 1e-9


def test_score_by_cluster_uniform_when_all_distinct():
    cands = [
        _cv(cid=f"c{i}", cluster_id=f"K_{i}") for i in range(4)
    ]
    s = score_by_cluster(cands)
    assert all(abs(v - 0.25) < 1e-9 for v in s.values())


def test_score_by_cluster_empty():
    assert score_by_cluster([]) == {}


# ---------------------------------------------------------------------------
# Selector C (heuristic)
# ---------------------------------------------------------------------------


def test_score_by_heuristic_repro_pass_wins_over_fail():
    a = _cv(cid="a", repro_status="pass", new_failures=0)
    b = _cv(cid="b", repro_status="fail", new_failures=0)
    s = score_by_heuristic([a, b])
    assert s["a"] > s["b"]


def test_score_by_heuristic_no_regressions_wins():
    a = _cv(cid="a", repro_status="pass", new_failures=0)
    b = _cv(cid="b", repro_status="pass", new_failures=3)
    s = score_by_heuristic([a, b])
    assert s["a"] > s["b"]


def test_score_by_heuristic_smaller_diff_wins_on_tie():
    a = _cv(cid="a", repro_status="pass", new_failures=0, additions=2, deletions=1)
    b = _cv(cid="b", repro_status="pass", new_failures=0, additions=10, deletions=5)
    s = score_by_heuristic([a, b])
    assert s["a"] > s["b"]


def test_score_by_heuristic_handles_none_repro():
    a = _cv(cid="a", repro_status=None, new_failures=0)
    b = _cv(cid="b", repro_status="pass", new_failures=0)
    s = score_by_heuristic([a, b])
    # b has repro_pass=True, a has False → b should rank higher
    assert s["b"] > s["a"]


def test_score_by_heuristic_apply_failed_lowest():
    """A candidate where suite_ran_at_base=False (apply failed) must
    rank below one with a real suite run."""
    apply_failed = _cv(
        cid="apply_failed", repro_status=None, suite_ran=False,
        new_failures=0,
    )
    real = _cv(cid="real", repro_status="pass", new_failures=0)
    s = score_by_heuristic([apply_failed, real])
    assert s["real"] > s["apply_failed"]


# ---------------------------------------------------------------------------
# Combiner end-to-end
# ---------------------------------------------------------------------------


def test_select_picks_best_candidate_k1():
    """K=1: trivially picks the sole candidate."""
    cands = [_cv(cid="only_one")]
    result = select(cands)
    assert isinstance(result, SelectionResult)
    assert result.chosen_candidate_id == "only_one"
    assert result.n_candidates == 1


def test_select_combines_b_and_c():
    """Selector A is disabled (K=1 default) → uniform 1/N. Selector B
    (cluster vote) and Selector C (heuristic) decide. With cluster
    homogeneity equal, heuristic wins."""
    a = _cv(cid="a", repro_status="pass", new_failures=0, cluster_id="X")
    b = _cv(cid="b", repro_status="fail", new_failures=2, cluster_id="X")
    result = select([a, b])
    assert result.chosen_candidate_id == "a"
    assert "selector_a" in result.per_selector_scores
    assert "selector_b" in result.per_selector_scores
    assert "selector_c" in result.per_selector_scores


def test_select_ties_broken_deterministically_by_candidate_id():
    """Two candidates with identical features → tied combined score
    → tie-broken by candidate_id ascending."""
    a = _cv(cid="zzz", repro_status="pass", new_failures=0, cluster_id="X")
    b = _cv(cid="aaa", repro_status="pass", new_failures=0, cluster_id="X")
    result = select([a, b])
    assert result.chosen_candidate_id == "aaa"


def test_select_is_deterministic():
    cands = [
        _cv(cid="a", repro_status="pass", new_failures=0, cluster_id="X"),
        _cv(cid="b", repro_status="fail", new_failures=2, cluster_id="X"),
        _cv(cid="c", repro_status="pass", new_failures=1, cluster_id="Y"),
    ]
    r1 = select(cands)
    r2 = select(cands)
    assert r1.chosen_candidate_id == r2.chosen_candidate_id
    assert r1.combined_score == r2.combined_score


def test_select_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        select([])


def test_select_with_selector_a_scores_provided():
    """When selector_a_scores is supplied, the combiner uses it
    in the weighted vote with the configured weight."""
    a = _cv(cid="a", repro_status="fail", new_failures=2, cluster_id="X")
    b = _cv(cid="b", repro_status="fail", new_failures=2, cluster_id="X")
    # Selector A says a is much better even though heuristic is tied.
    result = select(
        [a, b],
        enable_selector_a=True,
        selector_a_scores={"a": 1.0, "b": 0.0},
    )
    assert result.chosen_candidate_id == "a"


def test_default_weights_sum_to_one():
    assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9
