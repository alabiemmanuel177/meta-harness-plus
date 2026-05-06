"""Phase 5 Selector C — heuristic.

Per docs/V10_DESIGN_PHASE5.md §3.1: highest score on
``(repro_signal.status == 'pass') AND
(public_suite_signal.new_failures_count == 0)`` wins. Ties
broken by smallest diff, then by static_signal cleanliness
(True > None > False).

Pure function. Takes only ``CandidateView[]``. No LLM, no I/O.
"""

from __future__ import annotations

from harness.views import CandidateView


def _heuristic_features(cv: CandidateView) -> tuple:
    """Return a tuple sortable in descending order of preference.

    Order (most-preferred first):
      1. repro pass + 0 regressions  (best)
      2. repro pass, some regressions
      3. repro fail/errored, 0 regressions
      4. apply_failed (worst — public_suite_signal.suite_ran_at_base
         is False)
    """
    repro_pass = cv.repro_signal is not None and cv.repro_signal.status == "pass"
    no_regressions = cv.public_suite_signal.new_failures_count == 0
    suite_ran = cv.public_suite_signal.suite_ran_at_base
    diff_size = cv.diff_stats.additions + cv.diff_stats.deletions
    ruff_pref = {True: 2, None: 1, False: 0}.get(cv.static_signal.ruff_clean, 1)
    mypy_pref = {True: 2, None: 1, False: 0}.get(cv.static_signal.mypy_clean, 1)
    # Higher = better. Sort descending. Diff size: smaller is better, so negate.
    return (
        int(repro_pass),
        int(no_regressions),
        int(suite_ran),
        -diff_size,           # smallest diff wins
        ruff_pref,
        mypy_pref,
    )


def score_by_heuristic(candidates: list[CandidateView]) -> dict[str, float]:
    """Return per-candidate score in [0, 1]. Higher is better.

    Candidates with identical features get the SAME score (true tie)
    so the combiner's downstream candidate_id tie-break can fire.
    Otherwise score decays linearly by group rank.
    """
    if not candidates:
        return {}
    # Sort by features descending, candidate_id ascending as a stable
    # secondary key (so re-ordering input doesn't change scores).
    feature_keyed = [(c.candidate_id, _heuristic_features(c), c) for c in candidates]
    feature_keyed.sort(key=lambda t: (tuple(-x if isinstance(x, int) else x for x in t[1]), t[0]))
    # Group by features tuple — true ties get the same rank.
    rank_for_group: dict[tuple, int] = {}
    next_rank = 0
    for cid, feats, _ in feature_keyed:
        if feats not in rank_for_group:
            rank_for_group[feats] = next_rank
            next_rank += 1
    n_groups = max(1, next_rank)
    return {
        cid: 1.0 - (rank_for_group[feats] / n_groups)
        for cid, feats, _ in feature_keyed
    }


__all__ = ["score_by_heuristic"]
