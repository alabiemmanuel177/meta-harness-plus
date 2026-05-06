"""Phase 5 Selector B — cluster-then-vote.

Per docs/V10_DESIGN_PHASE5.md §3.1: group candidates by
``CandidateView.cluster_id`` (set by Phase 4 to the AST-normalized
hash, truncated). Pick the largest cluster's representative by
smallest diff. AST-hash equality is the clustering signal; tie
breaker is diff size ascending.

Pure function. Takes only ``CandidateView[]``. No LLM, no I/O.
"""

from __future__ import annotations

from collections import Counter

from harness.views import CandidateView


def score_by_cluster(candidates: list[CandidateView]) -> dict[str, float]:
    """Return per-candidate score in [0, 1].

    Score = (cluster_size / total) for each candidate; the candidate
    in the largest cluster gets the highest score. Within a cluster,
    all members tie on score; the combiner's downstream tiebreaker
    handles intra-cluster ordering.
    """
    if not candidates:
        return {}
    counts: Counter = Counter()
    for c in candidates:
        counts[c.cluster_id or "_unclustered_"] += 1
    total = len(candidates)
    return {
        c.candidate_id: counts[c.cluster_id or "_unclustered_"] / total
        for c in candidates
    }


__all__ = ["score_by_cluster"]
