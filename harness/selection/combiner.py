"""Phase 5 — selector combiner + final pick.

Per docs/V10_DESIGN_PHASE5.md §3.2: weighted vote across selectors.

V0 weights:
  selector_a (LLM reviewer):  0.5  (deferred when K=1 — no LLM call)
  selector_b (cluster vote):  0.25
  selector_c (heuristic):     0.25

Final pick: argmax sum(w_i * score_i). Ties broken deterministically
by candidate_id ascending.

The combiner is a pure function of CandidateView[] when Selector A
is disabled (K=1). When K>1 the LLM-reviewer call is hashed-cached
so re-runs are reproducible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from harness.selection.selector_b_cluster import score_by_cluster
from harness.selection.selector_c_heuristic import score_by_heuristic
from harness.views import CandidateView


log = logging.getLogger(__name__)


DEFAULT_WEIGHTS: dict[str, float] = {
    "selector_a": 0.5,
    "selector_b": 0.25,
    "selector_c": 0.25,
}


@dataclass(frozen=True)
class SelectionResult:
    """The final selection's choice + per-selector contribution audit."""

    chosen_candidate_id: str
    combined_score: float
    per_selector_scores: dict        # {selector_name: {candidate_id: score}}
    weights: dict
    n_candidates: int
    escalated: bool = False


def select(
    candidates: list[CandidateView],
    *,
    weights: dict[str, float] | None = None,
    enable_selector_a: bool = False,
    selector_a_scores: dict[str, float] | None = None,
) -> SelectionResult:
    """Pick the best candidate from ``candidates``.

    Args:
      candidates: PatchCandidate views from Phase 4 for this instance.
      weights: per-selector weight dict. Defaults to
        ``DEFAULT_WEIGHTS``.
      enable_selector_a: when False (V0 K=1 default) selector_a is
        treated as "every candidate ties at 1.0 / N" — equivalent to
        a no-op weighting.
      selector_a_scores: pre-computed LLM-reviewer scores per
        candidate_id when available (cached). When None and
        enable_selector_a=True, the combiner falls back to uniform
        scoring with a WARNING log.

    Returns ``SelectionResult`` with the chosen candidate id and the
    per-selector audit.
    """
    if not candidates:
        raise ValueError("select() called with empty candidates list")

    weights = weights or DEFAULT_WEIGHTS

    scores_a: dict[str, float] = {}
    if enable_selector_a:
        if selector_a_scores is not None:
            scores_a = selector_a_scores
        else:
            log.warning(
                "[selection.combiner] enable_selector_a=True but no "
                "selector_a_scores supplied; falling back to uniform"
            )
            scores_a = {c.candidate_id: 1.0 / len(candidates) for c in candidates}
    else:
        # K=1 path or LLM reviewer skipped — uniform.
        scores_a = {c.candidate_id: 1.0 / len(candidates) for c in candidates}

    scores_b = score_by_cluster(candidates)
    scores_c = score_by_heuristic(candidates)

    combined: dict[str, float] = {}
    for c in candidates:
        cid = c.candidate_id
        combined[cid] = (
            weights.get("selector_a", 0.0) * scores_a.get(cid, 0.0)
            + weights.get("selector_b", 0.0) * scores_b.get(cid, 0.0)
            + weights.get("selector_c", 0.0) * scores_c.get(cid, 0.0)
        )

    # Tie-breaker: candidate_id ascending. So sort by (-score, id).
    chosen = sorted(
        combined.items(), key=lambda kv: (-kv[1], kv[0])
    )[0]
    chosen_id, chosen_score = chosen

    return SelectionResult(
        chosen_candidate_id=chosen_id,
        combined_score=chosen_score,
        per_selector_scores={
            "selector_a": scores_a,
            "selector_b": scores_b,
            "selector_c": scores_c,
        },
        weights=dict(weights),
        n_candidates=len(candidates),
        escalated=False,
    )


__all__ = [
    "DEFAULT_WEIGHTS",
    "SelectionResult",
    "select",
]
