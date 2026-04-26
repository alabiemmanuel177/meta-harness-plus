"""Component synergy discovery — beyond drop-one attribution.

Drop-one attribution measures *each component's* individual value. But
two components may only be valuable *together* — e.g. compressed CoT
formatter only helps when paired with an LLM reranker, because the
reranker filters out items that confuse the compressed reasoning. This
is *synergy*, and drop-one cannot detect it.

Drop-pair attribution measures synergy directly. For a pair `(a, b)` of
component kinds, we compute four scores on the same eval subset:

    s_full   = score(harness)                      both a and b present
    s_drop_a = score(harness with kind_a → no-op)  only b
    s_drop_b = score(harness with kind_b → no-op)  only a
    s_neither = score(harness with both a and b → no-op)

The synergy delta is

    Δ_synergy(a, b) = s_full − s_neither
                      − [(s_drop_b − s_neither) + (s_drop_a − s_neither)]

Interpretation:
  Δ > 0:  super-additive synergy (the pair helps more than the sum of
          its parts) — these components only really work together.
  Δ ≈ 0:  approximately additive (drop-one captures the value).
  Δ < 0:  redundant (the pair partially cancels — having both is
          worse than either alone).

The cost is `O(K²)` ablation calls per harness, vs `O(K)` for drop-one.
We default to running synergy analysis only on a *small set of
candidates of interest* (typically the top of the Pareto frontier),
not on every search candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .harness import Component, Harness
from .scorer import ScoreVector, Scorer
from .task import TaskExample


@dataclass
class SynergySnapshot:
    """Per-pair synergy on a single candidate harness."""
    candidate_id: str
    kind_a: str
    kind_b: str
    s_full: float       # accuracy with both a and b
    s_drop_a: float     # accuracy with a → no-op
    s_drop_b: float     # accuracy with b → no-op
    s_neither: float    # accuracy with both a and b → no-op
    synergy_delta: float  # full − neither − (drop_a − neither + drop_b − neither)


@dataclass
class SynergyStats:
    pair: tuple[str, str]
    n: int = 0
    mean_synergy: float = 0.0

    def update(self, delta: float) -> None:
        self.n += 1
        self.mean_synergy += (delta - self.mean_synergy) / self.n


class SynergyTracker:
    """Drop-pair ablation + per-pair running stats.

    Pair the SynergyTracker with an existing AttributionTracker; the two
    are complementary (drop-one captures additive value, synergy captures
    pair interactions).
    """

    def __init__(
        self,
        scorer: Scorer,
        baseline_factory: Callable[[str], Component],
    ):
        self.scorer = scorer
        self.baseline_factory = baseline_factory
        self.stats: dict[tuple[str, str], SynergyStats] = {}
        self.snapshots: list[SynergySnapshot] = []

    def analyze(
        self,
        candidate_id: str,
        harness: Harness,
        examples: Sequence[TaskExample],
        full_score: ScoreVector | None = None,
        *,
        n_repeats: int = 1,
        max_workers: int = 1,
    ) -> list[SynergySnapshot]:
        """Drop-pair ablation on ``harness`` against ``examples``.

        Returns one ``SynergySnapshot`` per ordered pair (a, b) where a < b
        lexicographically by kind name (to dedup the symmetric pair).
        """
        # Score full harness once.
        if (full_score is None
                or full_score.n_evaluated != len(examples)
                or full_score.n_repeats != n_repeats):
            full_score = self.scorer.score(harness, examples, n_repeats=n_repeats,
                                           max_workers=max_workers)

        # Get unique (kind, component) pairs that have a baseline available.
        kinds: list[str] = []
        seen: set[str] = set()
        for c in harness.components:
            if c.kind in seen:
                continue
            seen.add(c.kind)
            if self.baseline_factory(c.kind) is not None:
                kinds.append(c.kind)

        # Cache single-drop scores so we don't re-run them for every pair.
        single_drop_score: dict[str, float] = {}
        for k in kinds:
            ablated = harness.swap(k, self.baseline_factory(k))
            single_drop_score[k] = self.scorer.score(
                ablated, examples, n_repeats=n_repeats, max_workers=max_workers,
            ).accuracy

        out: list[SynergySnapshot] = []
        # Iterate sorted unique pairs.
        for i in range(len(kinds)):
            for j in range(i + 1, len(kinds)):
                ka, kb = kinds[i], kinds[j]
                # Sort lexicographically so (a, b) and (b, a) collapse.
                if ka > kb:
                    ka, kb = kb, ka
                # Drop both
                ab_dropped = (
                    harness
                    .swap(ka, self.baseline_factory(ka))
                    .swap(kb, self.baseline_factory(kb))
                )
                s_neither = self.scorer.score(
                    ab_dropped, examples, n_repeats=n_repeats, max_workers=max_workers,
                ).accuracy

                synergy = (
                    full_score.accuracy - s_neither
                    - ((single_drop_score[kb] - s_neither)
                       + (single_drop_score[ka] - s_neither))
                )
                snap = SynergySnapshot(
                    candidate_id=candidate_id,
                    kind_a=ka, kind_b=kb,
                    s_full=full_score.accuracy,
                    s_drop_a=single_drop_score[ka],
                    s_drop_b=single_drop_score[kb],
                    s_neither=s_neither,
                    synergy_delta=synergy,
                )
                out.append(snap)
                self.snapshots.append(snap)
                self.stats.setdefault((ka, kb), SynergyStats(pair=(ka, kb))).update(synergy)
        return out

    def ranking(self, top_n: int | None = None) -> list[SynergyStats]:
        """Pairs ranked by mean synergy delta (descending)."""
        ranked = sorted(self.stats.values(), key=lambda s: s.mean_synergy, reverse=True)
        return ranked if top_n is None else ranked[:top_n]
