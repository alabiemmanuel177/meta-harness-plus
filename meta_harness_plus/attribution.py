"""Component-level credit assignment via drop-one ablation.

Contribution C3. When a candidate reaches the Pareto frontier, we run cheap
ablations on the screening subset: for each component kind in the harness,
re-score a version with that kind replaced by a minimal baseline component.
The accuracy delta is the component's attributed value on this candidate.

Running averages per kind are exposed so the proposer can bias mutations
toward high-value components (and prune low- or negative-value ones).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from .harness import Component, Harness
from .scorer import ScoreVector, Scorer
from .task import TaskExample


@dataclass
class AttributionSnapshot:
    """Per-kind value on a single candidate: accuracy delta vs baseline-swapped variant."""
    candidate_id: str
    kind: str
    full_score: ScoreVector
    ablated_score: ScoreVector
    # Positive = component helped accuracy; negative = it hurt.
    accuracy_delta: float


@dataclass
class AttributionStats:
    kind: str
    n: int = 0
    mean_delta: float = 0.0
    # Keep running mean + squared deviations for a stable variance read-out.
    m2: float = 0.0
    # Exponentially-weighted moving average — weights recent ablation
    # snapshots more heavily than old ones. Better signal for the
    # proposer when the search is mid-flight: "what's working LATELY"
    # is more actionable than "what's worked on average since iter 0",
    # because the Pareto frontier shifts over the course of a search.
    ewma_alpha: float = 0.3
    ewma_delta: float = 0.0

    def update(self, delta: float) -> None:
        self.n += 1
        d = delta - self.mean_delta
        self.mean_delta += d / self.n
        d2 = delta - self.mean_delta
        self.m2 += d * d2
        # EWMA: first sample initializes; subsequent samples blend.
        if self.n == 1:
            self.ewma_delta = delta
        else:
            self.ewma_delta = (
                self.ewma_alpha * delta
                + (1.0 - self.ewma_alpha) * self.ewma_delta
            )

    @property
    def variance(self) -> float:
        return self.m2 / self.n if self.n > 1 else 0.0


class AttributionTracker:
    """Runs drop-one ablations + maintains per-kind running statistics."""

    def __init__(
        self,
        scorer: Scorer,
        baseline_factory: Callable[[str], Component],
    ):
        self.scorer = scorer
        self.baseline_factory = baseline_factory
        self.stats: dict[str, AttributionStats] = {}
        self.snapshots: list[AttributionSnapshot] = []

    def analyze(
        self,
        candidate_id: str,
        harness: Harness,
        examples: Sequence[TaskExample],
        full_score: ScoreVector | None = None,
        *,
        n_repeats: int = 1,
        max_workers: int = 1,
    ) -> list[AttributionSnapshot]:
        """Drop-one ablation on ``harness`` against ``examples``.

        IMPORTANT: ``full_score`` is only used as a cached value when it was
        measured on the same ``examples`` AND the same ``n_repeats`` budget.
        Otherwise we re-score so deltas are apples-to-apples.

        ``n_repeats`` (branch: reproducibility): when > 1, both the full and
        the ablated harness are evaluated ``n_repeats`` times with median-
        accuracy aggregation. Cuts the noise floor on drop-one deltas — the
        chief complaint from the RESULTS.md bakeoff where 6-item screens
        produced near-zero attribution signal.

        ``max_workers`` (branch: parallel-scoring): forwarded to the
        Scorer so per-example LLM calls fan out across a thread pool.
        """
        cache_valid = (
            full_score is not None
            and full_score.n_evaluated == len(examples)
            and full_score.n_repeats == n_repeats
        )
        if not cache_valid:
            full_score = self.scorer.score(harness, examples, n_repeats=n_repeats,
                                           max_workers=max_workers)
        out: list[AttributionSnapshot] = []
        seen_kinds: set[str] = set()
        for comp in harness.components:
            if comp.kind in seen_kinds:
                continue
            seen_kinds.add(comp.kind)
            baseline = self.baseline_factory(comp.kind)
            if baseline is None:
                # If no baseline exists for this kind, skip the component rather
                # than dropping — dropping a formatter, say, can make the harness
                # meaningless and the attribution misleading.
                continue
            ablated_harness = harness.swap(comp.kind, baseline)
            ablated_score = self.scorer.score(ablated_harness, examples,
                                              n_repeats=n_repeats,
                                              max_workers=max_workers)
            delta = full_score.accuracy - ablated_score.accuracy
            snap = AttributionSnapshot(
                candidate_id=candidate_id,
                kind=comp.kind,
                full_score=full_score,
                ablated_score=ablated_score,
                accuracy_delta=delta,
            )
            out.append(snap)
            self.snapshots.append(snap)
            self.stats.setdefault(comp.kind, AttributionStats(kind=comp.kind)).update(delta)
        return out

    def ranking(self) -> list[AttributionStats]:
        """Kinds ranked by mean attributed value (descending)."""
        return sorted(self.stats.values(), key=lambda s: s.mean_delta, reverse=True)

    def mutation_weights(self, temperature: float = 1.0) -> dict[str, float]:
        """Turn attribution stats into a probability distribution over kinds.

        We softmax over mean_delta — high-value kinds get mutated more.
        Negative-attribution kinds still get non-zero probability (so the
        proposer can prune them), just low.
        """
        import math
        stats = list(self.stats.values())
        if not stats:
            return {}
        # Shift for numerical stability.
        best = max(s.mean_delta for s in stats)
        exps = {s.kind: math.exp((s.mean_delta - best) / max(temperature, 1e-6)) for s in stats}
        z = sum(exps.values())
        return {k: v / z for k, v in exps.items()}
