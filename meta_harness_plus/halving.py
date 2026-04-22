"""Successive halving for budget-aware harness evaluation.

Contribution C2. Given a population of candidates:
  round 0: evaluate all on a cheap screen (k_0 examples each)
  round 1: keep top 1/eta, evaluate on k_1 = k_0 * eta examples
  round r: keep top 1/eta^r, evaluate on k_r = k_0 * eta^r examples
Stop when |survivors| <= final_keep.

Ranking between rounds uses the Pareto-aware score: we rank by
*non-dominated-rank* (front 0 is best, then front 1, etc.), then within
the same front by a Pareto-compatible scalar tiebreaker (accuracy / log-cost).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log
from typing import Callable, Generic, Sequence, TypeVar

from .pareto import dominates
from .scorer import ScoreVector

T = TypeVar("T")


@dataclass
class HalvingResult(Generic[T]):
    survivors: list[T]
    all_scores: dict[int, list[tuple[T, ScoreVector, int]]] = field(default_factory=dict)
    total_evaluations: int = 0


def _front_ranks(scores: list[ScoreVector]) -> list[int]:
    """Assign each index its non-dominated front number (0 = best front)."""
    n = len(scores)
    rank = [0] * n
    remaining = set(range(n))
    current_rank = 0
    while remaining:
        front = []
        for i in remaining:
            if not any(dominates(scores[j], scores[i]) for j in remaining if j != i):
                front.append(i)
        for i in front:
            rank[i] = current_rank
            remaining.discard(i)
        if not front:
            # Pathological (shouldn't happen), fall back.
            for i in remaining:
                rank[i] = current_rank
            break
        current_rank += 1
    return rank


def _tiebreak(s: ScoreVector) -> float:
    """Scalar within-front tiebreaker; higher is better.

    Rewards accuracy, penalizes log of (tokens + latency). The log dampens
    huge cost differences so accuracy still dominates when they're similar.
    """
    cost = max(1.0, s.tokens + s.latency_ms)
    return s.accuracy - 0.01 * log(cost)


class SuccessiveHalving:
    def __init__(
        self,
        k0: int = 5,
        eta: int = 2,
        final_keep: int = 1,
        max_rounds: int = 10,
    ):
        if k0 <= 0 or eta < 2 or final_keep < 1:
            raise ValueError("k0 > 0, eta >= 2, final_keep >= 1 required")
        self.k0 = k0
        self.eta = eta
        self.final_keep = final_keep
        self.max_rounds = max_rounds

    def run(
        self,
        candidates: Sequence[T],
        evaluate: Callable[[T, int], ScoreVector],
    ) -> HalvingResult[T]:
        """Evaluate candidates with increasing budget, keeping survivors each round.

        ``evaluate(candidate, k)`` must return a ScoreVector measured on k items.
        """
        survivors = list(candidates)
        per_round: dict[int, list[tuple[T, ScoreVector, int]]] = {}
        total = 0
        k = self.k0
        for r in range(self.max_rounds):
            scored: list[tuple[T, ScoreVector]] = []
            for cand in survivors:
                s = evaluate(cand, k)
                scored.append((cand, s))
                total += k
            per_round[r] = [(c, s, k) for c, s in scored]

            if len(survivors) <= self.final_keep:
                break

            # Rank by (front number ascending, tiebreak descending).
            scores_only = [s for _, s in scored]
            ranks = _front_ranks(scores_only)
            decorated = sorted(
                range(len(scored)),
                key=lambda i: (ranks[i], -_tiebreak(scored[i][1])),
            )
            keep_count = max(self.final_keep, len(scored) // self.eta)
            survivors = [scored[i][0] for i in decorated[:keep_count]]
            k *= self.eta

        return HalvingResult(
            survivors=survivors,
            all_scores=per_round,
            total_evaluations=total,
        )
