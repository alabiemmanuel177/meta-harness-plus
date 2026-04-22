"""Pareto frontier over ScoreVectors.

Contribution C1. Non-dominated sort over (accuracy, -tokens, -latency).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .scorer import ScoreVector


def _normalize(s: ScoreVector) -> tuple[float, float, float]:
    """Apply maximize/minimize signs so "bigger is better" on every axis."""
    sa, st, sl = ScoreVector.objective_signs()
    return (sa * s.accuracy, st * s.tokens, sl * s.latency_ms)


def dominates(a: ScoreVector, b: ScoreVector) -> bool:
    """True iff ``a`` weakly dominates ``b`` on every objective AND strictly on at least one.

    Uses normalized signs so the check is uniform "≥ on all, > on some."
    """
    na, nb = _normalize(a), _normalize(b)
    better_or_equal = all(x >= y for x, y in zip(na, nb))
    strictly_better = any(x > y for x, y in zip(na, nb))
    return better_or_equal and strictly_better


@dataclass
class FrontierEntry:
    candidate_id: str
    score: ScoreVector
    # Free-form payload — harness describe(), attribution snapshot, etc.
    meta: dict


class ParetoFrontier:
    """Maintains the current non-dominated set.

    ``offer(entry)`` returns True if ``entry`` was admitted. Admission removes
    any previously-admitted entries it dominates.
    """

    def __init__(self) -> None:
        self.entries: list[FrontierEntry] = []

    def offer(self, entry: FrontierEntry) -> bool:
        # If any existing entry dominates the candidate, reject.
        for existing in self.entries:
            if dominates(existing.score, entry.score):
                return False
        # Reject exact score-tuple duplicates — only the first representative
        # of a given (acc, tokens, lat) tuple is admitted. This keeps the
        # frontier interpretable instead of cluttered with structurally-
        # different harnesses that happen to score identically.
        new_tuple = entry.score.as_tuple()
        for existing in self.entries:
            if existing.score.as_tuple() == new_tuple:
                return False
        # Otherwise admit and evict dominated entries.
        survivors = [e for e in self.entries if not dominates(entry.score, e.score)]
        survivors.append(entry)
        self.entries = survivors
        return True

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def best_by_accuracy(self) -> FrontierEntry | None:
        if not self.entries:
            return None
        return max(self.entries, key=lambda e: e.score.accuracy)

    def best_by_cost(self) -> FrontierEntry | None:
        if not self.entries:
            return None
        return min(self.entries, key=lambda e: e.score.tokens)

    def knee(self) -> FrontierEntry | None:
        """Return the entry closest to the utopia point on the normalized frontier.

        Useful as a single 'recommended' pick when a scalar is demanded.
        """
        if not self.entries:
            return None
        points = [ _normalize(e.score) for e in self.entries ]
        # Min-max normalize per axis within the frontier.
        axes = list(zip(*points))
        mins = [min(a) for a in axes]
        maxs = [max(a) for a in axes]
        def norm(p):
            out = []
            for v, lo, hi in zip(p, mins, maxs):
                if hi == lo:
                    out.append(1.0)
                else:
                    out.append((v - lo) / (hi - lo))
            return out
        best_i, best_d = 0, float("inf")
        for i, p in enumerate(points):
            np_ = norm(p)
            d = sum((1.0 - x) ** 2 for x in np_) ** 0.5
            if d < best_d:
                best_i, best_d = i, d
        return self.entries[best_i]


def non_dominated_filter(entries: Iterable[FrontierEntry]) -> list[FrontierEntry]:
    """Standalone non-dominated filter — handy in tests."""
    front = ParetoFrontier()
    for e in entries:
        front.offer(e)
    return list(front.entries)
