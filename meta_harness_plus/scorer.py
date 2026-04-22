"""Multi-objective scoring.

Contribution C1: scoring is a vector, not a scalar. Pareto comparison happens
in ``pareto.py``. Accuracy is maximize; tokens and latency are minimize —
encoded below as ``higher_is_better`` flags so Pareto logic stays symmetric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .harness import Harness
from .task import Task, TaskExample


@dataclass(frozen=True)
class ScoreVector:
    accuracy: float          # [0, 1], maximize
    tokens: float            # count, minimize
    latency_ms: float        # ms, minimize
    n_evaluated: int         # how many examples this score was computed over

    @staticmethod
    def objective_signs() -> tuple[int, int, int]:
        """(+1 maximize, -1 minimize) for (accuracy, tokens, latency)."""
        return (+1, -1, -1)

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.accuracy, self.tokens, self.latency_ms)


class Scorer:
    """Run a harness over a set of examples and aggregate costs + accuracy."""

    def __init__(self, task: Task):
        self.task = task

    def score(self, harness: Harness, examples: Sequence[TaskExample]) -> ScoreVector:
        if not examples:
            return ScoreVector(0.0, 0.0, 0.0, 0)
        correct = 0
        total_tokens = 0
        total_latency = 0.0
        for ex in examples:
            ctx = harness.run(ex)
            if ctx.prediction == ex.label:
                correct += 1
            total_tokens += ctx.tokens
            total_latency += ctx.latency_ms
        n = len(examples)
        return ScoreVector(
            accuracy=correct / n,
            tokens=total_tokens / n,
            latency_ms=total_latency / n,
            n_evaluated=n,
        )
