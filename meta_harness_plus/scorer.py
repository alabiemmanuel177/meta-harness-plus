"""Multi-objective scoring.

Contribution C1: scoring is a vector, not a scalar. Pareto comparison happens
in ``pareto.py``. Accuracy is maximize; tokens and latency are minimize —
encoded below as ``higher_is_better`` flags so Pareto logic stays symmetric.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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
    # Reproducibility fields (populated only when n_repeats > 1):
    n_repeats: int = 1
    accuracy_spread: float = 0.0   # max(acc) - min(acc) across repeats

    @staticmethod
    def objective_signs() -> tuple[int, int, int]:
        """(+1 maximize, -1 minimize) for (accuracy, tokens, latency)."""
        return (+1, -1, -1)

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.accuracy, self.tokens, self.latency_ms)


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


class Scorer:
    """Run a harness over a set of examples and aggregate costs + accuracy.

    Reproducibility note (the raison d'être of this branch): with
    ``n_repeats > 1``, the harness is evaluated the full sweep ``n_repeats``
    times and the aggregate is **median accuracy** + **mean tokens** +
    **mean latency**.  Median is robust to the one-in-N outlier runs that
    plague temperature=0 local models due to GPU batching non-determinism.
    Mean is fine for costs — they're approximately Gaussian across repeats.

    With ``n_repeats == 1`` (default) the behavior is identical to the
    single-shot Scorer on ``main``, so existing tests and workflows are
    unaffected.
    """

    def __init__(self, task: Task):
        self.task = task

    def score(
        self,
        harness: Harness,
        examples: Sequence[TaskExample],
        *,
        n_repeats: int = 1,
        max_workers: int = 1,
    ) -> ScoreVector:
        """Evaluate ``harness`` on ``examples``.

        ``max_workers > 1`` (parallel-scoring branch): when set, fans out
        per-example evaluation across a thread pool. Useful for cloud-API
        backends where each ``harness.run(ex)`` is dominated by network
        latency — one thread per example with concurrent HTTP requests
        gets a 5-10× wall-clock speedup. Default 1 preserves the
        sequential, deterministic-with-ScriptedClient behaviour.

        Threading is safe with ``LLMPredictor`` over a real network client
        (urllib releases the GIL during I/O). It is NOT safe with
        ``ScriptedClient`` (its response queue has shared state); tests
        that use ScriptedClient must keep ``max_workers=1``.
        """
        if not examples:
            return ScoreVector(0.0, 0.0, 0.0, 0, n_repeats=max(1, n_repeats))
        if n_repeats < 1:
            raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")

        def eval_one(ex: TaskExample) -> tuple[int, float, float]:
            ctx = harness.run(ex)
            return (
                1 if ctx.prediction == ex.label else 0,
                float(ctx.tokens),
                float(ctx.latency_ms),
            )

        def eval_sweep() -> list[tuple[int, float, float]]:
            if max_workers == 1:
                return [eval_one(ex) for ex in examples]
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                # pool.map preserves the input order — important for
                # determinism vs the sequential path.
                return list(pool.map(eval_one, examples))

        n = len(examples)

        if n_repeats == 1:
            results = eval_sweep()
            correct = sum(r[0] for r in results)
            total_tokens = sum(r[1] for r in results)
            total_latency = sum(r[2] for r in results)
            return ScoreVector(
                accuracy=correct / n,
                tokens=total_tokens / n,
                latency_ms=total_latency / n,
                n_evaluated=n,
            )

        # Multi-repeat path.
        accs: list[float] = []
        tok_sums: list[float] = []
        lat_sums: list[float] = []
        for _ in range(n_repeats):
            results = eval_sweep()
            correct = sum(r[0] for r in results)
            tok_sum = sum(r[1] for r in results)
            lat_sum = sum(r[2] for r in results)
            accs.append(correct / n)
            tok_sums.append(tok_sum / n)
            lat_sums.append(lat_sum / n)
        return ScoreVector(
            accuracy=_median(accs),
            tokens=sum(tok_sums) / n_repeats,
            latency_ms=sum(lat_sums) / n_repeats,
            n_evaluated=n,
            n_repeats=n_repeats,
            accuracy_spread=max(accs) - min(accs),
        )
