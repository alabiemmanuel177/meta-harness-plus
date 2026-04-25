"""Statistical helpers (Tier 1.2).

Bootstrap confidence intervals + paired difference tests. Zero deps,
stdlib only.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class BootstrapCI:
    mean: float
    low: float       # lower bound at confidence level
    high: float      # upper bound
    confidence: float
    n_resamples: int


def bootstrap_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> BootstrapCI:
    """Non-parametric bootstrap CI for the sample mean.

    Resamples ``values`` with replacement ``n_resamples`` times, computes
    the mean of each resample, returns the (low, high) percentiles at the
    requested confidence level.

    With < 2 values, returns the trivial (mean, mean, mean) — bootstrap
    is undefined.
    """
    if not values:
        return BootstrapCI(mean=0.0, low=0.0, high=0.0,
                           confidence=confidence, n_resamples=n_resamples)
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return BootstrapCI(mean=mean, low=mean, high=mean,
                           confidence=confidence, n_resamples=n_resamples)
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_resamples):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = 1.0 - confidence
    low_idx = int(math.floor((alpha / 2.0) * n_resamples))
    high_idx = int(math.ceil((1.0 - alpha / 2.0) * n_resamples)) - 1
    low_idx = max(0, low_idx)
    high_idx = min(n_resamples - 1, high_idx)
    return BootstrapCI(
        mean=mean,
        low=means[low_idx],
        high=means[high_idx],
        confidence=confidence,
        n_resamples=n_resamples,
    )


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> BootstrapCI:
    """Bootstrap CI for the mean of (a[i] - b[i]).

    Pair-resamples indices so each resample uses the same indices in a
    and b, preserving the pairing (e.g. same eval items scored under two
    harnesses). Returns the CI on the per-item difference.
    """
    if len(a) != len(b):
        raise ValueError(f"paired vectors must have equal length: {len(a)} vs {len(b)}")
    if not a:
        return BootstrapCI(mean=0.0, low=0.0, high=0.0,
                           confidence=confidence, n_resamples=n_resamples)
    diffs = [a[i] - b[i] for i in range(len(a))]
    return bootstrap_ci(diffs, confidence=confidence,
                        n_resamples=n_resamples, seed=seed)


def pearson_correlation(a: Sequence[float], b: Sequence[float]) -> float:
    """Pearson correlation; returns 0 for degenerate inputs."""
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    va = sum((a[i] - ma) ** 2 for i in range(n)) ** 0.5
    vb = sum((b[i] - mb) ** 2 for i in range(n)) ** 0.5
    if va == 0 or vb == 0:
        return 0.0
    return cov / (va * vb)
