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


@dataclass(frozen=True)
class PairedTestResult:
    """Paired t-test on (a[i] - b[i]) returning Cohen's d as effect size.

    No SciPy dependency — t and p computed via stdlib math.
    """
    n: int
    mean_diff: float
    std_diff: float
    t_statistic: float
    p_value_two_sided: float
    cohens_d: float


def paired_t_test(
    a: Sequence[float], b: Sequence[float],
) -> PairedTestResult:
    """Paired t-test for repeated measures.

    Returns the t statistic, two-sided p-value (Student's t CDF
    via the regularized incomplete beta function — accurate to ~1e-7),
    and Cohen's d effect size (mean_diff / std_diff). Cohen's d
    interpretation: 0.2 = small, 0.5 = medium, 0.8 = large.

    Both arguments must have equal length. Returns zero-filled result
    for degenerate inputs rather than raising.
    """
    if len(a) != len(b):
        raise ValueError(f"paired vectors must match length: {len(a)} vs {len(b)}")
    n = len(a)
    if n < 2:
        return PairedTestResult(n, 0.0, 0.0, 0.0, 1.0, 0.0)
    diffs = [a[i] - b[i] for i in range(n)]
    mean_diff = sum(diffs) / n
    variance = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
    std_diff = variance ** 0.5
    if std_diff == 0.0:
        if mean_diff == 0.0:
            return PairedTestResult(n, 0.0, 0.0, 0.0, 1.0, 0.0)
        return PairedTestResult(n, mean_diff, 0.0, float("inf"), 0.0, float("inf"))
    se = std_diff / (n ** 0.5)
    t = mean_diff / se
    df = n - 1
    # Student's t two-sided p via regularized incomplete beta:
    #   p = I_x(df/2, 1/2)  where x = df / (df + t^2).
    x = df / (df + t * t)
    p = _incomplete_beta_regularized(df / 2.0, 0.5, x)
    cohens_d = mean_diff / std_diff
    return PairedTestResult(
        n=n, mean_diff=mean_diff, std_diff=std_diff,
        t_statistic=t, p_value_two_sided=p, cohens_d=cohens_d,
    )


def _incomplete_beta_regularized(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b), for the Student's t CDF.

    Continued-fraction implementation from Numerical Recipes — accurate
    to ~1e-7 for typical statistical use cases. No SciPy dependency.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _betacf(a: float, b: float, x: float, max_iter: int = 200, eps: float = 3e-7) -> float:
    """Continued-fraction expansion for the incomplete beta function."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    return h
