"""Bootstrap CI + paired diff helpers (Tier 1.2)."""

from __future__ import annotations

import unittest

from meta_harness_plus.statistics import (
    bootstrap_ci,
    paired_bootstrap_diff,
    pearson_correlation,
)


class TestBootstrapCI(unittest.TestCase):
    def test_empty_input(self):
        ci = bootstrap_ci([])
        self.assertEqual(ci.mean, 0.0)
        self.assertEqual((ci.low, ci.high), (0.0, 0.0))

    def test_single_value(self):
        ci = bootstrap_ci([0.5])
        self.assertEqual(ci.mean, 0.5)
        self.assertEqual((ci.low, ci.high), (0.5, 0.5))

    def test_constant_vector_zero_width(self):
        ci = bootstrap_ci([0.7] * 10, n_resamples=500)
        self.assertAlmostEqual(ci.mean, 0.7)
        self.assertAlmostEqual(ci.low, 0.7)
        self.assertAlmostEqual(ci.high, 0.7)

    def test_widening_with_variance(self):
        narrow = bootstrap_ci([0.7, 0.7, 0.7, 0.7, 0.7], n_resamples=500, seed=1)
        wide = bootstrap_ci([0.4, 0.5, 0.6, 0.7, 0.9, 1.0], n_resamples=500, seed=1)
        self.assertLess(narrow.high - narrow.low, wide.high - wide.low)

    def test_mean_unbiased(self):
        # The CI's mean field is the sample mean (not bootstrapped).
        vals = [0.1, 0.5, 0.9]
        ci = bootstrap_ci(vals)
        self.assertAlmostEqual(ci.mean, 0.5)

    def test_confidence_level_threading(self):
        ci_99 = bootstrap_ci([0.4, 0.5, 0.6, 0.7, 0.9], confidence=0.99, seed=2)
        ci_80 = bootstrap_ci([0.4, 0.5, 0.6, 0.7, 0.9], confidence=0.80, seed=2)
        # 99% CI should be wider than 80% CI on the same data.
        self.assertGreaterEqual(ci_99.high - ci_99.low,
                                ci_80.high - ci_80.low)


class TestPairedBootstrapDiff(unittest.TestCase):
    def test_zero_diff(self):
        a = [0.5, 0.6, 0.7]
        ci = paired_bootstrap_diff(a, a, n_resamples=300)
        self.assertAlmostEqual(ci.mean, 0.0)
        self.assertAlmostEqual(ci.low, 0.0)
        self.assertAlmostEqual(ci.high, 0.0)

    def test_constant_offset(self):
        a = [0.7, 0.7, 0.7, 0.7]
        b = [0.5, 0.5, 0.5, 0.5]
        ci = paired_bootstrap_diff(a, b, n_resamples=500)
        self.assertAlmostEqual(ci.mean, 0.2)
        self.assertAlmostEqual(ci.low, 0.2)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            paired_bootstrap_diff([0.1, 0.2], [0.3])


class TestPearsonCorrelation(unittest.TestCase):
    def test_perfect_positive(self):
        c = pearson_correlation([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0])
        self.assertAlmostEqual(c, 1.0, places=5)

    def test_perfect_negative(self):
        c = pearson_correlation([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0])
        self.assertAlmostEqual(c, -1.0, places=5)

    def test_uncorrelated(self):
        c = pearson_correlation([1.0, 2.0, 3.0, 4.0], [3.0, 1.0, 4.0, 2.0])
        self.assertAlmostEqual(c, 0.0, places=5)

    def test_degenerate_returns_zero(self):
        self.assertEqual(pearson_correlation([1.0, 1.0], [1.0, 1.0]), 0.0)
        self.assertEqual(pearson_correlation([1.0], [1.0]), 0.0)
        self.assertEqual(pearson_correlation([1.0, 2.0], [1.0]), 0.0)


if __name__ == "__main__":
    unittest.main()
