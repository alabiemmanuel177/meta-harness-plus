"""Paired t-test + Cohen's d (better-aggregator branch)."""

from __future__ import annotations

import math
import unittest

from meta_harness_plus.statistics import (
    PairedTestResult,
    _incomplete_beta_regularized,
    paired_t_test,
)


class TestPairedTTest(unittest.TestCase):
    def test_identical_inputs_zero_t_p_one(self):
        r = paired_t_test([0.5, 0.6, 0.7], [0.5, 0.6, 0.7])
        self.assertEqual(r.t_statistic, 0.0)
        self.assertEqual(r.p_value_two_sided, 1.0)
        self.assertEqual(r.cohens_d, 0.0)

    def test_constant_offset_finite_d(self):
        # All differences = 0.04, std=0 → cohen's d = inf, t = inf
        r = paired_t_test(
            [0.92, 0.92, 0.92, 0.94, 0.92],
            [0.88, 0.88, 0.88, 0.88, 0.88],
        )
        # Variance is non-zero (one element differs from the rest), so
        # this isn't the strict-constant edge case.
        self.assertGreater(r.t_statistic, 5.0)
        self.assertLess(r.p_value_two_sided, 0.01)
        self.assertGreater(r.cohens_d, 1.0)
        self.assertEqual(r.n, 5)

    def test_pure_constant_diff_handles_zero_std(self):
        # Every diff exactly the same → std=0; reports inf t with d=inf.
        r = paired_t_test([1.0, 1.0, 1.0], [0.5, 0.5, 0.5])
        self.assertEqual(r.std_diff, 0.0)
        self.assertEqual(r.t_statistic, float("inf"))
        self.assertEqual(r.cohens_d, float("inf"))

    def test_zero_diff_constant(self):
        # All diffs zero, std zero → t=0 p=1 d=0
        r = paired_t_test([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
        self.assertEqual(r.t_statistic, 0.0)
        self.assertEqual(r.p_value_two_sided, 1.0)
        self.assertEqual(r.cohens_d, 0.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            paired_t_test([0.1, 0.2], [0.3])

    def test_n_below_2_handled(self):
        r = paired_t_test([0.5], [0.4])
        self.assertEqual(r.n, 1)
        self.assertEqual(r.p_value_two_sided, 1.0)

    def test_known_textbook_value(self):
        """Known case: paired t-test on a 5-pair sample with known stats."""
        a = [3.0, 4.0, 5.0, 4.5, 6.0]
        b = [2.0, 3.5, 4.0, 4.0, 5.0]
        # Diffs = [1, 0.5, 1, 0.5, 1] → mean=0.8, std≈0.274
        r = paired_t_test(a, b)
        self.assertAlmostEqual(r.mean_diff, 0.8, places=4)
        # std of [1, 0.5, 1, 0.5, 1] sample stddev (n-1) = sqrt(0.075) ≈ 0.2739
        self.assertAlmostEqual(r.std_diff, 0.27386, places=3)
        # t = 0.8 / (0.27386/sqrt(5)) = 6.532
        self.assertAlmostEqual(r.t_statistic, 6.532, places=2)
        # p two-sided << 0.01 for t=6.5, df=4
        self.assertLess(r.p_value_two_sided, 0.005)


class TestIncompleteBeta(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(_incomplete_beta_regularized(2.0, 3.0, 0.0), 0.0)
        self.assertEqual(_incomplete_beta_regularized(2.0, 3.0, 1.0), 1.0)

    def test_known_value(self):
        # I_{0.5}(2, 2) = 0.5 by symmetry
        v = _incomplete_beta_regularized(2.0, 2.0, 0.5)
        self.assertAlmostEqual(v, 0.5, places=5)


if __name__ == "__main__":
    unittest.main()
