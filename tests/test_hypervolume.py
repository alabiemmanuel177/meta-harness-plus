"""Hypervolume metric on the Pareto frontier (Tier 4.2)."""

from __future__ import annotations

import unittest

from meta_harness_plus.pareto import FrontierEntry, ParetoFrontier, hypervolume
from meta_harness_plus.scorer import ScoreVector


def _entry(cid: str, acc: float, tok: float, lat: float) -> FrontierEntry:
    return FrontierEntry(
        candidate_id=cid,
        score=ScoreVector(accuracy=acc, tokens=tok, latency_ms=lat,
                          n_evaluated=20),
        meta={},
    )


class TestHypervolume(unittest.TestCase):
    def test_empty_frontier_zero(self):
        self.assertEqual(hypervolume([]), 0.0)

    def test_single_point_box(self):
        # Reference (0, 1000, 10000); point (0.8, 100, 50) gives box
        # 0.8 × 900 × 9950 = 7164000.
        e = _entry("a", 0.8, 100, 50)
        hv = hypervolume([e], reference=(0.0, 1000.0, 10_000.0))
        expected = 0.8 * 900 * 9950
        self.assertAlmostEqual(hv, expected, places=2)

    def test_dominated_point_no_extra_volume(self):
        # Strictly worse than another — adding it shouldn't increase HV.
        a = _entry("a", 0.9, 100, 50)
        b = _entry("b", 0.5, 200, 100)  # strictly worse
        hv_a = hypervolume([a])
        hv_both = hypervolume([a, b])
        self.assertAlmostEqual(hv_a, hv_both, places=2)

    def test_two_non_dominated_points(self):
        # a is high-acc-high-cost; b is low-acc-low-cost. Both contribute
        # area. Total HV strictly greater than either alone.
        a = _entry("a", 0.9, 200, 100)
        b = _entry("b", 0.6,  50,  10)
        hv_a = hypervolume([a])
        hv_b = hypervolume([b])
        hv_both = hypervolume([a, b])
        self.assertGreater(hv_both, max(hv_a, hv_b))

    def test_below_reference_acc_excluded(self):
        # Point with accuracy=0 contributes zero (acc must exceed reference).
        e = _entry("a", 0.0, 50, 10)
        hv = hypervolume([e])
        self.assertEqual(hv, 0.0)

    def test_above_reference_cost_excluded(self):
        # Point with tokens=2000 (above reference cap of 1000) contributes 0.
        e = _entry("a", 0.9, 2000, 50)
        hv = hypervolume([e], reference=(0.0, 1000.0, 10_000.0))
        self.assertEqual(hv, 0.0)

    def test_monotone_under_frontier_extension(self):
        """Adding a non-dominated point strictly increases HV."""
        a = _entry("a", 0.8, 100, 50)
        hv1 = hypervolume([a])
        # Add a higher-accuracy point at same cost — strict improvement.
        b = _entry("b", 0.95, 100, 50)
        hv2 = hypervolume([a, b])
        self.assertGreater(hv2, hv1)

    def test_works_with_pareto_frontier_object(self):
        f = ParetoFrontier()
        f.offer(_entry("a", 0.9, 200, 100))
        f.offer(_entry("b", 0.6, 50, 10))
        hv = hypervolume(f.entries)
        self.assertGreater(hv, 0.0)


if __name__ == "__main__":
    unittest.main()
