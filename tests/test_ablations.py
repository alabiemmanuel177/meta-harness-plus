"""Ablation harness tests."""

from __future__ import annotations

import unittest

from meta_harness_plus.ablations import (
    AblationConfig,
    AblationResult,
    ScalarAccuracyFrontier,
    make_frontier,
    summarize_frontier,
)
from meta_harness_plus.pareto import FrontierEntry, ParetoFrontier
from meta_harness_plus.scorer import ScoreVector


def _entry(cid: str, acc: float, tok: float, lat: float) -> FrontierEntry:
    return FrontierEntry(
        candidate_id=cid,
        score=ScoreVector(accuracy=acc, tokens=tok, latency_ms=lat,
                          n_evaluated=20),
        meta={},
    )


class TestScalarAccuracyFrontier(unittest.TestCase):
    def test_keeps_only_best_accuracy(self):
        f = ScalarAccuracyFrontier()
        # First candidate admitted.
        self.assertTrue(f.offer(_entry("a", 0.6, 100, 50)))
        # Higher accuracy replaces — even at higher cost.
        self.assertTrue(f.offer(_entry("b", 0.8, 500, 1000)))
        self.assertEqual(len(f), 1)
        self.assertEqual(f.entries[0].candidate_id, "b")
        # Lower accuracy rejected even at much cheaper cost.
        self.assertFalse(f.offer(_entry("c", 0.7, 50, 10)))
        self.assertEqual(f.entries[0].candidate_id, "b")

    def test_tie_broken_by_lower_tokens(self):
        f = ScalarAccuracyFrontier()
        f.offer(_entry("a", 0.8, 200, 100))
        # Same accuracy, fewer tokens → replace.
        f.offer(_entry("b", 0.8, 100, 100))
        self.assertEqual(f.entries[0].candidate_id, "b")
        # Same accuracy, more tokens → keep current.
        f.offer(_entry("c", 0.8, 300, 50))
        self.assertEqual(f.entries[0].candidate_id, "b")


class TestAblationConfig(unittest.TestCase):
    def test_default_label(self):
        self.assertEqual(AblationConfig().label, "full-MH++")

    def test_single_ablations(self):
        self.assertEqual(AblationConfig(disable_pareto=True).label, "no-C1")
        self.assertEqual(AblationConfig(disable_halving=True).label, "no-C2")
        self.assertEqual(AblationConfig(disable_attribution=True).label, "no-C3")

    def test_combined_label(self):
        c = AblationConfig(disable_pareto=True, disable_halving=True)
        self.assertEqual(c.label, "no-C1+no-C2")
        c = AblationConfig(disable_pareto=True, disable_halving=True,
                           disable_attribution=True)
        self.assertEqual(c.label, "no-C1+no-C2+no-C3")


class TestMakeFrontier(unittest.TestCase):
    def test_default_returns_pareto(self):
        cfg = AblationConfig()
        f = make_frontier(cfg)
        self.assertIsInstance(f, ParetoFrontier)
        self.assertNotIsInstance(f, ScalarAccuracyFrontier)

    def test_disable_pareto_returns_scalar(self):
        cfg = AblationConfig(disable_pareto=True)
        f = make_frontier(cfg)
        self.assertIsInstance(f, ScalarAccuracyFrontier)


class TestSummarizeFrontier(unittest.TestCase):
    def test_empty_frontier(self):
        r = summarize_frontier("test", ParetoFrontier())
        self.assertEqual(r.best_acc, 0.0)
        self.assertEqual(r.n_frontier_points, 0)

    def test_picks_best_accuracy(self):
        f = ParetoFrontier()
        f.offer(_entry("a", 0.6, 50, 10))
        f.offer(_entry("b", 0.9, 200, 100))
        f.offer(_entry("c", 0.7, 100, 50))
        r = summarize_frontier("test", f)
        self.assertAlmostEqual(r.best_acc, 0.9)
        self.assertAlmostEqual(r.best_acc_tokens, 200)
        # All three entries non-dominated → frontier size 3.
        self.assertEqual(r.n_frontier_points, 3)


if __name__ == "__main__":
    unittest.main()
