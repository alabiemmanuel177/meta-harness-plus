"""Successive halving invariants."""

from __future__ import annotations

import random
import unittest

from meta_harness_plus.halving import SuccessiveHalving
from meta_harness_plus.scorer import ScoreVector


class TestHalving(unittest.TestCase):
    def test_respects_final_keep(self):
        """With final_keep=1 and enough rounds, exactly 1 candidate survives."""
        candidates = list(range(8))
        def evaluate(c: int, k: int) -> ScoreVector:
            # Candidate index encodes quality: lower idx = worse.
            return ScoreVector(accuracy=c / 10, tokens=1.0, latency_ms=1.0, n_evaluated=k)
        sh = SuccessiveHalving(k0=2, eta=2, final_keep=1, max_rounds=8)
        result = sh.run(candidates, evaluate)
        self.assertEqual(len(result.survivors), 1)

    def test_top_candidate_survives_on_clean_signal(self):
        """When accuracy is monotone in candidate index, halving must keep the top."""
        candidates = list(range(8))
        def evaluate(c: int, k: int) -> ScoreVector:
            return ScoreVector(accuracy=c / 10, tokens=1.0, latency_ms=1.0, n_evaluated=k)
        sh = SuccessiveHalving(k0=2, eta=2, final_keep=1, max_rounds=8)
        result = sh.run(candidates, evaluate)
        self.assertEqual(result.survivors[0], 7)

    def test_budget_accounting(self):
        """total_evaluations equals sum over rounds of (|survivors_at_round| * k_round)."""
        candidates = list(range(8))
        calls: list[tuple[int, int]] = []
        def evaluate(c: int, k: int) -> ScoreVector:
            calls.append((c, k))
            return ScoreVector(c / 10, 1.0, 1.0, k)
        sh = SuccessiveHalving(k0=2, eta=2, final_keep=1, max_rounds=8)
        result = sh.run(candidates, evaluate)
        expected = sum(k for _, k in calls)
        self.assertEqual(result.total_evaluations, expected)
        # And crucially, halving should be CHEAPER than full eval of all candidates
        # at the final budget k_final. With k0=2, eta=2, final=1, 8 candidates:
        # round 0: 8 * 2 = 16
        # round 1: 4 * 4 = 16
        # round 2: 2 * 8 = 16
        # round 3: 1 * 16 = 16   (stops because |survivors| <= final_keep)
        # Total 64. Naive full-eval at k=16 on all 8 would be 128.
        self.assertLess(result.total_evaluations, 8 * 16)

    def test_pareto_ranking_preserves_tradeoffs(self):
        """Halving uses non-dominated sort — two non-dominated candidates should
        both make it past the first cut even if their scalar is similar."""
        candidates = ["cheap_low_acc", "expensive_high_acc", "dominated_mid"]
        def evaluate(c: str, k: int) -> ScoreVector:
            return {
                "cheap_low_acc":        ScoreVector(0.5,  10.0, 2.0, k),
                "expensive_high_acc":   ScoreVector(0.9, 200.0, 30.0, k),
                "dominated_mid":        ScoreVector(0.5, 100.0, 20.0, k),
            }[c]
        # With final_keep=2 and 3 candidates, one should be dropped — and it
        # should be the dominated one.
        sh = SuccessiveHalving(k0=2, eta=2, final_keep=2, max_rounds=3)
        result = sh.run(candidates, evaluate)
        self.assertEqual(len(result.survivors), 2)
        self.assertNotIn("dominated_mid", result.survivors)


if __name__ == "__main__":
    unittest.main()
