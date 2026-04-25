"""Pareto frontier invariants."""

from __future__ import annotations

import random
import unittest

from meta_harness_plus.pareto import FrontierEntry, ParetoFrontier, dominates, non_dominated_filter
from meta_harness_plus.scorer import ScoreVector


def mkentry(cid: str, acc: float, tokens: float, lat: float) -> FrontierEntry:
    return FrontierEntry(
        candidate_id=cid,
        score=ScoreVector(accuracy=acc, tokens=tokens, latency_ms=lat, n_evaluated=20),
        meta={},
    )


class TestDominance(unittest.TestCase):
    def test_strict_domination(self):
        a = ScoreVector(0.8, 50, 10, 20)
        b = ScoreVector(0.7, 60, 15, 20)
        self.assertTrue(dominates(a, b))
        self.assertFalse(dominates(b, a))

    def test_non_domination_when_tradeoff(self):
        # a has more accuracy but also more tokens — neither dominates.
        a = ScoreVector(0.8, 100, 10, 20)
        b = ScoreVector(0.7, 50, 10, 20)
        self.assertFalse(dominates(a, b))
        self.assertFalse(dominates(b, a))

    def test_equal_does_not_dominate(self):
        s = ScoreVector(0.8, 50, 10, 20)
        self.assertFalse(dominates(s, s))


class TestFrontier(unittest.TestCase):
    def test_admit_and_evict(self):
        f = ParetoFrontier()
        self.assertTrue(f.offer(mkentry("a", 0.7, 60, 15)))
        # Strictly better on every axis — should evict "a".
        self.assertTrue(f.offer(mkentry("b", 0.8, 50, 10)))
        ids = {e.candidate_id for e in f.entries}
        self.assertEqual(ids, {"b"})

    def test_tradeoff_coexistence(self):
        f = ParetoFrontier()
        f.offer(mkentry("hi_acc", 0.9, 200, 30))   # best accuracy
        f.offer(mkentry("cheap",  0.5,  20,  5))   # best cost
        f.offer(mkentry("mid",    0.7,  80, 12))   # non-dominated middle
        self.assertEqual(len(f), 3)

    def test_dominated_rejected(self):
        f = ParetoFrontier()
        f.offer(mkentry("a", 0.8, 50, 10))
        self.assertFalse(f.offer(mkentry("b", 0.7, 60, 15)))  # worse on all axes
        self.assertEqual(len(f), 1)

    def test_fuzz_non_dominated_filter(self):
        """Random candidates: result must contain no dominated pairs."""
        rng = random.Random(42)
        entries = [mkentry(f"c{i}", rng.random(), rng.uniform(10, 200), rng.uniform(1, 30))
                   for i in range(200)]
        kept = non_dominated_filter(entries)
        # Sanity: none in kept dominates another in kept.
        for i, a in enumerate(kept):
            for j, b in enumerate(kept):
                if i != j:
                    self.assertFalse(dominates(a.score, b.score))
        # Sanity: every dropped entry is dominated by at least one kept entry.
        kept_ids = {e.candidate_id for e in kept}
        dropped = [e for e in entries if e.candidate_id not in kept_ids]
        for d in dropped:
            self.assertTrue(any(dominates(k.score, d.score) for k in kept))

    def test_variance_gated_admission_rejects_unstable(self):
        """ParetoFrontier(max_accuracy_spread=...) rejects high-spread points."""
        f = ParetoFrontier(max_accuracy_spread=0.05)
        # Stable: spread=0.02 <= threshold → admitted.
        stable_score = ScoreVector(0.9, 100, 50, 20,
                                   n_repeats=3, accuracy_spread=0.02)
        self.assertTrue(f.offer(FrontierEntry("stable", stable_score, {})))
        # Unstable: spread=0.10 > threshold → rejected even though it would
        # otherwise be non-dominated (different accuracy).
        unstable_score = ScoreVector(0.95, 80, 40, 20,
                                     n_repeats=3, accuracy_spread=0.10)
        self.assertFalse(f.offer(FrontierEntry("unstable", unstable_score, {})))
        self.assertEqual(len(f), 1)
        self.assertEqual(f.entries[0].candidate_id, "stable")

    def test_variance_gate_off_by_default(self):
        """Without max_accuracy_spread set, high-spread candidates admitted."""
        f = ParetoFrontier()  # no spread cap
        unstable = ScoreVector(0.95, 80, 40, 20,
                               n_repeats=3, accuracy_spread=0.50)
        self.assertTrue(f.offer(FrontierEntry("unstable", unstable, {})))

    def test_dedup_exact_score_tuple(self):
        """Two entries with identical score tuples: only the first is kept."""
        f = ParetoFrontier()
        self.assertTrue(f.offer(mkentry("a", 0.8, 50, 10)))
        self.assertFalse(f.offer(mkentry("a_dup", 0.8, 50, 10)))
        self.assertEqual(len(f), 1)
        self.assertEqual(f.entries[0].candidate_id, "a")

    def test_knee_is_on_frontier(self):
        f = ParetoFrontier()
        f.offer(mkentry("a", 0.9, 200, 30))
        f.offer(mkentry("b", 0.8,  80, 15))
        f.offer(mkentry("c", 0.5,  20,  5))
        knee = f.knee()
        self.assertIsNotNone(knee)
        self.assertIn(knee.candidate_id, {e.candidate_id for e in f.entries})


if __name__ == "__main__":
    unittest.main()
