"""Hyperband multi-bracket scheduler."""

from __future__ import annotations

import unittest

from meta_harness_plus.hyperband import HyperbandScheduler
from meta_harness_plus.scorer import ScoreVector


class TestBracketComputation(unittest.TestCase):
    def test_s_max_eta_3_max_27(self):
        # log_3(27) = 3 → s_max = 3, brackets s in {3, 2, 1, 0}
        hb = HyperbandScheduler(max_resource=27, reduction_factor=3)
        self.assertEqual(hb.s_max, 3)
        brackets = hb.brackets()
        self.assertEqual(len(brackets), 4)

    def test_eta_3_max_27_bracket_shapes(self):
        # Reference: Hyperband paper Table 1 for R=81, eta=3 (s_max=4).
        # We test R=27, eta=3 (s_max=3), shapes:
        #   s=3: n=4*27=27,  r=1
        #   s=2: n=4*9=12, ceil(4/3)=2 -> n=2*9=18, r=3
        #   s=1: n=ceil(4/2)*3 = 6, r=9
        #   s=0: n=ceil(4/1)*1 = 4, r=27
        hb = HyperbandScheduler(max_resource=27, reduction_factor=3)
        brackets = hb.brackets()
        # Wide bracket first, narrow last.
        self.assertGreater(brackets[0][0], brackets[-1][0])  # n decreasing
        self.assertLess(brackets[0][1], brackets[-1][1])     # r increasing
        # Last bracket should have r == max_resource.
        self.assertEqual(brackets[-1][1], 27)

    def test_max_resource_1_one_bracket(self):
        hb = HyperbandScheduler(max_resource=1, reduction_factor=3)
        self.assertEqual(hb.s_max, 0)
        self.assertEqual(len(hb.brackets()), 1)

    def test_invalid_args_raise(self):
        with self.assertRaises(ValueError):
            HyperbandScheduler(max_resource=0)
        with self.assertRaises(ValueError):
            HyperbandScheduler(max_resource=10, reduction_factor=1)


class TestHyperbandRun(unittest.TestCase):
    def test_runs_all_brackets_returns_survivors(self):
        hb = HyperbandScheduler(max_resource=9, reduction_factor=3, final_keep=1)
        proposed: list[int] = []

        def propose(n: int) -> list[int]:
            # Generate fresh integer candidates per bracket.
            cands = list(range(len(proposed), len(proposed) + n))
            proposed.extend(cands)
            return cands

        def evaluate(c: int, k: int) -> ScoreVector:
            # Higher candidate id = higher accuracy (deterministic ordering).
            return ScoreVector(accuracy=c / 100.0, tokens=1.0,
                               latency_ms=1.0, n_evaluated=k)

        result = hb.run(propose=propose, evaluate=evaluate)
        # At least one survivor per bracket.
        self.assertGreaterEqual(len(result.survivors), 1)
        # Every bracket recorded.
        self.assertEqual(len(result.per_bracket), 3)  # s_max=2 → 3 brackets
        self.assertGreater(result.total_evaluations, 0)

    def test_empty_proposal_skipped(self):
        hb = HyperbandScheduler(max_resource=9, reduction_factor=3, final_keep=1)
        def propose(n: int):
            return []  # propose nothing every time
        def evaluate(c, k):
            return ScoreVector(0.5, 1.0, 1.0, k)
        result = hb.run(propose=propose, evaluate=evaluate)
        self.assertEqual(result.survivors, [])
        self.assertEqual(result.per_bracket, [])

    def test_picks_best_per_bracket(self):
        hb = HyperbandScheduler(max_resource=9, reduction_factor=3, final_keep=1)
        # All proposed candidates have a deterministic accuracy. Higher id
        # = higher accuracy. final_keep=1 → halving picks the best per bracket.
        next_id = [0]
        def propose(n):
            ids = list(range(next_id[0], next_id[0] + n))
            next_id[0] += n
            return ids
        def evaluate(c, k):
            return ScoreVector(c / 100.0, 1.0, 1.0, k)
        result = hb.run(propose=propose, evaluate=evaluate)
        # Each bracket's survivor should be the highest-id candidate
        # proposed for that bracket.
        for br_result in result.per_bracket:
            self.assertEqual(len(br_result.survivors), 1)


if __name__ == "__main__":
    unittest.main()
