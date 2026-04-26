"""Tests for SynergyTracker (drop-pair ablation)."""
from __future__ import annotations

import unittest

from meta_harness_plus.components import (
    BagOfWordsRetriever, MajorityVoter, MockLLMPredictor,
    NullFewShot, NullRetriever, NullVoter, SimpleFormatter, TopKFewShot,
    baseline_for,
)
from meta_harness_plus.harness import Harness
from meta_harness_plus.scorer import Scorer
from meta_harness_plus.synergy import SynergyTracker
from meta_harness_plus.tasks import build_toy_task, mock_llm


class TestSynergyTracker(unittest.TestCase):
    def setUp(self):
        self.task = build_toy_task(seed=0)
        self.scorer = Scorer(self.task)
        self.llm = mock_llm()

    def test_analyze_returns_one_per_pair(self):
        h = Harness(components=[
            BagOfWordsRetriever(corpus=self.task.train, k=2),
            TopKFewShot(k=1),
            SimpleFormatter(),
            MockLLMPredictor(llm_fn=self.llm),
            NullVoter(),
        ])
        tracker = SynergyTracker(self.scorer, baseline_for)
        snaps = tracker.analyze(
            candidate_id="cand_001",
            harness=h,
            examples=list(self.task.eval_set),
            n_repeats=1,
        )
        # Kinds with baselines: retriever, fewshot, voter (formatter has no
        # null baseline, predictor too). With 3 ablate-able kinds, expect
        # C(3,2) = 3 pairs.
        # baseline_for has baselines for retriever, fewshot, voter, formatter
        # → 4 ablate-able kinds, C(4,2) = 6 pairs.
        self.assertEqual(len(snaps), 6)
        # Pairs should be sorted lexicographically.
        for s in snaps:
            self.assertLess(s.kind_a, s.kind_b)
        # Stats updated — one per pair.
        self.assertEqual(len(tracker.stats), 6)

    def test_synergy_delta_signed_correctly(self):
        h = Harness(components=[
            BagOfWordsRetriever(corpus=self.task.train, k=2),
            TopKFewShot(k=1),
            SimpleFormatter(),
            MockLLMPredictor(llm_fn=self.llm),
            NullVoter(),
        ])
        tracker = SynergyTracker(self.scorer, baseline_for)
        snaps = tracker.analyze(
            candidate_id="cand_001",
            harness=h,
            examples=list(self.task.eval_set),
            n_repeats=1,
        )
        # Definition: synergy = s_full - s_neither - (s_drop_b-s_neither + s_drop_a-s_neither)
        # = s_full + s_neither - s_drop_a - s_drop_b
        for s in snaps:
            recomputed = s.s_full + s.s_neither - s.s_drop_a - s.s_drop_b
            self.assertAlmostEqual(s.synergy_delta, recomputed, places=6)

    def test_ranking_returns_descending_synergy(self):
        h = Harness(components=[
            BagOfWordsRetriever(corpus=self.task.train, k=2),
            TopKFewShot(k=1),
            SimpleFormatter(),
            MockLLMPredictor(llm_fn=self.llm),
            NullVoter(),
        ])
        tracker = SynergyTracker(self.scorer, baseline_for)
        tracker.analyze(
            candidate_id="cand_001",
            harness=h,
            examples=list(self.task.eval_set),
            n_repeats=1,
        )
        ranked = tracker.ranking()
        self.assertEqual(len(ranked), 6)
        for i in range(1, len(ranked)):
            self.assertGreaterEqual(ranked[i - 1].mean_synergy,
                                    ranked[i].mean_synergy)


if __name__ == "__main__":
    unittest.main()
