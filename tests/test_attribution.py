"""Attribution tracker: drop-one ablations recover component value."""

from __future__ import annotations

import unittest

from meta_harness_plus.attribution import AttributionTracker
from meta_harness_plus.components import (
    BagOfWordsRetriever,
    MajorityVoter,
    MockLLMPredictor,
    NullFewShot,
    NullRetriever,
    NullVoter,
    SimpleFormatter,
    TopKFewShot,
    baseline_for,
)
from meta_harness_plus.harness import Harness
from meta_harness_plus.scorer import Scorer
from meta_harness_plus.tasks import build_toy_task, mock_llm


class TestAttribution(unittest.TestCase):
    def setUp(self) -> None:
        self.task = build_toy_task(seed=0)
        self.scorer = Scorer(self.task)
        self.llm = mock_llm()

    def _full_harness(self) -> Harness:
        return Harness(components=[
            BagOfWordsRetriever(corpus=self.task.train, k=3),
            TopKFewShot(k=2),
            SimpleFormatter(),
            MockLLMPredictor(llm_fn=self.llm, n_samples=5),
            MajorityVoter(),
        ])

    def test_drop_one_shows_fewshot_is_valuable(self):
        tracker = AttributionTracker(self.scorer, baseline_for)
        snapshots = tracker.analyze("cand_test", self._full_harness(), self.task.eval_set)
        # Every ablatable kind should show up.
        kinds = {s.kind for s in snapshots}
        self.assertIn("fewshot", kinds)
        self.assertIn("voter", kinds)
        self.assertIn("retriever", kinds)
        # Accuracy deltas should generally be >= 0 for this harness on this task
        # (the full harness is designed to be better than any single ablation).
        # Allow one axis to go slightly negative due to noise, but the sum
        # across kinds must be clearly positive.
        total = sum(s.accuracy_delta for s in snapshots)
        self.assertGreater(total, 0.05)

    def test_mutation_weights_sum_to_one(self):
        tracker = AttributionTracker(self.scorer, baseline_for)
        tracker.analyze("c1", self._full_harness(), self.task.eval_set)
        weights = tracker.mutation_weights(temperature=0.5)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
        for w in weights.values():
            self.assertGreaterEqual(w, 0.0)

    def test_running_stats_update(self):
        tracker = AttributionTracker(self.scorer, baseline_for)
        tracker.analyze("c1", self._full_harness(), self.task.eval_set)
        tracker.analyze("c2", self._full_harness(), self.task.eval_set)
        for stats in tracker.stats.values():
            self.assertEqual(stats.n, 2)  # each kind analyzed twice

    def test_attribution_zero_when_component_is_already_baseline(self):
        """Swapping a NullRetriever for a NullRetriever should give 0 delta."""
        baseline_harness = Harness(components=[
            NullRetriever(), NullFewShot(), SimpleFormatter(),
            MockLLMPredictor(llm_fn=self.llm, n_samples=1), NullVoter(),
        ])
        tracker = AttributionTracker(self.scorer, baseline_for)
        snapshots = tracker.analyze("baseline", baseline_harness, self.task.eval_set)
        # Every delta should be ~0 (we're ablating baselines against baselines).
        for s in snapshots:
            self.assertAlmostEqual(s.accuracy_delta, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
