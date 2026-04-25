"""Early-stopping on hypervolume convergence + EWMA attribution.

Both are framework-level upgrades targeting "make MH better everywhere":
- Early stopping saves compute when frontier has plateaued.
- EWMA attribution gives the proposer recency-weighted signal — recent
  ablations matter more than ancient ones because the frontier shifts.
"""

from __future__ import annotations

import unittest

from meta_harness_plus.attribution import AttributionStats, AttributionTracker
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
from meta_harness_plus.runner import SearchConfig, SearchRunner
from meta_harness_plus.scorer import Scorer
from meta_harness_plus.search.mock_proposer import (
    AttributionGuidedMutationProposer,
)
from meta_harness_plus.tasks import build_toy_task, mock_llm


class TestAttributionEWMA(unittest.TestCase):
    def test_first_sample_initializes_ewma(self):
        s = AttributionStats(kind="x")
        s.update(0.5)
        self.assertAlmostEqual(s.ewma_delta, 0.5)
        self.assertAlmostEqual(s.mean_delta, 0.5)

    def test_ewma_weights_recent_higher(self):
        """EWMA(0.3): new = 0.3 * sample + 0.7 * prev. After [+1, +1, +1, -1],
        the ewma should be much closer to -1 than the mean (0.5)."""
        s = AttributionStats(kind="x", ewma_alpha=0.3)
        s.update(1.0)
        s.update(1.0)
        s.update(1.0)
        s.update(-1.0)
        # Mean: (1+1+1-1)/4 = 0.5
        self.assertAlmostEqual(s.mean_delta, 0.5)
        # EWMA: 1, 1, 1, 0.3*-1 + 0.7*1 = 0.4 — biased toward most recent.
        self.assertLess(s.ewma_delta, s.mean_delta)
        self.assertAlmostEqual(s.ewma_delta, 0.4, places=4)

    def test_ewma_alpha_threading(self):
        s_high = AttributionStats(kind="x", ewma_alpha=0.9)
        s_low = AttributionStats(kind="x", ewma_alpha=0.1)
        for v in [1.0, 1.0, 1.0, -1.0]:
            s_high.update(v)
            s_low.update(v)
        # alpha=0.9 → almost just the latest sample.
        self.assertLess(s_high.ewma_delta, 0.0)  # last was -1
        # alpha=0.1 → mostly old samples.
        self.assertGreater(s_low.ewma_delta, 0.5)


class TestAttributionTrackerEWMA(unittest.TestCase):
    def test_tracker_propagates_ewma(self):
        from meta_harness_plus.tasks import build_toy_task
        from meta_harness_plus.components import MockLLMPredictor, NullRetriever, NullFewShot, SimpleFormatter, NullVoter

        task = build_toy_task(seed=0)
        scorer = Scorer(task)
        tracker = AttributionTracker(scorer, baseline_for)

        h = Harness(components=[
            BagOfWordsRetriever(corpus=task.train, k=3), TopKFewShot(k=2),
            SimpleFormatter(), MockLLMPredictor(llm_fn=mock_llm(), n_samples=3),
            MajorityVoter(),
        ])
        # Run analyze multiple times — at least one kind should accumulate
        # non-zero ewma (some ablations legitimately produce 0 delta when
        # the harness's component is already the baseline).
        for i in range(3):
            tracker.analyze(f"c{i}", h, task.eval_set[:5])
        # Every kind has its update count incremented.
        for stats in tracker.stats.values():
            self.assertEqual(stats.n, 3, f"{stats.kind} n != 3")
        # At least one kind shows non-zero ewma signal.
        nonzero = [s for s in tracker.stats.values() if s.ewma_delta != 0.0]
        self.assertGreater(len(nonzero), 0,
                           "no ewma signal across any kind")


class TestEarlyStop(unittest.TestCase):
    def test_off_by_default(self):
        cfg = SearchConfig()
        self.assertEqual(cfg.early_stop_patience, 0)
        self.assertEqual(cfg.early_stop_min_hv_delta, 0.0)

    def test_runs_full_iterations_when_disabled(self):
        task = build_toy_task(seed=0)
        scorer = Scorer(task)
        attr = AttributionTracker(scorer, baseline_for)
        mutators = {
            "retriever": [lambda: NullRetriever(),
                          lambda: BagOfWordsRetriever(corpus=task.train, k=3)],
            "fewshot": [lambda: NullFewShot(), lambda: TopKFewShot(k=2)],
            "voter": [lambda: NullVoter(), lambda: MajorityVoter()],
            "predictor": [lambda: MockLLMPredictor(llm_fn=mock_llm(), n_samples=1)],
        }
        proposer = AttributionGuidedMutationProposer(
            mutators=mutators, seed=0, depth_weights={1: 0.5, 2: 0.5},
        )
        runner = SearchRunner(
            task=task, scorer=scorer, proposer=proposer, attribution=attr,
            config=SearchConfig(
                n_iterations=4, proposals_per_iter=3, screen_size=3,
                halving_k0=2, halving_eta=2, halving_final_keep=1,
                # Early stop OFF.
            ),
            seed_harnesses=[Harness(components=[
                NullRetriever(), NullFewShot(), SimpleFormatter(),
                MockLLMPredictor(llm_fn=mock_llm(), n_samples=1), NullVoter(),
            ])],
        )
        state = runner.run()
        # All 4 iterations recorded.
        self.assertEqual(len(state.hypervolume_per_iter), 4)

    def test_stops_early_on_hv_plateau(self):
        """When patience=2 and HV doesn't move, terminate before n_iterations."""
        task = build_toy_task(seed=0)
        scorer = Scorer(task)
        attr = AttributionTracker(scorer, baseline_for)
        mutators = {
            "retriever": [lambda: NullRetriever()],
            "fewshot": [lambda: NullFewShot()],
            "voter": [lambda: NullVoter()],
            "predictor": [lambda: MockLLMPredictor(llm_fn=mock_llm(), n_samples=1)],
        }
        # Pathological case: only one mutator option per kind, so every
        # proposal is identical. HV plateaus immediately.
        proposer = AttributionGuidedMutationProposer(
            mutators=mutators, seed=0, depth_weights={1: 1.0},
        )
        runner = SearchRunner(
            task=task, scorer=scorer, proposer=proposer, attribution=attr,
            config=SearchConfig(
                n_iterations=10,  # would normally run 10
                proposals_per_iter=2, screen_size=3,
                halving_k0=2, halving_eta=2, halving_final_keep=1,
                early_stop_patience=2,           # 2 flat iters → stop
                early_stop_min_hv_delta=0.0,
            ),
            seed_harnesses=[Harness(components=[
                NullRetriever(), NullFewShot(), SimpleFormatter(),
                MockLLMPredictor(llm_fn=mock_llm(), n_samples=1), NullVoter(),
            ])],
        )
        state = runner.run()
        # Stopped before all 10 — proves early stop fired.
        self.assertLess(len(state.hypervolume_per_iter), 10)
        # An "early_stop" event should be in history.
        self.assertTrue(any(h.get("phase") == "early_stop" for h in state.history),
                        f"no early_stop event in history: {state.history}")


if __name__ == "__main__":
    unittest.main()
