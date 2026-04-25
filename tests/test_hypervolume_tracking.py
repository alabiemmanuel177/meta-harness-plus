"""Hypervolume per-iteration tracking in SearchRunner (Tier 4.2)."""

from __future__ import annotations

import unittest

from meta_harness_plus.attribution import AttributionTracker
from meta_harness_plus.baselines import bare_baseline
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
from meta_harness_plus.runner import SearchConfig, SearchRunner
from meta_harness_plus.scorer import Scorer
from meta_harness_plus.search.mock_proposer import (
    AttributionGuidedMutationProposer,
)
from meta_harness_plus.tasks import build_toy_task, mock_llm


class TestHypervolumeTracking(unittest.TestCase):
    def test_runner_records_hv_per_iteration(self):
        task = build_toy_task(seed=0)
        llm = mock_llm()
        scorer = Scorer(task)
        attr = AttributionTracker(scorer, baseline_for)
        mutators = {
            "retriever": [
                lambda: NullRetriever(),
                lambda: BagOfWordsRetriever(corpus=task.train, k=3),
            ],
            "fewshot": [lambda: NullFewShot(), lambda: TopKFewShot(k=2)],
            "voter": [lambda: NullVoter(), lambda: MajorityVoter()],
            "predictor": [
                lambda: MockLLMPredictor(llm_fn=llm, n_samples=1),
                lambda: MockLLMPredictor(llm_fn=llm, n_samples=5),
            ],
        }
        proposer = AttributionGuidedMutationProposer(
            mutators=mutators, seed=0,
            depth_weights={1: 0.4, 2: 0.4, 3: 0.2},
        )
        runner = SearchRunner(
            task=task, scorer=scorer, proposer=proposer, attribution=attr,
            config=SearchConfig(
                n_iterations=3, proposals_per_iter=4, screen_size=4,
                halving_k0=2, halving_eta=2, halving_final_keep=2,
            ),
            seed_harnesses=[bare_baseline(
                task, MockLLMPredictor(llm_fn=llm, n_samples=1),
            )],
        )
        state = runner.run()
        # 3 iterations -> 3 HV samples.
        self.assertEqual(len(state.hypervolume_per_iter), 3)
        # All HV values must be non-negative.
        for hv in state.hypervolume_per_iter:
            self.assertGreaterEqual(hv, 0.0)

    def test_hv_does_not_decrease_across_iterations(self):
        """Pareto frontier only ever grows or evicts dominated points;
        adding non-dominated points monotonically increases HV. So the
        per-iteration HV series should be monotone non-decreasing."""
        task = build_toy_task(seed=0)
        llm = mock_llm()
        scorer = Scorer(task)
        attr = AttributionTracker(scorer, baseline_for)
        mutators = {
            "retriever": [lambda: NullRetriever(),
                          lambda: BagOfWordsRetriever(corpus=task.train, k=3)],
            "fewshot": [lambda: NullFewShot(), lambda: TopKFewShot(k=2)],
            "voter": [lambda: NullVoter(), lambda: MajorityVoter()],
            "predictor": [lambda: MockLLMPredictor(llm_fn=llm, n_samples=1),
                          lambda: MockLLMPredictor(llm_fn=llm, n_samples=3)],
        }
        proposer = AttributionGuidedMutationProposer(
            mutators=mutators, seed=2,
            depth_weights={1: 0.3, 2: 0.5, 3: 0.2},
        )
        runner = SearchRunner(
            task=task, scorer=scorer, proposer=proposer, attribution=attr,
            config=SearchConfig(
                n_iterations=4, proposals_per_iter=4, screen_size=4,
                halving_k0=2, halving_eta=2, halving_final_keep=2,
            ),
            seed_harnesses=[bare_baseline(
                task, MockLLMPredictor(llm_fn=llm, n_samples=1),
            )],
        )
        state = runner.run()
        prev = -1.0
        for hv in state.hypervolume_per_iter:
            self.assertGreaterEqual(
                hv, prev,
                msg=f"HV regressed: {state.hypervolume_per_iter}",
            )
            prev = hv


if __name__ == "__main__":
    unittest.main()
