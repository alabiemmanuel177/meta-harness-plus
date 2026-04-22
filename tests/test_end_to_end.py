"""End-to-end: the search must improve over a baseline-only frontier on the toy task."""

from __future__ import annotations

import shutil
import tempfile
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
from meta_harness_plus.pareto import dominates
from meta_harness_plus.runner import SearchConfig, SearchRunner
from meta_harness_plus.scorer import Scorer
from meta_harness_plus.search.mock_proposer import AttributionGuidedMutationProposer
from meta_harness_plus.tasks import build_toy_task, mock_llm


def _seed_harness(task, llm) -> Harness:
    """A deliberately weak starting point so the search has room to improve."""
    return Harness(components=[
        NullRetriever(),
        NullFewShot(),
        SimpleFormatter(),
        MockLLMPredictor(llm_fn=llm, n_samples=1),
        NullVoter(),
    ])


def _mutators(task, llm):
    """Candidate mutations per kind. The proposer picks one to swap in."""
    return {
        "retriever": [
            lambda: NullRetriever(),
            lambda: BagOfWordsRetriever(corpus=task.train, k=2),
            lambda: BagOfWordsRetriever(corpus=task.train, k=3),
            lambda: BagOfWordsRetriever(corpus=task.train, k=5),
        ],
        "fewshot": [
            lambda: NullFewShot(),
            lambda: TopKFewShot(k=1),
            lambda: TopKFewShot(k=2),
            lambda: TopKFewShot(k=3),
        ],
        "voter": [
            lambda: NullVoter(),
            lambda: MajorityVoter(),
        ],
        "predictor": [
            lambda: MockLLMPredictor(llm_fn=llm, n_samples=1),
            lambda: MockLLMPredictor(llm_fn=llm, n_samples=3),
            lambda: MockLLMPredictor(llm_fn=llm, n_samples=5),
            lambda: MockLLMPredictor(llm_fn=llm, n_samples=7),
        ],
    }


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_search_improves_over_baseline(self):
        task = build_toy_task(seed=0)
        llm = mock_llm()
        scorer = Scorer(task)
        baseline_score = scorer.score(_seed_harness(task, llm), task.eval_set)

        attribution = AttributionTracker(scorer, baseline_for)
        proposer = AttributionGuidedMutationProposer(
            mutators=_mutators(task, llm), seed=0, epsilon=0.3, temperature=0.5,
        )
        runner = SearchRunner(
            task=task,
            scorer=scorer,
            proposer=proposer,
            attribution=attribution,
            config=SearchConfig(
                n_iterations=10,
                proposals_per_iter=8,
                screen_size=6,
                halving_k0=3,
                halving_eta=2,
                halving_final_keep=2,
                run_dir=self.tmp,
            ),
            seed_harnesses=[_seed_harness(task, llm)],
        )
        state = runner.run()

        # The frontier must contain at least one entry strictly dominating the baseline.
        self.assertGreater(len(state.frontier), 0)
        best_acc = state.frontier.best_by_accuracy().score.accuracy
        self.assertGreater(
            best_acc,
            baseline_score.accuracy,
            msg=f"search did not improve over baseline {baseline_score.accuracy:.2f}",
        )

        # Frontier must itself be valid (no internal domination).
        entries = list(state.frontier.entries)
        for i, a in enumerate(entries):
            for j, b in enumerate(entries):
                if i != j:
                    self.assertFalse(dominates(a.score, b.score))

        # Run logger must have materialized disk state.
        import os
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "frontier.json")))
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "attribution_stats.json")))
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "history.jsonl")))

    def test_attribution_guides_mutation_correctly(self):
        """After a few iterations, high-accuracy components accrue positive attribution."""
        task = build_toy_task(seed=1)
        llm = mock_llm()
        scorer = Scorer(task)
        attribution = AttributionTracker(scorer, baseline_for)
        proposer = AttributionGuidedMutationProposer(
            mutators=_mutators(task, llm), seed=1, epsilon=0.2, temperature=0.5,
        )
        runner = SearchRunner(
            task=task, scorer=scorer, proposer=proposer, attribution=attribution,
            config=SearchConfig(n_iterations=5, proposals_per_iter=5, screen_size=6,
                                halving_k0=3, halving_eta=2, halving_final_keep=2),
            seed_harnesses=[_seed_harness(task, llm)],
        )
        runner.run()

        # fewshot is the biggest accuracy lever on the toy task — after enough
        # ablations, its mean attributed value should be > 0 (not necessarily
        # the max, but materially positive).
        fewshot_stats = attribution.stats.get("fewshot")
        self.assertIsNotNone(fewshot_stats, "attribution should have seen fewshot")
        self.assertGreater(
            fewshot_stats.mean_delta, 0.0,
            msg=f"fewshot mean_delta={fewshot_stats.mean_delta:.3f} should be positive",
        )


if __name__ == "__main__":
    unittest.main()
