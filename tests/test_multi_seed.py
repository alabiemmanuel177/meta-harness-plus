"""Multi-seed runner + held-out test split (Tier 1.2)."""

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
from meta_harness_plus.multi_seed import MultiSeedRunner, split_holdout
from meta_harness_plus.runner import SearchConfig
from meta_harness_plus.scorer import Scorer
from meta_harness_plus.search.mock_proposer import (
    AttributionGuidedMutationProposer,
)
from meta_harness_plus.tasks import build_toy_task, mock_llm


class TestSplitHoldout(unittest.TestCase):
    def test_size_split(self):
        task = build_toy_task(seed=0)
        original = len(task.eval_set)
        search_task, holdout = split_holdout(task, holdout_frac=0.3, seed=0)
        # Stratified 30% split per class — total should be ~30% but
        # rounded up per class so exact total depends on class counts.
        self.assertEqual(len(search_task.eval_set) + len(holdout), original)
        self.assertGreater(len(holdout), 0)
        self.assertGreater(len(search_task.eval_set), 0)

    def test_no_overlap(self):
        task = build_toy_task(seed=0)
        search_task, holdout = split_holdout(task, holdout_frac=0.3, seed=0)
        search_inputs = {e.input for e in search_task.eval_set}
        for ho in holdout:
            self.assertNotIn(ho.input, search_inputs)

    def test_train_unchanged(self):
        task = build_toy_task(seed=0)
        search_task, _ = split_holdout(task, holdout_frac=0.3, seed=0)
        self.assertEqual(search_task.train, task.train)

    def test_stratified_balance(self):
        """Each class appears in both partitions when class has >= 2 items."""
        task = build_toy_task(seed=0)
        search_task, holdout = split_holdout(task, holdout_frac=0.3, seed=0)
        from collections import Counter
        search_classes = set(Counter(e.label for e in search_task.eval_set))
        holdout_classes = set(Counter(e.label for e in holdout))
        # Classes with >= 2 items should be in both — toy task has many.
        self.assertEqual(search_classes, holdout_classes,
                         f"stratification missed class: search={search_classes}, "
                         f"holdout={holdout_classes}")

    def test_invalid_frac_raises(self):
        task = build_toy_task(seed=0)
        with self.assertRaises(ValueError):
            split_holdout(task, holdout_frac=0.0)
        with self.assertRaises(ValueError):
            split_holdout(task, holdout_frac=1.0)


class TestMultiSeedRunner(unittest.TestCase):
    def setUp(self):
        self.task = build_toy_task(seed=0)
        self.llm = mock_llm()
        self.mutators = {
            "retriever": [
                lambda: NullRetriever(),
                lambda: BagOfWordsRetriever(corpus=self.task.train, k=3),
            ],
            "fewshot": [lambda: NullFewShot(), lambda: TopKFewShot(k=2)],
            "voter": [lambda: NullVoter(), lambda: MajorityVoter()],
            "predictor": [
                lambda: MockLLMPredictor(llm_fn=self.llm, n_samples=1),
                lambda: MockLLMPredictor(llm_fn=self.llm, n_samples=5),
            ],
        }

    def _seed_harness(self) -> Harness:
        return Harness(components=[
            NullRetriever(), NullFewShot(), SimpleFormatter(),
            MockLLMPredictor(llm_fn=self.llm, n_samples=1), NullVoter(),
        ])

    def _proposer_factory(self, seed: int):
        return AttributionGuidedMutationProposer(
            mutators=self.mutators, seed=seed,
            depth_weights={1: 0.3, 2: 0.5, 3: 0.2},
        )

    def test_runs_one_seed(self):
        runner = MultiSeedRunner(
            task=self.task,
            proposer_factory=self._proposer_factory,
            config=SearchConfig(
                n_iterations=2, proposals_per_iter=4, screen_size=4,
                halving_k0=2, halving_eta=2, halving_final_keep=1,
            ),
            seed_harnesses=[self._seed_harness()],
            holdout_frac=0.3, holdout_seed=0,
            n_eval_repeats_holdout=1,
        )
        result = runner.run([0])
        self.assertEqual(len(result.per_seed), 1)
        self.assertEqual(result.per_seed[0].seed, 0)
        # Held-out scoring should produce one score per seed.
        self.assertEqual(len(result.held_out_top_acc_per_seed), 1)

    def test_runs_multiple_seeds(self):
        runner = MultiSeedRunner(
            task=self.task,
            proposer_factory=self._proposer_factory,
            config=SearchConfig(
                n_iterations=2, proposals_per_iter=4, screen_size=4,
                halving_k0=2, halving_eta=2, halving_final_keep=1,
            ),
            seed_harnesses=[self._seed_harness()],
            holdout_frac=0.3, holdout_seed=0,
            n_eval_repeats_holdout=1,
        )
        result = runner.run([0, 1, 2])
        self.assertEqual(len(result.per_seed), 3)
        self.assertEqual([r.seed for r in result.per_seed], [0, 1, 2])
        self.assertEqual(len(result.held_out_top_acc_per_seed), 3)
        # Bootstrap CI populated.
        self.assertGreater(result.best_acc_ci.confidence, 0.0)

    def test_summary_renders(self):
        runner = MultiSeedRunner(
            task=self.task,
            proposer_factory=self._proposer_factory,
            config=SearchConfig(
                n_iterations=1, proposals_per_iter=3, screen_size=3,
                halving_k0=2, halving_eta=2, halving_final_keep=1,
            ),
            seed_harnesses=[self._seed_harness()],
            holdout_frac=0.3, holdout_seed=0,
            n_eval_repeats_holdout=1,
        )
        result = runner.run([0, 1])
        s = result.summary()
        self.assertIn("seeds=", s)
        self.assertIn("CI", s)


if __name__ == "__main__":
    unittest.main()
