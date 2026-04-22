"""Bundled symptom task: dataset integrity + reward surface + mini end-to-end."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from collections import Counter

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
from meta_harness_plus.runner import SearchConfig, SearchRunner
from meta_harness_plus.scorer import Scorer
from meta_harness_plus.search.mock_proposer import AttributionGuidedMutationProposer
from meta_harness_plus.tasks import SYMPTOM_CLASSES, build_symptom_task, symptom_mock_llm


class TestDatasetIntegrity(unittest.TestCase):
    def setUp(self):
        self.task = build_symptom_task()

    def test_size(self):
        self.assertEqual(len(self.task.train), 30)
        self.assertEqual(len(self.task.eval_set), 20)

    def test_classes(self):
        self.assertEqual(tuple(self.task.classes), SYMPTOM_CLASSES)

    def test_class_balance(self):
        train_counts = Counter(e.label for e in self.task.train)
        eval_counts = Counter(e.label for e in self.task.eval_set)
        # 30 train / 5 classes = 6 per class; 20 eval / 5 = 4 per class.
        for klass in SYMPTOM_CLASSES:
            self.assertEqual(train_counts[klass], 6, f"train class {klass} imbalanced")
            self.assertEqual(eval_counts[klass], 4, f"eval class {klass} imbalanced")

    def test_no_duplicate_inputs(self):
        all_inputs = [e.input for e in self.task.train] + [e.input for e in self.task.eval_set]
        self.assertEqual(len(all_inputs), len(set(all_inputs)),
                         "duplicate input strings across train + eval set")

    def test_inputs_nonempty(self):
        for e in self.task.train + self.task.eval_set:
            self.assertGreater(len(e.input), 10, f"suspiciously short input: {e.input!r}")


class TestRewardSurface(unittest.TestCase):
    """Sanity-check that the mock LLM actually produces a gradient the
    framework can climb. Not a correctness test of the dataset — a
    calibration test of the mock."""

    def setUp(self):
        self.task = build_symptom_task()
        self.llm = symptom_mock_llm()
        self.scorer = Scorer(self.task)

    def _harness(self, retriever, fewshot, n_samples, voter):
        return Harness(components=[
            retriever, fewshot, SimpleFormatter(),
            MockLLMPredictor(llm_fn=self.llm, n_samples=n_samples), voter,
        ])

    def test_voting_lifts_baseline(self):
        baseline = self._harness(NullRetriever(), NullFewShot(), 1, NullVoter())
        voting = self._harness(
            BagOfWordsRetriever(corpus=self.task.train, k=3),
            TopKFewShot(k=2),
            5, MajorityVoter(),
        )
        s_b = self.scorer.score(baseline, self.task.eval_set)
        s_v = self.scorer.score(voting, self.task.eval_set)
        self.assertGreater(s_v.accuracy, s_b.accuracy,
                           msg=f"voting {s_v.accuracy} did not beat baseline {s_b.accuracy}")

    def test_baseline_above_chance(self):
        """Keyword signal should get the bare harness above 1/5 = 0.2."""
        baseline = self._harness(NullRetriever(), NullFewShot(), 1, NullVoter())
        s = self.scorer.score(baseline, self.task.eval_set)
        self.assertGreater(s.accuracy, 0.4,
                           msg=f"baseline {s.accuracy} suspiciously low for keyword-rich task")


class TestMiniEndToEnd(unittest.TestCase):
    """Search should improve over baseline on the symptom task with mock LLM."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_search_improves_frontier(self):
        task = build_symptom_task()
        llm = symptom_mock_llm()
        scorer = Scorer(task)
        baseline = Harness(components=[
            NullRetriever(), NullFewShot(), SimpleFormatter(),
            MockLLMPredictor(llm_fn=llm, n_samples=1), NullVoter(),
        ])
        baseline_score = scorer.score(baseline, task.eval_set)

        mutators = {
            "retriever": [lambda: NullRetriever(),
                          lambda: BagOfWordsRetriever(corpus=task.train, k=3)],
            "fewshot":   [lambda: NullFewShot(), lambda: TopKFewShot(k=2)],
            "voter":     [lambda: NullVoter(), lambda: MajorityVoter()],
            "predictor": [
                lambda: MockLLMPredictor(llm_fn=llm, n_samples=1),
                lambda: MockLLMPredictor(llm_fn=llm, n_samples=5),
            ],
        }
        attribution = AttributionTracker(scorer, baseline_for)
        # Bias toward depth-2+ mutations: escaping baseline's domination region
        # on this task requires simultaneously flipping predictor (n>1) AND
        # voter (MajorityVoter) — a single-slot mutation won't suffice.
        proposer = AttributionGuidedMutationProposer(
            mutators=mutators, seed=3, epsilon=0.3, temperature=0.5,
            depth_weights={1: 0.2, 2: 0.5, 3: 0.3},
        )
        runner = SearchRunner(
            task=task, scorer=scorer, proposer=proposer, attribution=attribution,
            config=SearchConfig(
                n_iterations=8, proposals_per_iter=8, screen_size=6,
                halving_k0=3, halving_eta=2, halving_final_keep=2,
                run_dir=self.tmp,
            ),
            seed_harnesses=[baseline],
        )
        state = runner.run()
        best = state.frontier.best_by_accuracy()
        self.assertIsNotNone(best)
        self.assertGreater(
            best.score.accuracy, baseline_score.accuracy,
            msg=f"search peak {best.score.accuracy} did not beat baseline {baseline_score.accuracy}"
        )


if __name__ == "__main__":
    unittest.main()
