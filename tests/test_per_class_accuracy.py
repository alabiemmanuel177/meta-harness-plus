"""Per-class accuracy tracking + LLMProposer prompt integration."""

from __future__ import annotations

import unittest

from meta_harness_plus.components import (
    BagOfWordsRetriever,
    MockLLMPredictor,
    NullFewShot,
    NullRetriever,
    NullVoter,
    SimpleFormatter,
    TopKFewShot,
)
from meta_harness_plus.harness import Harness
from meta_harness_plus.llm.proposer import DiagnosticContext
from meta_harness_plus.scorer import Scorer
from meta_harness_plus.tasks import build_toy_task, mock_llm


class TestScorerPerClass(unittest.TestCase):
    def test_per_class_populated(self):
        task = build_toy_task(seed=0)
        llm = mock_llm()
        h = Harness(components=[
            NullRetriever(), NullFewShot(), SimpleFormatter(),
            MockLLMPredictor(llm_fn=llm, n_samples=1), NullVoter(),
        ])
        s = Scorer(task).score(h, task.eval_set)
        # At least one class represented in eval (toy task generation
        # may not sample every class into every eval set).
        self.assertGreater(len(s.per_class_accuracy), 0)
        # All values are accuracies in [0, 1].
        for klass, acc in s.per_class_accuracy:
            self.assertGreaterEqual(acc, 0.0)
            self.assertLessEqual(acc, 1.0)

    def test_per_class_dict_helper(self):
        task = build_toy_task(seed=0)
        llm = mock_llm()
        h = Harness(components=[
            NullRetriever(), NullFewShot(), SimpleFormatter(),
            MockLLMPredictor(llm_fn=llm, n_samples=1), NullVoter(),
        ])
        s = Scorer(task).score(h, task.eval_set)
        d = s.per_class_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(set(d), {k for k, _ in s.per_class_accuracy})

    def test_per_class_aggregates_across_repeats(self):
        task = build_toy_task(seed=0)
        llm = mock_llm()
        h = Harness(components=[
            BagOfWordsRetriever(corpus=task.train, k=3), TopKFewShot(k=2),
            SimpleFormatter(),
            MockLLMPredictor(llm_fn=llm, n_samples=3), NullVoter(),
        ])
        s = Scorer(task).score(h, task.eval_set, n_repeats=3)
        # Aggregated counts mean each class's denominator is n × n_repeats.
        # All accuracies still in [0, 1].
        for _, acc in s.per_class_accuracy:
            self.assertGreaterEqual(acc, 0.0)
            self.assertLessEqual(acc, 1.0)
        # n_repeats field carried.
        self.assertEqual(s.n_repeats, 3)

    def test_empty_examples_safe(self):
        task = build_toy_task(seed=0)
        llm = mock_llm()
        h = Harness(components=[
            NullRetriever(), NullFewShot(), SimpleFormatter(),
            MockLLMPredictor(llm_fn=llm, n_samples=1), NullVoter(),
        ])
        s = Scorer(task).score(h, [])
        # Empty examples → empty per_class.
        self.assertEqual(s.per_class_accuracy, ())


class TestProposerPromptPerClass(unittest.TestCase):
    """The LLMProposer prompt includes per-class breakdown of the best
    accuracy frontier point — directly actionable failure-mode signal."""

    def test_per_class_section_present(self):
        ctx = DiagnosticContext(
            frontier=[{
                "candidate_id": "best",
                "score": {
                    "accuracy": 0.85, "tokens": 200, "latency_ms": 100,
                    "per_class_accuracy": [
                        ("cardiology", 0.9),
                        ("dermatology", 1.0),
                        ("orthopedics", 0.6),  # weak class
                        ("neurology", 0.8),
                        ("gastroenterology", 0.95),
                    ],
                },
                "describe": [],
            }],
            attribution={},
            available=[],
            exploration_gap=[],
            iteration=2,
            n_requested=4,
        )
        prompt = ctx.to_user_prompt()
        self.assertIn("Per-class accuracy", prompt)
        self.assertIn("orthopedics", prompt)
        self.assertIn("0.60", prompt)  # the weakest class's accuracy
        # Weakest class should appear before strongest in the sorted list.
        weak_pos = prompt.find("orthopedics")
        strong_pos = prompt.find("dermatology")
        self.assertLess(weak_pos, strong_pos,
                        "weakest class should be listed first")

    def test_section_omitted_when_per_class_empty(self):
        ctx = DiagnosticContext(
            frontier=[{
                "candidate_id": "best",
                "score": {"accuracy": 0.85, "tokens": 200, "latency_ms": 100},
                "describe": [],
            }],
            attribution={}, available=[], exploration_gap=[],
            iteration=0, n_requested=4,
        )
        prompt = ctx.to_user_prompt()
        # No per-class data → no "Per-class accuracy" header
        self.assertNotIn("Per-class accuracy of best", prompt)


if __name__ == "__main__":
    unittest.main()
