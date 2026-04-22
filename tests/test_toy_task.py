"""Toy classification task determinism + reward-surface shape."""

from __future__ import annotations

import unittest

from meta_harness_plus.components import (
    BagOfWordsRetriever,
    MajorityVoter,
    MockLLMPredictor,
    NullFewShot,
    NullRetriever,
    NullVoter,
    SimpleFormatter,
    TopKFewShot,
)
from meta_harness_plus.harness import Harness
from meta_harness_plus.scorer import Scorer
from meta_harness_plus.tasks import build_toy_task, mock_llm


class TestToyTask(unittest.TestCase):
    def test_determinism(self):
        t1 = build_toy_task(seed=42)
        t2 = build_toy_task(seed=42)
        self.assertEqual([e.input for e in t1.eval_set], [e.input for e in t2.eval_set])
        self.assertEqual([e.label for e in t1.eval_set], [e.label for e in t2.eval_set])

    def test_different_seeds_differ(self):
        t1 = build_toy_task(seed=0)
        t2 = build_toy_task(seed=1)
        self.assertNotEqual([e.input for e in t1.eval_set], [e.input for e in t2.eval_set])

    def test_reward_surface_ordering(self):
        """Retrieval+few-shot+voting > few-shot alone > baseline."""
        task = build_toy_task(seed=0)
        scorer = Scorer(task)
        llm = mock_llm()

        def mk(retriever, fewshot, n_samples, voter):
            return Harness(components=[
                retriever, fewshot, SimpleFormatter(),
                MockLLMPredictor(llm_fn=llm, n_samples=n_samples), voter,
            ])

        baseline = mk(NullRetriever(), NullFewShot(), 1, NullVoter())
        rf = mk(BagOfWordsRetriever(corpus=task.train, k=3), TopKFewShot(k=2), 1, NullVoter())
        full = mk(BagOfWordsRetriever(corpus=task.train, k=3), TopKFewShot(k=2), 5, MajorityVoter())

        s_b = scorer.score(baseline, task.eval_set)
        s_rf = scorer.score(rf, task.eval_set)
        s_full = scorer.score(full, task.eval_set)

        # Strict ordering is what the framework must be able to rediscover.
        self.assertGreater(s_rf.accuracy, s_b.accuracy)
        self.assertGreater(s_full.accuracy, s_rf.accuracy)
        # Cost ordering is the flip side — the good harness costs more.
        self.assertGreater(s_full.tokens, s_b.tokens)
        self.assertGreater(s_full.latency_ms, s_b.latency_ms)


if __name__ == "__main__":
    unittest.main()
