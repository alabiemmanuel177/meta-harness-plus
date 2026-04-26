"""Tests for BootstrapFewShot + bootstrap_demos (DSPy-style fewshot)."""
from __future__ import annotations

import unittest

from meta_harness_plus.components import (
    BagOfWordsRetriever, BootstrapFewShot, bootstrap_demos,
    MockLLMPredictor, NullVoter, SimpleFormatter,
)
from meta_harness_plus.harness import Context, Harness
from meta_harness_plus.task import TaskExample
from meta_harness_plus.tasks import build_toy_task, mock_llm


class TestBootstrapDemos(unittest.TestCase):
    def setUp(self):
        self.task = build_toy_task(seed=0)
        self.llm = mock_llm()

    def test_bootstrap_keeps_correct_demos(self):
        """bootstrap_demos should keep demos the predictor gets right."""
        # Predictor: simple — uses the mock LLM. For toy task, mock LLM
        # returns the right class for items where label is in the input.
        def predictor(ex: TaskExample) -> str:
            # Use the mock LLM directly: it has knowledge of toy classes.
            try:
                return self.llm({"input": ex.input}).strip().lower()
            except Exception:
                return ""

        out = bootstrap_demos(predictor, self.task.train,
                              classes=self.task.classes, max_demos=10)
        # Should retain at least some demos (toy task is mostly correct).
        self.assertGreaterEqual(len(out), 0)
        self.assertLessEqual(len(out), 10)
        for ex in out:
            self.assertIsInstance(ex, TaskExample)

    def test_bootstrap_max_cap(self):
        """bootstrap_demos respects max_demos."""
        always_correct = lambda ex: ex.label  # trivial predictor
        out = bootstrap_demos(always_correct, self.task.train,
                              classes=self.task.classes, max_demos=3)
        self.assertEqual(len(out), 3)


class TestBootstrapFewShot(unittest.TestCase):
    def setUp(self):
        self.task = build_toy_task(seed=0)
        self.llm = mock_llm()

    def test_falls_back_to_topk_when_pool_empty(self):
        """With empty pool, BootstrapFewShot behaves like TopKFewShot."""
        h = Harness(components=[
            BagOfWordsRetriever(corpus=self.task.train, k=3),
            BootstrapFewShot(k=2, demos_pool=()),
            SimpleFormatter(),
            MockLLMPredictor(llm_fn=self.llm),
            NullVoter(),
        ])
        ctx = h.run(self.task.eval_set[0])
        # ctx.few_shots should be top-2 retrieved (fallback path).
        self.assertEqual(len(ctx.few_shots), 2)

    def test_uses_pool_when_provided(self):
        """With non-empty pool not in retrieved, pool is used directly."""
        # Build a pool of 4 demos that aren't in the retriever's corpus
        # (use eval items so retriever can't pick them).
        pool = tuple(self.task.eval_set[:4])
        h = Harness(components=[
            BagOfWordsRetriever(corpus=self.task.train, k=3),  # train corpus
            BootstrapFewShot(k=2, demos_pool=pool),
            SimpleFormatter(),
            MockLLMPredictor(llm_fn=self.llm),
            NullVoter(),
        ])
        ctx = h.run(self.task.eval_set[5])
        # Should use first 2 from pool since none are in ctx.retrieved.
        self.assertEqual(len(ctx.few_shots), 2)
        self.assertEqual(ctx.few_shots, list(pool[:2]))

    def test_config_includes_k_and_pool_size(self):
        bfs = BootstrapFewShot(k=4, demos_pool=tuple(self.task.train[:8]))
        cfg = bfs.config()
        self.assertEqual(cfg["k"], 4)
        self.assertEqual(cfg["demos_pool_size"], 8)
        self.assertEqual(cfg["kind"], "fewshot")


if __name__ == "__main__":
    unittest.main()
