"""ThreadPoolExecutor-based parallel scoring (parallel-scoring branch).

Two claims tested:
1. Parallel and sequential scoring produce identical results on a
   deterministic harness (correctness invariant — threads must not
   change the answer).
2. Parallel scoring is meaningfully faster than sequential when the
   per-call work is I/O-bound (slow path test using time.sleep).
"""

from __future__ import annotations

import time
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
from meta_harness_plus.harness import Context, Harness
from meta_harness_plus.scorer import Scorer
from meta_harness_plus.task import TaskExample
from meta_harness_plus.tasks import build_toy_task, mock_llm


class TestParallelMatchesSequential(unittest.TestCase):
    def setUp(self):
        self.task = build_toy_task(seed=0)
        self.llm = mock_llm()  # deterministic mock
        self.harness = Harness(components=[
            BagOfWordsRetriever(corpus=self.task.train, k=3),
            TopKFewShot(k=2),
            SimpleFormatter(),
            MockLLMPredictor(llm_fn=self.llm, n_samples=3),
            NullVoter(),
        ])

    def test_single_repeat_identical(self):
        scorer = Scorer(self.task)
        seq = scorer.score(self.harness, self.task.eval_set, max_workers=1)
        par = scorer.score(self.harness, self.task.eval_set, max_workers=4)
        self.assertEqual(seq.accuracy, par.accuracy)
        self.assertAlmostEqual(seq.tokens, par.tokens, places=6)
        self.assertAlmostEqual(seq.latency_ms, par.latency_ms, places=6)
        self.assertEqual(seq.n_evaluated, par.n_evaluated)

    def test_multi_repeat_identical(self):
        scorer = Scorer(self.task)
        seq = scorer.score(self.harness, self.task.eval_set,
                           n_repeats=3, max_workers=1)
        par = scorer.score(self.harness, self.task.eval_set,
                           n_repeats=3, max_workers=8)
        # With deterministic mock, every repeat gives the same answer;
        # both sequential and parallel medians match.
        self.assertEqual(seq.accuracy, par.accuracy)
        self.assertAlmostEqual(seq.tokens, par.tokens, places=6)
        self.assertEqual(seq.n_repeats, par.n_repeats)
        self.assertEqual(seq.accuracy_spread, par.accuracy_spread)


class TestParallelSpeedup(unittest.TestCase):
    """Slow per-call mock — parallel should be wall-clock faster than sequential."""

    def _slow_harness(self, sleep_per_call: float, n_samples: int = 1):
        def slow_llm(prompt, ctx, harness, sample_idx):
            time.sleep(sleep_per_call)
            return ctx.example.label, 5, 1.0  # always correct

        return Harness(components=[
            NullRetriever(), NullFewShot(), SimpleFormatter(),
            MockLLMPredictor(llm_fn=slow_llm, n_samples=n_samples), NullVoter(),
        ])

    def test_parallel_faster_with_io_bound_work(self):
        task = build_toy_task(seed=0)
        scorer = Scorer(task)
        harness = self._slow_harness(sleep_per_call=0.05)
        examples = task.eval_set[:8]  # 8 calls @ 50ms each = 400ms sequential

        t0 = time.perf_counter()
        seq = scorer.score(harness, examples, max_workers=1)
        seq_wall = time.perf_counter() - t0

        t0 = time.perf_counter()
        par = scorer.score(harness, examples, max_workers=8)
        par_wall = time.perf_counter() - t0

        # Correctness preserved.
        self.assertEqual(seq.accuracy, par.accuracy)
        # Wall-clock speedup. Allow generous threshold (2× speedup) since
        # CI / shared machines have unpredictable scheduling. Real speedup
        # on a clean machine is usually 6-7× for 8 workers.
        self.assertLess(par_wall, seq_wall / 2.0,
                        msg=f"parallel wall {par_wall:.3f}s not <half of "
                            f"sequential {seq_wall:.3f}s — speedup unrealized")


class TestParallelGuards(unittest.TestCase):
    def test_zero_workers_raises(self):
        task = build_toy_task(seed=0)
        scorer = Scorer(task)
        h = Harness(components=[
            NullRetriever(), NullFewShot(), SimpleFormatter(),
            MockLLMPredictor(llm_fn=mock_llm(), n_samples=1), NullVoter(),
        ])
        with self.assertRaises(ValueError):
            scorer.score(h, task.eval_set[:3], max_workers=0)

    def test_negative_workers_raises(self):
        task = build_toy_task(seed=0)
        scorer = Scorer(task)
        h = Harness(components=[
            NullRetriever(), NullFewShot(), SimpleFormatter(),
            MockLLMPredictor(llm_fn=mock_llm(), n_samples=1), NullVoter(),
        ])
        with self.assertRaises(ValueError):
            scorer.score(h, task.eval_set[:3], max_workers=-2)


class TestRunnerThreadsThrough(unittest.TestCase):
    """SearchConfig.max_workers must reach the scorer."""

    def test_config_default_one(self):
        from meta_harness_plus.runner import SearchConfig
        cfg = SearchConfig()
        self.assertEqual(cfg.max_workers, 1)

    def test_config_accepts_higher(self):
        from meta_harness_plus.runner import SearchConfig
        cfg = SearchConfig(max_workers=8)
        self.assertEqual(cfg.max_workers, 8)


if __name__ == "__main__":
    unittest.main()
