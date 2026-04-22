"""Tests for the reproducibility branch contribution: n_repeats support.

We model Ollama-style temperature=0 nondeterminism with a mock LLM that,
despite being "deterministic in spirit," flips its answer with some
probability per call. Single-shot evaluation bounces; median-over-repeats
should stabilize.
"""

from __future__ import annotations

import hashlib
import random
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
from meta_harness_plus.harness import Context, Harness
from meta_harness_plus.scorer import Scorer, ScoreVector, _median
from meta_harness_plus.task import Task, TaskExample
from meta_harness_plus.tasks import build_toy_task


def noisy_llm(flip_prob: float = 0.3, seed: int = 0):
    """Mock LLM that flips its prediction with ``flip_prob`` each call.

    Critically, the noise varies across invocations — we mix a global call
    counter into the hash so the same (prompt, sample_idx) pair can produce
    different predictions on different evaluations. This simulates Ollama's
    GPU-batching nondeterminism at temperature=0.
    """
    counter = {"n": 0}
    rng_seed = seed

    def llm_fn(prompt: str, ctx: Context, harness: Harness, sample_idx: int):
        counter["n"] += 1
        # True class signal from the query's keywords.
        from meta_harness_plus.tasks.toy_classification import _class_signal, CLASSES
        sig = _class_signal(ctx.example.input)
        best = max(sig, key=lambda k: sig[k])
        # Noise: flip to a different class with probability flip_prob.
        h = hashlib.sha1(f"{prompt}|{sample_idx}|{counter['n']}|{rng_seed}".encode()).hexdigest()
        v = int(h[:8], 16) / 0xFFFFFFFF
        if v < flip_prob:
            others = [c for c in CLASSES if c != best]
            idx = int(h[8:12], 16) % len(others)
            pred = others[idx]
        else:
            pred = best
        return pred, 5, 3.0
    return llm_fn


class TestMedianHelper(unittest.TestCase):
    def test_odd_count(self):
        self.assertEqual(_median([0.3, 0.5, 0.7]), 0.5)

    def test_even_count_averages_middle(self):
        self.assertAlmostEqual(_median([0.2, 0.4, 0.6, 0.8]), 0.5)

    def test_handles_out_of_order(self):
        self.assertEqual(_median([0.9, 0.1, 0.5]), 0.5)

    def test_empty(self):
        self.assertEqual(_median([]), 0.0)


class TestSingleRepeatUnchanged(unittest.TestCase):
    """n_repeats=1 must be byte-compatible with the pre-branch Scorer."""

    def test_single_shot_sets_metadata(self):
        task = build_toy_task(seed=0)
        scorer = Scorer(task)
        h = Harness(components=[
            NullRetriever(), NullFewShot(), SimpleFormatter(),
            MockLLMPredictor(llm_fn=noisy_llm(flip_prob=0.0), n_samples=1),
            NullVoter(),
        ])
        s = scorer.score(h, task.eval_set[:5])
        self.assertEqual(s.n_repeats, 1)
        self.assertEqual(s.accuracy_spread, 0.0)


class TestMedianAggregation(unittest.TestCase):
    """With a noisy LLM, single-shot accuracy varies run-to-run; median over
    many repeats stabilizes — and the variance spread field reports the noise."""

    def setUp(self):
        self.task = build_toy_task(seed=0)
        self.examples = self.task.eval_set[:10]

    def _run_single_shot_many_times(self, llm_fn, n_trials: int = 15) -> list[float]:
        """Run score(n_repeats=1) many times. Returns the distribution of accuracies."""
        accs: list[float] = []
        for _ in range(n_trials):
            h = Harness(components=[
                NullRetriever(), NullFewShot(), SimpleFormatter(),
                MockLLMPredictor(llm_fn=llm_fn, n_samples=1),
                NullVoter(),
            ])
            s = Scorer(self.task).score(h, self.examples, n_repeats=1)
            accs.append(s.accuracy)
        return accs

    def test_noisy_llm_causes_single_shot_variance(self):
        """Sanity: without the fix, single-shot has real variance."""
        llm_fn = noisy_llm(flip_prob=0.3, seed=42)
        accs = self._run_single_shot_many_times(llm_fn)
        spread = max(accs) - min(accs)
        # 30% flip rate + small sample → spread should be > 0.
        self.assertGreater(spread, 0.0,
            msg=f"expected nondeterminism to produce variance; got accs={accs}")

    def test_n_repeats_reports_spread(self):
        """A multi-repeat call fills in the accuracy_spread field."""
        llm_fn = noisy_llm(flip_prob=0.3, seed=123)
        h = Harness(components=[
            NullRetriever(), NullFewShot(), SimpleFormatter(),
            MockLLMPredictor(llm_fn=llm_fn, n_samples=1),
            NullVoter(),
        ])
        s = Scorer(self.task).score(h, self.examples, n_repeats=9)
        self.assertEqual(s.n_repeats, 9)
        # With flip_prob=0.3, we expect non-zero spread across 9 repeats.
        self.assertGreater(s.accuracy_spread, 0.0)

    def test_mean_cost_not_median(self):
        """Tokens/latency are averaged across repeats, not median'd."""
        # Use a deterministic llm_fn so we can reason about the aggregate.
        def det_llm(prompt, ctx, harness, sample_idx):
            return ctx.example.label, 10, 2.0  # always correct, fixed cost
        h = Harness(components=[
            NullRetriever(), NullFewShot(), SimpleFormatter(),
            MockLLMPredictor(llm_fn=det_llm, n_samples=1),
            NullVoter(),
        ])
        # SimpleFormatter adds prompt-length tokens and 2.0 latency;
        # Predictor adds 10 tokens + 2.0 latency. We only care that
        # aggregates are stable across repeats.
        s1 = Scorer(self.task).score(h, self.examples, n_repeats=1)
        s5 = Scorer(self.task).score(h, self.examples, n_repeats=5)
        # Deterministic cost → mean over 5 repeats must equal single-shot.
        self.assertAlmostEqual(s1.tokens, s5.tokens, places=6)
        self.assertAlmostEqual(s1.latency_ms, s5.latency_ms, places=6)
        self.assertEqual(s5.n_repeats, 5)
        self.assertEqual(s5.accuracy_spread, 0.0)

    def test_rejects_zero_repeats(self):
        scorer = Scorer(self.task)
        h = Harness(components=[
            NullRetriever(), NullFewShot(), SimpleFormatter(),
            MockLLMPredictor(llm_fn=noisy_llm(0.0), n_samples=1),
            NullVoter(),
        ])
        with self.assertRaises(ValueError):
            scorer.score(h, self.examples, n_repeats=0)


class TestAttributionWithRepeats(unittest.TestCase):
    """Attribution analyze() passes n_repeats through and caches full_score correctly."""

    def setUp(self):
        self.task = build_toy_task(seed=0)
        self.scorer = Scorer(self.task)
        self.llm = noisy_llm(flip_prob=0.2, seed=7)

    def _harness(self) -> Harness:
        return Harness(components=[
            BagOfWordsRetriever(corpus=self.task.train, k=3),
            TopKFewShot(k=2),
            SimpleFormatter(),
            MockLLMPredictor(llm_fn=self.llm, n_samples=3),
            MajorityVoter(),
        ])

    def test_cache_invalidated_when_repeats_differ(self):
        tracker = AttributionTracker(self.scorer, baseline_for)
        h = self._harness()
        examples = self.task.eval_set[:8]
        # Prime with n_repeats=1.
        full_1 = self.scorer.score(h, examples, n_repeats=1)
        # Request analysis at n_repeats=3 — the cached score shouldn't be reused.
        snapshots = tracker.analyze("c1", h, examples, full_score=full_1, n_repeats=3)
        # Every snapshot's full_score must reflect n_repeats=3 (fresh compute),
        # not the cached n_repeats=1 value.
        for snap in snapshots:
            self.assertEqual(snap.full_score.n_repeats, 3)

    def test_cache_used_when_shape_matches(self):
        tracker = AttributionTracker(self.scorer, baseline_for)
        h = self._harness()
        examples = self.task.eval_set[:8]
        full_3 = self.scorer.score(h, examples, n_repeats=3)
        snapshots = tracker.analyze("c1", h, examples, full_score=full_3, n_repeats=3)
        for snap in snapshots:
            # Reused cached score — object-identical accuracy.
            self.assertEqual(snap.full_score.accuracy, full_3.accuracy)


class TestRunnerConfigWiring(unittest.TestCase):
    """Smoke: SearchConfig accepts the new fields; runner threads them through."""

    def test_config_defaults_backward_compatible(self):
        from meta_harness_plus.runner import SearchConfig
        cfg = SearchConfig()
        self.assertEqual(cfg.eval_repeats, 1)
        self.assertEqual(cfg.screen_repeats, 1)
        self.assertEqual(cfg.attribution_repeats, 1)
        self.assertIsNone(cfg.attribution_screen_size)

    def test_attribution_screen_defaults_to_screen_set(self):
        """When attribution_screen_size is None, uses the plain screen subset."""
        from meta_harness_plus.runner import SearchConfig, SearchRunner
        from meta_harness_plus.attribution import AttributionTracker

        task = build_toy_task(seed=0)
        scorer = Scorer(task)
        tracker = AttributionTracker(scorer, baseline_for)

        class DummyProposer:
            def propose(self, frontier, attribution, n):
                from meta_harness_plus.search.proposer import ProposalResult
                return ProposalResult(harnesses=[])

        runner = SearchRunner(
            task=task, scorer=scorer, proposer=DummyProposer(), attribution=tracker,
            config=SearchConfig(n_iterations=0, screen_size=4),
        )
        self.assertEqual(
            [e.input for e in runner._attribution_set()],
            [e.input for e in runner._screen_set()],
        )

    def test_larger_attribution_screen_uses_different_slice(self):
        from meta_harness_plus.runner import SearchConfig, SearchRunner
        from meta_harness_plus.attribution import AttributionTracker

        task = build_toy_task(seed=0)
        scorer = Scorer(task)
        tracker = AttributionTracker(scorer, baseline_for)

        class DummyProposer:
            def propose(self, frontier, attribution, n):
                from meta_harness_plus.search.proposer import ProposalResult
                return ProposalResult(harnesses=[])

        runner = SearchRunner(
            task=task, scorer=scorer, proposer=DummyProposer(), attribution=tracker,
            config=SearchConfig(n_iterations=0, screen_size=4, attribution_screen_size=12),
        )
        screen = runner._screen_set()
        attr_screen = runner._attribution_set()
        self.assertEqual(len(screen), 4)
        self.assertEqual(len(attr_screen), 12)


if __name__ == "__main__":
    unittest.main()
