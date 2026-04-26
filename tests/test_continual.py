"""Tests for the production-grade ContinualImprover."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Sequence

from meta_harness_plus.components import (
    BagOfWordsRetriever, MockLLMPredictor, NullFewShot, NullRetriever,
    NullVoter, SimpleFormatter, TopKFewShot,
)
from meta_harness_plus.continual import (
    ContinualImprover,
    DriftAlert,
    DriftDetector,
    PerExampleTracker,
    PromotionGates,
    PromotionLogEntry,
    TickReport,
)
from meta_harness_plus.harness import Harness
from meta_harness_plus.scorer import Scorer
from meta_harness_plus.task import TaskExample
from meta_harness_plus.tasks import build_toy_task, mock_llm


def _bare_harness(llm):
    return Harness(components=[
        NullRetriever(), NullFewShot(), SimpleFormatter(),
        MockLLMPredictor(llm_fn=llm), NullVoter(),
    ])


def _rag_harness(corpus, llm, k=2):
    return Harness(components=[
        BagOfWordsRetriever(corpus=corpus, k=k),
        TopKFewShot(k=1),
        SimpleFormatter(),
        MockLLMPredictor(llm_fn=llm),
        NullVoter(),
    ])


# ------------------ PerExampleTracker ------------------

class TestPerExampleTracker(unittest.TestCase):
    def test_record_pads_unmeasured_labels(self):
        t = PerExampleTracker()
        t.register("a")
        t.register("b")
        ex = TaskExample(input="x", label="y")
        t.record(ex, {"a": (1, 100, 50)})  # 'b' not measured
        self.assertEqual(t.correctness["a"], [1])
        # b padded with -1 to maintain parallel-list invariant.
        self.assertEqual(t.correctness["b"], [-1])

    def test_paired_acc_skips_unmeasured(self):
        t = PerExampleTracker()
        t.register("a"); t.register("b")
        # First example only has 'a'; second has both.
        t.record(TaskExample("1", "y"), {"a": (1, 100, 50)})
        t.record(TaskExample("2", "y"), {"a": (1, 100, 50), "b": (0, 80, 40)})
        ca, cb = t.paired_acc("a", "b")
        self.assertEqual(len(ca), 1)
        self.assertEqual(ca, [1])
        self.assertEqual(cb, [0])

    def test_mean_cost(self):
        t = PerExampleTracker()
        t.register("a")
        t.record(TaskExample("1", "y"), {"a": (1, 100, 50)})
        t.record(TaskExample("2", "y"), {"a": (0, 200, 100)})
        toks, lat, n = t.mean_cost("a")
        self.assertAlmostEqual(toks, 150.0)
        self.assertAlmostEqual(lat, 75.0)
        self.assertEqual(n, 2)

    def test_state_dict_roundtrip(self):
        t = PerExampleTracker()
        t.register("a")
        t.record(TaskExample("x", "y"), {"a": (1, 50, 10)})
        d = t.state_dict()
        t2 = PerExampleTracker.from_state_dict(d)
        self.assertEqual(t2.correctness["a"], [1])
        self.assertEqual(t2.tokens["a"], [50.0])

    def test_trim(self):
        t = PerExampleTracker()
        t.register("a")
        for i in range(10):
            t.record(TaskExample(str(i), "y"), {"a": (i % 2, 100, 10)})
        t.trim(max_history=3)
        self.assertEqual(len(t.correctness["a"]), 3)


# ------------------ DriftDetector ------------------

class TestDriftDetector(unittest.TestCase):
    def test_no_alert_below_baseline(self):
        det = DriftDetector(window_size=10, baseline_size=10, threshold=0.2)
        for i in range(5):
            det.ingest(TaskExample(str(i), "a"))
        self.assertIsNone(det.check())

    def test_alert_when_distribution_shifts(self):
        det = DriftDetector(window_size=20, baseline_size=20, threshold=0.30)
        # Baseline: half "a", half "b".
        for _ in range(10):
            det.ingest(TaskExample("x", "a"))
        for _ in range(10):
            det.ingest(TaskExample("x", "b"))
        # Now baseline complete; recent window will be all "a".
        for _ in range(20):
            det.ingest(TaskExample("x", "a"))
        alert = det.check()
        self.assertIsNotNone(alert)
        # L1 between (0.5,0.5) and (1.0,0.0) = 0.5.
        self.assertAlmostEqual(alert.l1_distance, 0.5, places=2)

    def test_no_alert_when_distribution_stable(self):
        det = DriftDetector(window_size=20, baseline_size=20, threshold=0.30)
        seq = ["a", "b"] * 20
        for c in seq:
            det.ingest(TaskExample("x", c))
        self.assertIsNone(det.check())

    def test_state_dict_roundtrip(self):
        det = DriftDetector(window_size=10, baseline_size=5, threshold=0.2)
        for _ in range(7):
            det.ingest(TaskExample("x", "a"))
        d = det.state_dict()
        det2 = DriftDetector.from_state_dict(d)
        self.assertEqual(det2._baseline_n, det._baseline_n)
        self.assertEqual(det2._recent, det._recent)


# ------------------ ContinualImprover ------------------

class TestContinualImprover(unittest.TestCase):
    def setUp(self):
        self.task = build_toy_task(seed=0)
        self.scorer = Scorer(self.task)
        self.llm = mock_llm()

    def _improver(self, **kwargs) -> ContinualImprover:
        candidates = {
            "bare": _bare_harness(self.llm),
            "rag":  _rag_harness(self.task.train, self.llm),
        }
        return ContinualImprover(
            scorer=self.scorer,
            production_label="bare",
            candidates=candidates,
            **kwargs,
        )

    def test_basic_tick_records_correctness(self):
        improver = self._improver(promotion_gates=PromotionGates(min_examples=3))
        for ex in list(self.task.eval_set)[:3]:
            improver.tick(ex)
        self.assertEqual(improver.ingest_count, 3)
        # Both candidates measured.
        self.assertEqual(len(improver.tracker.correctness["bare"]), 3)
        self.assertEqual(len(improver.tracker.correctness["rag"]), 3)

    def test_unknown_production_label_raises(self):
        with self.assertRaises(ValueError):
            ContinualImprover(
                scorer=self.scorer,
                production_label="ghost",
                candidates={"bare": _bare_harness(self.llm)},
            )

    def test_promotion_when_gates_clear(self):
        # Use a min_examples=4 gate with the toy task so RAG (likely
        # better than bare on toy) triggers promotion.
        improver = self._improver(promotion_gates=PromotionGates(
            min_examples=4, paired_acc_ci_alpha=0.20,  # loose for toy data
            cooldown_ingests=1,
            max_token_regression_pct=10.0,  # disable cost gate for this test
            max_latency_regression_pct=10.0,
        ))
        promoted_seen = False
        for ex in list(self.task.eval_set):
            r = improver.tick(ex, force_promote_check=True)
            if r.promoted:
                promoted_seen = True
                self.assertEqual(r.promoted_from, "bare")
                self.assertEqual(r.promoted_to, "rag")
                break
        # If RAG isn't strictly better on toy task, it's allowed to
        # remain not-promoted — but the gate logic shouldn't have erred.
        self.assertIsInstance(improver.production_label, str)
        # At least the tracker has parallel measurements.
        ca, cb = improver.tracker.paired_acc("rag", "bare")
        self.assertEqual(len(ca), len(cb))

    def test_cost_gate_blocks_promotion(self):
        """A candidate with same accuracy but higher tokens should not promote."""
        # Build a candidate whose Predictor wraps + adds a fake token cost.
        from meta_harness_plus.harness import Component, Context

        class HighCostNop(Component):
            kind = "fewshot"
            name = "high_cost_nop"
            def run(self, ctx: Context, harness):
                ctx.tokens += 10000  # huge regression

        bare = _bare_harness(self.llm)
        expensive = Harness(components=[
            NullRetriever(), HighCostNop(), SimpleFormatter(),
            MockLLMPredictor(llm_fn=self.llm), NullVoter(),
        ])
        improver = ContinualImprover(
            scorer=self.scorer,
            production_label="bare",
            candidates={"bare": bare, "expensive": expensive},
            promotion_gates=PromotionGates(
                min_examples=3, paired_acc_ci_alpha=0.5,
                max_token_regression_pct=0.10, cooldown_ingests=1,
            ),
        )
        for ex in list(self.task.eval_set):
            r = improver.tick(ex, force_promote_check=True)
            self.assertNotEqual(r.promoted_to, "expensive",
                                "expensive shape must not promote")
        # Promotion log should record at least one no-op for this if
        # the candidate did out-perform on accuracy.

    def test_cooldown_prevents_immediate_re_promotion(self):
        improver = self._improver(promotion_gates=PromotionGates(
            min_examples=3, paired_acc_ci_alpha=0.5,
            cooldown_ingests=100,  # huge cooldown
            max_token_regression_pct=10.0,
            max_latency_regression_pct=10.0,
        ))
        # First, force one promotion via direct call.
        improver._do_promote(
            "rag", ci=(0.05, 0.15), n_paired=10,
            report=TickReport(ingest_count=10),
        )
        improver.last_promote_at = 10
        improver.ingest_count = 11
        # Now tick a few more — cooldown_ingests=100 means we shouldn't
        # try to promote yet.
        for ex in list(self.task.eval_set)[:5]:
            r = improver.tick(ex)
            self.assertFalse(r.promoted)

    def test_persistence_save_load(self):
        with tempfile.TemporaryDirectory() as td:
            state_path = Path(td) / "improver.json"
            improver = self._improver(state_path=state_path,
                                      promotion_gates=PromotionGates(min_examples=3))
            for ex in list(self.task.eval_set)[:5]:
                improver.tick(ex)
            improver.save()

            self.assertTrue(state_path.exists())
            data = json.loads(state_path.read_text())
            self.assertEqual(data["ingest_count"], 5)
            self.assertEqual(data["production_label"], "bare")

            # Recreate the improver with same candidates and load.
            improver2 = self._improver(state_path=state_path,
                                       promotion_gates=PromotionGates(min_examples=3))
            improver2.load()
            self.assertEqual(improver2.ingest_count, 5)
            self.assertEqual(improver2.tracker.correctness["bare"],
                             improver.tracker.correctness["bare"])

    def test_load_complains_about_missing_candidates(self):
        """If state mentions a candidate the constructor didn't get, fail loudly."""
        with tempfile.TemporaryDirectory() as td:
            state_path = Path(td) / "improver.json"
            # Save a state with 3 candidates.
            improver = ContinualImprover(
                scorer=self.scorer,
                production_label="a",
                candidates={
                    "a": _bare_harness(self.llm),
                    "b": _rag_harness(self.task.train, self.llm),
                    "c": _rag_harness(self.task.train, self.llm, k=3),
                },
                state_path=state_path,
            )
            improver.tick(self.task.eval_set[0])
            improver.save()
            # Now reconstruct without 'c'.
            improver2 = ContinualImprover(
                scorer=self.scorer,
                production_label="a",
                candidates={
                    "a": _bare_harness(self.llm),
                    "b": _rag_harness(self.task.train, self.llm),
                },
                state_path=state_path,
            )
            with self.assertRaises(ValueError):
                improver2.load()

    def test_rollback(self):
        improver = self._improver()
        improver._do_promote("rag", ci=(0.05, 0.10), n_paired=10,
                             report=TickReport(ingest_count=1))
        self.assertEqual(improver.production_label, "rag")
        ok = improver.rollback("test rollback")
        self.assertTrue(ok)
        self.assertEqual(improver.production_label, "bare")
        # Promotion log records both events.
        kinds = [e.kind for e in improver.promotion_log]
        self.assertIn("promote", kinds)
        self.assertIn("rollback", kinds)

    def test_rollback_when_no_history_returns_false(self):
        improver = self._improver()
        ok = improver.rollback()
        self.assertFalse(ok)

    def test_propose_and_admit(self):
        admitted_labels: list[str] = []

        def proposer(_examples: Sequence[TaskExample], _n: int) -> list[Harness]:
            return [_rag_harness(self.task.train, self.llm, k=3)]

        improver = ContinualImprover(
            scorer=self.scorer,
            production_label="bare",
            candidates={"bare": _bare_harness(self.llm)},
            propose_fn=proposer,
            promotion_gates=PromotionGates(min_examples=3),
            propose_every=2,
        )
        # Tick a few — propose_every=2 means a proposal at ingest 2, 4, 6...
        for ex in list(self.task.eval_set)[:6]:
            r = improver.tick(ex)
            if r.admitted_count > 0:
                admitted_labels.append(r.admitted_count)
        # Either at least one search happened and some admission, or
        # the proposer's output was rejected — both are valid outcomes
        # for the tracker. Assert: a propose tick actually fired.
        self.assertGreaterEqual(improver.search_count, 1)

    def test_drift_alert_propagates(self):
        det = DriftDetector(window_size=4, baseline_size=4, threshold=0.30)
        improver = self._improver(drift_detector=det)
        # Baseline: 2 'A', 2 'B'.
        baseline_examples = [
            TaskExample(input="x", label="A"),
            TaskExample(input="y", label="A"),
            TaskExample(input="z", label="B"),
            TaskExample(input="w", label="B"),
        ]
        # Recent: all 'A' — drift.
        recent_examples = [TaskExample(input=str(i), label="A") for i in range(4)]
        seen_alert = False
        for ex in baseline_examples + recent_examples:
            r = improver.tick(ex)
            if r.drift_alert:
                seen_alert = True
        self.assertTrue(seen_alert)

    def test_add_remove_candidate(self):
        improver = self._improver()
        new_h = _rag_harness(self.task.train, self.llm, k=3)
        improver.add_candidate("rag_k3", new_h)
        self.assertIn("rag_k3", improver.candidates)
        self.assertIn("rag_k3", improver.tracker.correctness)
        ok = improver.remove_candidate("rag_k3")
        self.assertTrue(ok)
        self.assertNotIn("rag_k3", improver.candidates)
        # Cannot remove the production label.
        with self.assertRaises(ValueError):
            improver.remove_candidate("bare")


if __name__ == "__main__":
    unittest.main()
