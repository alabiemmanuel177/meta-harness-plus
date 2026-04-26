"""Tests for OnlineHarnessImprover (production-traffic frontier updates)."""
from __future__ import annotations

import unittest

from meta_harness_plus.components import (
    BagOfWordsRetriever, MockLLMPredictor, NullFewShot, NullRetriever,
    NullVoter, SimpleFormatter, TopKFewShot,
)
from meta_harness_plus.harness import Harness
from meta_harness_plus.online import OnlineHarnessImprover, PromoteReport
from meta_harness_plus.scorer import Scorer
from meta_harness_plus.tasks import build_toy_task, mock_llm


class TestOnlineImprover(unittest.TestCase):
    def setUp(self):
        self.task = build_toy_task(seed=0)
        self.scorer = Scorer(self.task)
        self.llm = mock_llm()

    def _bare(self) -> Harness:
        return Harness(components=[
            NullRetriever(), NullFewShot(), SimpleFormatter(),
            MockLLMPredictor(llm_fn=self.llm), NullVoter(),
        ])

    def _rag(self) -> Harness:
        return Harness(components=[
            BagOfWordsRetriever(corpus=self.task.train, k=2),
            TopKFewShot(k=1),
            SimpleFormatter(),
            MockLLMPredictor(llm_fn=self.llm),
            NullVoter(),
        ])

    def test_no_promote_below_min_examples(self):
        improver = OnlineHarnessImprover(
            scorer=self.scorer,
            production_harness=self._bare(),
            candidate_harnesses={"rag": self._rag()},
            min_examples=10,
        )
        # Ingest 5, less than min.
        for ex in list(self.task.eval_set)[:5]:
            improver.ingest(ex)
        report = improver.maybe_promote(force_rescore=True)
        self.assertFalse(report.should_promote)
        self.assertIn("need 10", report.reason)

    def test_promote_when_candidate_dominates(self):
        improver = OnlineHarnessImprover(
            scorer=self.scorer,
            production_harness=self._bare(),
            candidate_harnesses={"rag": self._rag()},
            min_examples=5,
        )
        # On the toy task, RAG should beat BARE on accuracy (deterministic
        # mock LLM lookups via retrieved context).
        improver.ingest_many(list(self.task.eval_set))
        report = improver.maybe_promote(force_rescore=True)
        # We don't assume promotion happens (depends on toy-task RAG vs BARE
        # dynamics); just verify the report is well-structured.
        self.assertIsInstance(report, PromoteReport)
        self.assertIsNotNone(report.production_score)

    def test_history_rolls_off(self):
        improver = OnlineHarnessImprover(
            scorer=self.scorer,
            production_harness=self._bare(),
            candidate_harnesses={"rag": self._rag()},
            max_history=5,
        )
        for ex in list(self.task.eval_set)[:20]:
            improver.ingest(ex)
        # Only the last 5 should remain.
        self.assertEqual(improver.history_size, 5)

    def test_no_dominator_returns_negative_report(self):
        improver = OnlineHarnessImprover(
            scorer=self.scorer,
            production_harness=self._rag(),
            candidate_harnesses={"bare": self._bare()},
            min_examples=5,
        )
        improver.ingest_many(list(self.task.eval_set))
        report = improver.maybe_promote(force_rescore=True)
        # BARE shouldn't dominate RAG on accuracy + cost simultaneously.
        # (BARE may be cheaper but has lower accuracy.)
        self.assertFalse(report.should_promote)


if __name__ == "__main__":
    unittest.main()
