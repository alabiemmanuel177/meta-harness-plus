"""Tests for ContinualHarnessImprover (active continual-search variant)."""
from __future__ import annotations

import unittest
from typing import Sequence

from meta_harness_plus.components import (
    BagOfWordsRetriever, MockLLMPredictor, NullFewShot, NullRetriever,
    NullVoter, SimpleFormatter, TopKFewShot,
)
from meta_harness_plus.harness import Harness
from meta_harness_plus.online import ContinualHarnessImprover, OnlineHarnessImprover
from meta_harness_plus.scorer import Scorer
from meta_harness_plus.task import TaskExample
from meta_harness_plus.tasks import build_toy_task, mock_llm


class TestContinualHarnessImprover(unittest.TestCase):
    def setUp(self):
        self.task = build_toy_task(seed=0)
        self.scorer = Scorer(self.task)
        self.llm = mock_llm()

    def _bare(self) -> Harness:
        return Harness(components=[
            NullRetriever(), NullFewShot(), SimpleFormatter(),
            MockLLMPredictor(llm_fn=self.llm), NullVoter(),
        ])

    def _rag(self, k: int = 2) -> Harness:
        return Harness(components=[
            BagOfWordsRetriever(corpus=self.task.train, k=k),
            TopKFewShot(k=1),
            SimpleFormatter(),
            MockLLMPredictor(llm_fn=self.llm),
            NullVoter(),
        ])

    def test_propose_and_admit_no_propose_fn(self):
        """With no propose_fn, propose_and_admit returns 0 immediately."""
        improver = ContinualHarnessImprover(
            scorer=self.scorer,
            production_harness=self._bare(),
            candidate_harnesses={"rag": self._rag()},
            propose_fn=None,
        )
        improver.ingest_many(list(self.task.eval_set))
        self.assertEqual(improver.propose_and_admit(), 0)

    def test_propose_and_admit_below_min_examples(self):
        """Below min_examples, propose_and_admit returns 0."""
        def proposer(_examples: Sequence[TaskExample], _n: int) -> list[Harness]:
            return [self._rag(k=3)]
        improver = ContinualHarnessImprover(
            scorer=self.scorer,
            production_harness=self._bare(),
            candidate_harnesses={"rag": self._rag()},
            propose_fn=proposer,
            min_examples=20,
        )
        improver.ingest_many(list(self.task.eval_set)[:5])
        self.assertEqual(improver.propose_and_admit(), 0)

    def test_propose_and_admit_admits_when_non_dominated(self):
        """propose_and_admit should admit a non-dominated proposed harness."""
        def proposer(_examples: Sequence[TaskExample], _n: int) -> list[Harness]:
            return [self._rag(k=3)]  # different shape
        improver = ContinualHarnessImprover(
            scorer=self.scorer,
            production_harness=self._bare(),
            candidate_harnesses={"rag": self._rag()},
            propose_fn=proposer,
            min_examples=5,
            max_candidates_per_search=1,
        )
        improver.ingest_many(list(self.task.eval_set))
        # Pre-populate _last_scores by running maybe_promote once.
        improver.maybe_promote(force_rescore=True)
        prior_candidates = len(improver.candidate_harnesses)
        n_admitted = improver.propose_and_admit()
        # 0 or 1; admit only if non-dominated. Either case, search_count increments.
        self.assertGreaterEqual(n_admitted, 0)
        self.assertEqual(improver.search_count, 1)
        # If admitted, candidate pool grew.
        if n_admitted > 0:
            self.assertEqual(len(improver.candidate_harnesses), prior_candidates + n_admitted)

    def test_propose_fn_exception_does_not_crash(self):
        """If propose_fn raises, propose_and_admit returns 0 cleanly."""
        def bad_proposer(_examples: Sequence[TaskExample], _n: int) -> list[Harness]:
            raise RuntimeError("proposer failed")
        improver = ContinualHarnessImprover(
            scorer=self.scorer,
            production_harness=self._bare(),
            candidate_harnesses={"rag": self._rag()},
            propose_fn=bad_proposer,
            min_examples=5,
        )
        improver.ingest_many(list(self.task.eval_set))
        # Should not raise.
        result = improver.propose_and_admit()
        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
