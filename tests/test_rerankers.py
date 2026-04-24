"""Reranker component tests (null, diversity, LLM)."""

from __future__ import annotations

import unittest

from meta_harness_plus.components import (
    DiversityReranker,
    NullReranker,
    baseline_for,
)
from meta_harness_plus.harness import Context
from meta_harness_plus.llm.client import ScriptedClient
from meta_harness_plus.llm.reranker import LLMReranker
from meta_harness_plus.task import TaskExample


def ctx_with(items: list[tuple[str, str]]) -> Context:
    c = Context(example=TaskExample(input="q", label="a"))
    c.retrieved = [TaskExample(input=x, label=l) for x, l in items]
    return c


class TestNullReranker(unittest.TestCase):
    def test_passes_through(self):
        c = ctx_with([("x1", "a"), ("x2", "b")])
        NullReranker().run(c, None)
        self.assertEqual([e.label for e in c.retrieved], ["a", "b"])


class TestDiversityReranker(unittest.TestCase):
    def test_puts_unique_labels_first(self):
        c = ctx_with([("x1", "a"), ("x2", "a"), ("x3", "b"),
                      ("x4", "c"), ("x5", "a")])
        DiversityReranker().run(c, None)
        # First three: one per unique label (a, b, c); rest: remaining a's.
        first_three = [e.label for e in c.retrieved[:3]]
        self.assertEqual(set(first_three), {"a", "b", "c"})

    def test_empty_retrieved_noop(self):
        c = ctx_with([])
        DiversityReranker().run(c, None)
        self.assertEqual(c.retrieved, [])

    def test_kind(self):
        self.assertEqual(DiversityReranker().kind, "reranker")


class TestLLMReranker(unittest.TestCase):
    def test_picks_top_m_and_moves_to_front(self):
        c = ctx_with([("x0", "a"), ("x1", "b"), ("x2", "c"), ("x3", "d")])
        client = ScriptedClient(["picks:\n\n2 0"])
        LLMReranker(client=client, m=2).run(c, None)
        # Expect indices 2, 0 first; the rest (1, 3) after.
        self.assertEqual([e.input for e in c.retrieved[:2]], ["x2", "x0"])

    def test_invalid_parse_degrades_to_noop(self):
        c = ctx_with([("x0", "a"), ("x1", "b"), ("x2", "c")])
        original = list(c.retrieved)
        client = ScriptedClient(["completely unparseable text with no indices"])
        LLMReranker(client=client, m=2).run(c, None)
        # Parser gets nothing, retrieved unchanged.
        self.assertEqual([e.input for e in c.retrieved],
                         [e.input for e in original])

    def test_skip_if_m_gte_len(self):
        c = ctx_with([("x0", "a"), ("x1", "b")])
        client = ScriptedClient(["should not be called"])
        LLMReranker(client=client, m=5).run(c, None)
        self.assertEqual(client._idx, 0)  # never called

    def test_accounts_cost(self):
        c = ctx_with([("x0", "a"), ("x1", "b"), ("x2", "c"), ("x3", "d")])
        client = ScriptedClient(["0 1"], tokens_per_char=1.0, fake_latency_ms=50.0)
        LLMReranker(client=client, m=2).run(c, None)
        self.assertGreater(c.tokens, 0)
        self.assertEqual(c.latency_ms, 50.0)


class TestBaselineForReranker(unittest.TestCase):
    def test_reranker_baseline_is_null(self):
        baseline = baseline_for("reranker")
        self.assertIsNotNone(baseline)
        self.assertIsInstance(baseline, NullReranker)


if __name__ == "__main__":
    unittest.main()
