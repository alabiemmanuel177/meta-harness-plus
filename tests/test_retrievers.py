"""TF-IDF + BM25 retrievers (Tier 4.3)."""

from __future__ import annotations

import unittest

from meta_harness_plus.components import (
    BagOfWordsRetriever,
    BM25Retriever,
    TFIDFRetriever,
)
from meta_harness_plus.harness import Context
from meta_harness_plus.task import TaskExample


def _corpus():
    return [
        TaskExample(input="cat eats fish daily", label="animal"),
        TaskExample(input="dog chases cat in park", label="animal"),
        TaskExample(input="car drives fast on highway", label="vehicle"),
        TaskExample(input="bike rolls slow on trail", label="vehicle"),
        TaskExample(input="fish swims deep in ocean", label="animal"),
        TaskExample(input="truck hauls heavy load", label="vehicle"),
    ]


class TestTFIDF(unittest.TestCase):
    def test_idf_built_from_corpus(self):
        r = TFIDFRetriever(corpus=_corpus(), k=3)
        # 'fish' appears in 2/6 docs → IDF for fish < IDF for unique terms.
        self.assertGreater(len(r._idf), 0)
        # 'fish' should have lower IDF than 'truck' (1/6).
        self.assertLess(r._idf["fish"], r._idf["truck"])

    def test_retrieves_relevant_docs(self):
        r = TFIDFRetriever(corpus=_corpus(), k=2)
        ctx = Context(example=TaskExample(input="fish in water", label="animal"))
        r.run(ctx, None)
        self.assertEqual(len(ctx.retrieved), 2)
        # Both retrieved should mention "fish".
        for ex in ctx.retrieved:
            self.assertIn("fish", ex.input.lower())

    def test_empty_corpus_returns_empty(self):
        r = TFIDFRetriever(corpus=[], k=3)
        ctx = Context(example=TaskExample(input="anything", label="x"))
        r.run(ctx, None)
        self.assertEqual(ctx.retrieved, [])

    def test_no_match_returns_empty(self):
        r = TFIDFRetriever(corpus=_corpus(), k=3)
        ctx = Context(example=TaskExample(input="zzzzzz qqqq xxxxx", label="x"))
        r.run(ctx, None)
        self.assertEqual(ctx.retrieved, [])

    def test_config_emits_k(self):
        r = TFIDFRetriever(corpus=_corpus(), k=5)
        cfg = r.config()
        self.assertEqual(cfg["k"], 5)
        self.assertEqual(cfg["name"], "tfidf_retriever")
        self.assertEqual(cfg["kind"], "retriever")


class TestBM25(unittest.TestCase):
    def test_basic_retrieval(self):
        r = BM25Retriever(corpus=_corpus(), k=2)
        ctx = Context(example=TaskExample(input="bike on the trail", label="vehicle"))
        r.run(ctx, None)
        self.assertEqual(len(ctx.retrieved), 2)
        # 'bike rolls slow on trail' should be top-1 (most overlap).
        self.assertEqual(ctx.retrieved[0].input, "bike rolls slow on trail")

    def test_doc_length_normalization(self):
        # Two docs with same terms but different lengths should score
        # differently — shorter doc should rank higher.
        short = TaskExample(input="cat fish", label="x")
        long_ = TaskExample(input="cat fish " + " ".join(["filler"] * 20),
                            label="x")
        r = BM25Retriever(corpus=[short, long_], k=2)
        ctx = Context(example=TaskExample(input="cat fish", label="x"))
        r.run(ctx, None)
        # Short doc first.
        self.assertEqual(ctx.retrieved[0].input, "cat fish")

    def test_idf_high_for_rare_terms(self):
        r = BM25Retriever(corpus=_corpus(), k=3)
        # 'truck' appears in 1 doc, 'cat' appears in 2 docs.
        self.assertGreater(r._idf["truck"], r._idf["cat"])

    def test_config_threads_k1_and_b(self):
        r = BM25Retriever(corpus=_corpus(), k=3, k1=1.2, b=0.5)
        cfg = r.config()
        self.assertEqual(cfg["k1"], 1.2)
        self.assertEqual(cfg["b"], 0.5)


class TestRetrieverParity(unittest.TestCase):
    """Sanity: all three retrievers should top-rank obvious-match docs."""

    def test_obvious_match(self):
        ctx_template = TaskExample(input="fish in water", label="animal")
        for R in (BagOfWordsRetriever, TFIDFRetriever, BM25Retriever):
            r = R(corpus=_corpus(), k=3)
            ctx = Context(example=ctx_template)
            r.run(ctx, None)
            top_inputs = [e.input for e in ctx.retrieved]
            # Both fish-mentioning docs should be in top-3.
            fish_docs = [d for d in top_inputs if "fish" in d.lower()]
            self.assertEqual(len(fish_docs), 2,
                             f"{R.__name__} returned {top_inputs}")


if __name__ == "__main__":
    unittest.main()
