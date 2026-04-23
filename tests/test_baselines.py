"""Tests for named baseline harnesses + CoT formatter."""

from __future__ import annotations

import unittest

from meta_harness_plus.baselines import bare_baseline, rag_baseline
from meta_harness_plus.components import (
    BagOfWordsRetriever,
    CoTFormatter,
    MockLLMPredictor,
    NullFewShot,
    NullRetriever,
    NullVoter,
    SimpleFormatter,
    TopKFewShot,
    baseline_for,
)
from meta_harness_plus.harness import Context
from meta_harness_plus.task import TaskExample
from meta_harness_plus.tasks import build_symptom_hard_task, build_symptom_task


class TestBaselines(unittest.TestCase):
    def setUp(self):
        self.task = build_symptom_task()
        self.predictor = MockLLMPredictor(llm_fn=lambda *a, **k: ("cardiology", 5, 2.0), n_samples=1)

    def test_rag_baseline_shape(self):
        h = rag_baseline(self.task, self.predictor, retriever_k=3, fewshot_k=2)
        self.assertEqual([c.kind for c in h.components],
                         ["retriever", "fewshot", "formatter", "predictor", "voter"])
        # RAG uses BoW + TopK + SimpleFormatter + NullVoter by default.
        names = [c.name for c in h.components]
        self.assertEqual(names[0], "bow_retriever")
        self.assertEqual(names[1], "topk_fewshot")
        self.assertEqual(names[2], "simple_formatter")
        self.assertEqual(names[4], "null_voter")

    def test_rag_baseline_parameters_respected(self):
        h = rag_baseline(self.task, self.predictor, retriever_k=5, fewshot_k=3)
        # Retriever k=5, fewshot k=3.
        self.assertEqual(h.components[0].k, 5)
        self.assertEqual(h.components[1].k, 3)

    def test_bare_baseline_has_no_structure(self):
        h = bare_baseline(self.task, self.predictor)
        # All null — only the predictor does anything.
        self.assertIsInstance(h.components[0], NullRetriever)
        self.assertIsInstance(h.components[1], NullFewShot)
        self.assertIsInstance(h.components[4], NullVoter)


class TestCoTFormatter(unittest.TestCase):
    def test_has_formatter_kind(self):
        self.assertEqual(CoTFormatter().kind, "formatter")
        self.assertEqual(CoTFormatter().name, "cot_formatter")

    def test_hint_is_cot_flavored(self):
        hint = CoTFormatter().system_hint
        self.assertIn("analyze", hint.lower())
        self.assertIn("class name", hint.lower())

    def test_formats_prompt_with_query_and_fewshots(self):
        f = CoTFormatter()
        ctx = Context(example=TaskExample(input="chest pain", label="cardiology"))
        ctx.few_shots = [TaskExample(input="heart racing", label="cardiology")]
        f.run(ctx, harness=None)
        self.assertIn("chest pain", ctx.prompt)
        self.assertIn("heart racing", ctx.prompt)
        self.assertIn(f.system_hint, ctx.prompt)

    def test_ablation_baseline_is_simple_formatter(self):
        """Drop-one attribution on a CoT-formatter harness should swap in
        SimpleFormatter — the meaningful counterfactual for 'did CoT help?'."""
        baseline = baseline_for("formatter")
        self.assertIsNotNone(baseline)
        self.assertIsInstance(baseline, SimpleFormatter)


class TestSymptomHardTask(unittest.TestCase):
    def setUp(self):
        self.task = build_symptom_hard_task()

    def test_size(self):
        self.assertEqual(len(self.task.train), 50)  # 30 original + 20 disambiguating
        self.assertEqual(len(self.task.eval_set), 15)

    def test_train_class_balance(self):
        from collections import Counter
        counts = Counter(e.label for e in self.task.train)
        for klass in self.task.classes:
            self.assertEqual(counts[klass], 10, f"train class {klass} imbalanced")

    def test_eval_items_are_distinct_from_train(self):
        train_inputs = {e.input for e in self.task.train}
        for ev in self.task.eval_set:
            self.assertNotIn(ev.input, train_inputs,
                             f"eval item leaked from train: {ev.input!r}")

    def test_eval_includes_known_hard_cases(self):
        """Sanity: our named adversarial cases are actually in the eval set."""
        inputs = [e.input for e in self.task.eval_set]
        # The DVT case (labeled cardiology, not orthopedics despite "calf pain")
        self.assertTrue(any("calf" in i for i in inputs))
        # The MSK chest pain (labeled orthopedics, not cardiology)
        self.assertTrue(any("pressing on the chest wall" in i for i in inputs))


if __name__ == "__main__":
    unittest.main()
