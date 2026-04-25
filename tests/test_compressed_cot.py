"""CompressedCoTFormatter (framework-upgrades branch).

Builds the prompt with an explicit reasoning-token budget. Same Pareto
frontier slot as CoTFormatter; ablation-baseline still SimpleFormatter.
"""

from __future__ import annotations

import unittest

from meta_harness_plus.components import (
    CompressedCoTFormatter,
    CoTFormatter,
    SimpleFormatter,
    baseline_for,
)
from meta_harness_plus.harness import Context
from meta_harness_plus.task import TaskExample


class TestCompressedCoT(unittest.TestCase):
    def test_kind_and_name(self):
        c = CompressedCoTFormatter()
        self.assertEqual(c.kind, "formatter")
        self.assertEqual(c.name, "compressed_cot_formatter")

    def test_default_budget_in_hint(self):
        c = CompressedCoTFormatter()
        self.assertIn("at most 15 words", c.system_hint)
        self.assertIn("Be terse", c.system_hint)

    def test_budget_threads_through(self):
        c = CompressedCoTFormatter(max_reasoning_words=8)
        self.assertIn("at most 8 words", c.system_hint)

    def test_config_includes_budget(self):
        cfg = CompressedCoTFormatter(max_reasoning_words=12).config()
        self.assertEqual(cfg["max_reasoning_words"], 12)
        self.assertEqual(cfg["name"], "compressed_cot_formatter")

    def test_ablation_baseline_is_simple(self):
        # Drop-one attribution swaps any formatter to SimpleFormatter —
        # measures "did this formatter help vs plain prompt?".
        baseline = baseline_for("formatter")
        self.assertIsInstance(baseline, SimpleFormatter)

    def test_formats_prompt_with_query_and_fewshots(self):
        f = CompressedCoTFormatter()
        ctx = Context(example=TaskExample(input="chest pain", label="cardio"))
        ctx.few_shots = [TaskExample(input="palpitations", label="cardio")]
        f.run(ctx, harness=None)
        self.assertIn("chest pain", ctx.prompt)
        self.assertIn("palpitations", ctx.prompt)
        self.assertIn(f.system_hint, ctx.prompt)

    def test_distinct_from_cot_formatter(self):
        # Both use a CoT-flavored hint, but the system_hints differ.
        cot = CoTFormatter().system_hint
        comp = CompressedCoTFormatter().system_hint
        self.assertNotEqual(cot, comp)
        self.assertIn("at most", comp)
        self.assertNotIn("at most", cot)


class TestRegistrationInLlmSearchRegistry(unittest.TestCase):
    def test_registry_exposes_compressed_cot(self):
        from meta_harness_plus.llm.client import ScriptedClient
        from meta_harness_plus.llm.registry import llm_search_registry
        from meta_harness_plus.tasks import build_toy_task

        task = build_toy_task(seed=0)
        reg = llm_search_registry(task, ScriptedClient(["x"]))
        self.assertIn("compressed_cot_formatter", reg.available_for_kind("formatter"))

    def test_registry_factory_constructs(self):
        from meta_harness_plus.llm.client import ScriptedClient
        from meta_harness_plus.llm.registry import llm_search_registry
        from meta_harness_plus.tasks import build_toy_task

        task = build_toy_task(seed=0)
        reg = llm_search_registry(task, ScriptedClient(["x"]))
        c = reg.instantiate({
            "kind": "formatter",
            "name": "compressed_cot_formatter",
            "config": {"max_reasoning_words": 25},
        })
        self.assertEqual(c.config()["max_reasoning_words"], 25)


if __name__ == "__main__":
    unittest.main()
