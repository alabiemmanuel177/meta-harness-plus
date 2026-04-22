"""LLMPredictor: class-parsing + cost accounting against ScriptedClient."""

from __future__ import annotations

import unittest

from meta_harness_plus.harness import Context
from meta_harness_plus.llm.client import ScriptedClient
from meta_harness_plus.llm.predictor import LLMPredictor, _parse_class
from meta_harness_plus.task import TaskExample


class TestParseClass(unittest.TestCase):
    classes = ("sports", "cooking", "finance", "health", "tech")

    def test_exact_last_line(self):
        self.assertEqual(_parse_class("sports", self.classes), "sports")

    def test_case_insensitive(self):
        self.assertEqual(_parse_class("SPORTS", self.classes), "sports")

    def test_prose_plus_label(self):
        text = "Thinking about the keywords...\n\nThe class is sports"
        # Last line starts with "the" — falls back to substring match.
        self.assertEqual(_parse_class(text, self.classes), "sports")

    def test_multiline_picks_last(self):
        self.assertEqual(_parse_class("I think it's cooking\nsports", self.classes), "sports")

    def test_fallback_to_first_if_no_match(self):
        self.assertEqual(_parse_class("unrelated blather", self.classes), "sports")


class TestLLMPredictor(unittest.TestCase):
    classes = ("sports", "cooking", "finance", "health", "tech")

    def test_calls_client_n_times_and_accounts_cost(self):
        client = ScriptedClient(["sports", "sports", "cooking"],
                                tokens_per_char=1.0, fake_latency_ms=7.0)
        pred = LLMPredictor(client=client, classes=self.classes, n_samples=3)
        ctx = Context(example=TaskExample(input="q", label="sports"))
        ctx.prompt = "Classify this."
        pred.run(ctx, harness=None)  # harness unused
        self.assertEqual(ctx.candidate_predictions, ["sports", "sports", "cooking"])
        # Three calls, each adds fake_latency_ms to the context.
        self.assertAlmostEqual(ctx.latency_ms, 3 * 7.0)
        # Three calls each contribute (input_tokens + output_tokens) to ctx.tokens.
        self.assertGreater(ctx.tokens, 0)

    def test_respects_classes_via_parser(self):
        client = ScriptedClient(["I would say SPORTS"])
        pred = LLMPredictor(client=client, classes=self.classes, n_samples=1)
        ctx = Context(example=TaskExample(input="q", label="sports"))
        ctx.prompt = "Q"
        pred.run(ctx, harness=None)
        self.assertEqual(ctx.candidate_predictions, ["sports"])


if __name__ == "__main__":
    unittest.main()
