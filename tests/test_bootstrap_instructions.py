"""Tests for bootstrap_instructions (OPRO-style pre-generation)."""
from __future__ import annotations

import unittest

from meta_harness_plus.components import bootstrap_instructions
from meta_harness_plus.llm.client import ScriptedClient
from meta_harness_plus.tasks import build_toy_task


class TestBootstrapInstructions(unittest.TestCase):
    def setUp(self):
        self.task = build_toy_task(seed=0)

    def test_returns_seed_plus_n_unique(self):
        """Should return seed + up to n unique instructions."""
        client = ScriptedClient([
            "instruction one\n",
            "instruction two\n",
            "instruction three\n",
            "instruction one\n",  # dup; should be skipped
        ])
        out = bootstrap_instructions(client, self.task, n=4,
                                     seed_instruction="seed.",
                                     temperature=0.7)
        self.assertGreaterEqual(len(out), 1)
        self.assertEqual(out[0], "seed.")
        # No duplicates.
        self.assertEqual(len(set(out)), len(out))

    def test_handles_client_failure_gracefully(self):
        """If client.complete raises, the function returns what it has."""
        class BrokenClient:
            def complete(self, *, system, user, max_tokens, temperature):
                raise RuntimeError("LLM down")

        out = bootstrap_instructions(BrokenClient(), self.task, n=4,
                                     seed_instruction="seed.")
        # Just the seed.
        self.assertEqual(out, ["seed."])

    def test_takes_first_line_of_response(self):
        """If LLM returns multi-line, only first line is used."""
        client = ScriptedClient(["line one\nline two\nline three\n"])
        out = bootstrap_instructions(client, self.task, n=1,
                                     seed_instruction="seed.")
        # Seed + first-line-of-response.
        self.assertEqual(out, ["seed.", "line one"])


if __name__ == "__main__":
    unittest.main()
