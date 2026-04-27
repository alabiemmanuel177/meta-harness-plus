"""Deterministic tests for AIME-25 loader + answer parser.

No network in tests — uses the bundled fixture or a cached JSONL.
"""

from __future__ import annotations

import unittest

from meta_harness_plus.tasks.aime_task import (
    _coerce_aime,
    build_aime25_fixture_task,
    build_aime25_task,
    parse_aime_answer,
)


class TestParseAIMEAnswer(unittest.TestCase):
    def test_boxed(self):
        self.assertEqual(parse_aime_answer(r"so the answer is \boxed{42}"), 42)

    def test_hash_marker(self):
        self.assertEqual(parse_aime_answer("steps...\n#### 17"), 17)

    def test_answer_is_phrase(self):
        self.assertEqual(parse_aime_answer("Therefore, the answer is 999."), 999)
        self.assertEqual(parse_aime_answer("Answer: 0"), 0)

    def test_falls_back_to_last_int(self):
        self.assertEqual(parse_aime_answer("Some thinking 12 then 99"), 99)

    def test_out_of_range_returns_none(self):
        self.assertIsNone(parse_aime_answer(r"\boxed{1000}"))
        self.assertIsNone(parse_aime_answer(r"\boxed{-3}"))

    def test_garbage_returns_none(self):
        self.assertIsNone(parse_aime_answer(""))
        self.assertIsNone(parse_aime_answer(None))
        self.assertIsNone(parse_aime_answer("no digits at all"))

    def test_zero_is_valid(self):
        self.assertEqual(parse_aime_answer("the answer is 0"), 0)
        self.assertEqual(_coerce_aime("0"), 0)


class TestAIME25Fixture(unittest.TestCase):
    def test_fixture_has_5_problems(self):
        task = build_aime25_fixture_task()
        self.assertEqual(task.name, "aime25_fixture")
        self.assertEqual(len(task.eval_set), 5)
        for ex in task.eval_set:
            self.assertGreater(len(ex.input), 0)
            self.assertTrue(0 <= int(ex.label) <= 999)

    def test_fixture_classes_empty_for_open_generation(self):
        task = build_aime25_fixture_task()
        self.assertEqual(task.classes, [])


class TestAIME25LoaderVerification(unittest.TestCase):
    """If the cached JSONL exists, verify all answers are integers 0-999."""

    def test_real_loader_or_fixture_passes_verify(self):
        # build_aime25_task with verify=True should not raise on
        # well-formed data. We exercise the verify path either against
        # cached real data or the bundled fixture rows (both clean).
        task = build_aime25_task(verify=True)
        self.assertGreaterEqual(len(task.eval_set), 5)
        # All eval items have valid int answers.
        for ex in task.eval_set:
            v = int(ex.label)
            self.assertTrue(0 <= v <= 999, f"label out of range: {ex.label}")

    def test_n_train_split(self):
        task = build_aime25_task(n_train=2, verify=True)
        # Whatever the source has, 2 should be moved to train.
        self.assertEqual(len(task.train), 2)
        self.assertGreaterEqual(len(task.eval_set), 3)


if __name__ == "__main__":
    unittest.main()
