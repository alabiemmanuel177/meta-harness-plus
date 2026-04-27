"""Deterministic tests for HMMT-Feb-2025 loader + grader."""

from __future__ import annotations

import unittest

from meta_harness_plus.tasks.hmmt_task import (
    _is_integer_answer,
    build_hmmt_feb2025_task,
    build_hmmt_fixture_task,
    parse_hmmt_int_answer,
)


class TestIsIntegerAnswer(unittest.TestCase):
    def test_integers(self):
        for s in ("0", "1", "999", "-7", "  42 "):
            self.assertTrue(_is_integer_answer(s), s)

    def test_non_integers(self):
        for s in (r"\frac{1}{2}", r"\sqrt{5}", "1.5", "", None):
            self.assertFalse(_is_integer_answer(s), s)


class TestParseHMMTAnswer(unittest.TestCase):
    def test_boxed(self):
        self.assertEqual(parse_hmmt_int_answer(r"\boxed{103}"), 103)

    def test_negative(self):
        self.assertEqual(parse_hmmt_int_answer(r"\boxed{-7}"), -7)

    def test_large_int(self):
        self.assertEqual(parse_hmmt_int_answer(r"answer is 6300"), 6300)

    def test_runaway_rejected(self):
        # Beyond ±1e6 returns None — guard against latex garbage that
        # accidentally parses as a giant number.
        self.assertIsNone(parse_hmmt_int_answer("answer: 9" + "9" * 20))

    def test_empty(self):
        self.assertIsNone(parse_hmmt_int_answer(""))
        self.assertIsNone(parse_hmmt_int_answer(None))


class TestHMMTFixture(unittest.TestCase):
    def test_fixture_only_integer_problems(self):
        task = build_hmmt_fixture_task()
        self.assertEqual(task.name, "hmmt_fixture")
        self.assertEqual(len(task.eval_set), 3)
        for ex in task.eval_set:
            self.assertTrue(_is_integer_answer(ex.label))

    def test_real_loader_integer_only_default(self):
        # Either real cached/HF data (14 integer problems) or fixture (3).
        task = build_hmmt_feb2025_task(integer_only=True, verify=True)
        self.assertGreaterEqual(len(task.eval_set), 3)
        for ex in task.eval_set:
            self.assertTrue(_is_integer_answer(ex.label))

    def test_real_loader_full_includes_non_integers(self):
        task = build_hmmt_feb2025_task(integer_only=False, verify=True)
        # If we loaded real HF data, expect 30 problems with mixed
        # answers; fixture has 5.
        self.assertIn(len(task.eval_set), (5, 30))


if __name__ == "__main__":
    unittest.main()
