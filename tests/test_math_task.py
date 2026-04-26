"""Tests for math task wiring + numerical-answer extraction."""
from __future__ import annotations

import unittest

from meta_harness_plus.tasks.math_task import extract_math_answer, _normalize_num


class TestExtractMathAnswer(unittest.TestCase):
    def test_gsm8k_format(self):
        self.assertEqual(extract_math_answer("She makes 9 * 2 = 18.\n#### 18"), "18")

    def test_boxed_format(self):
        self.assertEqual(extract_math_answer("The answer is \\boxed{42}."), "42")

    def test_dollar_amount(self):
        self.assertEqual(extract_math_answer("She earns $2 per egg, total $18."), "18")

    def test_trailing_number(self):
        self.assertEqual(extract_math_answer("The answer is 7"), "7")

    def test_negative_number(self):
        self.assertEqual(extract_math_answer("#### -3"), "-3")

    def test_decimal_normalized_to_int(self):
        self.assertEqual(extract_math_answer("The answer is 18.0"), "18")

    def test_genuine_decimal_kept(self):
        self.assertEqual(extract_math_answer("The answer is 0.5"), "0.5")

    def test_empty_or_no_number(self):
        self.assertEqual(extract_math_answer(""), "")
        self.assertEqual(extract_math_answer("no numbers here"), "")

    def test_normalize_helper(self):
        self.assertEqual(_normalize_num("18.0"), "18")
        self.assertEqual(_normalize_num("+18"), "18")
        self.assertEqual(_normalize_num("3.14"), "3.14")


if __name__ == "__main__":
    unittest.main()
