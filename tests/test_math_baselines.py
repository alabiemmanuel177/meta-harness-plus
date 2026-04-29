"""Smoke tests for the Phase 2 math baselines runner.

Verifies module imports + per-baseline pure-logic helpers (no API).
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "_mb", _REPO / "examples" / "run_math_baselines.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class TestRunnerSurface(unittest.TestCase):
    def test_imports_and_has_main(self):
        m = _load()
        self.assertTrue(hasattr(m, "main"))
        self.assertEqual(set(m.BASELINES.keys()),
                         {"cot", "maj8", "maj16", "dspy", "opro", "random"})

    def test_majority_helper(self):
        m = _load()
        self.assertEqual(m._majority([1, 2, 1, None]), 1)
        self.assertEqual(m._majority([3, 3, 3]), 3)
        self.assertIsNone(m._majority([None, None]))
        self.assertIsNone(m._majority([]))

    def test_correct_helper(self):
        m = _load()
        self.assertTrue(m._correct(7, 7))
        self.assertFalse(m._correct(None, 7))
        self.assertFalse(m._correct(7, None))
        self.assertFalse(m._correct(7, 8))

    def test_gold_int(self):
        m = _load()
        self.assertEqual(m._gold_int("103"), 103)
        self.assertEqual(m._gold_int(" 0 "), 0)
        self.assertIsNone(m._gold_int("not an int"))


if __name__ == "__main__":
    unittest.main()
