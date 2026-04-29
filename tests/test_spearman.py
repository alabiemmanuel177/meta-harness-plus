"""Test the inline Spearman implementation in run_phase2_rank_correlation."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "_rc", _REPO / "examples" / "run_phase2_rank_correlation.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class TestSpearman(unittest.TestCase):
    def test_perfect_positive(self):
        m = _load()
        self.assertAlmostEqual(m._spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)

    def test_perfect_negative(self):
        m = _load()
        self.assertAlmostEqual(m._spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)

    def test_no_signal(self):
        m = _load()
        # All-equal y → rho = 0.
        self.assertEqual(m._spearman([1, 2, 3, 4], [5, 5, 5, 5]), 0.0)

    def test_partial_correlation(self):
        m = _load()
        rho = m._spearman([1, 2, 3, 4, 5], [2, 1, 4, 3, 5])
        self.assertGreater(rho, 0.5)
        self.assertLess(rho, 1.0)

    def test_short_input_returns_zero(self):
        m = _load()
        self.assertEqual(m._spearman([1], [1]), 0.0)
        self.assertEqual(m._spearman([], []), 0.0)


if __name__ == "__main__":
    unittest.main()
