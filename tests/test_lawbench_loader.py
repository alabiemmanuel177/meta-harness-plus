"""LawBench loader skeleton + bundled fixture (Tier 1.1)."""

from __future__ import annotations

import unittest
from collections import Counter

from meta_harness_plus.tasks.lawbench import (
    build_lawbench_fixture_task,
    build_lawbench_task,
    lawbench_classes,
    write_lawbench_fixture,
)


class TestLawbenchFixture(unittest.TestCase):
    def setUp(self):
        write_lawbench_fixture()  # idempotent
        self.task = build_lawbench_fixture_task()

    def test_basic_structure(self):
        self.assertEqual(self.task.name, "lawbench_fixture")
        self.assertEqual(len(self.task.train), 20)
        self.assertEqual(len(self.task.eval_set), 10)

    def test_classes(self):
        self.assertEqual(set(self.task.classes), set(lawbench_classes()))

    def test_class_balance_train(self):
        counts = Counter(e.label for e in self.task.train)
        for klass in lawbench_classes():
            self.assertEqual(counts[klass], 4, f"train class {klass} unbalanced")

    def test_class_balance_eval(self):
        counts = Counter(e.label for e in self.task.eval_set)
        for klass in lawbench_classes():
            self.assertEqual(counts[klass], 2, f"eval class {klass} unbalanced")

    def test_no_train_eval_leakage(self):
        train_inputs = {e.input for e in self.task.train}
        for ev in self.task.eval_set:
            self.assertNotIn(ev.input, train_inputs)


class TestLawbenchTaskMissing(unittest.TestCase):
    """Real LawBench data isn't bundled — loader must raise FileNotFoundError
    with a helpful message when the user hasn't run the download."""

    def test_missing_files_raise(self):
        with self.assertRaises(FileNotFoundError) as cm:
            build_lawbench_task(subtask="this-subtask-does-not-exist")
        # Error should reference download instructions.
        self.assertIn("lawbench.py docstring", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
