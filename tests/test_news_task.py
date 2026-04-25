"""News-headline classification task: dataset integrity."""

from __future__ import annotations

import unittest
from collections import Counter

from meta_harness_plus.tasks import (
    NEWS_CLASSES,
    build_news_hard_50_task,
    build_news_hard_task,
    build_news_task,
)


class TestNewsTask(unittest.TestCase):
    def setUp(self):
        self.task = build_news_task()

    def test_size(self):
        self.assertEqual(len(self.task.train), 40)
        self.assertEqual(len(self.task.eval_set), 20)

    def test_classes_covered(self):
        self.assertEqual(tuple(self.task.classes), NEWS_CLASSES)

    def test_class_balance(self):
        for split_name, items, expected in (
            ("train", self.task.train, 10),
            ("eval", self.task.eval_set, 5),
        ):
            counts = Counter(e.label for e in items)
            for klass in NEWS_CLASSES:
                self.assertEqual(counts[klass], expected,
                                 f"{split_name} class {klass} imbalanced: got {counts[klass]}")

    def test_no_train_eval_leakage(self):
        train_inputs = {e.input for e in self.task.train}
        for ev in self.task.eval_set:
            self.assertNotIn(ev.input, train_inputs)


class TestNewsHardTask(unittest.TestCase):
    def setUp(self):
        self.task = build_news_hard_task()

    def test_size(self):
        self.assertEqual(len(self.task.train), 56)  # 40 easy + 16 disambiguating
        self.assertEqual(len(self.task.eval_set), 15)

    def test_train_per_class_balanced(self):
        counts = Counter(e.label for e in self.task.train)
        for klass in NEWS_CLASSES:
            self.assertEqual(counts[klass], 14)

    def test_no_leakage(self):
        train_inputs = {e.input for e in self.task.train}
        for ev in self.task.eval_set:
            self.assertNotIn(ev.input, train_inputs)

    def test_known_adversarial_cases_in_eval(self):
        inputs = [e.input for e in self.task.eval_set]
        # Tennis athlete's investment — labeled sports, not business
        self.assertTrue(any("pickleball" in i for i in inputs))
        # Quantum IPO — labeled business, not tech
        self.assertTrue(any("Quantum" in i and "IPO" in i for i in inputs))


class TestNewsHard50Task(unittest.TestCase):
    """50-item bigger-eval variant — same train, expanded adversarial eval."""

    def setUp(self):
        self.task = build_news_hard_50_task()

    def test_size(self):
        self.assertEqual(len(self.task.train), 56)
        self.assertEqual(len(self.task.eval_set), 50)

    def test_eval_class_balance_within_one(self):
        """50 items / 4 classes; expect 12-13 per class (13/12/13/12)."""
        counts = Counter(e.label for e in self.task.eval_set)
        for klass in NEWS_CLASSES:
            self.assertGreaterEqual(counts[klass], 12,
                                    f"{klass} underrepresented: {counts[klass]}")
            self.assertLessEqual(counts[klass], 13,
                                 f"{klass} overrepresented: {counts[klass]}")

    def test_no_train_eval_leakage(self):
        train_inputs = {e.input for e in self.task.train}
        for ev in self.task.eval_set:
            self.assertNotIn(ev.input, train_inputs)

    def test_includes_15item_eval_as_subset(self):
        """The 50-item eval extends the original 15-item eval — so prior
        results stay comparable when restricted to the original subset."""
        original = build_news_hard_task()
        original_inputs = {e.input for e in original.eval_set}
        new_inputs = {e.input for e in self.task.eval_set}
        # All original items should be present in the bigger eval.
        missing = original_inputs - new_inputs
        self.assertEqual(missing, set(),
                         f"15-item eval items missing from 50-item: {missing}")


if __name__ == "__main__":
    unittest.main()
