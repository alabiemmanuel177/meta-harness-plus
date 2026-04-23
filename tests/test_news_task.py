"""News-headline classification task: dataset integrity."""

from __future__ import annotations

import unittest
from collections import Counter

from meta_harness_plus.tasks import NEWS_CLASSES, build_news_task


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


if __name__ == "__main__":
    unittest.main()
