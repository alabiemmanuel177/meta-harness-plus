"""Tests for the new public-dataset loaders + their synthetic fixtures.

We test the *infrastructure* (loader code paths, schema, class-set
inference) — not real downloaded data, since those require external
network access. The real-data path is exercised via
``scripts/download_extra_public_datasets.py`` and documented in
``RESULTS_FUTURE_WORK.md``.
"""

from __future__ import annotations

import json
import unittest

from meta_harness_plus.tasks import (
    LAWBENCH_CLASSIFICATION_SUBTASKS,
    MASSIVE_TOP8,
    USPTO50K_CLASSES,
    build_lawbench_fixture_task,
    build_lawbench_task,
    build_massive_fixture_task,
    build_uspto_fixture_task,
    list_available_lawbench_subtasks,
)


class TestUSPTOFixture(unittest.TestCase):
    def test_fixture_loads(self):
        task = build_uspto_fixture_task()
        self.assertEqual(task.name, "uspto_fixture")
        # 10 classes × 3 train + 2 eval = 30 + 20.
        self.assertEqual(len(task.train), 30)
        self.assertEqual(len(task.eval_set), 20)
        # Class set matches the declared USPTO-50k taxonomy.
        self.assertEqual(set(task.classes), set(USPTO50K_CLASSES))

    def test_fixture_balanced(self):
        task = build_uspto_fixture_task()
        from collections import Counter
        c = Counter(e.label for e in task.train)
        for cls in USPTO50K_CLASSES:
            self.assertEqual(c[cls], 3, f"{cls} should have 3 train examples")
        c = Counter(e.label for e in task.eval_set)
        for cls in USPTO50K_CLASSES:
            self.assertEqual(c[cls], 2, f"{cls} should have 2 eval examples")

    def test_real_loader_raises_when_files_missing(self):
        from meta_harness_plus.tasks.uspto import build_uspto50k_task, _DATA_DIR
        # Move the real files out of the way, if present.
        train = _DATA_DIR / "uspto50k_train.jsonl"
        if train.exists():
            self.skipTest("real USPTO-50k present — skip negative test")
        with self.assertRaises(FileNotFoundError):
            build_uspto50k_task()


class TestMASSIVEFixture(unittest.TestCase):
    def test_fixture_loads(self):
        task = build_massive_fixture_task()
        self.assertEqual(task.name, "massive_fixture")
        self.assertEqual(set(task.classes), set(MASSIVE_TOP8))
        self.assertEqual(len(task.train), 24)
        self.assertEqual(len(task.eval_set), 16)

    def test_eval_labels_match_class_set(self):
        task = build_massive_fixture_task()
        eval_labels = {e.label for e in task.eval_set}
        self.assertTrue(eval_labels.issubset(set(MASSIVE_TOP8)))

    def test_real_loader_raises_when_files_missing(self):
        from meta_harness_plus.tasks.massive import build_massive_task, _DATA_DIR
        # If real files exist (from a download), skip — we're testing
        # the negative path.
        if (_DATA_DIR / "massive_en-US_train.jsonl").exists():
            self.skipTest("real MASSIVE/en-US present — skip negative test")
        with self.assertRaises(FileNotFoundError):
            build_massive_task(locale="en-US")


class TestLawBenchMultiSubtask(unittest.TestCase):
    def test_fixture_still_loads(self):
        task = build_lawbench_fixture_task()
        self.assertEqual(task.name, "lawbench_fixture")
        self.assertEqual(set(task.classes), {"contract", "criminal", "family", "ip", "tort"})

    def test_list_available_subtasks_excludes_fixture(self):
        subs = list_available_lawbench_subtasks()
        self.assertNotIn("fixture", subs)
        # 2-2 is bundled in this repo.
        # (If somebody removed it, this assertion catches that regression.)
        self.assertIn("2-2", subs)

    def test_classification_subtask_constant_is_stable(self):
        # Sanity: 2-2 is the headline subtask and must be in the constant.
        self.assertIn("2-2", LAWBENCH_CLASSIFICATION_SUBTASKS)
        # Constant doesn't change without a deliberate edit.
        self.assertGreater(len(LAWBENCH_CLASSIFICATION_SUBTASKS), 5)

    def test_unknown_subtask_raises(self):
        with self.assertRaises(FileNotFoundError):
            build_lawbench_task(subtask="9-9-not-real")


class TestDownloadScriptInvariants(unittest.TestCase):
    """We can't run the actual download in CI, but we can import the
    script module and verify its arg parser + helper functions exist.
    """

    def test_module_imports(self):
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "_dl", Path(__file__).parent.parent / "scripts" / "download_extra_public_datasets.py"
        )
        mod = importlib.util.module_from_spec(spec)
        # Defer to avoid datasets dep at import-time during tests.
        try:
            spec.loader.exec_module(mod)
        except Exception as e:
            self.fail(f"download script failed to import: {e}")
        self.assertTrue(callable(mod.download_uspto50k))
        self.assertTrue(callable(mod.download_massive))
        self.assertTrue(callable(mod.download_lawbench_extra))
        self.assertTrue(callable(mod.balance_top_k))


if __name__ == "__main__":
    unittest.main()
