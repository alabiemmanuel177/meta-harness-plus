"""JSONL loader: happy path + validation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from meta_harness_plus.tasks.jsonl_loader import build_task_from_jsonl, load_jsonl


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


class TestLoadJSONL(unittest.TestCase):
    def test_happy_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.jsonl"
            _write(p, [
                {"input": "a", "label": "cat"},
                {"input": "b", "label": "dog"},
            ])
            out = load_jsonl(p)
            self.assertEqual(len(out), 2)
            self.assertEqual(out[0].input, "a")
            self.assertEqual(out[0].label, "cat")

    def test_skips_blank_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.jsonl"
            p.write_text('{"input":"a","label":"x"}\n\n{"input":"b","label":"y"}\n')
            out = load_jsonl(p)
            self.assertEqual(len(out), 2)

    def test_missing_file_raises_filenotfound(self):
        with self.assertRaises(FileNotFoundError):
            load_jsonl("/tmp/definitely-nonexistent-path-xyz.jsonl")

    def test_invalid_json_raises_with_lineno(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.jsonl"
            p.write_text('{"input":"ok","label":"x"}\nnot-json-here\n')
            with self.assertRaises(ValueError) as cm:
                load_jsonl(p)
            self.assertIn(":2:", str(cm.exception))

    def test_missing_required_key_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.jsonl"
            _write(p, [{"input": "a"}])  # no label
            with self.assertRaises(ValueError):
                load_jsonl(p)


class TestBuildTaskFromJSONL(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.train_p = Path(self.tmp) / "train.jsonl"
        self.eval_p = Path(self.tmp) / "eval.jsonl"
        _write(self.train_p, [
            {"input": "one", "label": "a"},
            {"input": "two", "label": "b"},
            {"input": "three", "label": "a"},
        ])
        _write(self.eval_p, [
            {"input": "x", "label": "a"},
            {"input": "y", "label": "b"},
        ])

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_infers_classes_sorted(self):
        t = build_task_from_jsonl("demo", self.train_p, self.eval_p)
        self.assertEqual(t.classes, ["a", "b"])  # sorted

    def test_explicit_classes_respected(self):
        t = build_task_from_jsonl("demo", self.train_p, self.eval_p, classes=["a", "b", "c"])
        self.assertEqual(t.classes, ["a", "b", "c"])  # preserves given order

    def test_extra_label_rejected_when_classes_specified(self):
        _write(self.eval_p, [
            {"input": "x", "label": "a"},
            {"input": "z", "label": "surprise"},
        ])
        with self.assertRaises(ValueError) as cm:
            build_task_from_jsonl("demo", self.train_p, self.eval_p, classes=["a", "b"])
        self.assertIn("surprise", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
