"""Deterministic tests for the SWE-bench adapter (no network, no Docker)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from meta_harness_plus.swebench_adapter import (
    SWEBenchInstance,
    build_user_prompt,
    extract_patch,
    load_swebench_verified,
    write_predictions,
)


SAMPLE_DIFF = (
    "diff --git a/foo.py b/foo.py\n"
    "--- a/foo.py\n"
    "+++ b/foo.py\n"
    "@@ -1,3 +1,3 @@\n"
    " def add(a, b):\n"
    "-    return a - b\n"
    "+    return a + b\n"
)


class TestExtractPatch(unittest.TestCase):
    def test_plain_diff(self):
        out = extract_patch(SAMPLE_DIFF)
        self.assertTrue(out.startswith("diff --git "))
        self.assertIn("return a + b", out)

    def test_fenced_diff(self):
        wrapped = f"```diff\n{SAMPLE_DIFF}```"
        out = extract_patch(wrapped)
        self.assertTrue(out.startswith("diff --git "))

    def test_plan_then_patch(self):
        wrapped = (
            "I will fix the bug in foo.py by swapping the operator.\n"
            "===PATCH===\n"
            f"{SAMPLE_DIFF}"
        )
        out = extract_patch(wrapped)
        self.assertTrue(out.startswith("diff --git "))
        self.assertNotIn("I will fix", out)

    def test_chatty_with_explanation_after(self):
        # If the model puts text after the diff, we keep it (harmless;
        # git apply tolerates trailing whitespace, anything after a
        # diff that isn't another `diff --git` line just becomes a
        # final hunk's last line).
        wrapped = SAMPLE_DIFF + "\nThis fixes the bug."
        out = extract_patch(wrapped)
        self.assertIn("return a + b", out)

    def test_no_diff_returns_empty(self):
        self.assertEqual(extract_patch("nothing here"), "")
        self.assertEqual(extract_patch(""), "")
        self.assertEqual(extract_patch(None), "")


class TestBuildUserPrompt(unittest.TestCase):
    def test_includes_repo_and_problem(self):
        inst = SWEBenchInstance(
            instance_id="x__y-1", repo="x/y", base_commit="abcdef",
            problem_statement="something is broken",
            hints_text="", test_patch="",
            fail_to_pass=[], pass_to_pass=[],
            version="1.0", gold_patch="",
        )
        prompt = build_user_prompt(inst)
        self.assertIn("x/y", prompt)
        self.assertIn("abcdef", prompt)
        self.assertIn("something is broken", prompt)

    def test_hints_included_by_default(self):
        inst = SWEBenchInstance(
            instance_id="x__y-1", repo="x/y", base_commit="a",
            problem_statement="bug", hints_text="hint here",
            test_patch="", fail_to_pass=[], pass_to_pass=[],
            version="1.0", gold_patch="",
        )
        self.assertIn("hint here", build_user_prompt(inst))
        self.assertNotIn("hint here",
                         build_user_prompt(inst, include_hints=False))


class TestPredictionsFile(unittest.TestCase):
    def test_write_predictions_jsonl_format(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "preds.jsonl"
            write_predictions(
                {"foo__bar-1": SAMPLE_DIFF, "foo__bar-2": ""},
                model_name="claude-sonnet-4-6",
                out_path=p,
            )
            lines = p.read_text().splitlines()
            self.assertEqual(len(lines), 2)
            r0 = json.loads(lines[0])
            self.assertEqual(set(r0.keys()),
                             {"instance_id", "model_name_or_path", "model_patch"})
            self.assertEqual(r0["model_name_or_path"], "claude-sonnet-4-6")


class TestLoaderUsesCache(unittest.TestCase):
    def test_loader_reads_cached_jsonl_without_network(self):
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td) / "fake_swebench.jsonl"
            cache.write_text(json.dumps({
                "instance_id": "x__y-1", "repo": "x/y",
                "base_commit": "abc", "problem_statement": "bug",
                "hints_text": "", "test_patch": "",
                "FAIL_TO_PASS": '["test_a"]', "PASS_TO_PASS": "[]",
                "version": "1.0", "patch": "",
            }) + "\n")
            insts = load_swebench_verified(cached_path=cache)
            self.assertEqual(len(insts), 1)
            self.assertEqual(insts[0].instance_id, "x__y-1")
            self.assertEqual(insts[0].fail_to_pass, ["test_a"])


if __name__ == "__main__":
    unittest.main()
