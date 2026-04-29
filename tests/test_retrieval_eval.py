"""Tests for harness.eval's retrieval-recall path.

The recall eval is the only place V10 reads the gold ``patch`` field.
These tests use synthetic patches and instance lists to exercise the
diff parser + scoring without depending on the real dataset's content.
"""

from __future__ import annotations

import pathlib

import pytest

from harness.eval import (
    RetrievalRecallEntry,
    RetrievalRecallReport,
    _gold_touched_files_from_patch,
    _normalize_path,
    evaluate_retrieval_recall,
)


# ---------------------------------------------------------------------------
# Diff parser
# ---------------------------------------------------------------------------


def test_gold_touched_files_extracts_paths_from_diff() -> None:
    patch = """diff --git a/django/contrib/auth/forms.py b/django/contrib/auth/forms.py
index abc..def 100644
--- a/django/contrib/auth/forms.py
+++ b/django/contrib/auth/forms.py
@@ -10,3 +10,5 @@
 some change
diff --git a/django/contrib/auth/models.py b/django/contrib/auth/models.py
index 111..222 100644
--- a/django/contrib/auth/models.py
+++ b/django/contrib/auth/models.py
@@ -20,3 +20,5 @@
 another change
"""
    files = _gold_touched_files_from_patch(patch)
    assert files == {
        "django/contrib/auth/forms.py",
        "django/contrib/auth/models.py",
    }


def test_gold_touched_files_handles_empty_patch() -> None:
    assert _gold_touched_files_from_patch("") == set()
    assert _gold_touched_files_from_patch(None or "") == set()


def test_gold_touched_files_skips_comment_lines_with_diff() -> None:
    """Only lines starting with ``diff --git`` count, not casual mentions."""
    patch = "this string mentions diff --git in prose, not a real diff line"
    files = _gold_touched_files_from_patch(patch)
    assert files == set()


# ---------------------------------------------------------------------------
# Path normalization
# ---------------------------------------------------------------------------


def test_normalize_path_strips_leading_dot_slash() -> None:
    assert _normalize_path("./foo/bar.py") == "foo/bar.py"
    assert _normalize_path("/foo/bar.py") == "foo/bar.py"
    assert _normalize_path("foo/bar.py") == "foo/bar.py"


# ---------------------------------------------------------------------------
# Recall report integration (uses the real dataset to read gold)
# ---------------------------------------------------------------------------


def test_evaluate_retrieval_recall_real_dataset_one_instance() -> None:
    """Run the real eval against the real dataset on a single
    well-known instance. psf__requests-2317's gold patch modifies
    ``requests/sessions.py``; we feed the eval a top-1 that includes
    it and verify all three Ks register a hit.
    """
    iid = "psf__requests-2317"
    retrieved = {
        iid: ["requests/sessions.py", "test_requests.py", "requests/auth.py"],
    }
    report = evaluate_retrieval_recall(retrieved_files_per_instance=retrieved)
    assert report.n_instances == 1
    entry = report.per_instance[0]
    assert "requests/sessions.py" in entry.gold_files
    assert report.n_top1 == 1
    assert report.n_top5 == 1
    assert report.n_top10 == 1


def test_evaluate_retrieval_recall_misses_record_correctly() -> None:
    """Feed a retrieval that doesn't include the gold; verify all
    three Ks register a miss."""
    iid = "psf__requests-2317"
    retrieved = {
        iid: ["unrelated.py", "also_wrong.py"],
    }
    report = evaluate_retrieval_recall(retrieved_files_per_instance=retrieved)
    assert report.n_top1 == 0
    assert report.n_top5 == 0
    assert report.n_top10 == 0


def test_evaluate_retrieval_recall_top5_distinct_from_top1() -> None:
    """If the gold is at rank 3, top-1 misses but top-5/top-10 hit."""
    iid = "psf__requests-2317"
    retrieved = {
        iid: ["wrong1.py", "wrong2.py", "requests/sessions.py", "wrong4.py"],
    }
    report = evaluate_retrieval_recall(retrieved_files_per_instance=retrieved)
    assert report.n_top1 == 0
    assert report.n_top5 == 1
    assert report.n_top10 == 1
