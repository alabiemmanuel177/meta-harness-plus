"""Phase 4 — diff_stats unit tests.

Pure-function tests; no sandbox or LLM."""

from __future__ import annotations

from harness.validation.diff_stats import (
    compute_ast_normalized_hash,
    compute_diff_stats,
)
from harness.views import DiffStats


_SAMPLE_DIFF = """\
diff --git a/src/foo.py b/src/foo.py
index abc..def 100644
--- a/src/foo.py
+++ b/src/foo.py
@@ -1,3 +1,4 @@
 def foo(x):
-    return None
+    if x is None:
+        raise ValueError("foo got None")
+    return x
"""


def test_compute_diff_stats_files_touched():
    s = compute_diff_stats(_SAMPLE_DIFF)
    assert s.files_touched == ("src/foo.py",)


def test_compute_diff_stats_additions_deletions():
    s = compute_diff_stats(_SAMPLE_DIFF)
    assert s.additions == 3
    assert s.deletions == 1


def test_compute_diff_stats_returns_DiffStats_instance():
    s = compute_diff_stats(_SAMPLE_DIFF)
    assert isinstance(s, DiffStats)
    assert isinstance(s.ast_normalized_hash, str)
    assert len(s.ast_normalized_hash) > 0


def test_compute_ast_normalized_hash_stable_under_whitespace():
    """Two diffs that differ only in whitespace within the additions
    should produce the same AST hash."""
    diff_a = _SAMPLE_DIFF
    diff_b = _SAMPLE_DIFF.replace("    if x is None:", "    if x is None :")
    h_a = compute_ast_normalized_hash(diff_a)
    h_b = compute_ast_normalized_hash(diff_b)
    assert h_a == h_b


def test_compute_ast_normalized_hash_distinguishes_different_logic():
    diff_a = _SAMPLE_DIFF
    diff_b = _SAMPLE_DIFF.replace("raise ValueError", "raise TypeError")
    assert compute_ast_normalized_hash(diff_a) != compute_ast_normalized_hash(diff_b)


def test_compute_diff_stats_multifile():
    multi = """\
diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1 +1 @@
-x
+y
diff --git a/b.py b/b.py
--- a/b.py
+++ b/b.py
@@ -1 +1 @@
-z
+w
"""
    s = compute_diff_stats(multi)
    assert s.files_touched == ("a.py", "b.py")
    assert s.additions == 2
    assert s.deletions == 2


def test_compute_diff_stats_empty_diff_yields_empty_files():
    s = compute_diff_stats("")
    assert s.files_touched == ()
    assert s.additions == 0
    assert s.deletions == 0


def test_ast_hash_falls_back_on_unparseable_addition_lines():
    """If the addition lines aren't valid Python (common for malformed
    patches), fall back to the whitespace-collapsed RAW hash. Should
    not raise."""
    bad = """\
diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1 +1 @@
-x
+def (((((
"""
    h = compute_ast_normalized_hash(bad)
    assert isinstance(h, str)
    assert len(h) > 0
