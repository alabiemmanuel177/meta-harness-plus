"""Tests for harness.bm25 — code-aware tokenizer + per-file BM25 index.

The synthetic-corpus tests are diagnostic: each one targets a
tokenization rule (snake_case, CamelCase, dotted access, alpha-digit)
where a naive tokenizer would give the wrong answer.
"""

from __future__ import annotations

import pytest

from harness.bm25 import (
    BM25IndexEntry,
    FileBM25Index,
    build_index_from_file_contents,
    tokenize,
)


# ---------------------------------------------------------------------------
# Tokenizer tests — each rule isolated
# ---------------------------------------------------------------------------


def test_tokenize_emits_compound_and_parts_for_snake_case() -> None:
    toks = tokenize("compute_score")
    assert "compute_score" in toks
    assert "compute" in toks
    assert "score" in toks


def test_tokenize_splits_camelcase_into_parts() -> None:
    toks = tokenize("UserCreationForm")
    # Compound preserved (lowercased)
    assert "usercreationform" in toks
    # Parts emitted
    assert "user" in toks
    assert "creation" in toks
    assert "form" in toks


def test_tokenize_splits_dotted_access() -> None:
    toks = tokenize("django.db.models.Manager")
    # Compound preserved
    assert "django.db.models.manager" in toks
    # Parts
    assert "django" in toks
    assert "db" in toks
    assert "models" in toks
    assert "manager" in toks


def test_tokenize_splits_alpha_digit_boundaries() -> None:
    toks = tokenize("http2_proxy")
    # Compound preserved
    assert "http2_proxy" in toks
    # Parts after _ split + alpha/digit split
    assert "http" in toks
    # 'http2' is also valid (split on '_' but not on 2 since 'http2' is already a part)
    # Single-char tokens dropped, so '2' won't appear


def test_tokenize_lowercases_all_output() -> None:
    toks = tokenize("FooBar AnotherClass")
    for t in toks:
        assert t == t.lower()


def test_tokenize_drops_single_chars_and_punct_only() -> None:
    toks = tokenize("a , b ; ()")
    # 'a' and 'b' are single-char; punct-only drops too
    assert "a" not in toks
    assert "b" not in toks
    assert "()" not in toks
    assert "," not in toks


def test_tokenize_empty() -> None:
    assert tokenize("") == []
    assert tokenize("   ") == []


# ---------------------------------------------------------------------------
# Synthetic-corpus tests — verify the right answer requires this
# tokenization, not a naive whitespace-only one.
# ---------------------------------------------------------------------------


def test_synthetic_corpus_query_matches_compound_via_part() -> None:
    """Query for a PART of a compound identifier should match the
    file containing the compound — naive whitespace tokenization
    would miss this."""
    files = [
        ("a.py", "def compute_score(x): return x * 2"),
        ("b.py", "def helper(x): return x + 1"),
        ("c.py", "import os; print('hello world')"),
    ]
    idx = build_index_from_file_contents(
        [f for f, _ in files], [c for _, c in files],
    )
    # Query "score" — should rank a.py top because compute_score → ..., score
    hits = idx.query(tokenize("score"), top_k=3)
    assert len(hits) >= 1
    assert hits[0][0] == "a.py", f"expected a.py first, got {hits!r}"


def test_synthetic_corpus_camelcase_part_match() -> None:
    """Query for a part of a CamelCase symbol should match the file."""
    files = [
        ("a.py", "class UserCreationForm: pass"),
        ("b.py", "class OtherForm: pass"),
        ("c.py", "import os"),
    ]
    idx = build_index_from_file_contents(
        [f for f, _ in files], [c for _, c in files],
    )
    # Query "creation" — should match a.py (UserCreationForm → ..., creation)
    hits = idx.query(tokenize("creation"), top_k=3)
    assert hits[0][0] == "a.py"


def test_synthetic_corpus_dotted_access_match() -> None:
    files = [
        ("a.py", "from django.db.models import Manager"),
        ("b.py", "from collections import OrderedDict"),
        ("c.py", "import os"),
    ]
    idx = build_index_from_file_contents(
        [f for f, _ in files], [c for _, c in files],
    )
    # Query "django" — should match a.py
    hits = idx.query(tokenize("django"), top_k=3)
    assert hits[0][0] == "a.py"


def test_synthetic_corpus_compound_query_match() -> None:
    """Query for the WHOLE compound should also match — both
    compound and parts are indexed."""
    files = [
        ("a.py", "def compute_score(x): return x"),
        ("b.py", "def compute(y): return y * 2"),
    ]
    idx = build_index_from_file_contents(
        [f for f, _ in files], [c for _, c in files],
    )
    # Query "compute_score" — should rank a.py FIRST (exact compound match)
    # over b.py (which has only "compute").
    hits = idx.query(tokenize("compute_score"), top_k=2)
    assert hits[0][0] == "a.py", f"expected a.py first for compound query, got {hits!r}"


# ---------------------------------------------------------------------------
# Index API
# ---------------------------------------------------------------------------


def test_index_query_returns_ranked_top_k() -> None:
    files = [
        ("a.py", "score score score"),
        ("b.py", "score score"),
        ("c.py", "score"),
        ("d.py", "unrelated content here"),
    ]
    idx = build_index_from_file_contents(
        [f for f, _ in files], [c for _, c in files],
    )
    hits = idx.query(["score"], top_k=3)
    assert len(hits) == 3
    paths = [h[0] for h in hits]
    assert paths == ["a.py", "b.py", "c.py"]
    # Ranks are 1-indexed
    assert [h[2] for h in hits] == [1, 2, 3]


def test_index_query_empty_query_returns_empty() -> None:
    files = [("a.py", "anything")]
    idx = build_index_from_file_contents(
        [f for f, _ in files], [c for _, c in files],
    )
    assert idx.query([]) == []


def test_build_index_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        build_index_from_file_contents(["a.py", "b.py"], ["x"])


def test_index_rejects_empty_corpus() -> None:
    with pytest.raises(ValueError):
        FileBM25Index([])
