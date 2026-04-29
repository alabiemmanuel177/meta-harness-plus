"""Code-aware BM25 file-level retrieval for Phase 1 stage 1b.

The tokenizer is the load-bearing piece. SWE-bench issues frequently
mention identifiers in their original code form (``compute_score``,
``UserCreationForm``, ``django.db.models.Manager``); a naive
whitespace-only tokenizer would index ``compute_score`` as a single
opaque token and miss queries for ``score`` alone. A naive whitespace+
snake-case tokenizer would index only the parts and miss queries for
the original compound form (which appears verbatim in the issue text).

We index BOTH: the compound token AND its parts. Concretely the
tokenizer:

  1. Splits on whitespace.
  2. For each whitespace-delimited token, ALSO splits on:
     - ``_``  (snake_case)
     - ``.``  (dotted access: ``foo.bar.baz``)
     - lowercase→uppercase boundary (CamelCase: ``FooBar`` → ``Foo``, ``Bar``)
     - alpha→digit boundary (``http2`` → ``http``, ``2``)
  3. Emits BOTH the original compound token AND each part, all
     lowercased.
  4. Drops single-character tokens and pure-punctuation tokens.

Examples (from synthetic corpus tests):

  ``compute_score``      → ``compute_score``, ``compute``, ``score``
  ``UserCreationForm``   → ``usercreationform``, ``user``, ``creation``, ``form``
  ``module.submodule.f`` → ``module.submodule.f``, ``module``, ``submodule`` (single ``f`` dropped)
  ``http2_proxy``        → ``http2_proxy``, ``http2``, ``proxy``, ``http``, ``2`` (then ``2`` dropped as single char)

This mirrors how a human reader of code searches for symbols: they
might type the compound form they remember, or just one part of it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from rank_bm25 import BM25Okapi


# Pre-compiled token splitters.
_WHITESPACE_SPLIT = re.compile(r"\s+")
# After whitespace split, we re-split on these intra-word boundaries:
#   _  .  ,  ;  :  (  )  [  ]  {  }  '  "  /  \  -
# Plus camelCase and alpha-digit transitions handled separately below.
_INTRA_TOKEN_PUNCT = re.compile(r"[_.,;:()\[\]{}\"'/\\\-]+")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z])(?=[A-Z])")
_ALPHA_DIGIT_BOUNDARY = re.compile(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])")


def tokenize(text: str) -> list[str]:
    """Code-aware tokenization. See module docstring for the rules.

    Returns a list of lowercased tokens. Empty string → empty list.
    Order is preserved (BM25 doesn't depend on order, but downstream
    debug logs benefit from stable ordering).
    """
    if not text:
        return []
    out: list[str] = []
    for raw in _WHITESPACE_SPLIT.split(text.strip()):
        if not raw:
            continue
        # Compound token (lowercased).
        compound = raw.lower()
        # Strip pure-punctuation compounds.
        if not any(c.isalnum() for c in compound):
            continue
        # Drop single-character compounds (we treat 'a', 'b', etc. as
        # noise, same as we do for sub-parts).
        if len(compound) < 2:
            continue
        out.append(compound)

        # Split on intra-token punct.
        parts: list[str] = [p for p in _INTRA_TOKEN_PUNCT.split(raw) if p]
        # Further split each part on CamelCase + alpha/digit boundaries.
        more_parts: list[str] = []
        for p in parts:
            tmp = _CAMEL_BOUNDARY.split(p)
            for q in tmp:
                more_parts.extend(_ALPHA_DIGIT_BOUNDARY.split(q))

        # Emit each lowercased sub-part if it differs from the compound
        # and is at least 2 characters of alphanumerics.
        for sp in more_parts:
            sp_low = sp.lower()
            if sp_low == compound:
                continue
            if len(sp_low) < 2:
                continue
            if not any(c.isalnum() for c in sp_low):
                continue
            out.append(sp_low)
    return out


# ---------------------------------------------------------------------------
# Per-file BM25 index over a repo's Python source.
# ---------------------------------------------------------------------------


@dataclass
class BM25IndexEntry:
    file_path: str
    tokens: list[str]


class FileBM25Index:
    """BM25 index over per-file token lists. The corpus is the set of
    Python files at base_commit; the query is the issue text (or a
    derivation — see retrieval.QUERY_STRATEGIES)."""

    def __init__(self, entries: list[BM25IndexEntry]):
        if not entries:
            raise ValueError("FileBM25Index requires at least one entry")
        self._entries = entries
        self._corpus_tokens = [e.tokens for e in entries]
        self._bm25 = BM25Okapi(self._corpus_tokens)

    @property
    def n_files(self) -> int:
        return len(self._entries)

    def query(self, query_tokens: Iterable[str], top_k: int = 30) -> list[tuple[str, float, int]]:
        """Return the top-K (file_path, score, rank) tuples for the
        given query_tokens. rank is 1-indexed."""
        toks = list(query_tokens)
        if not toks:
            return []
        scores = self._bm25.get_scores(toks)
        # Sort descending by score; tie-break by file_path for
        # reproducibility.
        ranked = sorted(
            range(len(self._entries)),
            key=lambda i: (-scores[i], self._entries[i].file_path),
        )
        out: list[tuple[str, float, int]] = []
        for rank, i in enumerate(ranked[:top_k], start=1):
            out.append((self._entries[i].file_path, float(scores[i]), rank))
        return out


def build_index_from_file_contents(
    file_paths: list[str],
    file_contents: list[str],
) -> FileBM25Index:
    """Build a per-file BM25 index from parallel lists of file paths
    and contents. Tokenization runs per file via ``tokenize``.
    """
    if len(file_paths) != len(file_contents):
        raise ValueError(
            f"file_paths and file_contents length mismatch: "
            f"{len(file_paths)} vs {len(file_contents)}"
        )
    entries: list[BM25IndexEntry] = []
    for path, content in zip(file_paths, file_contents):
        tokens = tokenize(content)
        entries.append(BM25IndexEntry(file_path=path, tokens=tokens))
    return FileBM25Index(entries)


__all__ = [
    "tokenize",
    "BM25IndexEntry",
    "FileBM25Index",
    "build_index_from_file_contents",
]
