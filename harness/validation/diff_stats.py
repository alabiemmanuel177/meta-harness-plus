"""Phase 4 — diff_stats + AST-normalized hash.

Per docs/V10_DESIGN_PHASE4.md §3.2 step 2 and step 6. Pure functions
operating on the diff text alone — no sandbox access, no LLM calls.
"""

from __future__ import annotations

import hashlib
import re
import tokenize
from io import StringIO

from harness.views import DiffStats


_DIFF_GIT_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)$", re.MULTILINE)
_HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@")


def _files_touched(diff: str) -> tuple[str, ...]:
    """Parse the ``diff --git a/<path> b/<path>`` headers to get the
    set of files the patch modifies."""
    files: list[str] = []
    seen: set[str] = set()
    for m in _DIFF_GIT_RE.finditer(diff):
        path = m.group(2)
        if path not in seen:
            seen.add(path)
            files.append(path)
    return tuple(files)


def _additions_deletions(diff: str) -> tuple[int, int]:
    """Count `+`/`-` lines in the diff, excluding the file headers
    and hunk headers."""
    additions = 0
    deletions = 0
    for line in diff.splitlines():
        if not line:
            continue
        if line.startswith("+++ ") or line.startswith("--- "):
            continue
        if line.startswith("+"):
            additions += 1
        elif line.startswith("-"):
            deletions += 1
    return additions, deletions


def compute_ast_normalized_hash(diff: str) -> str:
    """Tokenize the diff body, drop whitespace + comments, hash.

    Used by Phase 5 Selector B (cluster-then-vote) for grouping
    semantically-equivalent diffs that differ only in formatting.
    Phase 4 just produces the hash; the selector consumes it.

    On tokenization failure (e.g., a diff that contains invalid
    Python in its `+` lines — common when the patch is malformed),
    fall back to a SHA256 of the whitespace-collapsed diff text.
    """
    body_chars: list[str] = []
    for line in diff.splitlines():
        if not line:
            continue
        if line.startswith("diff --git"):
            continue
        if line.startswith("index "):
            continue
        if line.startswith("+++ ") or line.startswith("--- "):
            continue
        if _HUNK_HEADER_RE.match(line):
            continue
        # Keep `+` and `-` as part of the body — they distinguish the
        # change direction.
        body_chars.append(line.rstrip())
    body = "\n".join(body_chars)

    # Try AST-aware token normalization on the additions only.
    addition_lines = [
        ln[1:] for ln in body_chars if ln.startswith("+") and not ln.startswith("+++")
    ]
    addition_src = "\n".join(addition_lines)

    try:
        tokens: list[str] = []
        for tok in tokenize.generate_tokens(StringIO(addition_src).readline):
            kind, val = tok.type, tok.string
            if kind in (
                tokenize.NEWLINE, tokenize.NL,
                tokenize.INDENT, tokenize.DEDENT,
                tokenize.COMMENT,
                tokenize.ENCODING,
                tokenize.ENDMARKER,
            ):
                continue
            tokens.append(val)
        norm = " ".join(tokens)
        return hashlib.sha256(("AST:" + norm).encode("utf-8")).hexdigest()[:32]
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        # Fallback: whitespace-collapsed diff body
        collapsed = re.sub(r"\s+", " ", body).strip()
        return hashlib.sha256(("RAW:" + collapsed).encode("utf-8")).hexdigest()[:32]


def compute_diff_stats(diff: str) -> DiffStats:
    """Build a DiffStats from the diff text alone."""
    files = _files_touched(diff)
    additions, deletions = _additions_deletions(diff)
    ast_hash = compute_ast_normalized_hash(diff)
    return DiffStats(
        files_touched=files,
        additions=additions,
        deletions=deletions,
        ast_normalized_hash=ast_hash,
    )


__all__ = [
    "compute_ast_normalized_hash",
    "compute_diff_stats",
]
