"""Stage 1b file-level retrieval: code-aware BM25 + local embedding +
three query strategies.

Per V10_DESIGN.md §3.2 stage 1b. Three query strategies
(``full_issue``, ``first_paragraph``, ``extracted_symbols``) run in
parallel; each emits BM25Hit signals tagged with the strategy that
produced them. Embedding retrieval runs once on the issue (full text)
against a BM25-pre-shortlisted set of files, to keep cost bounded on
django/sympy-class repos.

Design choices:

  - File content is read once per sandbox session via ``sb.run_shell``
    + a streaming dump. This avoids per-file ``read_file`` round-trips
    that scale poorly on 2700-file django repos.

  - BM25 runs on every file in the corpus. The code-aware tokenizer
    (``harness.bm25.tokenize``) handles the snake/camel/dot case.

  - Embedding runs on the BM25 top-200 shortlist to bound cost. On
    bge-large-en-v1.5 CPU, embedding ~200 files is order-of-seconds
    per instance; embedding 2700 files would be order-of-minutes.

  - Aggregation is per-file: a candidate is the union of files named
    by ANY of {3 BM25 strategies, embedding}, with per-signal best
    rank tracked. Used by Stage 1g rerank.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from harness.bm25 import build_index_from_file_contents, tokenize
from harness.localization_signals import (
    BM25Hit,
    EmbeddingHit,
    QUERY_STRATEGIES,
    STAGE_1B_FILE_CANDIDATES,
)
from harness.views import InstanceView


BM25_TOP_K_PER_QUERY = 30
EMBEDDING_TOP_K = 30
EMBEDDING_SHORTLIST_FROM_BM25 = 200  # cap on files passed to embedder


# ---------------------------------------------------------------------------
# Query-strategy derivations
# ---------------------------------------------------------------------------


def _query_full_issue(issue_text: str) -> str:
    return issue_text.strip()


def _query_first_paragraph(issue_text: str) -> str:
    """First paragraph = up to the first blank line, else first 600 chars."""
    text = issue_text.strip()
    parts = re.split(r"\n\s*\n", text, maxsplit=1)
    para = parts[0] if parts else text
    if len(para) > 600:
        para = para[:600]
    return para


# Identifier extraction patterns
_CAMEL_IDENTIFIER = re.compile(r"\b[A-Z][a-zA-Z0-9_]{2,}\b")
_SNAKE_IDENTIFIER = re.compile(r"\b[a-z_][a-z0-9_]{3,}_[a-z_][a-z0-9_]*\b")
_DOTTED_REFERENCE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]+(?:\.[a-zA-Z_][a-zA-Z0-9_]+)+\b")
_BACKTICK_CODE = re.compile(r"`+([^`\n]+?)`+")
_QUOTED_CODE = re.compile(r"['\"]([A-Za-z_][A-Za-z0-9_./]+)['\"]")

# Common English words that look like identifiers but aren't diagnostic.
_STOPWORD_IDENTIFIERS = frozenset({
    "True", "False", "None", "Error", "Exception", "Test",
    "Should", "When", "Then", "Given", "Following",
})


def _extracted_symbols(issue_text: str) -> str:
    """Pull out identifier-shaped tokens, dotted references, and
    backtick / quoted code from the issue. Returns a space-separated
    string of unique candidates (preserves first-occurrence order)."""
    seen: set[str] = set()
    out: list[str] = []

    def _add(s: str) -> None:
        s = s.strip()
        if not s or s in seen or s in _STOPWORD_IDENTIFIERS:
            return
        seen.add(s)
        out.append(s)

    for m in _BACKTICK_CODE.finditer(issue_text):
        _add(m.group(1))
    for m in _QUOTED_CODE.finditer(issue_text):
        _add(m.group(1))
    for m in _DOTTED_REFERENCE.finditer(issue_text):
        _add(m.group(0))
    for m in _CAMEL_IDENTIFIER.finditer(issue_text):
        _add(m.group(0))
    for m in _SNAKE_IDENTIFIER.finditer(issue_text):
        _add(m.group(0))
    return " ".join(out)


def query_for_strategy(strategy: str, issue_text: str) -> str:
    if strategy == "full_issue":
        return _query_full_issue(issue_text)
    if strategy == "first_paragraph":
        return _query_first_paragraph(issue_text)
    if strategy == "extracted_symbols":
        return _extracted_symbols(issue_text)
    raise ValueError(f"unknown query strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# In-container file dump
# ---------------------------------------------------------------------------


_FILE_DUMP_SCRIPT = r"""
import json, pathlib, sys

root = pathlib.Path('/testbed')
EXCLUDED = {'.git', '__pycache__', '.tox', '.eggs', 'build', 'dist', '.venv'}

def is_excluded(path):
    parts = set(path.parts)
    if any(p in parts for p in EXCLUDED):
        return True
    return any(p.startswith('.') and p not in {'.', '..'} for p in parts)

# Output: list of {path, content}.
out = []
for path in sorted(root.rglob('*.py')):
    if is_excluded(path):
        continue
    try:
        text = path.read_text(errors='replace')
    except Exception:
        continue
    rel = str(path.relative_to(root))
    out.append({'path': rel, 'content': text})
print(json.dumps(out))
"""


def _read_repo_files(sandbox) -> tuple[list[str], list[str]]:
    """Read every .py file in /testbed via a single in-container script.
    Returns (paths, contents) — parallel lists, sorted by path.
    """
    res = sandbox.run_shell(
        "python3 -c " + _shell_quote(_FILE_DUMP_SCRIPT),
        timeout_s=240,
    )
    if res.exit_code != 0:
        raise RuntimeError(
            f"file dump probe failed: stderr={(res.stderr or '')[:400]}"
        )
    rows = __import__("json").loads(res.stdout)
    paths = [r["path"] for r in rows]
    contents = [r["content"] for r in rows]
    return paths, contents


def _shell_quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class _CandidateAccumulator:
    file_path: str
    upstream_signals: set[str] = field(default_factory=set)
    upstream_best_rank: dict[str, int] = field(default_factory=dict)
    bm25_max_score: float = 0.0
    embedding_max_sim: float = 0.0


@dataclass(frozen=True)
class RetrievalResult:
    candidates: tuple                  # tuple of dicts (file_path, upstream_signals, upstream_best_rank, bm25_max_score, embedding_max_sim)
    bm25_hits_per_strategy: dict       # {strategy: list[BM25Hit]}
    embedding_hits: list               # list[EmbeddingHit]
    n_files_indexed: int


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def run_stage_1b_retrieval(
    view: InstanceView,
    sandbox,
    *,
    embedder=None,
    bm25_top_k: int = BM25_TOP_K_PER_QUERY,
    embedding_top_k: int = EMBEDDING_TOP_K,
    trajectory_writer=None,
) -> RetrievalResult:
    """Run BM25 across all 3 query strategies + embedding (if embedder
    provided) and aggregate candidates with per-signal attribution.

    Returns a ``RetrievalResult`` carrying the typed signal lists. If
    ``trajectory_writer`` is given, signals are also written to the
    trajectory log via ``write_signal``.
    """
    paths, contents = _read_repo_files(sandbox)
    if not paths:
        return RetrievalResult(
            candidates=(), bm25_hits_per_strategy={}, embedding_hits=[],
            n_files_indexed=0,
        )

    bm25_index = build_index_from_file_contents(paths, contents)

    candidates: dict[str, _CandidateAccumulator] = {}

    def _bump_signal(file_path: str, signal_kind: str, rank: int, score: float) -> None:
        acc = candidates.get(file_path)
        if acc is None:
            acc = _CandidateAccumulator(file_path=file_path)
            candidates[file_path] = acc
        acc.upstream_signals.add(signal_kind)
        prev_rank = acc.upstream_best_rank.get(signal_kind)
        if prev_rank is None or rank < prev_rank:
            acc.upstream_best_rank[signal_kind] = rank
        if signal_kind == "bm25":
            acc.bm25_max_score = max(acc.bm25_max_score, score)
        elif signal_kind == "embedding":
            acc.embedding_max_sim = max(acc.embedding_max_sim, score)

    # BM25 across the 3 strategies
    bm25_hits_per_strategy: dict[str, list[BM25Hit]] = {}
    for strategy in QUERY_STRATEGIES:
        query_text = query_for_strategy(strategy, view.problem_statement)
        if not query_text:
            bm25_hits_per_strategy[strategy] = []
            continue
        query_tokens = tokenize(query_text)
        if not query_tokens:
            bm25_hits_per_strategy[strategy] = []
            continue
        hits = bm25_index.query(query_tokens, top_k=bm25_top_k)
        signals: list[BM25Hit] = []
        for file_path, score, rank in hits:
            sig = BM25Hit(
                stage=STAGE_1B_FILE_CANDIDATES,
                file_path=file_path,
                score=score,
                rank=rank,
                query_basis="issue_text",
                query_strategy=strategy,
            )
            signals.append(sig)
            if trajectory_writer is not None:
                trajectory_writer.write_signal(sig)
            _bump_signal(file_path, "bm25", rank, score)
        bm25_hits_per_strategy[strategy] = signals

    # Embedding retrieval on a BM25-pre-shortlisted subset (cost guard
    # for big-repo instances).
    embedding_hits: list[EmbeddingHit] = []
    if embedder is not None:
        # Build the shortlist from union of BM25 top-N across strategies.
        shortlist_paths: list[str] = []
        seen: set[str] = set()
        for strategy_hits in bm25_hits_per_strategy.values():
            for sig in strategy_hits[:EMBEDDING_SHORTLIST_FROM_BM25 // 3]:
                if sig.file_path not in seen:
                    seen.add(sig.file_path)
                    shortlist_paths.append(sig.file_path)
                if len(shortlist_paths) >= EMBEDDING_SHORTLIST_FROM_BM25:
                    break
            if len(shortlist_paths) >= EMBEDDING_SHORTLIST_FROM_BM25:
                break
        if shortlist_paths:
            path_to_content = dict(zip(paths, contents))
            shortlist_contents = [path_to_content[p] for p in shortlist_paths]
            from harness.embedding import retrieve_by_embedding
            results = retrieve_by_embedding(
                embedder,
                file_paths=shortlist_paths,
                file_contents=shortlist_contents,
                query=view.problem_statement,
                top_k=embedding_top_k,
            )
            for r in results:
                sig = EmbeddingHit(
                    stage=STAGE_1B_FILE_CANDIDATES,
                    file_path=r.file_path,
                    cosine_similarity=r.cosine_similarity,
                    rank=r.rank,
                    model=getattr(embedder, "model_name", "unknown"),
                    chunk_basis="whole_file",
                )
                embedding_hits.append(sig)
                if trajectory_writer is not None:
                    trajectory_writer.write_signal(sig)
                _bump_signal(r.file_path, "embedding", r.rank, r.cosine_similarity)

    # Build the aggregated candidate list.
    aggregated: list[dict] = []
    for fp, acc in candidates.items():
        aggregated.append({
            "file_path": fp,
            "upstream_signals": tuple(sorted(acc.upstream_signals)),
            "upstream_best_rank": dict(acc.upstream_best_rank),
            "bm25_max_score": acc.bm25_max_score,
            "embedding_max_sim": acc.embedding_max_sim,
        })
    # Sort: prefer files named by more upstream signals, then lower
    # best rank across signals, then higher BM25 score.
    aggregated.sort(
        key=lambda c: (
            -len(c["upstream_signals"]),
            min(c["upstream_best_rank"].values()) if c["upstream_best_rank"] else 99,
            -c["bm25_max_score"],
        )
    )

    return RetrievalResult(
        candidates=tuple(aggregated),
        bm25_hits_per_strategy=bm25_hits_per_strategy,
        embedding_hits=embedding_hits,
        n_files_indexed=len(paths),
    )


__all__ = [
    "RetrievalResult",
    "run_stage_1b_retrieval",
    "query_for_strategy",
    "BM25_TOP_K_PER_QUERY",
    "EMBEDDING_TOP_K",
    "EMBEDDING_SHORTLIST_FROM_BM25",
]
