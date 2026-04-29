"""Structured signal types for Phase 1 localization.

Per V10_DESIGN.md §3.2, Phase 1 produces a sequence of localization
signals (skeleton retrieval, BM25 / embedding ranks, traceback frames,
symbol grep, git archaeology, dep-graph expansion, LLM rerank). Each
signal is recorded to the trajectory log as a typed JSON record so the
ablation table can be auto-generated from logs (Phase 1 acceptance
criterion: structured records, not free-form text).

Schema commitments (frozen, validated):

  BM25Hit            — one BM25 retrieval result against the issue query
  EmbeddingHit       — one embedding-similarity hit (text-embedding-3
                       large or open equivalent)
  TracebackFrame     — one stack frame extracted from the issue text
  SymbolGrepHit      — one ripgrep match for an identifier / error
                       string / quoted code snippet
  GitArchaeologyHit  — one commit reference from `git log -S<token>` or
                       `git log -- <file>`
  DepGraphHop        — one edge in the AST/import dep-graph expansion
                       around a localization candidate
  RerankerOutput     — the top-K LLM rerank of aggregated candidates

All signals share these properties:

  - ``frozen=True`` dataclasses; immutable once constructed.
  - ``stage`` field naming the Phase 1 sub-stage that emitted the
    signal (1a-1g per V10_DESIGN.md).
  - ``as_dict()`` returns a JSON-serializable dict with the signal
    class name in ``_signal``, suitable for ``trajectory.write_signal``.
  - Field-name validation against ``FORBIDDEN_TOKENS`` via the same
    mechanism the views use.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Union


from harness.views import FORBIDDEN_TOKENS, ForbiddenFieldError


def _assert_no_forbidden_field_names(cls: type) -> None:
    bad: list[str] = []
    for f in fields(cls):
        folded = f.name.casefold()
        for tok in FORBIDDEN_TOKENS:
            if tok in folded:
                bad.append(f"{cls.__name__}.{f.name} (matches {tok!r})")
    if bad:
        raise ForbiddenFieldError(
            "localization signal carries forbidden field(s): " + ", ".join(bad)
        )


# Phase 1 sub-stage labels matching V10_DESIGN.md §3.2.
LocalizationStage = str
STAGE_1A_SKELETON = "1a_skeleton"
STAGE_1B_FILE_CANDIDATES = "1b_file_candidates"
STAGE_1C_TRACEBACK = "1c_traceback"
STAGE_1D_SYMBOL_GREP = "1d_symbol_grep"
STAGE_1E_GIT_ARCHEOLOGY = "1e_git_archeology"
STAGE_1F_DEP_GRAPH = "1f_dep_graph"
STAGE_1G_RERANK = "1g_rerank"
STAGE_1H_FUNCTION_LINE = "1h_function_line"

ALL_STAGES: tuple[str, ...] = (
    STAGE_1A_SKELETON,
    STAGE_1B_FILE_CANDIDATES,
    STAGE_1C_TRACEBACK,
    STAGE_1D_SYMBOL_GREP,
    STAGE_1E_GIT_ARCHEOLOGY,
    STAGE_1F_DEP_GRAPH,
    STAGE_1G_RERANK,
    STAGE_1H_FUNCTION_LINE,
)


# ---------------------------------------------------------------------------
# Mixin for serialization + field validation. Plain function avoids
# inheritance + frozen-dataclass edge cases.
# ---------------------------------------------------------------------------


def _signal_as_dict(signal) -> dict:
    """Convert a signal dataclass instance to a JSON-friendly dict.

    Adds ``_signal`` key naming the class so trajectory readers can
    dispatch on type without import. ``stage`` field is preserved.
    """
    d = asdict(signal)
    d["_signal"] = type(signal).__name__
    return d


# ---------------------------------------------------------------------------
# 1b — file candidates (BM25 + embedding)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BM25Hit:
    """One BM25 retrieval result against the issue query.

    Phase 1 stage 1b. The query is the issue's tokenized text; the
    document is per-file content. ``query_strategy`` records which of
    the three Stage 1b query shapes this hit came from (full issue /
    first paragraph / extracted symbols), so per-strategy ablations
    can be done from logs alone.
    """

    stage: LocalizationStage
    file_path: str
    score: float
    rank: int                # 1-indexed
    query_basis: str         # 'issue_text' | 'issue_summary' | 'symbol_only' | etc.
    query_strategy: str = "full_issue"  # 'full_issue' | 'first_paragraph' | 'extracted_symbols'

    def __post_init__(self) -> None:
        _assert_no_forbidden_field_names(type(self))
        if self.rank < 1:
            raise ValueError("BM25Hit.rank is 1-indexed")
        if self.query_strategy not in QUERY_STRATEGIES:
            raise ValueError(
                f"BM25Hit.query_strategy invalid: {self.query_strategy!r}; "
                f"must be one of {QUERY_STRATEGIES}"
            )

    def as_dict(self) -> dict:
        return _signal_as_dict(self)


QUERY_STRATEGIES: tuple[str, ...] = (
    "full_issue",
    "first_paragraph",
    "extracted_symbols",
)


@dataclass(frozen=True)
class EmbeddingHit:
    """One embedding-similarity hit.

    Phase 1 stage 1b. ``model`` records which embedding model was used
    so Phase 1 ablations can attribute lift correctly.
    """

    stage: LocalizationStage
    file_path: str
    cosine_similarity: float
    rank: int
    model: str           # 'text-embedding-3-large' | 'bge-large-en' | etc.
    chunk_basis: str     # 'whole_file' | 'function_chunks' | 'header_chunks'

    def __post_init__(self) -> None:
        _assert_no_forbidden_field_names(type(self))
        if self.rank < 1:
            raise ValueError("EmbeddingHit.rank is 1-indexed")

    def as_dict(self) -> dict:
        return _signal_as_dict(self)


# ---------------------------------------------------------------------------
# 1c — traceback parser
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TracebackFrame:
    """One stack frame extracted from the issue text.

    Phase 1 stage 1c. Frames closer to the bottom of the traceback
    (deepest call) get higher prior weight.
    """

    stage: LocalizationStage
    file_path: str       # as it appears in the traceback (may be relative)
    line_no: int
    function_name: str
    frame_index: int     # 0 = outermost, n-1 = innermost
    excerpt: str = ""    # the source line as quoted by the traceback

    def __post_init__(self) -> None:
        _assert_no_forbidden_field_names(type(self))

    def as_dict(self) -> dict:
        return _signal_as_dict(self)


# ---------------------------------------------------------------------------
# 1d — symbol grep
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolGrepHit:
    """One ripgrep match for an identifier / error string / quoted
    code snippet pulled from the issue.

    Phase 1 stage 1d. ``rarity_score`` is the inverse-frequency weight:
    rare tokens (e.g., a unique class name) score high; common tokens
    (e.g., 'self', 'def') are filtered out before grep.
    """

    stage: LocalizationStage
    needle: str          # the searched-for string
    file_path: str
    line_no: int
    line_text: str
    rarity_score: float  # higher = rarer = more diagnostic

    def __post_init__(self) -> None:
        _assert_no_forbidden_field_names(type(self))

    def as_dict(self) -> dict:
        return _signal_as_dict(self)


# ---------------------------------------------------------------------------
# 1e — git archaeology
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GitArchaeologyHit:
    """One commit reference from ``git log -S<token>`` or
    ``git log -- <candidate_file>``.

    Phase 1 stage 1e. Past commits that touched candidate files are
    strong priors, especially when the commit message contains 'fix:',
    'bug', 'closes #', or 'fixes #'.
    """

    stage: LocalizationStage
    file_path: str
    sha: str
    parent_sha: str
    commit_message_summary: str   # first line of commit msg (truncated)
    fix_signal: bool              # message contains 'fix:' / 'bug' / etc.
    age_days: int                 # days between commit and base_commit
    files_changed: int            # number of files in the commit

    def __post_init__(self) -> None:
        _assert_no_forbidden_field_names(type(self))

    def as_dict(self) -> dict:
        return _signal_as_dict(self)


# ---------------------------------------------------------------------------
# 1f — dep-graph expansion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DepGraphHop:
    """One edge in the AST / import dep-graph expansion around a
    localization candidate.

    Phase 1 stage 1f. Expands candidate files one hop in both
    directions (importers and importees) to catch cases where the bug
    is in a caller or callee of the localized file.
    """

    stage: LocalizationStage
    source_file: str
    target_file: str
    edge_type: str       # 'imports' | 'imported_by' | 'calls' | 'called_by'
    hop_distance: int    # 1 for direct neighbors, 2 for two-hop
    confidence: float    # 0-1; how confident the static analysis is

    def __post_init__(self) -> None:
        _assert_no_forbidden_field_names(type(self))
        if self.hop_distance < 1:
            raise ValueError("DepGraphHop.hop_distance must be >= 1")

    def as_dict(self) -> dict:
        return _signal_as_dict(self)


# ---------------------------------------------------------------------------
# 1g — LLM rerank
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# RankedFile — per-candidate record carrying upstream-signal attribution.
# Replaces the earlier flat 3-tuple in RerankerOutput.ranked_files so the
# ablation table can be auto-generated from RerankerOutput records alone,
# without joining across upstream signal records.
# ---------------------------------------------------------------------------

UPSTREAM_SIGNAL_KINDS: frozenset[str] = frozenset({
    "bm25",
    "embedding",
    "traceback",
    "symbol_grep",
    "git_archaeology",
    "dep_graph",
})


@dataclass(frozen=True)
class RankedFile:
    """One ranked candidate emitted by the Phase 1 reranker.

    Carries enough upstream-signal attribution to reconstruct, from a
    single RerankerOutput log line, which Stage 1b–1f signals named
    this file and at what rank — i.e., the ablation pivot.

    Fields:
      file_path             — relative path inside the repo at base_commit
      final_score           — reranker's combined score (model-defined)
      rationale             — human-readable string from the reranker
      upstream_signals      — tuple of stage labels that named this file
                              (subset of UPSTREAM_SIGNAL_KINDS)
      upstream_best_rank    — None if no upstream rank info, else dict
                              {signal_kind: best_rank_seen} where
                              best_rank_seen is 1-indexed
    """

    file_path: str
    final_score: float
    rationale: str
    upstream_signals: tuple[str, ...]
    upstream_best_rank: dict | None = None  # dict[str, int] | None

    def __post_init__(self) -> None:
        if not self.file_path:
            raise ValueError("RankedFile.file_path required")
        if not isinstance(self.upstream_signals, tuple):
            raise TypeError(
                f"RankedFile.upstream_signals must be tuple, "
                f"got {type(self.upstream_signals).__name__}"
            )
        unknown = set(self.upstream_signals) - UPSTREAM_SIGNAL_KINDS
        if unknown:
            raise ValueError(
                f"RankedFile.upstream_signals contains unknown kinds: "
                f"{sorted(unknown)}; must be subset of "
                f"{sorted(UPSTREAM_SIGNAL_KINDS)}"
            )
        if self.upstream_best_rank is not None:
            unknown_rank_keys = set(self.upstream_best_rank.keys()) - UPSTREAM_SIGNAL_KINDS
            if unknown_rank_keys:
                raise ValueError(
                    f"RankedFile.upstream_best_rank has unknown keys: "
                    f"{sorted(unknown_rank_keys)}"
                )
            for k, v in self.upstream_best_rank.items():
                if not isinstance(v, int) or v < 1:
                    raise ValueError(
                        f"RankedFile.upstream_best_rank[{k!r}] must be int >= 1, "
                        f"got {v!r}"
                    )


@dataclass(frozen=True)
class RerankerOutput:
    """The top-K LLM rerank of aggregated candidates from stages 1b-1f.

    Phase 1 stage 1g. Recorded per LLM call: the reranker sees the
    aggregated candidate list and picks the top-K with rationale.
    Each ranked entry is a typed RankedFile (refactored from the earlier
    flat 3-tuple in commit 3b) — this makes per-upstream-signal
    ablations auto-generatable from RerankerOutput records alone.
    """

    stage: LocalizationStage
    model: str                          # role-resolved model name
    input_tokens: int
    output_tokens: int
    prompt_token_count: int             # for context-budget audits
    ranked_files: tuple                 # tuple of RankedFile
    skeleton_compression: str           # 'full' | 'top-30-symbols' | etc.

    def __post_init__(self) -> None:
        _assert_no_forbidden_field_names(type(self))
        for i, entry in enumerate(self.ranked_files):
            if not isinstance(entry, RankedFile):
                raise TypeError(
                    f"RerankerOutput.ranked_files[{i}] must be RankedFile, "
                    f"got {type(entry).__name__}"
                )

    def as_dict(self) -> dict:
        d = {
            "_signal": "RerankerOutput",
            "stage": self.stage,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "prompt_token_count": self.prompt_token_count,
            "skeleton_compression": self.skeleton_compression,
            "ranked_files": [
                {
                    "file_path": rf.file_path,
                    "final_score": rf.final_score,
                    "rationale": rf.rationale,
                    "upstream_signals": list(rf.upstream_signals),
                    "upstream_best_rank": (
                        dict(rf.upstream_best_rank)
                        if rf.upstream_best_rank is not None
                        else None
                    ),
                }
                for rf in self.ranked_files
            ],
        }
        return d


# ---------------------------------------------------------------------------
# Type alias + registry
# ---------------------------------------------------------------------------


LocalizationSignal = Union[
    BM25Hit,
    EmbeddingHit,
    TracebackFrame,
    SymbolGrepHit,
    GitArchaeologyHit,
    DepGraphHop,
    RerankerOutput,
]


SIGNAL_CLASSES: tuple[type, ...] = (
    BM25Hit,
    EmbeddingHit,
    TracebackFrame,
    SymbolGrepHit,
    GitArchaeologyHit,
    DepGraphHop,
    RerankerOutput,
)


__all__ = [
    "LocalizationStage",
    "STAGE_1A_SKELETON",
    "STAGE_1B_FILE_CANDIDATES",
    "STAGE_1C_TRACEBACK",
    "STAGE_1D_SYMBOL_GREP",
    "STAGE_1E_GIT_ARCHEOLOGY",
    "STAGE_1F_DEP_GRAPH",
    "STAGE_1G_RERANK",
    "STAGE_1H_FUNCTION_LINE",
    "ALL_STAGES",
    "QUERY_STRATEGIES",
    "UPSTREAM_SIGNAL_KINDS",
    "BM25Hit",
    "EmbeddingHit",
    "TracebackFrame",
    "SymbolGrepHit",
    "GitArchaeologyHit",
    "DepGraphHop",
    "RankedFile",
    "RerankerOutput",
    "LocalizationSignal",
    "SIGNAL_CLASSES",
]
