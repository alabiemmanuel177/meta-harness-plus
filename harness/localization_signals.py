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
    document is per-file content (or a per-file summary, depending on
    the reranker config — recorded in ``query_basis``).
    """

    stage: LocalizationStage
    file_path: str
    score: float
    rank: int            # 1-indexed
    query_basis: str     # 'issue_text' | 'issue_summary' | 'symbol_only' | etc.

    def __post_init__(self) -> None:
        _assert_no_forbidden_field_names(type(self))
        if self.rank < 1:
            raise ValueError("BM25Hit.rank is 1-indexed")

    def as_dict(self) -> dict:
        return _signal_as_dict(self)


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


@dataclass(frozen=True)
class RerankerOutput:
    """The top-K LLM rerank of aggregated candidates from stages 1b-1f.

    Phase 1 stage 1g. Recorded per LLM call: the reranker sees the
    aggregated candidate list and picks the top-K with rationale.
    Aggregation rationale is logged so ablations can show which
    upstream signal (BM25, embedding, grep, archaeology, dep-graph)
    contributed to each kept candidate.
    """

    stage: LocalizationStage
    model: str                          # role-resolved model name
    input_tokens: int
    output_tokens: int
    prompt_token_count: int             # for context-budget audits
    ranked_files: tuple                 # tuple of (file_path, score, rationale)
    skeleton_compression: str           # 'full' | 'top-30-symbols' | etc.

    def __post_init__(self) -> None:
        _assert_no_forbidden_field_names(type(self))
        # Validate ranked_files shape: each entry is a 3-tuple of (str, float, str).
        for i, entry in enumerate(self.ranked_files):
            if not isinstance(entry, tuple) or len(entry) != 3:
                raise ValueError(
                    f"RerankerOutput.ranked_files[{i}] must be a 3-tuple"
                )
            fp, score, rationale = entry
            if not isinstance(fp, str) or not isinstance(score, (int, float)):
                raise ValueError(
                    f"RerankerOutput.ranked_files[{i}]: "
                    f"({type(fp).__name__}, {type(score).__name__}, ...)"
                )
            if not isinstance(rationale, str):
                raise ValueError(
                    f"RerankerOutput.ranked_files[{i}].rationale must be str"
                )

    def as_dict(self) -> dict:
        return _signal_as_dict(self)


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
    "BM25Hit",
    "EmbeddingHit",
    "TracebackFrame",
    "SymbolGrepHit",
    "GitArchaeologyHit",
    "DepGraphHop",
    "RerankerOutput",
    "LocalizationSignal",
    "SIGNAL_CLASSES",
]
