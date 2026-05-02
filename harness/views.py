"""Typed views — the contamination firewall's type layer.

`InstanceView` is what prompt builders see. `CandidateView` is what
selectors see. Neither carries any field that could reveal the eval
oracle (FAIL_TO_PASS, PASS_TO_PASS, hints_text, gold patch, evaluator
verdict). Both are frozen dataclasses; both validate field names against
the forbidden-token list at construction time.

If a future change adds a forbidden field to either view, construction
raises ``ForbiddenFieldError`` and the firewall test (which scans the
class fields statically) also fails — defense in depth.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Tuple


# ---------------------------------------------------------------------------
# Forbidden tokens — the single source of truth, used by views, sandbox
# runtime guard, and the firewall test. Substring + casefold matching.
# ---------------------------------------------------------------------------

FORBIDDEN_TOKENS: tuple[str, ...] = (
    "fail_to_pass",
    "pass_to_pass",
    "test_patch",
    "hints_text",
    "gold_patch",
)
# Per V10_DESIGN_PHASE2.md §8.4 calibration (commit 17a) and the
# 17c firewall-test pass: bare English words like ``hints`` and
# ``resolved`` were dropped from FORBIDDEN_TOKENS. Reasoning:
#   1. They aren't actual SWE-bench dataset field names. The real
#      field is ``hints_text`` (compound; still in the list); there
#      is no column literally named ``resolved`` in
#      swebench_verified.jsonl.
#   2. Under word-boundary matching they would trip on legitimate
#      English usage in issue text ("any hints would be appreciated",
#      "once this is resolved") — over-block with no security gain.
#   3. The compound form ``hints_text`` catches the actual leak
#      vector (someone copying the JSON column name) under
#      word-boundary matching.


class ForbiddenFieldError(TypeError):
    """A view dataclass has a field whose name matches a forbidden token."""


def _assert_no_forbidden_field_names(cls: type) -> None:
    bad: list[str] = []
    for f in fields(cls):
        folded = f.name.casefold()
        for tok in FORBIDDEN_TOKENS:
            if tok in folded:
                bad.append(f"{cls.__name__}.{f.name} (matches {tok!r})")
    if bad:
        raise ForbiddenFieldError(
            "view dataclass carries forbidden field(s): " + ", ".join(bad)
        )


# ---------------------------------------------------------------------------
# Building-block dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestDirectives:
    """Where the public test suite lives in this repo, derived from repo
    conventions only (filesystem inspection at base_commit + per-repo
    override map). Never derived from a dataset row.
    """

    dirs: tuple[str, ...]
    source: str  # human-readable provenance: "override:<artifact>" or "discovery:<n_dirs>"

    def __post_init__(self) -> None:
        _assert_no_forbidden_field_names(type(self))
        if not self.dirs:
            raise ValueError("TestDirectives.dirs must be non-empty")
        for d in self.dirs:
            if not isinstance(d, str) or not d.strip():
                raise ValueError(f"TestDirectives.dirs entry invalid: {d!r}")


@dataclass(frozen=True)
class FunctionSummary:
    name: str
    line_start: int
    line_end: int


@dataclass(frozen=True)
class ClassSummary:
    name: str
    line_start: int
    line_end: int
    methods: tuple[FunctionSummary, ...] = ()


@dataclass(frozen=True)
class FileSummary:
    path: str
    classes: tuple[ClassSummary, ...] = ()
    functions: tuple[FunctionSummary, ...] = ()


@dataclass(frozen=True)
class RepoSkeleton:
    """Compact tree of files / classes / functions at base_commit. Never
    includes test bodies — only file-level entries for test files."""

    repo: str
    base_commit: str
    files: tuple[FileSummary, ...] = ()

    def __post_init__(self) -> None:
        _assert_no_forbidden_field_names(type(self))


# ---------------------------------------------------------------------------
# Top-level views
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InstanceView:
    """The ONLY view of a SWE-bench instance that prompt builders may see.

    Forbidden fields (FAIL_TO_PASS, PASS_TO_PASS, test_patch, hints_text,
    gold_patch, resolved verdict, etc.) are dropped at the dataset
    boundary in ``harness.dataset._project_to_view``; they cannot reach
    this object by construction.

    ``dockerhub_tag`` is the SWE-bench Pro image tag (lookup field for
    ``jefzda/sweap-images:{tag}``); empty string for Verified instances,
    where the image name is derived from instance_id. It is harness
    infrastructure, not oracle data — see V10_DESIGN.md §13.2 (Pro
    delta).
    """

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    repo_skeleton: RepoSkeleton
    test_directives: TestDirectives
    dockerhub_tag: str = ""

    def __post_init__(self) -> None:
        _assert_no_forbidden_field_names(type(self))
        if not self.instance_id:
            raise ValueError("InstanceView.instance_id required")
        if not self.repo:
            raise ValueError("InstanceView.repo required")
        if not self.base_commit:
            raise ValueError("InstanceView.base_commit required")
        if not self.problem_statement:
            raise ValueError("InstanceView.problem_statement required")
        if self.repo_skeleton.repo != self.repo:
            raise ValueError(
                f"InstanceView.repo_skeleton.repo mismatch: "
                f"{self.repo_skeleton.repo!r} != {self.repo!r}"
            )


# ---- Candidate-side dataclasses ------------------------------------------


@dataclass(frozen=True)
class DiffStats:
    files_touched: tuple[str, ...]
    additions: int
    deletions: int
    ast_normalized_hash: str  # for clustering; not for selection


@dataclass(frozen=True)
class ReproSignal:
    """Result of running our self-generated reproduction test.

    Status enum: 'pass', 'fail', 'no_repro', 'errored'. ``no_repro``
    means we could not generate a repro that fails at base_commit; the
    selector treats this as a missing signal rather than a negative one.
    """

    status: str
    duration_s: float
    log_excerpt: str = ""

    def __post_init__(self) -> None:
        if self.status not in ("pass", "fail", "no_repro", "errored"):
            raise ValueError(f"ReproSignal.status invalid: {self.status!r}")


@dataclass(frozen=True)
class PublicSuiteSignal:
    """Delta of repo's PUBLIC test suite when running at base_commit vs.
    at base_commit+candidate_patch.

    NB: this counts public-suite outcomes only. We never load the
    FAIL_TO_PASS / PASS_TO_PASS lists, so the ``regressed_count`` here
    is a property of the public suite — not of the hidden eval set.
    """

    suite_ran_at_base: bool
    new_failures_count: int  # tests that pass at base, fail at base+patch
    new_passes_count: int    # tests that fail at base, pass at base+patch (rare; possible)
    flake_retries: int
    duration_s: float
    log_excerpt: str = ""


@dataclass(frozen=True)
class StaticSignal:
    ruff_clean: bool | None  # None if ruff not configured for this repo
    mypy_clean: bool | None  # None if mypy not configured
    duration_s: float


@dataclass(frozen=True)
class CandidateView:
    """The ONLY view of a candidate patch that selectors may see.

    Carries no eval-derived field. The grader's verdict (written to the
    isolated output directory described in V10_DESIGN.md §12.6) cannot
    reach here by construction — the firewall test scans for any field
    name resembling a forbidden token.
    """

    candidate_id: str
    diff: str
    diff_stats: DiffStats
    repro_signal: ReproSignal | None
    public_suite_signal: PublicSuiteSignal
    static_signal: StaticSignal
    cluster_id: str = ""
    notes: str = ""  # human-debug-only string, allowed (not used by selectors)

    def __post_init__(self) -> None:
        _assert_no_forbidden_field_names(type(self))
        if not self.candidate_id:
            raise ValueError("CandidateView.candidate_id required")
        if not self.diff:
            raise ValueError("CandidateView.diff required (use empty diff sentinel if intended)")


# ---------------------------------------------------------------------------
# Public re-exports for type checkers and the firewall test
# ---------------------------------------------------------------------------

VIEW_CLASSES: Tuple[type, ...] = (
    TestDirectives,
    RepoSkeleton,
    InstanceView,
    DiffStats,
    ReproSignal,
    PublicSuiteSignal,
    StaticSignal,
    CandidateView,
)


__all__ = [
    "FORBIDDEN_TOKENS",
    "ForbiddenFieldError",
    "TestDirectives",
    "FunctionSummary",
    "ClassSummary",
    "FileSummary",
    "RepoSkeleton",
    "InstanceView",
    "DiffStats",
    "ReproSignal",
    "PublicSuiteSignal",
    "StaticSignal",
    "CandidateView",
    "VIEW_CLASSES",
]
