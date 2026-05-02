"""Frozen dataclasses for Phase 3 patch generation.

Per docs/V10_DESIGN_PHASE3.md §3.2:

  - ``PatchCandidate`` — the frozen output of one generator attempt
    (pipeline temperature variant or one agent run). The Phase 4
    validator + Phase 5 selector consume PatchCandidate; Phase 3
    never reads other candidates' diffs during generation.
  - ``PatchGenContext`` — the assembled prompt context (problem
    statement + top-K file contents + test_directives). Built by
    ``harness.patch_gen.context.build_patch_gen_context_with_superset_check``;
    nothing else builds it. Persisted only to trajectory (audit),
    never re-fed to a model.
  - ``FileSnippet`` — one entry inside ``PatchGenContext.files``.

All dataclasses are field-name-validated against ``FORBIDDEN_TOKENS``
via ``harness.views._assert_no_forbidden_field_names``.

``PatchGenError`` is the public error class for any pre-LLM failure
(parse error, missing required field, etc.). ``ContextOversizeError``
is the dedicated subclass for the cap-hitter case (§3.2 oversize
handling — pipeline skips, agent path covers).
"""

from __future__ import annotations

from dataclasses import dataclass

from harness.views import _assert_no_forbidden_field_names


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PatchGenError(RuntimeError):
    """Public error class for Phase 3 pre-LLM failures (parse error,
    missing required field, etc.)."""


class ContextOversizeError(PatchGenError):
    """Raised by ``build_patch_gen_context_with_superset_check`` when
    the projected token count exceeds the per-instance limit. Per
    docs/V10_DESIGN_PHASE3.md §3.2: the pipeline path skips these
    instances; the agent path covers them in P3c (per-file streaming
    is the agent's natural shape)."""


class ContextSupersetError(PatchGenError):
    """Raised when the patch generator's file context does not
    superset the repro generator's file context for the same
    instance. Per docs/V10_DESIGN_PHASE2.md §3.1: any repro the
    generator wrote MUST be structurally satisfiable by the patch
    generator. Violation is a logic bug — fail fast."""


# ---------------------------------------------------------------------------
# FileSnippet (one file in the patch-gen context)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FileSnippet:
    """One file's contents as seen by the patch generator. Truncated
    to ``len(content)`` chars. Per ``DEFAULT_PER_FILE_CHAR_CAP`` in
    ``harness.patch_gen.context`` (default 12_000 chars ≈ 3000 tokens).

    ``final_score`` and ``rationale`` are passed through from
    Phase 1's reranker so the prompt can cite why each file was
    selected.
    """

    path: str
    content: str
    truncated: bool
    final_score: float
    rationale: str

    def __post_init__(self) -> None:
        _assert_no_forbidden_field_names(type(self))
        if not self.path:
            raise ValueError("FileSnippet.path required")


# ---------------------------------------------------------------------------
# PatchGenContext (input to the LLM call)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PatchGenContext:
    """Assembled context for the Phase 3 patch generator.

    Built ONLY by ``harness.patch_gen.context.build_patch_gen_context_with_superset_check``
    (the firewall test asserts this is the only entrypoint). Carries:

      - The instance's ``problem_statement`` and ``test_directives`` from
        ``InstanceView``.
      - ``files`` — top-K ``FileSnippet`` in rank order.
      - ``projected_token_count`` — used by callers to decide the
        oversize gate.
      - ``repro_context_files`` — the file paths the repro generator
        saw for this instance, when supplied. Used by the superset
        assertion. ``None`` means the caller did not pass it (e.g.,
        Phase 2 wasn't run for this instance, or the operator
        chose to bypass).
    """

    instance_id: str
    problem_statement: str
    test_directives: tuple[str, ...]
    files: tuple[FileSnippet, ...]
    projected_token_count: int
    repro_context_files: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        _assert_no_forbidden_field_names(type(self))
        if not self.instance_id:
            raise ValueError("PatchGenContext.instance_id required")
        if not isinstance(self.test_directives, tuple):
            raise TypeError(
                "PatchGenContext.test_directives must be tuple, "
                f"got {type(self.test_directives).__name__}"
            )
        if not isinstance(self.files, tuple):
            raise TypeError(
                "PatchGenContext.files must be tuple, "
                f"got {type(self.files).__name__}"
            )
        if self.projected_token_count < 0:
            raise ValueError(
                f"PatchGenContext.projected_token_count must be >=0, "
                f"got {self.projected_token_count}"
            )

    @property
    def file_paths(self) -> tuple[str, ...]:
        return tuple(f.path for f in self.files)


# ---------------------------------------------------------------------------
# PatchCandidate (output to Phase 4)
# ---------------------------------------------------------------------------


VALID_SOURCE_ROUTES: tuple[str, ...] = ("pipeline", "agent")


@dataclass(frozen=True)
class PatchCandidate:
    """One generator attempt's output. Frozen, hash-stable.

    ``candidate_id`` is a stable per-instance identifier:
      - Pipeline: ``"pipe_t0"``, ``"pipe_t05"``, ``"pipe_t07"`` etc.
        (temperature without decimal point).
      - Agent: ``"agent_a0_final"``, ``"agent_a1_final"``.

    ``diff`` is a unified diff (git-apply-compatible) with paths
    relative to the repo root. The diff field is NOT scanned for
    forbidden tokens at construction (patch contents may legitimately
    include code fragments containing words that look like forbidden
    substrings under naive matching). The boundary scan
    (``check_output_substring_hits``) runs at the I/O surface
    instead.
    """

    instance_id: str
    candidate_id: str
    diff: str
    source_route: str
    source_temperature: float | None
    source_attempt_index: int
    generator_model: str
    generator_input_tokens: int
    generator_output_tokens: int
    generation_cost_usd: float
    duration_s: float

    def __post_init__(self) -> None:
        _assert_no_forbidden_field_names(type(self))
        if not self.instance_id:
            raise ValueError("PatchCandidate.instance_id required")
        if not self.candidate_id:
            raise ValueError("PatchCandidate.candidate_id required")
        if not self.diff:
            raise ValueError("PatchCandidate.diff required (non-empty)")
        if self.source_route not in VALID_SOURCE_ROUTES:
            raise ValueError(
                f"PatchCandidate.source_route must be one of "
                f"{VALID_SOURCE_ROUTES}, got {self.source_route!r}"
            )
        if self.source_route == "pipeline" and self.source_temperature is None:
            raise ValueError(
                "PatchCandidate.source_temperature must be set for "
                "source_route='pipeline'"
            )
        if self.source_attempt_index < 0:
            raise ValueError(
                f"PatchCandidate.source_attempt_index must be >=0, "
                f"got {self.source_attempt_index}"
            )
        if self.generator_input_tokens < 0 or self.generator_output_tokens < 0:
            raise ValueError("PatchCandidate token counts must be >=0")
        if self.generation_cost_usd < 0:
            raise ValueError("PatchCandidate.generation_cost_usd must be >=0")
        if self.duration_s < 0:
            raise ValueError("PatchCandidate.duration_s must be >=0")


__all__ = [
    "ContextOversizeError",
    "ContextSupersetError",
    "FileSnippet",
    "PatchCandidate",
    "PatchGenContext",
    "PatchGenError",
    "VALID_SOURCE_ROUTES",
]
