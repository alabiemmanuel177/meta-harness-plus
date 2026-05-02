"""Phase 3 context assembly + patch-generator-context-superset assertion.

Per docs/V10_DESIGN_PHASE3.md §2.1 rule 3 + §2.2:

  ``build_patch_gen_context_with_superset_check`` is the SINGLE
  entrypoint for assembling Phase 3 context. The firewall test
  (added in P3b) confirms every prompt-build path under
  ``harness.patch_gen`` calls THIS function before any LLM call.
  Refactors that bypass the assertion fail the firewall test.

The assertion: whatever files the repro generator saw for instance
`i`, the patch generator MUST also see for the same `i`. This
guarantees any repro is structurally satisfiable. Caller passes
``repro_context_files=`` when known; ``None`` skips the assertion
(Phase 2 wasn't run for this instance, or the operator chose to
bypass).

Token-count gate: instances projected over ``DEFAULT_PROJECTED_TOKEN_LIMIT``
raise ``ContextOversizeError``. Per design §3.2: pipeline path
caller catches and skips; agent path covers them in P3c.
"""

from __future__ import annotations

import logging

from harness.localization_signals import RankedFile
from harness.patch_gen.views import (
    ContextOversizeError,
    ContextSupersetError,
    FileSnippet,
    PatchGenContext,
)
from harness.views import InstanceView


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------


DEFAULT_TOP_K: int = 10
DEFAULT_PER_FILE_CHAR_CAP: int = 12_000     # ≈ 3000 tokens
DEFAULT_PROJECTED_TOKEN_LIMIT: int = 700_000


# ---------------------------------------------------------------------------
# Token estimation (cheap; ratio-based)
# ---------------------------------------------------------------------------


def _estimate_tokens(s: str) -> int:
    """Approximate token count for context-budgeting only. Uses the
    BPE-typical 4 chars/token ratio. Not a substitute for the actual
    tokenizer — but good enough for the §3.2 oversize gate where
    we just need to know "is this pipeline-shaped or agent-shaped?".
    """
    return max(1, len(s) // 4)


# ---------------------------------------------------------------------------
# File-content read with truncation
# ---------------------------------------------------------------------------


def _read_one_file(sandbox, path: str, char_cap: int) -> tuple[str, bool]:
    """Read ``path`` from the sandbox, truncated to ``char_cap`` chars.

    Returns ``(content, truncated)``. If the file can't be read,
    returns ``("", False)`` so the caller can decide whether to drop
    the entry; we don't want a single missing file to abort the
    whole context.
    """
    try:
        # max_chars on the sandbox's read_file applies on the read
        # itself (avoids dumping a 10MB file into memory).
        res = sandbox.read_file(path, max_chars=char_cap * 2)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[patch-gen-context] read_file failed for %s: %s",
            path, exc,
        )
        return "", False
    if res.exit_code != 0 or not res.stdout:
        return "", False
    content = res.stdout
    if len(content) > char_cap:
        content = content[:char_cap]
        return content, True
    return content, False


# ---------------------------------------------------------------------------
# The single entrypoint (firewall-enforced)
# ---------------------------------------------------------------------------


def build_patch_gen_context_with_superset_check(
    *,
    view: InstanceView,
    ranked_files: list[RankedFile],
    sandbox,
    top_k: int = DEFAULT_TOP_K,
    per_file_char_cap: int = DEFAULT_PER_FILE_CHAR_CAP,
    projected_token_limit: int = DEFAULT_PROJECTED_TOKEN_LIMIT,
    repro_context_files: tuple[str, ...] | None = None,
) -> PatchGenContext:
    """Assemble the patch-gen context for one instance. Single
    entrypoint per §2.2 firewall.

    Per §2.1 rule 3 (patch-generator-context-superset constraint):
    if ``repro_context_files`` is supplied, the patch-gen file set
    MUST be a superset. Violation raises ``ContextSupersetError``.

    Per §3.2 oversize gate: if the projected token count exceeds
    ``projected_token_limit``, raises ``ContextOversizeError``. The
    pipeline path caller catches and skips; the agent path covers
    these instances naturally.

    Args:
      view: InstanceView (firewall-clean by construction).
      ranked_files: Phase 1 reranker output, top-K candidates.
      sandbox: a started Sandbox at base_commit. Used only for
        ``read_file`` calls.
      top_k: how many of the ranked files to read into context.
      per_file_char_cap: max chars per file (truncate beyond).
      projected_token_limit: oversize gate.
      repro_context_files: if supplied, the file paths the repro
        generator saw for this instance. Used by the superset
        assertion.
    """
    if not ranked_files:
        raise ValueError(
            f"build_patch_gen_context_with_superset_check: ranked_files "
            f"empty for {view.instance_id}; Phase 1 must produce at "
            f"least one candidate before Phase 3 can run"
        )

    selected = ranked_files[:top_k]
    snippets: list[FileSnippet] = []
    for rf in selected:
        content, truncated = _read_one_file(sandbox, rf.file_path, per_file_char_cap)
        if not content:
            # Skip files we couldn't read; log and move on.
            log.info(
                "[patch-gen-context] dropping unreadable file %s for %s",
                rf.file_path, view.instance_id,
            )
            continue
        snippets.append(
            FileSnippet(
                path=rf.file_path,
                content=content,
                truncated=truncated,
                final_score=float(rf.final_score),
                rationale=rf.rationale or "",
            )
        )

    if not snippets:
        raise ValueError(
            f"build_patch_gen_context_with_superset_check: no readable "
            f"files in top-{top_k} for {view.instance_id}; sandbox is "
            f"likely misconfigured"
        )

    # Project total token count: problem_statement + each file's content
    # + a small per-file framing overhead.
    proj = _estimate_tokens(view.problem_statement)
    for snip in snippets:
        proj += _estimate_tokens(snip.content)
        proj += 100  # ~100 tokens of framing per file (path + score line)

    if proj > projected_token_limit:
        raise ContextOversizeError(
            f"{view.instance_id}: projected context {proj} tokens > "
            f"limit {projected_token_limit}; pipeline path skips this "
            f"instance per V10_DESIGN_PHASE3.md §3.2 (agent path is "
            f"the natural shape for cap-hitters)"
        )

    # Patch-generator-context-superset assertion (§2.1 rule 3).
    if repro_context_files is not None:
        patch_files = {s.path for s in snippets}
        repro_files = set(repro_context_files)
        missing = repro_files - patch_files
        if missing:
            raise ContextSupersetError(
                f"{view.instance_id}: patch-gen context missing files "
                f"the repro generator saw: {sorted(missing)}. The "
                f"superset constraint guarantees any repro is "
                f"structurally satisfiable by the patch generator."
            )

    return PatchGenContext(
        instance_id=view.instance_id,
        problem_statement=view.problem_statement,
        test_directives=view.test_directives.dirs,
        files=tuple(snippets),
        projected_token_count=proj,
        repro_context_files=repro_context_files,
    )


__all__ = [
    "DEFAULT_PER_FILE_CHAR_CAP",
    "DEFAULT_PROJECTED_TOKEN_LIMIT",
    "DEFAULT_TOP_K",
    "build_patch_gen_context_with_superset_check",
]
