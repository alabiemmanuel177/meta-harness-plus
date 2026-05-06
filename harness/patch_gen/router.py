"""Phase 3 P3d-fix — patch-gen strategy router (two-way).

Dispatches each instance to one of two patch-gen strategies based on
CHEAP, deterministic features extracted from ``InstanceView`` +
``RankedFile[]``. No LLM calls. No network. No imports of repro,
eval, or memory.

History — why two-way and not three:

  P3d (commit 0762e75) shipped a three-way router with
  ``PIPELINE_ONE_SHOT`` as the third strategy. Rule 1 required
  ``candidate_file_count <= 3``, but Phase 1 always returns top-K=10
  reranked files. With ``candidate_file_count == 10`` for every
  instance, Rule 1 was unreachable — the router NEVER chose
  PIPELINE_ONE_SHOT on dev_50. The intent was to catch
  django-11206-style tight fixes via single-shot pipeline; that
  intent was lost.

  Per user decision after P3d hit 9/50 (18%) HARD STOP: drop
  PIPELINE_ONE_SHOT from the strategy enum entirely. The pipeline
  generator (``generate_pipeline_one_shot``) STAYS in the codebase
  — it's used internally by BOOTSTRAPPED_AGENT as the seed. Only
  the standalone dispatch role is removed.

The two strategies and their dev_50 rationale:

  - ``AGENT``: agent path WITHOUT a pipeline seed. Wins on long
    issues without tracebacks where the pipeline's first-shot
    guess can mislead the agent's apply_patch retries. Resolved 8
    instances in P3c. The 2 instances ONLY non-bootstrapped agent
    catches (``django__django-10880``, ``psf__requests-1142``) had
    long, multi-paragraph issues without explicit traceback frames.
  - ``BOOTSTRAPPED_AGENT``: pipeline-bootstrapped agent (the P3c-v2
    architecture). Default — best path on average (11/50 = 22% on
    dev_50). Wins on instances with mid-large surface area where
    the agent benefits from a starting diff but needs to iterate.

Routing thresholds are tunable constants at the top of this file.

Spec: docs/V10_DESIGN_PHASE3.md §3.1 (routes), §3.4 (difficulty
estimation; linear formula on cheap signals; escalation to
learned classifier deferred unless dev_50 routing accuracy <70%).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from harness.localization_signals import RankedFile
from harness.views import InstanceView


# ---------------------------------------------------------------------------
# Tunable thresholds — calibrated against the dev_50 distribution
# ---------------------------------------------------------------------------
#
# Distribution of features on dev_50 (50 instances, computed after P3c-v2
# from the cached top-10 reranked files + InstanceView fields):
#   issue_word_count:      p50 ≈ 220, p90 ≈ 700, max 2,400
#   candidate_file_count:  always 10 (top-10 from reranker)
#   top1_file_loc:         p50 ≈ 1,200, p90 ≈ 4,500, max 13,000
#   has_traceback:         ~30% of instances
#
# AGENT (no seed) is chosen when ALL three conditions hold:
#   - long issue (issue_word_count >= 300)
#   - no traceback in issue
#   - large top-1 file (top1_file_loc >= 800)
#
# Rationale: long, traceback-less issues with large affected files are
# the regime where a pipeline seed is most likely to mislead the agent.
# Better to let the agent explore from scratch with no anchor. This
# fires on roughly the upper-third of dev_50 instances.

AGENT_LONG_ISSUE_WORDS_MIN: int = 300
AGENT_LARGE_TOP1_LOC_MIN: int = 800


# ---------------------------------------------------------------------------
# Strategy enum (two-way after P3d-fix)
# ---------------------------------------------------------------------------


class PatchGenStrategy(Enum):
    """The two patch-gen paths the router can dispatch to.

    PIPELINE_ONE_SHOT was removed in P3d-fix — see the module
    docstring history note. The pipeline generator itself remains
    available; only its standalone dispatch role is gone.
    """

    AGENT = "agent"
    BOOTSTRAPPED_AGENT = "bootstrapped_agent"


# ---------------------------------------------------------------------------
# Feature extraction (cheap, deterministic, no external state)
# ---------------------------------------------------------------------------


_TRACEBACK_RE = re.compile(
    r"(Traceback \(most recent call last\)|"
    r"  File \"[^\"]+\", line \d+|"
    r"^\s+at .+\(.*?:\d+\)$)",
    re.MULTILINE,
)


def _has_traceback(text: str) -> bool:
    if not text:
        return False
    return bool(_TRACEBACK_RE.search(text))


@dataclass(frozen=True)
class RouterFeatures:
    """Cheap features extracted from InstanceView + RankedFile[].

    Field set is unchanged from P3d (commit 0762e75) even though the
    new two-way router only consults ``issue_word_count``,
    ``has_traceback_in_issue``, and ``top1_file_loc``. The remaining
    fields stay so a future routing iteration can use them without a
    schema refactor.
    """

    issue_word_count: int
    has_traceback_in_issue: bool
    candidate_file_count: int
    top1_file_loc: int
    repo_id: str
    skeleton_size_chars: int

    def __post_init__(self) -> None:
        if self.issue_word_count < 0:
            raise ValueError("issue_word_count must be >= 0")
        if self.candidate_file_count < 0:
            raise ValueError("candidate_file_count must be >= 0")
        if self.top1_file_loc < 0:
            raise ValueError("top1_file_loc must be >= 0")
        if self.skeleton_size_chars < 0:
            raise ValueError("skeleton_size_chars must be >= 0")


def extract_features(
    view: InstanceView,
    ranked_files: list[RankedFile],
    *,
    top1_file_loc: int = 0,
    skeleton_size_chars: int = 0,
) -> RouterFeatures:
    """Build a RouterFeatures from view + ranked_files.

    ``top1_file_loc`` and ``skeleton_size_chars`` are passed in by
    the caller (the eval script) because computing them requires a
    sandbox read; the router itself stays pure. When the caller
    can't provide them (e.g., no sandbox), pass 0 and the rules
    will treat them as "small" — same fallback as if the file was
    empty.
    """
    issue = view.problem_statement or ""
    word_count = len(issue.split())
    return RouterFeatures(
        issue_word_count=word_count,
        has_traceback_in_issue=_has_traceback(issue),
        candidate_file_count=len(ranked_files),
        top1_file_loc=top1_file_loc,
        repo_id=view.repo,
        skeleton_size_chars=skeleton_size_chars,
    )


# ---------------------------------------------------------------------------
# Router (pure function — no side effects)
# ---------------------------------------------------------------------------


def route(features: RouterFeatures) -> PatchGenStrategy:
    """Two-way router. Pure deterministic function — same input always
    returns the same strategy.

    Rules (in priority order):

      1. **Exploration-heavy regime → AGENT (no seed).** Long issues
         (≥300 words), without tracebacks, with large top-1 files
         (≥800 LOC). The pipeline seed is most likely to mislead the
         agent in this regime; better to let the agent explore from
         scratch.

      2. **Default → BOOTSTRAPPED_AGENT.** Best path on average on
         dev_50 (22% standalone). The pipeline-quality starting diff
         + apply_patch verification + agent revision loop handles
         the bulk of instances.
    """
    if (
        features.issue_word_count >= AGENT_LONG_ISSUE_WORDS_MIN
        and not features.has_traceback_in_issue
        and features.top1_file_loc >= AGENT_LARGE_TOP1_LOC_MIN
    ):
        return PatchGenStrategy.AGENT
    return PatchGenStrategy.BOOTSTRAPPED_AGENT


__all__ = [
    "AGENT_LARGE_TOP1_LOC_MIN",
    "AGENT_LONG_ISSUE_WORDS_MIN",
    "PatchGenStrategy",
    "RouterFeatures",
    "extract_features",
    "route",
]
