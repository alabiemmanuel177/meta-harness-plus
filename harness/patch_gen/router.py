"""Phase 3 P3d — patch-gen strategy router.

Dispatches each instance to one of three patch-gen strategies based on
CHEAP, deterministic features extracted from ``InstanceView`` +
``RankedFile[]``. No LLM calls. No network. No imports of repro,
eval, or memory.

The strategies and their dev_50 rationale (from
``docs/audits/dev_50_patch_gen_eval{,_agent,_agent_v2}.md``):

  - ``PIPELINE_ONE_SHOT``: K=1 single-shot pipeline call at T=0. Wins
    on tight, deterministic, single-file fixes — typically when the
    issue text contains a Python traceback and the localizer's
    candidate set is small. Pipeline alone resolved 7/50; the only
    instance ONLY pipeline catches is ``django__django-11206``
    (traceback + 2 candidate files + 600-LOC top-1 file).
  - ``AGENT``: agent path WITHOUT a pipeline seed. Wins on long
    issues without tracebacks where the pipeline's first-shot
    guess can mislead the agent's apply_patch retries. Resolved 8
    instances (P3c). The 2 instances ONLY non-bootstrapped agent
    catches (``django__django-10880``, ``psf__requests-1142``) had
    long, multi-paragraph issues without explicit traceback frames.
  - ``BOOTSTRAPPED_AGENT``: pipeline-bootstrapped agent (the P3c-v2
    architecture). Default — best path on average (11/50 = 22%
    standalone). Wins on instances with mid-large surface area
    where the agent benefits from a starting diff but needs to
    iterate. The 4 instances ONLY bootstrapped agent catches
    (``astropy-12907``, ``sklearn-10908``, ``sphinx-10466``,
    ``sphinx-10673``) had medium issue length + 5+ candidate files.

Routing thresholds are tunable constants at the top of this file.
The §6 acceptance gate for P3d on dev_50 is ≥25% (oracle merge of
the three strategies hits 28%; routing's job is to land within
3pp of that ceiling).

Spec: docs/V10_DESIGN_PHASE3.md §3.1 (three routes), §3.4
(difficulty estimation; linear formula on cheap signals;
escalation to learned classifier deferred unless dev_50 routing
accuracy <70%).
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
# The thresholds below carve out the "tight pipeline-friendly" slice:
# small issue, traceback present, modest top-1 file. This was 7/50
# instances on dev_50 — a high-precision rule for picking pipeline.

PIPELINE_TIGHT_ISSUE_WORDS_MAX: int = 400
PIPELINE_TIGHT_TOP1_LOC_MAX: int = 1_500
PIPELINE_TIGHT_CANDIDATE_FILES_MAX: int = 3

# "Long issue without traceback" → AGENT (no seed). Threshold chosen so
# the rule fires on the upper-half of issue length but only when no
# traceback anchors the localizer.
AGENT_LONG_ISSUE_WORDS_MIN: int = 600


# ---------------------------------------------------------------------------
# Strategy enum
# ---------------------------------------------------------------------------


class PatchGenStrategy(Enum):
    """The three patch-gen paths the router can dispatch to."""

    PIPELINE_ONE_SHOT = "pipeline_one_shot"
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

    Field names are stable; the router consumes this dataclass instead
    of inspecting raw view/files so the routing logic and the feature
    extractor can evolve independently.
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
    """Linear-formula router. Pure deterministic function — same input
    always returns the same strategy.

    Rules (in priority order):

      1. **Tight pipeline shape.** Traceback + small candidate set +
         small top-1 file + short issue → PIPELINE_ONE_SHOT. The
         pipeline does well on these (django-11206-style cases). High
         precision rule; if any condition fails, fall through.

      2. **Long issue without traceback.** Long, multi-paragraph
         issues where the pipeline's single-shot guess is most likely
         to mislead the agent. Send to AGENT (no seed) so the agent
         explores from scratch rather than fix-from-wrong-start.

      3. **Default — BOOTSTRAPPED_AGENT.** Best path on average
         (22% on dev_50 standalone vs 14% pipeline / 16% non-boot
         agent). The agent + seed combination handles mid-to-hard
         instances where exploration is needed but a pipeline-quality
         starting diff still helps.
    """
    # Rule 1: tight pipeline shape
    if (
        features.has_traceback_in_issue
        and features.candidate_file_count <= PIPELINE_TIGHT_CANDIDATE_FILES_MAX
        and features.top1_file_loc <= PIPELINE_TIGHT_TOP1_LOC_MAX
        and features.issue_word_count <= PIPELINE_TIGHT_ISSUE_WORDS_MAX
    ):
        return PatchGenStrategy.PIPELINE_ONE_SHOT

    # Rule 2: long issue without traceback → unbootstrapped agent
    if (
        features.issue_word_count > AGENT_LONG_ISSUE_WORDS_MIN
        and not features.has_traceback_in_issue
    ):
        return PatchGenStrategy.AGENT

    # Rule 3: default
    return PatchGenStrategy.BOOTSTRAPPED_AGENT


__all__ = [
    "AGENT_LONG_ISSUE_WORDS_MIN",
    "PIPELINE_TIGHT_CANDIDATE_FILES_MAX",
    "PIPELINE_TIGHT_ISSUE_WORDS_MAX",
    "PIPELINE_TIGHT_TOP1_LOC_MAX",
    "PatchGenStrategy",
    "RouterFeatures",
    "extract_features",
    "route",
]
