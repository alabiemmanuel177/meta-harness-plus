"""Stage 1c: traceback parser.

Per V10_DESIGN.md §3.2 stage 1c. Extract Python stack frames from the
issue text, then suffix-match each frame's quoted path against the
repo skeleton. If the issue says ``"expr.py"`` and the repo has 12
``expr.py`` files, all 12 become ``TracebackFrame`` signals — the
Stage 1g reranker disambiguates. We do not pick one based on
heuristics; that's the reranker's job.

Frame priority: the LAST frame in a traceback is where the exception
was actually raised, so it carries the highest signal. Earlier frames
are call-stack context. Reflect this by assigning lower (= higher
priority) ``rank`` values to deeper frames.

Two traceback formats are handled:

  1. Standard Python traceback:

         Traceback (most recent call last):
           File "/path/to/foo.py", line 42, in bar
             code_line
         SomeError: message

  2. GitHub-issue-rendered fragment (often missing the leading
     ``Traceback`` header, sometimes with truncated paths):

         File "sympy/core/expr.py", line 234, in evalf
             return self._eval_evalf(prec)

Both shapes use the same per-frame regex; the absence of a header
doesn't matter because we anchor on ``File "..."``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Optional

from harness.localization_signals import (
    STAGE_1C_TRACEBACK,
    TracebackFrame,
)
from harness.views import RepoSkeleton


# A frame line: ``File "<path>", line <N>(, in <func>)?``.
# Path may be quoted with " or '. Trailing ", in func" optional. Path
# may be relative ("expr.py"), partial ("sympy/core/expr.py"),
# truncated (".../sympy/core/expr.py"), or absolute.
_FRAME_RE = re.compile(
    r"""File\s+
        ["']                      # opening quote
        (?P<path>[^"'\n]+?)       # path (non-greedy, no quotes/newlines)
        ["']                      # closing quote
        ,?\s*                     # optional comma + whitespace
        line\s+(?P<lineno>\d+)    # line N
        (?:                       # optional ", in <func>"
            ,?\s*in\s+
            (?P<func>[A-Za-z_][A-Za-z0-9_<>]*)
        )?
    """,
    re.VERBOSE | re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedFrame:
    """One frame extracted from issue text. file_path is the path
    AS QUOTED in the issue (may be partial / truncated). frame_index
    is 0-based, in the order frames appear in the source text."""

    file_path_quoted: str
    line_no: int
    function_name: str
    frame_index: int
    source_excerpt: str   # the literal issue line(s) where this frame was extracted

    def normalized_quoted(self) -> str:
        """Strip leading "..." truncation markers and ./ prefix."""
        p = self.file_path_quoted.strip()
        if p.startswith("..."):
            p = p.lstrip(".").lstrip("/")
        return p.lstrip("./")


def parse_tracebacks(issue_text: str) -> list[ParsedFrame]:
    """Return parsed frames in source-order. ``frame_index`` is the
    position in source text — frame_index = N - 1 is the LAST/deepest
    frame across all tracebacks combined. (We treat multi-traceback
    issues as one stream; the deepest frame is most diagnostic.)
    """
    if not issue_text:
        return []
    frames: list[ParsedFrame] = []
    for match in _FRAME_RE.finditer(issue_text):
        path = match.group("path").strip()
        try:
            line_no = int(match.group("lineno"))
        except (TypeError, ValueError):
            continue
        func = match.group("func") or ""
        # Capture a short excerpt: from "File ..." up to the next
        # "File " or paragraph break (~120 chars).
        start = match.start()
        end = match.end()
        excerpt = issue_text[start:end]
        # Append following indented line if present (the actual source
        # line shown under the frame in CPython tracebacks).
        rest = issue_text[end:end + 200]
        next_line_match = re.match(r"\s*\n( {2,}|\t)([^\n]+)", rest)
        if next_line_match:
            excerpt = excerpt + "\n" + next_line_match.group(0).strip("\n")
        frames.append(ParsedFrame(
            file_path_quoted=path,
            line_no=line_no,
            function_name=func,
            frame_index=len(frames),
            source_excerpt=excerpt[:300],
        ))
    return frames


# ---------------------------------------------------------------------------
# Suffix matching against the repo skeleton
# ---------------------------------------------------------------------------


def _candidate_paths(skeleton: RepoSkeleton) -> list[str]:
    return [f.path for f in skeleton.files]


def suffix_match(quoted_path: str, candidate_paths: list[str]) -> list[str]:
    """Return all candidate paths that suffix-match the quoted path.

    Matching rules (in order, deduplicated by candidate path):

      1. Exact match — quoted == candidate.
      2. Quoted is a tail of candidate: candidate ends with "/<quoted>".
      3. Candidate's basename matches quoted's basename AND the parent
         path components in quoted appear in candidate (loose).

    Multi-match is by design: if 12 ``expr.py`` files exist in the
    repo and the issue quotes only ``expr.py``, all 12 are returned.
    Stage 1g rerank disambiguates.
    """
    quoted = quoted_path.strip()
    if quoted.startswith("..."):
        quoted = quoted.lstrip(".").lstrip("/")
    quoted = quoted.lstrip("./")
    if not quoted:
        return []

    matches: list[str] = []
    quoted_basename = PurePosixPath(quoted).name

    for cand in candidate_paths:
        cand_norm = cand.lstrip("./")
        if cand_norm == quoted:
            matches.append(cand)
            continue
        if cand_norm.endswith("/" + quoted):
            matches.append(cand)
            continue
        # Basename match — only if quoted is a bare filename (no '/'),
        # else require the directory components to be a tail of cand.
        if "/" not in quoted:
            if PurePosixPath(cand_norm).name == quoted_basename:
                matches.append(cand)
        else:
            # Already covered by endswith; nothing else.
            pass

    # Preserve first-occurrence order, dedupe.
    seen: set[str] = set()
    deduped: list[str] = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            deduped.append(m)
    return deduped


# ---------------------------------------------------------------------------
# End-to-end: emit TracebackFrame signals
# ---------------------------------------------------------------------------


def emit_traceback_frames(
    issue_text: str,
    skeleton: RepoSkeleton,
) -> list[TracebackFrame]:
    """Parse issue tracebacks and emit one TracebackFrame per
    (parsed_frame, matched_skeleton_path) pair.

    Frame priority is encoded in ``frame_index`` (0 = outermost,
    n-1 = innermost = highest signal). Stage 1c integrators should
    convert frame_index into a rank such that the LAST frame's
    matches get the LOWEST rank (highest priority).
    """
    parsed = parse_tracebacks(issue_text)
    if not parsed:
        return []
    candidates = _candidate_paths(skeleton)
    if not candidates:
        return []

    out: list[TracebackFrame] = []
    for f in parsed:
        normalized = f.normalized_quoted()
        matches = suffix_match(normalized, candidates)
        for resolved_path in matches:
            out.append(TracebackFrame(
                stage=STAGE_1C_TRACEBACK,
                file_path=resolved_path,
                line_no=f.line_no,
                function_name=f.function_name,
                frame_index=f.frame_index,
                excerpt=f.source_excerpt,
            ))
    return out


def rank_for_frame(
    frame: TracebackFrame,
    *,
    n_frames_total: int,
    match_index_within_frame: int,
    max_matches_per_frame: int = 30,
) -> int:
    """Compute the 1-indexed rank for a TracebackFrame in the candidate
    pool. Last frame's matches get ranks 1..M; second-to-last frame's
    get M+1..2M; etc.
    """
    # Distance from last frame (0 = last, increasing for earlier).
    distance_from_last = (n_frames_total - 1) - frame.frame_index
    return (
        distance_from_last * max_matches_per_frame
        + match_index_within_frame
        + 1
    )


__all__ = [
    "ParsedFrame",
    "parse_tracebacks",
    "suffix_match",
    "emit_traceback_frames",
    "rank_for_frame",
]
