"""Stage 1g — LLM rerank with per-strategy-rank features.

Designed around the Phase 1 finding (V10_DESIGN.md §9): aggregation by
flat score destroys signal. Stage 1g consumes the per-strategy ranks
from Stage 1b–1c (BM25 across 3 query strategies, embedding,
traceback, …) directly as features. A file ranked 1 in
`bm25_first_paragraph` and 7 in `embedding` is structurally different
from a file ranked 12 in everything; the reranker prompt makes that
distinction visible to the model.

The reranker is intentionally NOT given full file contents — only
file paths + one-line skeleton summaries + per-strategy ranks. That
keeps the prompt size predictable (~ 100 K tokens cap fits dev_50
even on django-class instances) and avoids context-budget blowups
flagged in V10_DESIGN.md §3.4.

Output: a tuple of ``RankedFile`` (typed dataclass from
``harness.localization_signals``) carrying the model's final scoring
+ rationale + which upstream signals supported each kept candidate.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from harness.localization_signals import (
    RankedFile,
    RerankerOutput,
    STAGE_1G_RERANK,
    UPSTREAM_SIGNAL_KINDS,
)
from harness.views import InstanceView


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateForRerank:
    """One file presented to the reranker.

    Carries the per-strategy ranks/scores and a one-line skeleton
    summary. NEVER carries full file content — the reranker scores by
    structure, not by reading code.
    """
    file_path: str
    per_strategy_rank: dict          # {strategy_label: int (1-indexed)}
    per_strategy_score: dict         # {strategy_label: float}
    file_summary: str                # one-line summary from the skeleton
    n_lines: int                     # file size hint


@dataclass
class RerankerInput:
    view: InstanceView
    candidates: tuple                # tuple[CandidateForRerank, ...]
    top_k: int = 5


# ---------------------------------------------------------------------------
# Token-budget guard
# ---------------------------------------------------------------------------


# Approximate chars/token; same heuristic V10 uses elsewhere.
CHARS_PER_TOKEN = 4
DEFAULT_PROMPT_TOKEN_CAP = 100_000


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _truncate_candidates_to_budget(
    candidates: tuple,
    issue_text: str,
    *,
    cap_tokens: int = DEFAULT_PROMPT_TOKEN_CAP,
) -> tuple:
    """Drop the lowest-priority candidates until the projected prompt
    fits under ``cap_tokens``. Priority = best (lowest) per-strategy
    rank across all strategies that named the file.
    """
    if not candidates:
        return candidates
    base_overhead = _estimate_tokens(issue_text) + 800  # system + framing
    per_candidate_chars = 250  # one row of the candidate table
    per_candidate_tokens = per_candidate_chars // CHARS_PER_TOKEN

    capacity = cap_tokens - base_overhead
    if capacity <= 0:
        # Issue alone already exceeds budget; emit just the top-1 and
        # accept truncation downstream.
        return tuple(_sort_by_priority(candidates)[:1])

    max_n = capacity // per_candidate_tokens
    if max_n >= len(candidates):
        return candidates

    sorted_cands = _sort_by_priority(candidates)
    return tuple(sorted_cands[:max_n])


def _sort_by_priority(candidates) -> list:
    """Sort candidates: best (lowest) per-strategy rank wins; ties by
    number of strategies that named the file (more = better)."""
    def _key(c):
        ranks = c.per_strategy_rank.values()
        best_rank = min(ranks) if ranks else 10**6
        return (best_rank, -len(c.per_strategy_rank))
    return sorted(candidates, key=_key)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = """You rank Python files by how likely each is to contain the bug \
described in a software-engineering issue.

You are given:
  - the issue text,
  - a list of CANDIDATE FILES, each with: file_path, n_lines, file_summary, \
and a per-strategy rank table.

Use per-strategy ranks as evidence. A file ranked 1 in first_paragraph \
but absent from embedding is STRUCTURALLY DIFFERENT from a file ranked \
7 in both — the former is a strong signal from a focused query, the \
latter is a weak signal from a broad one. Do not just pick files that \
appear in the most strategies; weigh STRENGTH of signal (low rank in \
ANY strategy = strong) over BREADTH (named in many strategies but \
ranked low in each = weak).

Strategies you may see:
  - bm25_full_issue      : BM25 over the full issue text
  - bm25_first_paragraph : BM25 over the issue's first paragraph
  - bm25_extracted_symbols: BM25 over identifiers extracted from the issue
  - embedding            : dense embedding similarity (BAAI/bge-large-en-v1.5)
  - traceback            : Python stack frames matched to repo files
  - symbol_grep          : ripgrep matches on rare identifiers (when present)
  - git_archeology       : commits referencing the file (when present)
  - dep_graph            : import-graph neighbors of other candidates (when present)

Output JSON ONLY in this exact shape:

{
  "ranked": [
    {
      "file_path": "...",
      "final_score": 0.95,
      "rationale": "ranked highly because <which strategies were decisive>",
      "upstream_signals": ["bm25", "embedding"],
      "upstream_best_rank": {"bm25": 1, "embedding": 7}
    },
    ...
  ]
}

upstream_signals must be a subset of: bm25, embedding, traceback, \
symbol_grep, git_archeology, dep_graph.
- "bm25_full_issue" / "bm25_first_paragraph" / "bm25_extracted_symbols" all \
collapse to "bm25" in upstream_signals (drop the strategy suffix).
- upstream_best_rank uses the same collapsed kinds; keep the BEST (lowest) \
rank seen across that kind's strategies.

Pick the top-K candidates. Higher final_score = more likely the bug is \
in this file. Score range [0, 1]."""


def _format_strategy_rank_row(c: CandidateForRerank) -> str:
    """Render the candidate's per-strategy ranks compactly."""
    parts = []
    # Sort strategy keys for stable output.
    for sk in sorted(c.per_strategy_rank.keys()):
        rank = c.per_strategy_rank[sk]
        parts.append(f"{sk}={rank}")
    return "[" + ", ".join(parts) + "]"


def build_user_prompt(inp: RerankerInput) -> str:
    """Build the user-facing prompt with the candidate table."""
    lines: list[str] = []
    lines.append(f"Repository: {inp.view.repo} @ {inp.view.base_commit[:12]}")
    lines.append("")
    lines.append("## Issue")
    lines.append(inp.view.problem_statement.strip())
    lines.append("")
    lines.append(f"## Candidate files ({len(inp.candidates)} total)")
    lines.append("")
    lines.append("Each row: `file_path | n_lines | file_summary | strategy_ranks`")
    lines.append("Strategy ranks: lower is better (rank 1 = top hit for that strategy).")
    lines.append("")
    for c in inp.candidates:
        ranks = _format_strategy_rank_row(c)
        summary = c.file_summary or "(no skeleton summary)"
        # Truncate summary to ~120 chars to keep the table tight.
        if len(summary) > 120:
            summary = summary[:117] + "..."
        lines.append(f"- `{c.file_path}` | {c.n_lines} lines | {summary} | {ranks}")
    lines.append("")
    lines.append(
        f"Pick the top-{inp.top_k} files most likely to contain the bug. "
        f"Return JSON only."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


_BM25_STRATEGY_PREFIX = "bm25_"


def _collapse_strategy_kind(strategy_label: str) -> str:
    """Collapse 'bm25_full_issue' → 'bm25', 'embedding' → 'embedding'."""
    if strategy_label.startswith(_BM25_STRATEGY_PREFIX):
        return "bm25"
    return strategy_label


def _extract_json_object(text: str) -> dict:
    """Tolerant JSON extractor — handles ```json fences and trailing
    prose."""
    text = text.strip()
    # Strip code fences.
    fence_match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$",
                            text, re.DOTALL | re.MULTILINE)
    if fence_match:
        text = fence_match.group(1).strip()
    # Try direct parse.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find the first balanced {...} block.
    start = text.find("{")
    if start < 0:
        raise ValueError(f"no JSON object found in output: {text[:200]!r}")
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError(f"unbalanced JSON in output: {text[:200]!r}")


def parse_reranker_output(text: str, *, top_k: int) -> tuple:
    """Parse the LLM's JSON output into a tuple of RankedFile.
    Raises ValueError on malformed output."""
    data = _extract_json_object(text)
    if "ranked" not in data:
        raise ValueError(f"reranker output missing 'ranked' key: {list(data.keys())}")
    out: list[RankedFile] = []
    for entry in data["ranked"][:top_k]:
        try:
            file_path = entry["file_path"]
            final_score = float(entry["final_score"])
            rationale = entry.get("rationale", "")
            upstream_signals = tuple(entry.get("upstream_signals", []))
            upstream_best_rank = entry.get("upstream_best_rank") or None
            # Coerce values to ints.
            if upstream_best_rank is not None:
                upstream_best_rank = {
                    k: int(v) for k, v in upstream_best_rank.items()
                }
            # Filter unknown signal kinds (model hallucinations).
            upstream_signals = tuple(
                s for s in upstream_signals if s in UPSTREAM_SIGNAL_KINDS
            )
            if upstream_best_rank is not None:
                upstream_best_rank = {
                    k: v for k, v in upstream_best_rank.items()
                    if k in UPSTREAM_SIGNAL_KINDS
                }
            out.append(RankedFile(
                file_path=file_path,
                final_score=final_score,
                rationale=rationale,
                upstream_signals=upstream_signals,
                upstream_best_rank=upstream_best_rank,
            ))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"reranker entry malformed: {entry!r} ({exc})"
            )
    return tuple(out)


# ---------------------------------------------------------------------------
# Top-level rerank()
# ---------------------------------------------------------------------------


@dataclass
class RerankResult:
    ranked_files: tuple             # tuple[RankedFile, ...]
    signal: RerankerOutput          # the typed signal for trajectory log
    duration_s: float
    cost_usd: float


class RerankerError(RuntimeError):
    """Raised when the reranker fails after retries."""


def rerank(
    inp: RerankerInput,
    *,
    role: str = "reranker",
    model_override: Optional[str] = None,
    cap_tokens: int = DEFAULT_PROMPT_TOKEN_CAP,
    n_retries: int = 1,
    trajectory_writer=None,
    cost_tracker=None,
) -> RerankResult:
    """Run the LLM rerank.

    Token budget enforcement:
      - Total prompt projected; candidates truncated by per-strategy
        priority until under ``cap_tokens``.
      - Default 100 K tokens (well under any 200K-context model).

    Output handling:
      - JSON-only response requested via response_format_json=True
      - Tolerant parser handles fenced output / trailing prose
      - Up to ``n_retries`` retries on parse failure (default 1 retry,
        i.e., 2 total attempts)
    """
    from harness.llm.clients import complete_chat, model_for_role, price_for_model

    candidates = _truncate_candidates_to_budget(
        inp.candidates, inp.view.problem_statement, cap_tokens=cap_tokens,
    )
    truncated_input = RerankerInput(
        view=inp.view, candidates=candidates, top_k=inp.top_k,
    )

    user_prompt = build_user_prompt(truncated_input)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    chosen_model = model_override or model_for_role(role)

    last_err: Optional[Exception] = None
    text = ""
    chat = None
    t0 = time.perf_counter()
    for attempt in range(n_retries + 1):
        chat = complete_chat(
            messages=messages,
            model=chosen_model,
            max_tokens=4096,
            temperature=0.0,
            response_format_json=True,
        )
        text = chat.text
        try:
            parsed = parse_reranker_output(text, top_k=inp.top_k)
            break
        except ValueError as exc:
            last_err = exc
            if attempt < n_retries:
                # Inject a follow-up instruction and try again.
                messages = messages + [
                    {"role": "assistant", "content": text},
                    {
                        "role": "user",
                        "content": (
                            "The previous response was not valid JSON in the "
                            "expected shape. Output ONLY the JSON object "
                            "with the 'ranked' key. No prose."
                        ),
                    },
                ]
            else:
                raise RerankerError(
                    f"reranker failed to produce valid JSON after "
                    f"{n_retries + 1} attempts: {last_err}"
                ) from last_err
    dur = time.perf_counter() - t0

    # Cost estimate.
    prices = price_for_model(chosen_model)
    cost = (
        chat.input_tokens * prices["input"] +
        chat.output_tokens * prices["output"]
    ) / 1_000_000

    sig = RerankerOutput(
        stage=STAGE_1G_RERANK,
        model=chosen_model,
        input_tokens=chat.input_tokens,
        output_tokens=chat.output_tokens,
        prompt_token_count=_estimate_tokens(user_prompt),
        ranked_files=parsed,
        skeleton_compression="symbol_summary_only",
    )

    if trajectory_writer is not None:
        trajectory_writer.write_signal(sig)

    if cost_tracker is not None:
        # Stage 1g rerank is a localization sub-stage per V10_DESIGN.md §3.2.
        cost_tracker.record(
            "localization",
            model=chosen_model,
            input_tokens=chat.input_tokens,
            output_tokens=chat.output_tokens,
        )

    return RerankResult(
        ranked_files=parsed,
        signal=sig,
        duration_s=dur,
        cost_usd=cost,
    )


__all__ = [
    "CandidateForRerank",
    "RerankerInput",
    "RerankResult",
    "RerankerError",
    "DEFAULT_PROMPT_TOKEN_CAP",
    "build_user_prompt",
    "parse_reranker_output",
    "rerank",
    "SYSTEM_PROMPT",
]
