"""Phase 3 pipeline-path patch generation — single-shot K candidates.

Per docs/V10_DESIGN_PHASE3.md §3.2:

  - K=2 in dev (temperatures 0.0, 0.5); K=4 in leaderboard (0.0, 0.3,
    0.7, 1.0).
  - Each candidate is one independent LLM call; no candidate sees
    any other.
  - Per-instance cost cap $1 dev / $5 production via
    ``harness.cost.CostTracker``. Exceeding cap mid-K returns the
    candidates produced so far (not an error).
  - Output JSON shape: ``{"rationale": "...", "diff": "..."}``.

Model dispatch: ``role="patch_generator_pipeline"`` resolves through
``harness.config.models.yaml`` to ``deepseek-chat`` by default. The
leaderboard run env-var-overrides this without touching code.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass

from harness.cost import CostTracker
from harness.localization_signals import RankedFile
from harness.patch_gen.context import (
    build_patch_gen_context_with_superset_check,
)
from harness.patch_gen.views import (
    ContextOversizeError,
    PatchCandidate,
    PatchGenContext,
    PatchGenError,
)
from harness.views import InstanceView


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------


DEFAULT_PIPELINE_TEMPERATURES: tuple[float, ...] = (0.0, 0.5)
DEFAULT_PIPELINE_COST_CAP_USD: float = 1.0
DEFAULT_MAX_OUTPUT_TOKENS: int = 4096


# ---------------------------------------------------------------------------
# Prompt build
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = """\
You are a software engineer fixing a bug in a Python codebase.

You will be given:
  - The issue text describing the bug.
  - The repo's test directory layout.
  - The contents of the files most likely to contain the bug
    (selected by the localization stage).

Produce a UNIFIED DIFF that fixes the bug. Rules:

  1. The diff MUST apply cleanly with `git apply` at the repo root.
     Use repo-relative paths (no leading `/` or `a/`/`b/` prefix
     manipulation beyond what `git diff` normally produces).
  2. Edit ONLY the files needed to fix the bug. Do not modify
     unrelated code.
  3. Do not modify test files. Tests are evaluated separately.
  4. Keep the diff MINIMAL — the smallest change that fixes the
     described bug.
  5. Match the repo's existing style (indentation, naming,
     docstrings).

Output ONLY a JSON object. No prose, no markdown fences.

  {
    "rationale": "<1-2 sentence explanation of the fix>",
    "diff": "<unified diff text — multiple lines OK>"
  }

The `diff` field is a single string containing the full unified
diff (newlines preserved). Begin with the standard `diff --git ...`
header for each file changed.
"""


def _build_user_prompt(ctx: PatchGenContext) -> str:
    parts: list[str] = []
    parts.append("# Issue\n\n")
    parts.append(ctx.problem_statement.strip())
    parts.append("\n\n# Test directories\n\n")
    for d in ctx.test_directives:
        parts.append(f"  - {d}\n")
    parts.append("\n# Likely-relevant files (top-K from localizer)\n\n")
    for snip in ctx.files:
        truncation_note = " (truncated)" if snip.truncated else ""
        parts.append(
            f"## {snip.path} (score={snip.final_score:.2f}){truncation_note}\n\n"
        )
        if snip.rationale:
            parts.append(f"_Localizer rationale: {snip.rationale}_\n\n")
        parts.append("```python\n")
        parts.append(snip.content)
        if not snip.content.endswith("\n"):
            parts.append("\n")
        parts.append("```\n\n")
    parts.append("# Output\n\nReturn the JSON object now.\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


def _parse_response(text: str) -> dict:
    """Parse the JSON-only response into ``{"rationale": ..., "diff": ...}``.

    Tolerates accidental markdown fences (some models add them despite
    the system prompt's instruction).
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PatchGenError(
            f"pipeline output is not valid JSON: {exc}"
        ) from exc
    for k in ("rationale", "diff"):
        if k not in data:
            raise PatchGenError(f"pipeline output missing key: {k!r}")
        if not isinstance(data[k], str):
            raise PatchGenError(
                f"pipeline output {k!r} must be string, "
                f"got {type(data[k]).__name__}"
            )
    if not data["diff"].strip():
        raise PatchGenError("pipeline output diff is empty")
    return data


# ---------------------------------------------------------------------------
# candidate_id construction
# ---------------------------------------------------------------------------


def _candidate_id_for_temperature(t: float) -> str:
    """Stable per-temperature candidate id. Strips the decimal point
    so '0.5' -> 'pipe_t05' and '0.0' -> 'pipe_t0'."""
    if t == 0.0:
        return "pipe_t0"
    s = f"{t:.2f}".rstrip("0").rstrip(".")
    s = s.replace(".", "")
    return f"pipe_t{s}"


def _cost_for_chat(model: str, input_tokens: int, output_tokens: int) -> float:
    from harness.llm.clients import price_for_model
    p = price_for_model(model)
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000.0


# ---------------------------------------------------------------------------
# Single-shot generator
# ---------------------------------------------------------------------------


def generate_pipeline_one_shot(
    *,
    ctx: PatchGenContext,
    temperature: float,
    attempt_index: int,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> PatchCandidate:
    """One LLM call, one PatchCandidate. No retry; no cost tracker
    (the K-loop owner manages cost). Raises ``PatchGenError`` if the
    response can't be parsed or is missing required fields."""
    from harness.llm.clients import complete_chat, model_for_role

    model = model_for_role("patch_generator_pipeline")
    user_prompt = _build_user_prompt(ctx)

    t_start = time.perf_counter()
    chat = complete_chat(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        max_tokens=max_output_tokens,
        temperature=temperature,
        response_format_json=True,
    )
    elapsed = time.perf_counter() - t_start

    data = _parse_response(chat.text)
    cost = _cost_for_chat(chat.model, chat.input_tokens, chat.output_tokens)

    return PatchCandidate(
        instance_id=ctx.instance_id,
        candidate_id=_candidate_id_for_temperature(temperature),
        diff=data["diff"],
        source_route="pipeline",
        source_temperature=temperature,
        source_attempt_index=attempt_index,
        generator_model=chat.model,
        generator_input_tokens=chat.input_tokens,
        generator_output_tokens=chat.output_tokens,
        generation_cost_usd=cost,
        duration_s=elapsed,
    )


# ---------------------------------------------------------------------------
# K-candidate orchestrator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineGenerationResult:
    """Result of running the K-candidate pipeline path on one instance."""

    candidates: tuple[PatchCandidate, ...]
    total_cost_usd: float
    cost_cap_hit: bool
    parse_failures: tuple[str, ...]   # one entry per failed attempt


def generate_pipeline(
    *,
    view: InstanceView,
    ranked_files: list[RankedFile],
    sandbox,
    temperatures: tuple[float, ...] = DEFAULT_PIPELINE_TEMPERATURES,
    cost_cap_usd: float = DEFAULT_PIPELINE_COST_CAP_USD,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    repro_context_files: tuple[str, ...] | None = None,
) -> PipelineGenerationResult:
    """Run the pipeline path for one instance.

    Steps:
      1. ``build_patch_gen_context_with_superset_check`` — single
         entrypoint for context assembly. Raises
         ``ContextOversizeError`` for cap-hitters; caller (the
         routing layer) catches and skips, falling through to the
         agent path in P3c.
      2. For each temperature:
           a. ``generate_pipeline_one_shot``.
           b. Track cost. If cumulative > cap, stop and return
              candidates produced so far.
           c. On parse error, record but continue to next
              temperature (don't abort the whole K-loop on one bad
              response).
      3. Return ``PipelineGenerationResult`` with all candidates
         + cost telemetry.

    The caller (Phase 3 orchestrator) is responsible for handing
    candidates to Phase 4 for validation. This function does NOT
    call validators — it just generates.

    Args:
      view: InstanceView (firewall-clean).
      ranked_files: Phase 1 reranker top-K.
      sandbox: started Sandbox at base_commit.
      temperatures: K-tuple of sampling temperatures.
      cost_cap_usd: per-instance cost cap. Default $1.
      max_output_tokens: per-call output cap. Default 4096.
      repro_context_files: forwarded to the context builder for the
        superset assertion. None bypasses the assertion.
    """
    ctx = build_patch_gen_context_with_superset_check(
        view=view,
        ranked_files=ranked_files,
        sandbox=sandbox,
        repro_context_files=repro_context_files,
    )

    tracker = CostTracker(instance_id=view.instance_id, cap_usd=cost_cap_usd)
    candidates: list[PatchCandidate] = []
    parse_failures: list[str] = []
    cost_cap_hit = False

    for attempt_idx, t in enumerate(temperatures):
        if tracker.hard_cap_reached:
            log.info(
                "[pipeline] %s: cost cap $%.4f reached after %d candidates; "
                "skipping remaining temperatures %s",
                view.instance_id, tracker.total_usd, len(candidates),
                temperatures[attempt_idx:],
            )
            cost_cap_hit = True
            break

        try:
            cand = generate_pipeline_one_shot(
                ctx=ctx,
                temperature=t,
                attempt_index=attempt_idx,
                max_output_tokens=max_output_tokens,
            )
        except PatchGenError as exc:
            log.info(
                "[pipeline] %s: parse failure at temperature=%.2f: %s",
                view.instance_id, t, exc,
            )
            parse_failures.append(f"t={t}: {exc}")
            continue

        tracker.record(
            stage="patch_gen_pipeline",
            model=cand.generator_model,
            input_tokens=cand.generator_input_tokens,
            output_tokens=cand.generator_output_tokens,
        )
        candidates.append(cand)

    return PipelineGenerationResult(
        candidates=tuple(candidates),
        total_cost_usd=tracker.total_usd,
        cost_cap_hit=cost_cap_hit,
        parse_failures=tuple(parse_failures),
    )


__all__ = [
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "DEFAULT_PIPELINE_COST_CAP_USD",
    "DEFAULT_PIPELINE_TEMPERATURES",
    "PipelineGenerationResult",
    "generate_pipeline",
    "generate_pipeline_one_shot",
]
