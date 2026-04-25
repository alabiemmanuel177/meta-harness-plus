"""LLM-backed harness proposer.

Reads the filesystem run log written by ``RunLogger`` (faithful to the
original Meta-Harness's proposer-facing filesystem interface), builds a
diagnostic prompt, asks the LLM for structured JSON proposals, validates
them through the registry, and returns ``Harness`` objects.

Contrast with the original paper: it lets the LLM access the log via raw
shell tools (``grep``, ``cat``) for maximum freedom. We instead assemble a
*curated* diagnostic prompt that includes:

1. the registry's available components (so the LLM knows the action space),
2. the current Pareto frontier (so it knows what's already good),
3. per-component attribution stats (so it knows which slots are high-value),
4. a short "exploration gap" summary (which registered components have
   NEVER appeared on the frontier — the motivation being the finding from
   main branch that attribution-guided greedy search can miss whole regions
   of the harness space).

That last item is our novel contribution on this branch: **explicit
exploration-gap surfacing.** Goes beyond both the original paper's
unstructured filesystem interface and the main-branch mutation proposer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..attribution import AttributionTracker
from ..harness import Harness
from ..pareto import ParetoFrontier
from ..search.proposer import ProposalResult
from .client import LLMClient, LLMResponse
from .registry import ComponentRegistry


SYSTEM = """You are a harness-search proposer for the Meta-Harness++ framework.
Each iteration you propose NEW candidate harnesses that advance the Pareto \
frontier over (accuracy ↑, tokens ↓, latency ↓).

A harness is an ordered pipeline. Components run in this order:
  retriever -> reranker -> fewshot -> formatter -> predictor -> voter

Reranker is optional (null_reranker is a no-op). When present, it reorders \
``ctx.retrieved`` between retrieval and few-shot selection — use it to \
surface a more diverse or higher-relevance set of examples than raw \
retrieval order.

Respond with STRICTLY VALID JSON of the form:
{
  "rationale": "one-sentence explanation of your strategy for this batch",
  "proposals": [
    {
      "components": [
        {"kind": "retriever", "name": "...", "config": {...}},
        {"kind": "reranker",  "name": "...", "config": {...}},
        {"kind": "fewshot",   "name": "...", "config": {...}},
        {"kind": "formatter", "name": "...", "config": {...}},
        {"kind": "predictor", "name": "...", "config": {...}},
        {"kind": "voter",     "name": "...", "config": {...}}
      ]
    },
    ...
  ]
}

No prose outside the JSON. Use only component kind+name pairs listed as \
available. Respect the pipeline order above. If you don't want a reranker, \
use reranker/null_reranker — don't omit the slot."""


@dataclass
class DiagnosticContext:
    """Parsed snapshot of the run log, used to build the prompt."""
    frontier: list[dict]
    attribution: dict[str, dict]   # kind -> {mean_delta, n, variance}
    available: list[dict]          # full component specs incl. allowed config fields
    exploration_gap: list[tuple[str, str]]  # (kind, name) never seen on frontier
    iteration: int
    n_requested: int

    def to_user_prompt(self) -> str:
        lines: list[str] = []
        lines.append(f"Iteration: {self.iteration}. Propose {self.n_requested} harnesses.")
        lines.append("")
        # Token-budget hint — explicit guidance on the cost axis. Without
        # this, the proposer biases toward expensive shapes (more retrieval,
        # more samples, more reasoning). The hint targets strict Pareto
        # dominance: discovered points should sit BELOW the cheapest current
        # frontier point on the token axis, not above.
        if self.frontier:
            best_acc_pt = max(self.frontier, key=lambda e: e.get("score", {}).get("accuracy", 0))
            cheapest_pt = min(self.frontier, key=lambda e: e.get("score", {}).get("tokens", float("inf")))
            best_acc = best_acc_pt.get("score", {}).get("accuracy", 0)
            best_acc_tok = best_acc_pt.get("score", {}).get("tokens", 0)
            cheap_acc = cheapest_pt.get("score", {}).get("accuracy", 0)
            cheap_tok = cheapest_pt.get("score", {}).get("tokens", 0)
            lines.append("# Token-budget guidance (priority signal)")
            lines.append(
                f"  Best accuracy on frontier: {best_acc:.2f} @ {best_acc_tok:.0f} tokens"
            )
            lines.append(
                f"  Cheapest on frontier:      {cheap_acc:.2f} @ {cheap_tok:.0f} tokens"
            )
            lines.append(
                f"  STRICT WIN target: accuracy >= {best_acc:.2f} at <= {cheap_tok:.0f} tokens. "
                f"Aim there. Heavier shapes only if they hit accuracy strictly above {best_acc:.2f}."
            )
            lines.append("")

            # Per-class accuracy of the best harness — failure-mode signal.
            # If a class is at 0.6 while others are 1.0, the proposer should
            # try components that fix that class (more retrieval for that
            # class's hard items, CoT, etc.).
            best_score_dict = best_acc_pt.get("score", {})
            per_class = best_score_dict.get("per_class_accuracy", [])
            if per_class:
                lines.append("# Per-class accuracy of best-accuracy frontier point")
                lines.append("  (focus your next proposals on the WEAKEST classes)")
                # Sort weakest first.
                pairs = sorted(per_class, key=lambda kv: kv[1] if isinstance(kv, (list, tuple)) else 0)
                for kv in pairs:
                    if isinstance(kv, (list, tuple)) and len(kv) == 2:
                        klass, acc = kv
                        bar = "#" * max(1, int(acc * 10))
                        lines.append(f"  {klass:20s}  {acc:.2f}  {bar}")
                lines.append("")
        lines.append("# Available components (kind/name — allowed config fields)")
        for spec in self.available:
            req = ", ".join(spec.get("required_config_fields", [])) or "–"
            allowed = ", ".join(spec.get("allowed_config_fields", [])) or "–"
            lines.append(
                f"  - {spec['kind']}/{spec['name']}  "
                f"required: {req}  |  allowed: {allowed}"
            )
        lines.append("")
        lines.append("IMPORTANT: Use ONLY config field names listed above. Unknown fields "
                     "are rejected. Use 'k' for retriever/fewshot k-values, NOT 'top_k'.")
        lines.append("")
        lines.append("# Current Pareto frontier")
        if not self.frontier:
            lines.append("  (empty)")
        for e in self.frontier:
            s = e.get("score", {})
            shape = " + ".join(
                f"{c['kind']}/{c.get('name','?')}"
                + (f"(k={c['k']})" if 'k' in c else "")
                + (f"(n={c['n_samples']})" if 'n_samples' in c else "")
                for c in e.get("describe", [])
            )
            lines.append(
                f"  - {e.get('candidate_id','?')}  "
                f"acc={s.get('accuracy',0):.2f}  "
                f"tok={s.get('tokens',0):.1f}  "
                f"lat={s.get('latency_ms',0):.1f}  :: {shape}"
            )
        lines.append("")
        lines.append("# Component attribution (drop-one ablation deltas)")
        if not self.attribution:
            lines.append("  (no attribution data yet)")
        for kind, stats in sorted(self.attribution.items(),
                                  key=lambda kv: -kv[1].get("ewma_delta",
                                                            kv[1].get("mean_delta", 0.0))):
            mean = stats.get("mean_delta", 0.0)
            ewma = stats.get("ewma_delta", mean)
            n = stats.get("n", 0)
            var = stats.get("variance", 0.0)
            lines.append(
                f"  - {kind}: ewma={ewma:+.3f}  mean={mean:+.3f}  n={n}  var={var:.3f}"
            )
        if self.attribution:
            lines.append("  (ewma weights recent ablations more — trust it over mean for this iter)")
        lines.append("")
        lines.append("# Exploration gaps (components NEVER on the frontier)")
        if not self.exploration_gap:
            lines.append("  (none — full coverage)")
        else:
            for k, n in self.exploration_gap:
                lines.append(f"  - {k}/{n}")
        lines.append("")
        lines.append("Propose harnesses that either (a) advance the frontier, or (b) "
                     "explore the gaps above — especially gaps whose kinds have high "
                     "or unknown attribution. Emit JSON.")
        return "\n".join(lines)


class LLMProposer:
    """Filesystem-reading LLM proposer."""

    def __init__(
        self,
        *,
        client: LLMClient,
        registry: ComponentRegistry,
        run_dir: str | Path,
        temperature: float = 0.4,
        max_tokens: int = 1500,
    ):
        self.client = client
        self.registry = registry
        self.run_dir = Path(run_dir)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.last_response: LLMResponse | None = None
        self._iter = 0

    # --- filesystem reading ---

    def _read_frontier(self) -> list[dict]:
        path = self.run_dir / "frontier.json"
        if not path.exists():
            return []
        return json.loads(path.read_text())

    def _read_attribution(self) -> dict[str, dict]:
        path = self.run_dir / "attribution_stats.json"
        if not path.exists():
            return {}
        raw = json.loads(path.read_text())
        # RunLogger writes the dict of AttributionStats verbatim; normalize field names.
        out = {}
        for kind, stats in raw.items():
            if isinstance(stats, dict):
                out[kind] = {
                    "mean_delta": stats.get("mean_delta", 0.0),
                    "ewma_delta": stats.get("ewma_delta", stats.get("mean_delta", 0.0)),
                    "n": stats.get("n", 0),
                    "variance": stats.get("m2", 0.0) / stats.get("n", 1)
                    if stats.get("n", 0) > 1 else 0.0,
                }
        return out

    def _exploration_gap(self, frontier: list[dict]) -> list[tuple[str, str]]:
        on_frontier: set[tuple[str, str]] = set()
        for e in frontier:
            for c in e.get("describe", []):
                on_frontier.add((c.get("kind", ""), c.get("name", "")))
        return [kn for kn in self.registry.available() if kn not in on_frontier]

    # --- prompt / response ---

    def _build_context(self, n: int) -> DiagnosticContext:
        frontier = self._read_frontier()
        attribution = self._read_attribution()
        available = self.registry.describe_available()
        gap = self._exploration_gap(frontier)
        return DiagnosticContext(
            frontier=frontier,
            attribution=attribution,
            available=available,
            exploration_gap=gap,
            iteration=self._iter,
            n_requested=n,
        )

    def _parse_json(self, text: str) -> dict:
        """Extract the first valid JSON object from the response."""
        # Strip common wrappers: markdown fences, leading prose.
        cleaned = text.strip()
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
        if fence:
            cleaned = fence.group(1)
        else:
            # Fall back to the first { … } balanced block.
            start = cleaned.find("{")
            if start < 0:
                raise ValueError(f"no JSON object in response: {text[:200]}")
            depth = 0
            end = -1
            for i, ch in enumerate(cleaned[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end < 0:
                raise ValueError("unbalanced JSON braces in response")
            cleaned = cleaned[start:end]
        return json.loads(cleaned)

    # --- Proposer protocol ---

    def propose(
        self,
        frontier: ParetoFrontier,
        attribution: AttributionTracker,
        n: int,
    ) -> ProposalResult:
        self._iter += 1
        ctx = self._build_context(n)
        resp = self.client.complete(
            system=SYSTEM,
            user=ctx.to_user_prompt(),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        self.last_response = resp

        try:
            payload = self._parse_json(resp.text)
        except (ValueError, json.JSONDecodeError) as e:
            return ProposalResult(harnesses=[], rationale=f"parse_error: {e}")

        rationale = payload.get("rationale", "")
        proposals = payload.get("proposals", [])
        harnesses: list[Harness] = []
        errors: list[str] = []
        for i, prop in enumerate(proposals):
            comps = prop.get("components", [])
            try:
                h = self.registry.build_harness(comps)
                harnesses.append(h)
            except Exception as e:  # noqa: BLE001 — registry raises ValueError; defensive catch-all
                errors.append(f"proposal[{i}]: {e}")
        if errors:
            rationale = (rationale + " | invalid: " + "; ".join(errors)).strip()
        return ProposalResult(harnesses=harnesses, rationale=rationale)
