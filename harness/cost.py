"""Per-stage, per-instance cost tracker.

Tracks token usage and approximate USD for every LLM call. Provides a
per-instance budget cap with graceful degradation: when cumulative
spend reaches a configured fraction of the cap, the per-turn cost
guard fires (V10_DESIGN.md §3.4) and the agent finalizes its best
partial candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# Approximate per-million-token prices, USD. These are intentionally
# rough — exact pricing belongs in the model adapter, this module just
# needs ballpark numbers for the budget guard.
_DEFAULT_PRICES: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-opus-4-7":     {"input": 15.0, "output": 75.0},
    "claude-opus-4-6":     {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-6":   {"input":  3.0, "output": 15.0},
    "claude-haiku-4-5":    {"input":  0.8, "output":  4.0},
    # DeepSeek (approximate as of dev cycle)
    "deepseek-reasoner":   {"input":  0.55, "output": 2.19},
    "deepseek-v4-pro":     {"input":  0.55, "output": 2.19},
    # Local / unknown
    "_unknown_":           {"input":  1.0,  "output":  3.0},
}


Stage = Literal[
    "localization",
    "repro_gen",
    "patch_gen_pipeline",
    "patch_gen_agent",
    "validation",
    "selection",
    "minimization",
    "adversarial",
    "repo_notes",
    "reviewer_escalation",
    "other",
]


@dataclass
class StageCost:
    stage: Stage
    model: str
    input_tokens: int
    output_tokens: int
    usd: float


@dataclass
class CostTracker:
    instance_id: str
    cap_usd: float = 35.0
    soft_guard_fraction: float = 0.80  # fire per-turn guard at 80% of cap
    entries: list[StageCost] = field(default_factory=list)

    @property
    def total_usd(self) -> float:
        return sum(e.usd for e in self.entries)

    @property
    def soft_cap_reached(self) -> bool:
        return self.total_usd >= self.cap_usd * self.soft_guard_fraction

    @property
    def hard_cap_reached(self) -> bool:
        return self.total_usd >= self.cap_usd

    def record(
        self,
        stage: Stage,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        prices: dict[str, dict[str, float]] | None = None,
    ) -> StageCost:
        prices = prices or _DEFAULT_PRICES
        p = prices.get(model, prices["_unknown_"])
        usd = (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000
        e = StageCost(
            stage=stage,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            usd=usd,
        )
        self.entries.append(e)
        return e

    def by_stage(self) -> dict[Stage, float]:
        out: dict[Stage, float] = {}
        for e in self.entries:
            out[e.stage] = out.get(e.stage, 0.0) + e.usd
        return out

    def summary(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "total_usd": round(self.total_usd, 4),
            "cap_usd": self.cap_usd,
            "soft_cap_reached": self.soft_cap_reached,
            "hard_cap_reached": self.hard_cap_reached,
            "by_stage": {k: round(v, 4) for k, v in self.by_stage().items()},
            "n_calls": len(self.entries),
        }


__all__ = ["CostTracker", "StageCost", "Stage"]
