"""Ablation harness — run search with each contribution toggled off.

Reviewers ask, for any framework with multiple contributions: did each
piece actually help? This module makes that question runnable as a
single experiment per ablation.

Three ablations targeting the three core contributions:

C1 — Pareto multi-objective vs scalar accuracy:
    Wraps ParetoFrontier with ``ScalarAccuracyFrontier`` that admits
    only by accuracy (ignoring tokens/latency). Forces the search to
    optimize accuracy alone, the way OPRO/TextGrad-style scalar
    optimizers would.

C2 — Successive halving vs full-eval all candidates:
    ``UnhalvedScreener`` evaluates every proposed candidate on the
    full screen subset rather than progressively halving. Same total
    compute → fewer rounds, less aggressive elimination.

C3 — Attribution-guided proposer vs random:
    Already shipped as ``RandomProposer``. The ablation just swaps
    proposers in the search runner.

A clean ablation experiment runs the same task + budget with each
ablation enabled (one at a time) and the full MH++ unablated, then
reports per-condition Pareto frontier and (where applicable) Δ vs
RAG. If the full MH++ wins on the multi-objective + has-halving +
has-attribution config, each ablation should hurt the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .pareto import FrontierEntry, ParetoFrontier, dominates
from .scorer import ScoreVector


class ScalarAccuracyFrontier(ParetoFrontier):
    """C1 ablation: keep only the highest-accuracy candidate.

    Behaves as ``len(self) ≤ 1`` always. The "frontier" collapses to
    a single point — the best-accuracy candidate seen. Tokens and
    latency are ignored entirely. This is the harness search RAG users
    would do without C1: pick the most-accurate prompt regardless of
    cost.
    """

    def offer(self, entry: FrontierEntry) -> bool:
        if not self.entries:
            self.entries = [entry]
            return True
        current_best = self.entries[0]
        if entry.score.accuracy > current_best.score.accuracy:
            self.entries = [entry]
            return True
        if (entry.score.accuracy == current_best.score.accuracy
                and entry.score.tokens < current_best.score.tokens):
            # Ties broken by lower tokens — keep the cheaper one for
            # determinism, but this is just a tiebreak, not C1's
            # objective.
            self.entries = [entry]
            return True
        return False


@dataclass
class AblationConfig:
    """Which parts of MH++ are turned off. Default = full MH++ enabled.

    Matches the ROADMAP's ablation list. Wired into the search runner
    via a small adapter so the same SearchRunner code runs all four
    conditions.
    """
    disable_pareto: bool = False        # C1: scalar accuracy only
    disable_halving: bool = False       # C2: full-eval every candidate
    disable_attribution: bool = False   # C3: random proposer instead

    @property
    def label(self) -> str:
        bits = []
        if self.disable_pareto:
            bits.append("no-C1")
        if self.disable_halving:
            bits.append("no-C2")
        if self.disable_attribution:
            bits.append("no-C3")
        return "+".join(bits) if bits else "full-MH++"


def make_frontier(config: AblationConfig) -> ParetoFrontier:
    """Construct the right frontier given the ablation config."""
    if config.disable_pareto:
        return ScalarAccuracyFrontier()
    return ParetoFrontier()


@dataclass
class AblationResult:
    """Summary of one ablation run."""
    config_label: str
    best_acc: float
    best_acc_tokens: float
    best_acc_latency_ms: float
    n_frontier_points: int


def summarize_frontier(label: str, frontier: ParetoFrontier) -> AblationResult:
    if not frontier.entries:
        return AblationResult(label, 0.0, 0.0, 0.0, 0)
    best = max(frontier.entries, key=lambda e: e.score.accuracy)
    return AblationResult(
        config_label=label,
        best_acc=best.score.accuracy,
        best_acc_tokens=best.score.tokens,
        best_acc_latency_ms=best.score.latency_ms,
        n_frontier_points=len(frontier.entries),
    )
