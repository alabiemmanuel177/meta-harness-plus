"""Ensemble of harness proposers with diversity pressure (Tier 5.2).

Addresses the single-proposer mode-collapse limit the original Meta-Harness
paper acknowledges: when one proposer carries the search, it tends to
re-propose variations on shapes it already saw work, missing whole regions
of the action space.

This module wraps N proposers (different seeds, personas, or LLM clients),
runs them in parallel each iteration, deduplicates the union by structural
signature, and returns the diversity-filtered candidates. The combined
budget is held equal to a single-proposer call: each child gets
``n // len(proposers)`` proposals, so the ensemble doesn't simply spend
more compute — it spends the same compute differently.

Empirical claim (testable on the toy task): ensemble of K proposers with
distinct seeds discovers a strict superset of the unique harness
signatures that any single member discovers at matched total compute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..attribution import AttributionTracker
from ..harness import Harness
from ..pareto import ParetoFrontier
from .proposer import ProposalResult, Proposer


def harness_signature(h: Harness) -> tuple:
    """Hashable shape-signature for diversity comparison.

    Captures (kind, name, key config knobs) per component. Two harnesses
    with the same signature are structurally identical from the search's
    point of view — same Pareto vector under deterministic eval.
    """
    sig = []
    for c in h.components:
        cfg = c.config()
        # Pull the config knobs that affect cost/accuracy:
        knobs = tuple(sorted(
            (k, v) for k, v in cfg.items()
            if k not in {"kind", "name"}
        ))
        sig.append((cfg.get("kind"), cfg.get("name"), knobs))
    return tuple(sig)


def diversity_count(harnesses: Sequence[Harness]) -> int:
    """Number of unique shape signatures in the set."""
    return len({harness_signature(h) for h in harnesses})


@dataclass
class EnsembleProposer:
    """Wraps multiple proposers, runs them per-iteration, dedupes by signature.

    ``proposers``: sequence of any objects matching the Proposer protocol.
    Each gets called with ``n // len(proposers)`` proposals (rounded up,
    minimum 1).

    ``dedup``: when True (default), the union is filtered so each unique
    shape signature appears at most once. Diversity over count.

    The ensemble itself satisfies the Proposer protocol, so it drops in
    wherever a proposer is expected (SearchRunner, bakeoff scripts).
    """
    proposers: Sequence[Proposer]
    dedup: bool = True

    def __post_init__(self) -> None:
        if not self.proposers:
            raise ValueError("EnsembleProposer needs >= 1 child proposer")

    def propose(
        self,
        frontier: ParetoFrontier,
        attribution: AttributionTracker,
        n: int,
    ) -> ProposalResult:
        per = max(1, n // len(self.proposers))
        bag: list[Harness] = []
        rationales: list[str] = []
        for i, p in enumerate(self.proposers):
            r = p.propose(frontier=frontier, attribution=attribution, n=per)
            bag.extend(r.harnesses)
            if r.rationale:
                rationales.append(f"[p{i}] {r.rationale}")

        if not self.dedup:
            return ProposalResult(harnesses=bag, rationale=" || ".join(rationales))

        seen: set[tuple] = set()
        unique: list[Harness] = []
        for h in bag:
            sig = harness_signature(h)
            if sig not in seen:
                seen.add(sig)
                unique.append(h)
        rationales.append(
            f"[ensemble] dedup {len(bag)} -> {len(unique)} unique signatures"
        )
        return ProposalResult(harnesses=unique, rationale=" || ".join(rationales))
