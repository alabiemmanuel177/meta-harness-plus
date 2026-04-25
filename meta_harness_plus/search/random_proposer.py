"""Random-search baseline (Tier 1.3 — strong baselines).

Standard reviewer challenge for any AutoML method: "does it beat random
search at matched compute?" A serious paper has to answer that. This
module provides a ``RandomProposer`` that uniformly samples harness
shapes from the same per-kind mutator pool the
``AttributionGuidedMutationProposer`` uses, *without* attribution-based
weighting.

If MH++ doesn't beat random search at the same number of total LLM
calls, the framework has nothing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

from ..attribution import AttributionTracker
from ..harness import Harness
from ..pareto import ParetoFrontier
from .proposer import ProposalResult, Proposer


@dataclass
class RandomProposer:
    """Proposes harnesses by sampling each component-kind uniformly.

    Unlike ``AttributionGuidedMutationProposer``, this does NOT condition
    on the frontier or attribution stats — every proposal is independent.
    That's the point: it's the null hypothesis for "search smarts matter."

    ``mutators``: dict from component kind to list of zero-arg factories.
    Same shape as the mutation proposer's mutators.
    ``parents``: optional initial harnesses to draw structure from. If
    empty, the random proposer needs every kind in mutators to be present
    (so it can build a full pipeline from scratch).
    """
    mutators: dict[str, list[Callable[[], object]]]
    seed: int = 0
    parents: Sequence[Harness] = ()
    _rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def propose(
        self,
        frontier: ParetoFrontier,
        attribution: AttributionTracker,
        n: int,
    ) -> ProposalResult:
        # If we have parent harnesses, mutate by replacing 1-3 random kinds.
        # If not, build from scratch by sampling one component per kind.
        if self.parents or len(frontier) > 0:
            seed_pool: list[Harness] = list(self.parents)
            for entry in frontier:
                h = entry.meta.get("harness")
                if h is not None:
                    seed_pool.append(h)
            if not seed_pool:
                return ProposalResult(harnesses=[], rationale="no parents")
            out: list[Harness] = []
            for _ in range(n):
                parent = self._rng.choice(seed_pool)
                child = parent
                # Random depth 1-3.
                depth = self._rng.choice([1, 2, 3])
                kinds_pool = list(self.mutators.keys())
                self._rng.shuffle(kinds_pool)
                for kind in kinds_pool[:depth]:
                    new_comp = self._rng.choice(self.mutators[kind])()
                    child = child.swap(kind, new_comp)
                out.append(child)
            return ProposalResult(
                harnesses=out,
                rationale=f"random over {len(seed_pool)} seed(s)",
            )

        # Build-from-scratch path: for each kind, pick one factory at random.
        out_scratch: list[Harness] = []
        kinds_in_order = list(self.mutators.keys())
        for _ in range(n):
            comps = []
            for kind in kinds_in_order:
                factories = self.mutators[kind]
                if factories:
                    comps.append(self._rng.choice(factories)())
            out_scratch.append(Harness(components=comps))
        return ProposalResult(
            harnesses=out_scratch,
            rationale="random from scratch",
        )
