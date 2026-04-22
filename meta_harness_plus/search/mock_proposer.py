"""Deterministic attribution-guided mutation proposer.

Offline, no LLM. Maintains diversity by mutating different slots with
probabilities pulled from the attribution tracker. This is the piece that
closes the loop between C3 (attribution) and sample-efficient search.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from ..attribution import AttributionTracker
from ..harness import Harness
from ..pareto import ParetoFrontier
from .proposer import ProposalResult


@dataclass
class AttributionGuidedMutationProposer:
    """Mutates one *or more* component-kinds per proposal.

    The choice of which kind to mutate is drawn from attribution weights
    (high-value kinds mutated more), with an epsilon floor for exploration.

    ``depth_weights`` controls how many mutations are applied to the parent
    in one proposal: ``{1: 0.5, 2: 0.35, 3: 0.15}`` means 50% of proposals
    mutate one slot, 35% mutate two, 15% mutate three. Compound mutations
    are necessary when single-slot changes can't escape the baseline's
    domination region (e.g. retriever-alone doesn't help accuracy, so it's
    dominated on cost; but retriever + fewshot together does).
    """
    mutators: dict[str, list[Callable[[], object]]]
    seed: int = 0
    epsilon: float = 0.2
    temperature: float = 0.5
    depth_weights: dict[int, float] = field(
        default_factory=lambda: {1: 0.4, 2: 0.4, 3: 0.2}
    )
    _rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def _pick_kind(self, attribution: AttributionTracker) -> str:
        kinds = list(self.mutators.keys())
        weights = attribution.mutation_weights(temperature=self.temperature)
        # Epsilon-greedy: with prob epsilon pick uniformly; else follow weights.
        if self._rng.random() < self.epsilon or not weights:
            return self._rng.choice(kinds)
        # Restrict to kinds we actually know how to mutate.
        filtered = [(k, weights.get(k, 0.0)) for k in kinds if k in weights]
        if not filtered or sum(w for _, w in filtered) == 0.0:
            return self._rng.choice(kinds)
        # Weighted choice.
        total = sum(w for _, w in filtered)
        r = self._rng.random() * total
        acc = 0.0
        for k, w in filtered:
            acc += w
            if r <= acc:
                return k
        return filtered[-1][0]

    def propose(
        self,
        frontier: ParetoFrontier,
        attribution: AttributionTracker,
        n: int,
    ) -> ProposalResult:
        # Seed corpus: harnesses on the frontier. If empty we can't propose.
        if len(frontier) == 0:
            return ProposalResult(harnesses=[], rationale="empty frontier")
        parents = [entry.meta["harness"] for entry in frontier.entries if "harness" in entry.meta]
        if not parents:
            return ProposalResult(harnesses=[], rationale="no harness payloads on frontier")
        out: list[Harness] = []
        rationales: list[str] = []
        depths = list(self.depth_weights.keys())
        depth_probs = list(self.depth_weights.values())
        for _ in range(n):
            parent: Harness = self._rng.choice(parents)
            depth = self._rng.choices(depths, weights=depth_probs, k=1)[0]
            child = parent
            applied: list[str] = []
            used_kinds: set[str] = set()
            for _ in range(depth):
                kind = self._pick_kind(attribution)
                # Avoid redundantly mutating the same kind twice within one proposal.
                tries = 0
                while kind in used_kinds and tries < 5:
                    kind = self._rng.choice(list(self.mutators.keys()))
                    tries += 1
                used_kinds.add(kind)
                mutator = self._rng.choice(self.mutators[kind])
                new_comp = mutator()  # type: ignore[operator]
                child = child.swap(kind, new_comp)  # type: ignore[arg-type]
                applied.append(f"{kind}->{getattr(new_comp, 'name', '?')}")
            out.append(child)
            rationales.append("+".join(applied))
        return ProposalResult(harnesses=out, rationale="; ".join(rationales))
