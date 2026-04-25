"""Hyperband multi-bracket scheduling (alternative to plain successive halving).

Standard SuccessiveHalving uses a single bracket: start with N candidates
at budget r, halve, double budget, repeat. This allocates compute fairly
but commits to a specific (n, r) tradeoff before seeing results.

Hyperband (Li et al. 2017) runs *multiple* brackets at different (n, r)
tradeoffs in parallel:

  - Bracket s_max: many candidates × small budget each (wide exploration)
  - Bracket s_max-1: fewer candidates × bigger budget each
  - ...
  - Bracket 0: few candidates × full budget each (deep exploitation)

Then picks the best across brackets. Hedges between exploration and
exploitation — "I don't know if the right strategy is to try lots of
shapes briefly or a few shapes carefully, so do both and trust the
winner."

For our search:
- "Resource" is the screen subset size (number of eval items)
- "Candidates" are harnesses proposed by the proposer
- max_resource is the screen_size config parameter

Returns survivors across all brackets — the SearchRunner can then
do a full-eval on each survivor and admit the best to the Pareto
frontier as usual.

Reference: Li, Jamieson, DeSalvo, Rostamizadeh, Talwalkar (2017),
"Hyperband: A Novel Bandit-Based Approach to Hyperparameter
Optimization." JMLR 18(185):1-52.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Generic, Sequence, TypeVar

from .halving import HalvingResult, SuccessiveHalving
from .scorer import ScoreVector

T = TypeVar("T")


@dataclass
class HyperbandResult(Generic[T]):
    survivors: list[T]
    per_bracket: list[HalvingResult[T]] = field(default_factory=list)
    total_evaluations: int = 0


class HyperbandScheduler:
    """Hyperband scheduler over a candidate pool.

    Parameters mirror the original paper:
    - ``max_resource`` (R): the largest budget allocatable per candidate
      (= screen size). Must be a positive integer.
    - ``reduction_factor`` (η, default 3): halving factor between rounds.
      The paper recommends η in [3, 4]; we default to 3 since with our
      typical eval sizes (50-100) η=3 produces 3-4 brackets, which is
      a useful range.

    The scheduler computes ``s_max = floor(log_η(R))`` and runs brackets
    s = s_max, s_max - 1, ..., 0. Bracket s starts with::

        n_s = ceil((s_max + 1) / (s + 1)) * η^s    candidates
        r_s = R / η^s                              initial budget per candidate

    Each bracket runs successive halving with reduction factor η.
    """

    def __init__(
        self,
        max_resource: int,
        reduction_factor: int = 3,
        final_keep: int = 1,
    ) -> None:
        if max_resource < 1:
            raise ValueError(f"max_resource must be >= 1, got {max_resource}")
        if reduction_factor < 2:
            raise ValueError(f"reduction_factor must be >= 2, got {reduction_factor}")
        self.max_resource = max_resource
        self.eta = reduction_factor
        self.final_keep = final_keep
        self.s_max = int(math.log(max_resource) / math.log(self.eta))

    def brackets(self) -> list[tuple[int, int]]:
        """Compute (n_candidates, r_initial) for every bracket.

        Brackets ordered from widest (most candidates, smallest budget
        each) to narrowest (fewest candidates, largest budget each).
        """
        out: list[tuple[int, int]] = []
        for s in range(self.s_max, -1, -1):
            n = int(math.ceil((self.s_max + 1) / (s + 1)) * (self.eta ** s))
            r = max(1, int(self.max_resource / (self.eta ** s)))
            out.append((n, r))
        return out

    def run(
        self,
        propose: Callable[[int], Sequence[T]],
        evaluate: Callable[[T, int], ScoreVector],
    ) -> HyperbandResult[T]:
        """Run Hyperband.

        ``propose(n)`` returns up to n new candidates. The scheduler asks
        for candidates separately per bracket so the proposer can adapt
        to bracket size (e.g. a smart proposer might propose more
        diverse shapes for the wide bracket and more refined shapes for
        the narrow bracket). For most uses the same proposer can ignore
        the bracket-specific n.

        ``evaluate(candidate, k)`` is the same signature as for
        SuccessiveHalving — returns a ScoreVector measured on k items.
        """
        all_survivors: list[T] = []
        per_bracket: list[HalvingResult[T]] = []
        total = 0
        for n, r in self.brackets():
            cands = list(propose(n))
            if not cands:
                continue
            sh = SuccessiveHalving(
                k0=r, eta=self.eta, final_keep=self.final_keep,
            )
            result = sh.run(cands, evaluate)
            per_bracket.append(result)
            all_survivors.extend(result.survivors)
            total += result.total_evaluations
        return HyperbandResult(
            survivors=all_survivors,
            per_bracket=per_bracket,
            total_evaluations=total,
        )
