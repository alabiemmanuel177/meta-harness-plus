"""Proposer interface.

An LLM-backed proposer (Claude Code-style, as in the original paper) would
subclass this and read the filesystem-laid-out run log written by the runner.
We ship a deterministic mutation-based proposer for tests; the LLM version is
a documented integration point, intentionally left as a stub here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..attribution import AttributionTracker
from ..harness import Harness
from ..pareto import ParetoFrontier


@dataclass
class ProposalResult:
    harnesses: list[Harness]
    rationale: str = ""


class Proposer(Protocol):
    def propose(
        self,
        frontier: ParetoFrontier,
        attribution: AttributionTracker,
        n: int,
    ) -> ProposalResult:
        """Return ``n`` candidate harnesses."""
        ...
