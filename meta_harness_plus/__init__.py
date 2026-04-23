"""Meta-Harness++: multi-objective, budget-aware, attribution-guided harness search."""

from .harness import Harness, Component
from .scorer import ScoreVector, Scorer
from .task import Task, TaskExample
from .pareto import ParetoFrontier
from .halving import SuccessiveHalving
from .attribution import AttributionTracker
from .runner import SearchRunner, SearchConfig
from .baselines import rag_baseline, bare_baseline

__all__ = [
    "Harness",
    "Component",
    "ScoreVector",
    "Scorer",
    "Task",
    "TaskExample",
    "ParetoFrontier",
    "SuccessiveHalving",
    "AttributionTracker",
    "SearchRunner",
    "SearchConfig",
    "rag_baseline",
    "bare_baseline",
]
