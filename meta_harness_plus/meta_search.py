"""Self-improving search — search the search (Tier 8.3).

The outer loop optimizes the *configuration of the inner harness search
itself*: which proposer, which depth_weights, which halving k0/eta/keep,
how many iterations, how many eval/attribution repeats, etc. The inner
loop is the standard SearchRunner already in this codebase.

Conceptually: one fixed point of "Meta-Harness optimizes harnesses" is
"a Meta-Harness search whose own search-config was found by another
Meta-Harness search." This module is the first concrete step toward
that fixed point.

Mechanics, kept deliberately simple for the prototype:
- ``MetaCandidate``: a bundle of (SearchConfig, proposer_factory,
  seed_harnesses_factory) — one configuration of the inner search.
- ``MetaScorer``: runs one MetaCandidate's inner search to completion,
  rescores the best harness on a held-out test set, returns a
  ``MetaScoreVector`` over (peak_accuracy ↑, search_compute ↓,
  wall_seconds ↓).
- ``meta_pareto_search``: exhaustive eval of a candidate list + Pareto
  filter on the meta-axes. This is the obvious O(N²) version; the
  recursive version would reuse SuccessiveHalving + AttributionTracker
  on the meta-frontier (left as a follow-on once the simple version is
  battle-tested).

ROADMAP.md tier 8.3: highest-leverage long-bet. Once "search the search"
works, the framework becomes recursively self-improving — each new task
or model can have its own optimized search-config without manual tuning.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .attribution import AttributionTracker
from .components import baseline_for as default_baseline_for
from .harness import Harness
from .runner import SearchConfig, SearchRunner
from .scorer import Scorer
from .search.proposer import Proposer
from .task import Task, TaskExample


@dataclass(frozen=True)
class MetaScoreVector:
    """How well an inner-search configuration performed.

    Three meta-axes for Pareto comparison:
    - ``peak_accuracy``: held-out accuracy of the inner search's best
      discovered harness. Maximize.
    - ``search_compute``: how much budget the inner search consumed
      (proxy: # candidates that were full-evaluated, including seeds).
      Minimize.
    - ``wall_seconds``: real-time elapsed for the inner search.
      Minimize.

    Intentionally separate from ``ScoreVector`` (per-harness) — the
    meta-axes have different semantics. The objective_signs convention
    matches: (+1, -1, -1).
    """
    peak_accuracy: float
    search_compute: int
    wall_seconds: float
    frontier_size: int = 0

    @staticmethod
    def objective_signs() -> tuple[int, int, int]:
        return (+1, -1, -1)


def _meta_dominates(a: MetaScoreVector, b: MetaScoreVector) -> bool:
    """``a`` strictly Pareto-dominates ``b`` on the meta-axes."""
    ge = (a.peak_accuracy >= b.peak_accuracy
          and a.search_compute <= b.search_compute
          and a.wall_seconds <= b.wall_seconds)
    gt = (a.peak_accuracy > b.peak_accuracy
          or a.search_compute < b.search_compute
          or a.wall_seconds < b.wall_seconds)
    return ge and gt


@dataclass
class MetaCandidate:
    """One configuration of the inner harness search.

    ``proposer_factory(seed) -> Proposer`` so each meta-eval can pin a
    deterministic seed for reproducibility.
    ``seed_harnesses_factory() -> list[Harness]`` lets the meta-search
    test different seeding strategies (BARE only, BARE+RAG, BARE+RAG+CoT
    seeds, etc.).
    """
    name: str
    config: SearchConfig
    proposer_factory: Callable[[int], Proposer]
    seed_harnesses_factory: Callable[[], list[Harness]] = field(
        default_factory=lambda: lambda: []
    )

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MetaCandidate) and self.name == other.name


@dataclass
class MetaCandidateResult:
    candidate: MetaCandidate
    score: MetaScoreVector


class MetaScorer:
    """Score a single MetaCandidate by running its inner search.

    Re-scores the inner search's best harness on ``holdout`` (which the
    inner search never sees). This separates "how well did the search
    optimize toward eval_set?" from "how well does its discovery
    generalize to held-out items?". Without this, meta-search would
    select for inner-search overfitting.
    """

    def __init__(
        self,
        task: Task,
        holdout: Sequence[TaskExample],
        *,
        seed: int = 0,
        n_repeats_holdout: int = 1,
        baseline_for_kind: Callable = default_baseline_for,
    ):
        self.task = task
        self.holdout = list(holdout)
        self.seed = seed
        self.n_repeats_holdout = n_repeats_holdout
        self.baseline_for_kind = baseline_for_kind

    def score(self, candidate: MetaCandidate) -> MetaScoreVector:
        scorer = Scorer(self.task)
        attribution = AttributionTracker(scorer, self.baseline_for_kind)
        proposer = candidate.proposer_factory(self.seed)
        seeds = candidate.seed_harnesses_factory()
        runner = SearchRunner(
            task=self.task,
            scorer=scorer,
            proposer=proposer,
            attribution=attribution,
            config=candidate.config,
            seed_harnesses=seeds,
        )
        t0 = time.time()
        state = runner.run()
        wall = time.time() - t0

        peak_acc = 0.0
        best = state.frontier.best_by_accuracy()
        if best is not None and self.holdout:
            hs = scorer.score(
                best.meta["harness"], self.holdout,
                n_repeats=self.n_repeats_holdout,
            )
            peak_acc = hs.accuracy

        # search_compute: count of full-eval/screen events in history.
        compute = sum(
            1 for h in state.history
            if h.get("phase") in {"seed", "survivor_full_eval"}
        )

        return MetaScoreVector(
            peak_accuracy=peak_acc,
            search_compute=compute,
            wall_seconds=wall,
            frontier_size=len(state.frontier),
        )


def meta_pareto_search(
    candidates: Sequence[MetaCandidate],
    scorer: MetaScorer,
) -> list[MetaCandidateResult]:
    """Exhaustive meta-search: score each candidate, return non-dominated set.

    Returns a list of ``MetaCandidateResult`` (candidate + score) where
    no result strictly dominates another. The simplest possible meta-
    search — useful when you have a small handful of search-configs to
    pick between (e.g. <20). For larger meta-spaces, swap in a recursive
    SuccessiveHalving / Pareto search; the score vector is already
    compatible with the existing framework.
    """
    scored = [(c, scorer.score(c)) for c in candidates]
    survivors: list[MetaCandidateResult] = []
    for i, (ci, si) in enumerate(scored):
        dominated = False
        for j, (cj, sj) in enumerate(scored):
            if i != j and _meta_dominates(sj, si):
                dominated = True
                break
        if not dominated:
            survivors.append(MetaCandidateResult(candidate=ci, score=si))
    return survivors


def best_meta_by_accuracy(
    results: Sequence[MetaCandidateResult],
) -> MetaCandidateResult | None:
    """Pick the highest-peak-accuracy survivor; tiebreak by lower compute."""
    if not results:
        return None
    return max(
        results,
        key=lambda r: (r.score.peak_accuracy, -r.score.search_compute),
    )
