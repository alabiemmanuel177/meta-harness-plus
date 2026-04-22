"""End-to-end search loop.

Ties together: propose → successive-halving screen → full-eval the survivor →
attribution on the survivor → update Pareto frontier → log everything to disk.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from .attribution import AttributionTracker
from .halving import SuccessiveHalving
from .harness import Harness
from .logging_utils import RunLogger
from .pareto import FrontierEntry, ParetoFrontier
from .scorer import Scorer, ScoreVector
from .search.proposer import Proposer
from .task import Task, TaskExample


@dataclass
class SearchConfig:
    n_iterations: int = 10
    proposals_per_iter: int = 6
    screen_size: int = 6
    full_eval_size: int | None = None  # None = use full eval_set
    halving_k0: int = 4
    halving_eta: int = 2
    halving_final_keep: int = 2
    screen_seed: int = 0
    run_dir: str | None = None

    # Reproducibility knobs (branch: reproducibility):
    # - eval_repeats > 1 runs full/screen evals multiple times and aggregates
    #   with median accuracy / mean cost. Combats nondeterministic LLM backends.
    # - attribution_screen_size: separate (typically larger) subset for
    #   drop-one ablation. Small screens make attribution noise-dominated;
    #   decoupling lets you spend more budget where it matters.
    eval_repeats: int = 1
    screen_repeats: int = 1            # repeats during successive-halving screen
    attribution_repeats: int = 1
    attribution_screen_size: int | None = None  # None = use screen_size


@dataclass
class SearchState:
    frontier: ParetoFrontier
    attribution: AttributionTracker
    history: list[dict] = field(default_factory=list)


class SearchRunner:
    def __init__(
        self,
        task: Task,
        scorer: Scorer,
        proposer: Proposer,
        attribution: AttributionTracker,
        config: SearchConfig,
        seed_harnesses: Sequence[Harness] = (),
    ):
        self.task = task
        self.scorer = scorer
        self.proposer = proposer
        self.attribution = attribution
        self.config = config
        self.seed_harnesses = list(seed_harnesses)
        self._id_counter = itertools.count(1)
        self.logger = RunLogger(config.run_dir) if config.run_dir else None

    # --- helpers ---
    def _next_id(self) -> str:
        return f"cand_{next(self._id_counter):04d}"

    def _full_eval_set(self) -> list[TaskExample]:
        if self.config.full_eval_size is None:
            return list(self.task.eval_set)
        return list(self.task.eval_set[: self.config.full_eval_size])

    def _screen_set(self) -> list[TaskExample]:
        return self.task.screen_subset(self.config.screen_size, seed=self.config.screen_seed)

    def _attribution_set(self) -> list[TaskExample]:
        """Possibly-larger subset for drop-one ablation.

        Defaults to the screen set if ``attribution_screen_size`` is unset.
        """
        size = self.config.attribution_screen_size
        if size is None:
            return self._screen_set()
        # Use a distinct seed so the attribution set and screen set don't
        # overlap completely — deliberately different slices reduce leakage
        # between what halving already saw and what attribution ablates on.
        return self.task.screen_subset(size, seed=self.config.screen_seed + 1)

    def _admit(self, frontier: ParetoFrontier, cand_id: str, harness: Harness, score: ScoreVector) -> bool:
        entry = FrontierEntry(
            candidate_id=cand_id,
            score=score,
            meta={"harness": harness, "describe": harness.describe()},
        )
        return frontier.offer(entry)

    # --- main loop ---
    def run(self) -> SearchState:
        frontier = ParetoFrontier()
        state = SearchState(frontier=frontier, attribution=self.attribution)

        # Seed the frontier with any initial harnesses.
        screen = self._screen_set()
        attribution_screen = self._attribution_set()
        full = self._full_eval_set()
        for h in self.seed_harnesses:
            cid = self._next_id()
            score = self.scorer.score(h, full, n_repeats=self.config.eval_repeats)
            admitted = self._admit(frontier, cid, h, score)
            self.attribution.analyze(
                cid, h, attribution_screen, full_score=None,
                n_repeats=self.config.attribution_repeats,
            )
            state.history.append({
                "phase": "seed",
                "candidate_id": cid,
                "admitted": admitted,
                "score": score.as_tuple(),
                "accuracy": score.accuracy,
            })
            if self.logger:
                self.logger.record_candidate(cid, h.describe())
                self.logger.record_score(cid, score)
                self.logger.event(phase="seed", candidate_id=cid, admitted=admitted,
                                  score=score.as_tuple())

        halving = SuccessiveHalving(
            k0=self.config.halving_k0,
            eta=self.config.halving_eta,
            final_keep=self.config.halving_final_keep,
        )

        for it in range(self.config.n_iterations):
            proposal = self.proposer.propose(
                frontier=frontier,
                attribution=self.attribution,
                n=self.config.proposals_per_iter,
            )
            if not proposal.harnesses:
                state.history.append({"phase": "propose", "iter": it, "n": 0})
                continue

            # Give each proposed harness a provisional id so eval logs are attributable.
            ids = [self._next_id() for _ in proposal.harnesses]

            # Screening evaluator closure — uses deterministic subset of screen set.
            def screen_eval(h: Harness, k: int) -> ScoreVector:
                return self.scorer.score(
                    h, screen[:k], n_repeats=self.config.screen_repeats,
                )

            result = halving.run(list(proposal.harnesses), screen_eval)

            # Map survivors back to ids.
            id_by_h = {id(h): i for h, i in zip(proposal.harnesses, ids)}
            for surv in result.survivors:
                cid = id_by_h[id(surv)]
                # Full eval on the survivor (median-over-repeats for robustness).
                full_score = self.scorer.score(
                    surv, full, n_repeats=self.config.eval_repeats,
                )
                admitted = self._admit(frontier, cid, surv, full_score)
                snapshots = self.attribution.analyze(
                    cid, surv, attribution_screen, full_score=full_score,
                    n_repeats=self.config.attribution_repeats,
                )
                state.history.append({
                    "phase": "survivor_full_eval",
                    "iter": it,
                    "candidate_id": cid,
                    "admitted": admitted,
                    "accuracy": full_score.accuracy,
                    "tokens": full_score.tokens,
                    "latency_ms": full_score.latency_ms,
                })
                if self.logger:
                    self.logger.record_candidate(cid, surv.describe())
                    self.logger.record_score(cid, full_score)
                    self.logger.record_attribution(cid, snapshots)

            # Record end-of-iter state to disk.
            if self.logger:
                self.logger.record_frontier([
                    {
                        "candidate_id": e.candidate_id,
                        "score": e.score,
                        "describe": e.meta.get("describe", []),
                    }
                    for e in frontier.entries
                ])
                self.logger.record_attribution_stats(self.attribution.stats)
                self.logger.event(
                    phase="iter_end",
                    iter=it,
                    frontier_size=len(frontier),
                    total_screen_evals=result.total_evaluations,
                )

        return state
