"""Multi-seed search runner + held-out test split (Tier 1.2).

Wraps SearchRunner: runs the same search N times with different seeds,
aggregates results, reports bootstrap confidence intervals on accuracy.

Held-out test split: takes a Task, returns (search_task, holdout_set).
Search runs against ``search_task.eval_set``; the final Pareto frontier
is then re-scored on ``holdout_set`` and that's what's reported. Cleanly
separates "what the search optimized for" from "what we measure."

This is tier 1.2 of ROADMAP.md — the statistical hygiene that turns a
single-seed point estimate into a defensible mean + CI.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .attribution import AttributionTracker
from .components import baseline_for as default_baseline_for
from .harness import Harness
from .pareto import FrontierEntry, ParetoFrontier
from .runner import SearchConfig, SearchRunner, SearchState
from .scorer import Scorer, ScoreVector
from .search.proposer import Proposer
from .statistics import BootstrapCI, bootstrap_ci
from .task import Task, TaskExample


def split_holdout(
    task: Task,
    *,
    holdout_frac: float = 0.3,
    seed: int = 0,
) -> tuple[Task, list[TaskExample]]:
    """Split task.eval_set into (search_eval, held_out_test).

    Returns a NEW Task whose eval_set is the search portion, and a list
    of held-out items the search never sees. Train set is unchanged
    (search legitimately uses train for retrieval/few-shot).

    Stratified by class: each class's items are split so both partitions
    have roughly the same class distribution.
    """
    if not 0.0 < holdout_frac < 1.0:
        raise ValueError(f"holdout_frac must be in (0, 1), got {holdout_frac}")

    rng = random.Random(seed)
    by_class: dict[str, list[TaskExample]] = {}
    for ex in task.eval_set:
        by_class.setdefault(ex.label, []).append(ex)

    search_eval: list[TaskExample] = []
    holdout: list[TaskExample] = []
    for klass, items in by_class.items():
        idx = list(range(len(items)))
        rng.shuffle(idx)
        n_holdout = max(1, int(round(holdout_frac * len(items))))
        holdout_idx = set(idx[:n_holdout])
        for i, item in enumerate(items):
            if i in holdout_idx:
                holdout.append(item)
            else:
                search_eval.append(item)

    new_task = Task(
        name=f"{task.name}_search",
        train=task.train,
        eval_set=search_eval,
        classes=task.classes,
    )
    return new_task, holdout


@dataclass
class SeedResult:
    seed: int
    state: SearchState
    holdout_scores: list[tuple[str, ScoreVector]]   # (candidate_id, score on holdout)


@dataclass
class MultiSeedResult:
    """Aggregated multi-seed search results."""
    seeds: list[int]
    per_seed: list[SeedResult]
    # CI for each candidate type (best-by-acc, best-by-cost, etc.) across seeds.
    best_acc_ci: BootstrapCI
    best_cost_at_top_acc_ci: BootstrapCI
    held_out_top_acc_per_seed: list[float]
    held_out_top_cost_per_seed: list[float]
    holdout_size: int

    def summary(self) -> str:
        return (
            f"seeds={self.seeds}  holdout={self.holdout_size}\n"
            f"held-out accuracy of search-best harness: "
            f"mean={self.best_acc_ci.mean:.3f} "
            f"[{self.best_acc_ci.low:.3f}, {self.best_acc_ci.high:.3f}] "
            f"({self.best_acc_ci.confidence:.0%} CI)\n"
            f"held-out tokens of search-best harness: "
            f"mean={self.best_cost_at_top_acc_ci.mean:.1f}"
        )


class MultiSeedRunner:
    """Runs the same search config N times with different seeds.

    Each seed gets:
    - a fresh proposer (built by ``proposer_factory(seed)``)
    - the same Task, Scorer, AttributionTracker (reset per seed) and SearchConfig

    After each search, the search-best harness (best by accuracy on the
    Pareto frontier) is re-scored on the held-out test set. The held-out
    accuracies and costs across seeds are aggregated with bootstrap CIs.
    """

    def __init__(
        self,
        task: Task,
        proposer_factory: Callable[[int], Proposer],
        config: SearchConfig,
        seed_harnesses: Sequence[Harness] = (),
        baseline_for_kind: Callable = default_baseline_for,
        holdout_frac: float = 0.3,
        holdout_seed: int = 0,
        n_eval_repeats_holdout: int = 3,
    ):
        self.task = task
        self.proposer_factory = proposer_factory
        self.config = config
        self.seed_harnesses = list(seed_harnesses)
        self.baseline_for_kind = baseline_for_kind
        self.holdout_frac = holdout_frac
        self.holdout_seed = holdout_seed
        self.n_eval_repeats_holdout = n_eval_repeats_holdout

    def run(self, seeds: Sequence[int]) -> MultiSeedResult:
        search_task, holdout = split_holdout(
            self.task, holdout_frac=self.holdout_frac, seed=self.holdout_seed,
        )
        scorer = Scorer(search_task)
        per_seed: list[SeedResult] = []
        held_out_top_acc: list[float] = []
        held_out_top_cost: list[float] = []

        for s in seeds:
            attribution = AttributionTracker(scorer, self.baseline_for_kind)
            proposer = self.proposer_factory(s)
            cfg = SearchConfig(**{**self.config.__dict__, "screen_seed": s})
            runner = SearchRunner(
                task=search_task,
                scorer=scorer,
                proposer=proposer,
                attribution=attribution,
                config=cfg,
                seed_harnesses=self.seed_harnesses,
            )
            state = runner.run()

            # Re-score the search-best harness on held-out.
            best = state.frontier.best_by_accuracy()
            holdout_scores: list[tuple[str, ScoreVector]] = []
            if best is not None:
                hs = scorer.score(
                    best.meta["harness"], holdout,
                    n_repeats=self.n_eval_repeats_holdout,
                )
                holdout_scores.append((best.candidate_id, hs))
                held_out_top_acc.append(hs.accuracy)
                held_out_top_cost.append(hs.tokens)

            per_seed.append(SeedResult(
                seed=s, state=state, holdout_scores=holdout_scores,
            ))

        return MultiSeedResult(
            seeds=list(seeds),
            per_seed=per_seed,
            best_acc_ci=bootstrap_ci(held_out_top_acc, seed=0),
            best_cost_at_top_acc_ci=bootstrap_ci(held_out_top_cost, seed=0),
            held_out_top_acc_per_seed=held_out_top_acc,
            held_out_top_cost_per_seed=held_out_top_cost,
            holdout_size=len(holdout),
        )
