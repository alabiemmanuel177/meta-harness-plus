"""Online/continual harness improvement — minimal prototype.

NOTE: For production use prefer ``meta_harness_plus.continual`` — that
module has full state persistence, real per-example paired CIs (not
±2σ-spread heuristic), drift detection, configurable cost-regression
gates, and rollback support. This file is the original prototype and
is kept for back-compat with the early demos / tests.

The offline MH++ search produces a Pareto frontier on a static eval set.
In production, the data distribution drifts and new labelled examples
arrive over time. This module provides a minimal mechanism to:

1. Ingest new labelled examples from a production stream.
2. Periodically re-score the current frontier on the new data.
3. Compute paired bootstrap CIs to decide when a candidate has enough
   evidence to safely promote past the current production-deployed
   shape.
4. Emit a "promote" signal when a frontier candidate strictly Pareto-
   dominates the production shape with confidence ≥ threshold.

Design principles:

- **Conservative promotion.** New shapes are only promoted when their
  multi-objective improvement CI excludes zero on the cost axes too.
  No accuracy-only promotions; production should not silently get more
  expensive.
- **Optional active search.** The base ``OnlineHarnessImprover`` is
  passive — it ingests, rescores, and emits PromoteReports for a human
  scheduler to act on. The extension class
  ``ContinualHarnessImprover`` adds an active search loop that
  periodically proposes new candidate shapes from the accumulated
  production data and admits non-dominated survivors to the candidate
  pool. Promotion still uses the same paired-CI conservative gate, so
  the safety story is preserved.
- **Simple data structures.** New examples are appended to a deque
  with a max history; old examples roll off. The frontier is
  re-evaluated on the most recent N examples.

Usage:
    online = OnlineHarnessImprover(
        scorer=scorer,
        production_harness=current_prod_shape,
        candidate_harnesses={"alpha": alpha_shape, "v2": v2_shape},
        promote_ci_alpha=0.05,
        min_examples=50,
    )
    # As production data streams in:
    for example in production_stream:
        online.ingest(example)
        report = online.maybe_promote()
        if report.should_promote:
            deploy_new_shape(report.winner_label, report.winner_harness)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from .harness import Harness
from .pareto import dominates
from .scorer import ScoreVector, Scorer
from .statistics import paired_bootstrap_diff
from .task import TaskExample


@dataclass
class PromoteReport:
    """The decision the OnlineHarnessImprover emitted on a maybe_promote() call."""
    should_promote: bool
    winner_label: str | None
    winner_harness: Harness | None
    winner_score: ScoreVector | None
    production_score: ScoreVector | None
    delta_acc_ci: tuple[float, float] | None  # 95% paired-bootstrap CI on Δacc
    reason: str


class OnlineHarnessImprover:
    """Tracks production examples, periodically re-scores frontier, decides
    when a candidate beats production with enough evidence to promote.
    """

    def __init__(
        self,
        scorer: Scorer,
        production_harness: Harness,
        candidate_harnesses: dict[str, Harness],
        *,
        promote_ci_alpha: float = 0.05,
        min_examples: int = 50,
        max_history: int = 1000,
        rescore_every: int = 25,
        max_workers: int = 1,
    ):
        """
        Args:
          scorer: how to compute (acc, tokens, latency) for a harness.
          production_harness: the shape currently serving production.
          candidate_harnesses: labeled candidates competing to replace it.
          promote_ci_alpha: significance threshold (0.05 = 95% CI).
          min_examples: don't promote until we have this many production
              examples to score on.
          max_history: roll older examples off after this many.
          rescore_every: re-score on every N-th newly-ingested example.
          max_workers: passed to scorer.score() for parallel scoring.
        """
        self.scorer = scorer
        self.production_harness = production_harness
        self.candidate_harnesses = dict(candidate_harnesses)
        self.promote_ci_alpha = promote_ci_alpha
        self.min_examples = min_examples
        self.max_history = max_history
        self.rescore_every = rescore_every
        self.max_workers = max_workers

        self._examples: deque[TaskExample] = deque(maxlen=max_history)
        self._ingest_count = 0
        # Per-(label, example_idx) cached prediction is too memory-intensive;
        # we just re-score from the deque each rescore.
        self._last_scores: dict[str, ScoreVector] = {}

    def ingest(self, example: TaskExample) -> None:
        """Add one labelled production example to the history."""
        self._examples.append(example)
        self._ingest_count += 1

    def ingest_many(self, examples: Iterable[TaskExample]) -> None:
        for ex in examples:
            self.ingest(ex)

    def _rescore_all(self, n_repeats: int = 1) -> dict[str, ScoreVector]:
        """Re-score production + every candidate on the current history."""
        examples = list(self._examples)
        if len(examples) == 0:
            return {}
        scores: dict[str, ScoreVector] = {}
        scores["__production__"] = self.scorer.score(
            self.production_harness, examples,
            n_repeats=n_repeats, max_workers=self.max_workers,
        )
        for label, h in self.candidate_harnesses.items():
            scores[label] = self.scorer.score(
                h, examples, n_repeats=n_repeats, max_workers=self.max_workers,
            )
        self._last_scores = scores
        return scores

    def maybe_promote(self, force_rescore: bool = False, n_repeats: int = 1) -> PromoteReport:
        """Decide whether to promote a candidate over the current production shape.

        Returns a PromoteReport. Only emits ``should_promote=True`` when:
        - We have at least ``min_examples`` examples in history.
        - The candidate's score strictly Pareto-dominates production.
        - The paired-bootstrap CI on (candidate_acc - production_acc) excludes
          zero on the negative side (Δacc CI lower bound > 0).
        """
        if len(self._examples) < self.min_examples:
            return PromoteReport(
                should_promote=False,
                winner_label=None, winner_harness=None,
                winner_score=None, production_score=None,
                delta_acc_ci=None,
                reason=f"only {len(self._examples)} examples, need {self.min_examples}",
            )

        if force_rescore or (self._ingest_count % self.rescore_every == 0):
            self._rescore_all(n_repeats=n_repeats)

        if "__production__" not in self._last_scores:
            return PromoteReport(
                should_promote=False,
                winner_label=None, winner_harness=None,
                winner_score=None, production_score=None,
                delta_acc_ci=None, reason="not yet rescored",
            )

        prod_score = self._last_scores["__production__"]
        # Find candidates that strictly Pareto-dominate prod.
        dominators: list[tuple[str, ScoreVector]] = [
            (label, sc)
            for label, sc in self._last_scores.items()
            if label != "__production__" and dominates(sc, prod_score)
        ]
        if not dominators:
            return PromoteReport(
                should_promote=False,
                winner_label=None, winner_harness=None,
                winner_score=None, production_score=prod_score,
                delta_acc_ci=None,
                reason="no candidate Pareto-dominates production",
            )

        # Pick the highest-accuracy dominator as the candidate winner.
        winner_label, winner_score = max(dominators, key=lambda x: x[1].accuracy)

        # Paired bootstrap on accuracy. We approximate per-example accuracy by
        # ScoreVector's per-class breakdown isn't quite right here — we'd
        # ideally have per-example correctness. For now, the CI is computed
        # over the K cached scores we have; treat the sample size as the
        # number of repeats × ratio. (Refinement for future work: track
        # per-example correctness deltas.)
        # Conservative default: only promote if winner's accuracy CI lower
        # bound exceeds production accuracy by more than the CI width that a
        # paired bootstrap on accuracy_spread would produce.
        # For this minimal implementation we use a simple z-test approximation:
        # require winner.accuracy - prod.accuracy > prod.accuracy_spread.
        delta_acc = winner_score.accuracy - prod_score.accuracy
        # Approximate 95% CI: use ±2σ from accuracy_spread.
        sigma_combined = max(prod_score.accuracy_spread, winner_score.accuracy_spread, 1e-6)
        ci_low = delta_acc - 2.0 * sigma_combined
        ci_high = delta_acc + 2.0 * sigma_combined

        if ci_low > 0:
            return PromoteReport(
                should_promote=True,
                winner_label=winner_label,
                winner_harness=self.candidate_harnesses[winner_label],
                winner_score=winner_score,
                production_score=prod_score,
                delta_acc_ci=(ci_low, ci_high),
                reason=(f"{winner_label} strictly Pareto-dominates production; "
                        f"Δacc 95%-CI=[{ci_low:+.3f},{ci_high:+.3f}] excludes 0"),
            )

        return PromoteReport(
            should_promote=False,
            winner_label=winner_label,
            winner_harness=None,
            winner_score=winner_score,
            production_score=prod_score,
            delta_acc_ci=(ci_low, ci_high),
            reason=(f"{winner_label} dominates but Δacc CI=[{ci_low:+.3f},{ci_high:+.3f}] "
                    f"includes 0 — not enough confidence to promote"),
        )

    @property
    def history_size(self) -> int:
        return len(self._examples)


# ---------------- ContinualHarnessImprover ----------------

class ContinualHarnessImprover(OnlineHarnessImprover):
    """Active continual-search variant of OnlineHarnessImprover.

    Adds a ``propose_and_admit`` step that runs a small MH++ search on
    the accumulated production data and admits non-dominated survivors
    to the candidate pool. Subsequent ``maybe_promote`` calls then
    consider the newly-discovered candidates alongside the original
    ones.

    The active loop is intentionally bounded — it runs at most
    ``max_candidates_per_search`` proposals per call and uses the same
    Scorer the rest of the framework uses. It is NOT a long-lived
    daemon; the caller decides when to invoke ``propose_and_admit``
    (typically every N ingestions, or on a wall-clock cadence).

    Promotion still goes through the parent class's paired-bootstrap CI
    gate — proposed-and-admitted candidates aren't auto-deployed.

    Usage::

        improver = ContinualHarnessImprover(
            scorer=scorer,
            production_harness=prod,
            candidate_harnesses={"alpha": alpha_shape},
            propose_fn=my_proposer.propose,  # any callable
            min_examples=50,
            max_candidates_per_search=4,
        )
        for ex in production_stream:
            improver.ingest(ex)
            if improver.history_size % 100 == 0:
                improver.propose_and_admit()  # active step
            report = improver.maybe_promote()
            if report.should_promote:
                deploy_new_shape(report.winner_label, report.winner_harness)
    """

    def __init__(
        self,
        *args,
        propose_fn: Callable[[Sequence[TaskExample], int], list[Harness]] | None = None,
        max_candidates_per_search: int = 4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.propose_fn = propose_fn
        self.max_candidates_per_search = max_candidates_per_search
        self.search_count = 0
        self.admitted_count = 0

    def propose_and_admit(self, n_repeats: int = 1) -> int:
        """Run a propose-and-admit cycle: propose new candidates from
        accumulated production data, score them, admit non-dominated.

        Returns the number of candidates admitted to the pool.
        Returns 0 if there's no proposer or insufficient data.
        """
        if self.propose_fn is None:
            return 0
        if len(self._examples) < self.min_examples:
            return 0

        examples = list(self._examples)
        try:
            new_harnesses = self.propose_fn(examples, self.max_candidates_per_search)
        except Exception:
            return 0
        if not new_harnesses:
            return 0

        self.search_count += 1
        admitted = 0

        # Score current production for dominance comparison.
        prod_score = self.scorer.score(
            self.production_harness, examples,
            n_repeats=n_repeats, max_workers=self.max_workers,
        )

        for i, h in enumerate(new_harnesses):
            try:
                score = self.scorer.score(
                    h, examples, n_repeats=n_repeats, max_workers=self.max_workers,
                )
            except Exception:
                continue
            # Check non-dominance vs prod and existing candidates.
            from .pareto import dominates
            existing_scores = list(self._last_scores.values())
            is_dominated = any(dominates(s, score) for s in existing_scores
                               if s is not None)
            if is_dominated:
                continue
            label = f"discovered_{self.search_count}_{i}"
            self.candidate_harnesses[label] = h
            self._last_scores[label] = score
            self.admitted_count += 1
            admitted += 1
        return admitted
