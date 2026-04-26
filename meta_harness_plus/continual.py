"""Production-grade continual harness improvement.

This module is the production-quality successor to ``online.py``'s
minimal prototype. It adds:

- **Persistence.** All state is serializable to / loadable from a
  single JSON file, so a restart resumes mid-stream. Persisted state:
  the production harness's identity (label), candidate-pool labels,
  per-(label, example_idx) correctness vector, promotion log, drift
  windows, and counters.
- **Per-example correctness tracking.** Promotion CIs are real paired
  bootstrap CIs over per-example correctness deltas, not the
  approximate ±2σ-spread heuristic the prototype used. The tracker
  evaluates each candidate on each new example as it streams in,
  amortizing the cost.
- **Drift detection.** A sliding-window class-distribution shift test:
  for each window of recent examples, we estimate the empirical class
  distribution and compare it to a reference (typically the first
  ``baseline_window`` examples). Drift is reported as L1-distance
  between class distributions; configurable threshold raises a
  ``DriftAlert`` for the caller.
- **Active proposal scheduling.** A ``ContinualImprover.tick()`` loop
  decides each turn whether to propose, score, gate, or rollback,
  based on time-since-last-proposal + drift signal + ingest count.
- **Conservative promotion gates.** A candidate is promoted iff its
  paired-acc CI lower bound > 0 AND its tokens/latency don't regress
  beyond configurable absolute caps (`max_token_regression`,
  `max_latency_regression_ms`). Otherwise the report explains why.
- **Rollback.** A ``rollback()`` API restores the previous production
  shape and writes a rollback record to the promotion log. Useful when
  a promoted shape later degrades on fresh production data.

The design is the active continual-improvement loop sketched in the
roadmap; it is unit-tested deterministically without API calls and
demonstrated end-to-end in ``examples/continual_demo.py``.

Example::

    improver = ContinualImprover(
        scorer=scorer,
        production_label="v1",
        candidates={"v1": v1_shape, "v2": v2_shape},
        propose_fn=my_proposer,
        state_path="state/improver.json",
        promotion_gates=PromotionGates(min_examples=50,
                                       max_token_regression_pct=0.10),
    )
    for ex in production_stream:
        report = improver.tick(ex)
        if report.promoted:
            log_promotion(report)
        if report.drift_alert:
            page_oncall(report.drift_alert)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .harness import Harness
from .scorer import ScoreVector, Scorer
from .statistics import paired_bootstrap_diff, paired_t_test
from .task import TaskExample


# ----------------- shared types -----------------

@dataclass
class PromotionGates:
    """Conservative gates a candidate must clear to be promoted."""
    min_examples: int = 50
    paired_acc_ci_alpha: float = 0.05
    # Cost regressions allowed *as fractions* of the production score.
    # E.g. 0.10 = candidate may use up to 110% of production tokens.
    max_token_regression_pct: float = 0.10
    max_latency_regression_pct: float = 0.20
    # Hard absolute caps (ms / token count). 0 disables.
    max_token_regression_abs: float = 0.0
    max_latency_regression_abs_ms: float = 0.0
    # Promote at most every N ingestions to avoid flapping.
    cooldown_ingests: int = 25


@dataclass
class DriftAlert:
    window_size: int
    baseline_size: int
    l1_distance: float
    classes_top: list[tuple[str, float]]   # top-3 deltas
    threshold: float


@dataclass
class TickReport:
    """What ``ContinualImprover.tick()`` did this turn."""
    ingest_count: int
    promoted: bool = False
    promoted_from: str | None = None
    promoted_to: str | None = None
    promotion_reason: str = ""
    rolled_back: bool = False
    rollback_reason: str = ""
    drift_alert: DriftAlert | None = None
    proposed_count: int = 0
    admitted_count: int = 0


@dataclass
class PromotionLogEntry:
    timestamp: float
    ingest_count: int
    kind: str               # "promote" | "rollback" | "no-op"
    from_label: str
    to_label: str
    reason: str
    delta_acc_ci: tuple[float, float] | None = None
    n_paired_examples: int = 0


# ----------------- correctness tracker -----------------

class PerExampleTracker:
    """Per-(candidate_label, example_index) correctness over the production stream.

    Each ingested example is scored under every candidate (and the
    current production), with a binary correctness recorded. The
    tracker holds these as parallel lists indexed by ingest order, so
    paired-bootstrap CIs are real per-example pairings.

    Memory grows O(n_examples × n_candidates). For long-lived loops the
    caller can call ``trim(max_history)`` to drop older entries.
    """

    def __init__(self):
        # label -> list[int (0/1)] in ingest order.
        self.correctness: dict[str, list[int]] = {}
        # label -> running token / latency totals per example.
        self.tokens: dict[str, list[float]] = {}
        self.latency_ms: dict[str, list[float]] = {}
        self.examples: list[TaskExample] = []

    def register(self, label: str) -> None:
        """Add a fresh candidate. Future examples will be scored under it.
        Existing examples are NOT back-filled (cost would explode); the
        tracker just tracks from this point forward.
        """
        self.correctness.setdefault(label, [])
        self.tokens.setdefault(label, [])
        self.latency_ms.setdefault(label, [])

    def deregister(self, label: str) -> None:
        self.correctness.pop(label, None)
        self.tokens.pop(label, None)
        self.latency_ms.pop(label, None)

    def record(
        self,
        example: TaskExample,
        scored: dict[str, tuple[int, float, float]],
    ) -> None:
        """Append per-(label) results for one example.

        ``scored`` maps label -> (correct: 0/1, tokens, latency_ms).
        Labels not in ``scored`` get padded with NaN-ish defaults
        (0/0/0) which ``paired_acc()`` etc. handle by truncating to
        the first-registered offset.
        """
        self.examples.append(example)
        # All known labels, even ones not in `scored`, get padding so
        # parallel-list invariant holds.
        for label in self.correctness.keys():
            if label in scored:
                c, tok, lat = scored[label]
                self.correctness[label].append(int(c))
                self.tokens[label].append(float(tok))
                self.latency_ms[label].append(float(lat))
            else:
                # Unknown — typically because the candidate registered
                # later. We pad with -1 to mark "not measured" so
                # callers can filter.
                self.correctness[label].append(-1)
                self.tokens[label].append(0.0)
                self.latency_ms[label].append(0.0)

    def paired_acc(self, a: str, b: str) -> tuple[list[int], list[int]]:
        """Return the parallel correctness vectors for (a, b) over the
        examples where BOTH were measured."""
        ca = self.correctness.get(a, [])
        cb = self.correctness.get(b, [])
        n = min(len(ca), len(cb))
        out_a, out_b = [], []
        for i in range(n):
            if ca[i] == -1 or cb[i] == -1:
                continue
            out_a.append(ca[i])
            out_b.append(cb[i])
        return out_a, out_b

    def mean_cost(self, label: str) -> tuple[float, float, int]:
        """Mean (tokens, latency_ms, n) over measured examples."""
        toks = [t for t, c in zip(self.tokens.get(label, []),
                                  self.correctness.get(label, []))
                if c != -1]
        lats = [l for l, c in zip(self.latency_ms.get(label, []),
                                  self.correctness.get(label, []))
                if c != -1]
        if not toks:
            return (0.0, 0.0, 0)
        return (sum(toks) / len(toks), sum(lats) / len(lats), len(toks))

    def trim(self, max_history: int) -> None:
        for label in list(self.correctness.keys()):
            self.correctness[label] = self.correctness[label][-max_history:]
            self.tokens[label] = self.tokens[label][-max_history:]
            self.latency_ms[label] = self.latency_ms[label][-max_history:]
        self.examples = self.examples[-max_history:]

    def state_dict(self) -> dict:
        return {
            "correctness": dict(self.correctness),
            "tokens": dict(self.tokens),
            "latency_ms": dict(self.latency_ms),
            # Examples are not persisted (they may carry PII / be large).
            # Restoring loses them but the parallel-vector invariant
            # holds because they're indexed by ingest order, not by
            # example identity.
            "n_examples": len(self.examples),
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> "PerExampleTracker":
        t = cls()
        t.correctness = {k: list(v) for k, v in d.get("correctness", {}).items()}
        t.tokens = {k: list(v) for k, v in d.get("tokens", {}).items()}
        t.latency_ms = {k: list(v) for k, v in d.get("latency_ms", {}).items()}
        # ``examples`` is empty after restore; new examples will append from here.
        return t


# ----------------- drift detection -----------------

@dataclass
class DriftDetector:
    """Sliding-window class-distribution shift detector.

    L1 distance between the recent window's class distribution and a
    fixed baseline window. Configurable threshold; emits ``DriftAlert``
    when exceeded. Class identity is the example's ``label``; for
    intent-classification streams that's the right granularity. For
    open-ended tasks, callers can subclass and override
    ``_class_of(example)``.
    """
    window_size: int = 200
    baseline_size: int = 200
    threshold: float = 0.30

    def __post_init__(self):
        self._baseline: dict[str, int] = {}
        self._baseline_n: int = 0
        self._recent: list[str] = []

    def _class_of(self, ex: TaskExample) -> str:
        return ex.label

    def ingest(self, ex: TaskExample) -> None:
        cls = self._class_of(ex)
        if self._baseline_n < self.baseline_size:
            self._baseline[cls] = self._baseline.get(cls, 0) + 1
            self._baseline_n += 1
            return
        self._recent.append(cls)
        if len(self._recent) > self.window_size:
            # ring buffer
            self._recent = self._recent[-self.window_size:]

    def _baseline_dist(self) -> dict[str, float]:
        if self._baseline_n == 0:
            return {}
        return {k: v / self._baseline_n for k, v in self._baseline.items()}

    def _recent_dist(self) -> dict[str, float]:
        if not self._recent:
            return {}
        n = len(self._recent)
        out: dict[str, float] = {}
        for c in self._recent:
            out[c] = out.get(c, 0.0) + 1.0 / n
        return out

    def check(self) -> DriftAlert | None:
        if self._baseline_n < self.baseline_size or len(self._recent) < self.window_size // 2:
            return None
        base = self._baseline_dist()
        recent = self._recent_dist()
        all_classes = set(base) | set(recent)
        deltas = sorted(
            ((c, recent.get(c, 0.0) - base.get(c, 0.0)) for c in all_classes),
            key=lambda x: -abs(x[1]),
        )
        l1 = 0.5 * sum(abs(recent.get(c, 0.0) - base.get(c, 0.0)) for c in all_classes)
        if l1 < self.threshold:
            return None
        return DriftAlert(
            window_size=len(self._recent),
            baseline_size=self._baseline_n,
            l1_distance=l1,
            classes_top=[(c, d) for c, d in deltas[:3]],
            threshold=self.threshold,
        )

    def state_dict(self) -> dict:
        return {
            "baseline": dict(self._baseline),
            "baseline_n": self._baseline_n,
            "recent": list(self._recent),
            "window_size": self.window_size,
            "baseline_size": self.baseline_size,
            "threshold": self.threshold,
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> "DriftDetector":
        det = cls(
            window_size=d.get("window_size", 200),
            baseline_size=d.get("baseline_size", 200),
            threshold=d.get("threshold", 0.30),
        )
        det._baseline = {k: int(v) for k, v in d.get("baseline", {}).items()}
        det._baseline_n = int(d.get("baseline_n", 0))
        det._recent = list(d.get("recent", []))
        return det


# ----------------- main improver -----------------

ProposeFn = Callable[[Sequence[TaskExample], int], list[Harness]]


class ContinualImprover:
    """Active continual harness improver with persistence + drift + paired CIs.

    Args:
      scorer: ``Scorer`` that knows how to evaluate a single example.
      production_label: the label of the harness currently serving
          production. Must be a key in ``candidates``.
      candidates: label -> Harness pool.
      propose_fn: optional callable ``(examples, n) -> list[Harness]``
          invoked when ``maybe_propose=True`` is set on tick(). If
          omitted, the improver only ranks/promotes existing candidates.
      promotion_gates: see ``PromotionGates``.
      drift_detector: optional ``DriftDetector``; if present, drift
          checks run on every tick.
      state_path: file path for persistence; if set, ``save()`` writes
          the full state and ``load()`` restores it.
      max_history: cap on per-example tracker history.
      propose_every: tick a search every N ingestions when propose_fn
          is set and gates allow.
    """

    def __init__(
        self,
        scorer: Scorer,
        production_label: str,
        candidates: dict[str, Harness],
        *,
        propose_fn: ProposeFn | None = None,
        promotion_gates: PromotionGates | None = None,
        drift_detector: DriftDetector | None = None,
        state_path: str | Path | None = None,
        max_history: int = 5000,
        propose_every: int = 100,
        max_candidates_per_search: int = 4,
    ):
        if production_label not in candidates:
            raise ValueError(f"production_label {production_label!r} not in candidates")
        self.scorer = scorer
        self.production_label = production_label
        self.candidates: dict[str, Harness] = dict(candidates)
        self.propose_fn = propose_fn
        self.gates = promotion_gates or PromotionGates()
        self.drift = drift_detector
        self.state_path = Path(state_path) if state_path else None
        self.max_history = max_history
        self.propose_every = propose_every
        self.max_candidates_per_search = max_candidates_per_search

        self.tracker = PerExampleTracker()
        for label in self.candidates:
            self.tracker.register(label)

        self.ingest_count = 0
        self.last_promote_at = -10**9
        self.search_count = 0
        self.promotion_log: list[PromotionLogEntry] = []
        # Stack of previous production labels — supports rollback().
        self._production_stack: list[str] = [production_label]

    # ----- io -----

    def save(self) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "production_label": self.production_label,
            "production_stack": list(self._production_stack),
            "candidates_labels": list(self.candidates.keys()),
            "ingest_count": self.ingest_count,
            "last_promote_at": self.last_promote_at,
            "search_count": self.search_count,
            "promotion_log": [asdict(e) for e in self.promotion_log],
            "tracker": self.tracker.state_dict(),
            "drift": self.drift.state_dict() if self.drift else None,
            "gates": asdict(self.gates),
        }
        self.state_path.write_text(json.dumps(state, indent=2))

    def load(self) -> None:
        """Restore state from ``state_path`` (no-op if file absent).

        Note: ``candidates`` (Harness objects) are not persisted, since
        they may close over arbitrary Python objects. The caller must
        re-register the same set of labels via the constructor before
        ``load()`` so the tracker's parallel vectors line up.
        """
        if self.state_path is None or not self.state_path.exists():
            return
        state = json.loads(self.state_path.read_text())
        on_disk_labels = set(state.get("candidates_labels", []))
        in_memory_labels = set(self.candidates.keys())
        missing = on_disk_labels - in_memory_labels
        if missing:
            raise ValueError(
                f"Cannot load state: candidates {sorted(missing)} were "
                f"persisted but not provided in this constructor. "
                f"Re-create ContinualImprover with the same candidate set."
            )
        self.production_label = state["production_label"]
        self._production_stack = list(state.get("production_stack", [self.production_label]))
        self.ingest_count = state["ingest_count"]
        self.last_promote_at = state["last_promote_at"]
        self.search_count = state.get("search_count", 0)
        self.promotion_log = [
            PromotionLogEntry(**e) for e in state.get("promotion_log", [])
        ]
        self.tracker = PerExampleTracker.from_state_dict(state["tracker"])
        # Re-register any labels missing from the loaded tracker (e.g. a
        # candidate added in this session that the saved state didn't know).
        for label in self.candidates:
            self.tracker.register(label)
        if state.get("drift") and self.drift is None:
            self.drift = DriftDetector.from_state_dict(state["drift"])
        elif state.get("drift") and self.drift is not None:
            # Replace the in-memory drift detector with the persisted state.
            self.drift = DriftDetector.from_state_dict(state["drift"])

    # ----- streaming API -----

    def tick(
        self,
        example: TaskExample,
        *,
        maybe_propose: bool | None = None,
        force_promote_check: bool = False,
    ) -> TickReport:
        """Process one production example end-to-end.

        Steps:
        1. Score every candidate on ``example``; record per-example correctness.
        2. Update drift detector.
        3. If ``maybe_propose`` is True (or it's a periodic propose tick)
           and we have a propose_fn, run a small search and admit
           non-dominated survivors.
        4. Check promotion gates; if a candidate clears them, promote.
        """
        self.ingest_count += 1

        scored: dict[str, tuple[int, float, float]] = {}
        for label, h in self.candidates.items():
            sv = self.scorer.score(h, [example], n_repeats=1, max_workers=1)
            correct = 1 if sv.accuracy >= 0.999 else 0  # 1 example: 0 or 1
            scored[label] = (correct, sv.tokens, sv.latency_ms)
        self.tracker.record(example, scored)
        self.tracker.trim(self.max_history)

        report = TickReport(ingest_count=self.ingest_count)

        if self.drift is not None:
            self.drift.ingest(example)
            alert = self.drift.check()
            if alert:
                report.drift_alert = alert

        if self.propose_fn is not None:
            do_propose = (
                bool(maybe_propose)
                or (self.ingest_count > 0 and self.ingest_count % self.propose_every == 0)
            )
            if do_propose:
                report.proposed_count, report.admitted_count = self._propose_and_admit()

        if force_promote_check or (self.ingest_count - self.last_promote_at) >= self.gates.cooldown_ingests:
            promoted = self._maybe_promote(report)
            # rollback isn't checked on every tick — caller drives it explicitly.
            if not promoted and report.drift_alert is not None:
                # If drift is severe we don't auto-roll-back, but record a no-op.
                self.promotion_log.append(PromotionLogEntry(
                    timestamp=time.time(),
                    ingest_count=self.ingest_count,
                    kind="no-op",
                    from_label=self.production_label,
                    to_label=self.production_label,
                    reason=f"drift detected (L1={report.drift_alert.l1_distance:.3f}); "
                           f"no candidate cleared promotion gates",
                ))

        if self.state_path is not None:
            self.save()

        return report

    def _propose_and_admit(self) -> tuple[int, int]:
        if self.propose_fn is None:
            return (0, 0)
        examples = list(self.tracker.examples)
        if len(examples) < self.gates.min_examples:
            return (0, 0)
        try:
            new_h = self.propose_fn(examples, self.max_candidates_per_search)
        except Exception:
            return (0, 0)
        if not new_h:
            return (0, 0)
        self.search_count += 1
        admitted = 0
        for i, h in enumerate(new_h):
            label = f"discovered_{self.search_count}_{i}"
            # Score on a screen subset so admission is cheap.
            screen = examples[-min(50, len(examples)):]
            try:
                sv = self.scorer.score(h, screen, n_repeats=1, max_workers=1)
            except Exception:
                continue
            # Admit if it doesn't hard-fail; gating happens at promote-time.
            if sv.accuracy <= 0.0:
                continue
            self.candidates[label] = h
            self.tracker.register(label)
            admitted += 1
        return (len(new_h), admitted)

    def _maybe_promote(self, report: TickReport) -> bool:
        """Check every non-production candidate against gates, promote the best."""
        prod = self.production_label
        prod_correct = self.tracker.correctness.get(prod, [])
        if len(prod_correct) < self.gates.min_examples:
            return False

        prod_tokens, prod_latency, _ = self.tracker.mean_cost(prod)
        candidates = [(l, c) for l, c in self.candidates.items() if l != prod]
        # Pick best by mean accuracy on measured examples first.
        best_label = None
        best_delta = -1.0
        for label, _ in candidates:
            ca, cb = self.tracker.paired_acc(label, prod)
            if len(ca) < self.gates.min_examples:
                continue
            mean_a = sum(ca) / len(ca)
            mean_b = sum(cb) / len(cb)
            delta = mean_a - mean_b
            if delta > best_delta:
                best_delta = delta
                best_label = label
        if best_label is None:
            return False

        # Real paired bootstrap CI.
        ca, cb = self.tracker.paired_acc(best_label, prod)
        ci = paired_bootstrap_diff(ca, cb, confidence=1 - self.gates.paired_acc_ci_alpha)
        if ci.low <= 0:
            return False

        cand_tokens, cand_latency, _ = self.tracker.mean_cost(best_label)
        # Cost-regression checks.
        if prod_tokens > 0:
            tok_pct = (cand_tokens - prod_tokens) / max(prod_tokens, 1e-9)
            if tok_pct > self.gates.max_token_regression_pct:
                self.promotion_log.append(PromotionLogEntry(
                    timestamp=time.time(),
                    ingest_count=self.ingest_count,
                    kind="no-op",
                    from_label=prod, to_label=best_label,
                    reason=f"token regression {tok_pct:.1%} exceeds gate "
                           f"{self.gates.max_token_regression_pct:.1%}",
                    delta_acc_ci=(ci.low, ci.high), n_paired_examples=len(ca),
                ))
                return False
        if self.gates.max_token_regression_abs > 0:
            if (cand_tokens - prod_tokens) > self.gates.max_token_regression_abs:
                return False
        if prod_latency > 0:
            lat_pct = (cand_latency - prod_latency) / max(prod_latency, 1e-9)
            if lat_pct > self.gates.max_latency_regression_pct:
                self.promotion_log.append(PromotionLogEntry(
                    timestamp=time.time(),
                    ingest_count=self.ingest_count,
                    kind="no-op",
                    from_label=prod, to_label=best_label,
                    reason=f"latency regression {lat_pct:.1%} exceeds gate "
                           f"{self.gates.max_latency_regression_pct:.1%}",
                    delta_acc_ci=(ci.low, ci.high), n_paired_examples=len(ca),
                ))
                return False
        if self.gates.max_latency_regression_abs_ms > 0:
            if (cand_latency - prod_latency) > self.gates.max_latency_regression_abs_ms:
                return False

        # All gates clear → promote.
        self._do_promote(best_label, ci=(ci.low, ci.high), n_paired=len(ca), report=report)
        return True

    def _do_promote(
        self,
        new_label: str,
        ci: tuple[float, float],
        n_paired: int,
        report: TickReport,
    ) -> None:
        prev = self.production_label
        self.production_label = new_label
        self._production_stack.append(new_label)
        self.last_promote_at = self.ingest_count
        report.promoted = True
        report.promoted_from = prev
        report.promoted_to = new_label
        report.promotion_reason = (
            f"paired-bootstrap Δacc CI=[{ci[0]:+.3f},{ci[1]:+.3f}] excludes 0 "
            f"over n={n_paired} paired examples; cost gates passed"
        )
        self.promotion_log.append(PromotionLogEntry(
            timestamp=time.time(),
            ingest_count=self.ingest_count,
            kind="promote",
            from_label=prev, to_label=new_label,
            reason=report.promotion_reason,
            delta_acc_ci=ci, n_paired_examples=n_paired,
        ))

    # ----- explicit operations -----

    def rollback(self, reason: str = "manual rollback") -> bool:
        """Roll back to the previous production label.

        Returns True if a rollback happened. Records the rollback in
        the promotion log.
        """
        if len(self._production_stack) < 2:
            return False
        promoted = self._production_stack.pop()  # current
        prev = self._production_stack[-1]
        self.production_label = prev
        self.last_promote_at = self.ingest_count  # cooldown after rollback too
        self.promotion_log.append(PromotionLogEntry(
            timestamp=time.time(),
            ingest_count=self.ingest_count,
            kind="rollback",
            from_label=promoted, to_label=prev,
            reason=reason,
        ))
        if self.state_path is not None:
            self.save()
        return True

    def add_candidate(self, label: str, harness: Harness) -> None:
        """Register a new candidate on the fly. Tracker starts measuring from now."""
        self.candidates[label] = harness
        self.tracker.register(label)

    def remove_candidate(self, label: str) -> bool:
        """Remove a candidate (e.g. after consistent under-performance)."""
        if label == self.production_label:
            raise ValueError("cannot remove the current production candidate")
        if label not in self.candidates:
            return False
        del self.candidates[label]
        self.tracker.deregister(label)
        return True
