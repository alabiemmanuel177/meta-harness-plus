"""Deterministic end-to-end demo of ContinualImprover.

Runs a synthetic production stream and shows the full lifecycle:

1. Two candidates are registered (a weak ``bare`` shape and a stronger
   ``rag`` shape).
2. Per-example correctness is tracked as the stream progresses.
3. Once we have enough paired evidence and the ``rag`` shape clears
   the conservative gates, it gets promoted.
4. We then inject a synthetic drift event (the class distribution
   shifts) and observe the drift alert.
5. We demonstrate ``rollback()``.

No network access. Useful as a smoke test, and as a runnable artifact
referenced from ``RESULTS_FUTURE_WORK.md``.

Usage::

    python3 examples/continual_demo.py
    python3 examples/continual_demo.py --state /tmp/imp.json --resume
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meta_harness_plus.components import (
    BagOfWordsRetriever, MockLLMPredictor, NullFewShot, NullRetriever,
    NullVoter, SimpleFormatter, TopKFewShot,
)
from meta_harness_plus.continual import (
    ContinualImprover,
    DriftDetector,
    PromotionGates,
)
from meta_harness_plus.harness import Harness
from meta_harness_plus.scorer import Scorer
from meta_harness_plus.task import TaskExample
from meta_harness_plus.tasks import build_toy_task, mock_llm


def _bare(llm) -> Harness:
    return Harness(components=[
        NullRetriever(), NullFewShot(), SimpleFormatter(),
        MockLLMPredictor(llm_fn=llm), NullVoter(),
    ])


def _rag(corpus, llm, k=2) -> Harness:
    return Harness(components=[
        BagOfWordsRetriever(corpus=corpus, k=k),
        TopKFewShot(k=1),
        SimpleFormatter(),
        MockLLMPredictor(llm_fn=llm),
        NullVoter(),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=None,
                    help="Path to persistence file. If set, the run "
                         "resumes from it (or seeds it).")
    ap.add_argument("--resume", action="store_true",
                    help="Reload from --state instead of starting fresh.")
    args = ap.parse_args()

    task = build_toy_task(seed=0)
    scorer = Scorer(task)
    llm = mock_llm()

    drift = DriftDetector(window_size=10, baseline_size=10, threshold=0.30)
    # Demo gates — lenient on tokens because RAG inherently uses ~4× the
    # tokens of bare on the toy task, but tight on accuracy CI so we
    # only promote when there's real paired evidence.
    gates = PromotionGates(
        min_examples=8,
        paired_acc_ci_alpha=0.20,
        cooldown_ingests=5,
        max_token_regression_pct=5.0,   # demo: tolerate 5× cost
        max_latency_regression_pct=5.0,
    )

    state_path = args.state or str(Path(tempfile.gettempdir()) / "mh_continual_demo.json")

    improver = ContinualImprover(
        scorer=scorer,
        production_label="bare",
        candidates={"bare": _bare(llm), "rag": _rag(task.train, llm)},
        promotion_gates=gates,
        drift_detector=drift,
        state_path=state_path,
    )
    if args.resume:
        improver.load()
        print(f"resumed from {state_path}: ingest_count={improver.ingest_count}")

    # Phase 1: stream a balanced 'production' batch.
    print("\n=== phase 1: balanced stream (20 examples)")
    for i, ex in enumerate(list(task.eval_set)):
        r = improver.tick(ex, force_promote_check=True)
        if r.promoted:
            print(f"  ingest #{r.ingest_count}: PROMOTED {r.promoted_from} → {r.promoted_to}")
            print(f"    {r.promotion_reason}")

    # Phase 2: inject drift — synthesize examples whose label distribution
    # collapses to one class. We ingest 12 to force the recent window
    # past the alert threshold.
    print("\n=== phase 2: synthetic drift (12 examples, all label='red')")
    for i in range(12):
        ex = TaskExample(input=f"red item {i}", label="red")
        r = improver.tick(ex)
        if r.drift_alert:
            a = r.drift_alert
            print(f"  ingest #{r.ingest_count}: DRIFT detected — "
                  f"L1={a.l1_distance:.3f} (threshold {a.threshold:.2f})")
            print(f"    top deltas: {a.classes_top}")
            break

    # Phase 3: rollback demo.
    print(f"\n=== phase 3: rollback demo (current production = {improver.production_label})")
    if len(improver._production_stack) > 1:
        ok = improver.rollback("demo: simulated production-fresh-data regression")
        print(f"  rollback() returned {ok}; production now = {improver.production_label}")
    else:
        print("  no promotions to roll back from in this run.")

    print(f"\nState persisted to: {state_path}")
    print(f"Promotion log entries: {len(improver.promotion_log)}")
    for e in improver.promotion_log:
        print(f"  - [ingest {e.ingest_count}] {e.kind}: {e.from_label} → {e.to_label}: {e.reason}")


if __name__ == "__main__":
    main()
