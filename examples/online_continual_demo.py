"""End-to-end integration demo for ContinualHarnessImprover.

Simulates a production stream where:
1. We deploy an initial harness (vanilla RAG).
2. Examples arrive over time. Each batch goes through ingest().
3. Periodically (every N ingestions) we trigger propose_and_admit() to
   actively search for better candidates from the accumulated production
   data.
4. After each ingest we check maybe_promote() — if a candidate strictly
   Pareto-dominates production with confidence, we deploy it.

This is the active continual-improvement loop, exercised end-to-end.
Uses the toy task (deterministic, no API needed) so the demo runs in
under 30 seconds. The same loop works with real LLM components — see
examples/demo.py for the LLM equivalent.

Usage:
    python3 examples/online_continual_demo.py
"""
from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from meta_harness_plus.components import (
    BagOfWordsRetriever, MockLLMPredictor, NullFewShot, NullRetriever,
    NullVoter, SimpleFormatter, TopKFewShot,
)
from meta_harness_plus.harness import Harness
from meta_harness_plus.online import ContinualHarnessImprover
from meta_harness_plus.scorer import Scorer
from meta_harness_plus.task import TaskExample
from meta_harness_plus.tasks import build_toy_task, mock_llm


def _bare(llm) -> Harness:
    return Harness(components=[
        NullRetriever(), NullFewShot(), SimpleFormatter(),
        MockLLMPredictor(llm_fn=llm), NullVoter(),
    ])


def _rag(corpus, llm, k: int = 2) -> Harness:
    return Harness(components=[
        BagOfWordsRetriever(corpus=corpus, k=k),
        TopKFewShot(k=1),
        SimpleFormatter(),
        MockLLMPredictor(llm_fn=llm),
        NullVoter(),
    ])


def make_proposer(corpus, llm):
    """Build a propose_fn that proposes RAG variants with different k."""
    rng = random.Random(0)

    def propose(_examples, n: int) -> list[Harness]:
        # Vary the k of the retriever — a simple discrete search.
        out = []
        for _ in range(n):
            k = rng.choice([1, 2, 3, 4, 5])
            out.append(_rag(corpus, llm, k=k))
        return out

    return propose


def main():
    print("=== ContinualHarnessImprover end-to-end demo ===\n")
    task = build_toy_task(seed=0)
    llm = mock_llm()
    scorer = Scorer(task)

    # Production deploys BARE (the floor).
    production = _bare(llm)

    improver = ContinualHarnessImprover(
        scorer=scorer,
        production_harness=production,
        candidate_harnesses={"rag_initial": _rag(task.train, llm, k=2)},
        propose_fn=make_proposer(task.train, llm),
        promote_ci_alpha=0.05,
        min_examples=10,
        max_history=200,
        rescore_every=5,
        max_candidates_per_search=4,
    )

    print(f"Starting state:")
    print(f"  production: BARE")
    print(f"  initial candidates: {list(improver.candidate_harnesses.keys())}")
    print()

    # Simulate a production stream.
    stream = list(task.eval_set) * 10  # repeat the eval set as the "stream"
    rng = random.Random(0)
    rng.shuffle(stream)

    promotions = 0
    searches = 0
    for i, ex in enumerate(stream):
        improver.ingest(ex)

        # Every 25 ingestions, run an active search.
        if improver.history_size > 0 and improver.history_size % 25 == 0:
            n_admitted = improver.propose_and_admit()
            searches += 1
            print(f"  step {i+1:3d}: search #{searches} ran, "
                  f"admitted {n_admitted} new candidates "
                  f"(pool now {len(improver.candidate_harnesses)})")

        # Every 10 ingestions, check promotion.
        if improver.history_size > 0 and improver.history_size % 10 == 0:
            report = improver.maybe_promote(force_rescore=True)
            if report.should_promote:
                promotions += 1
                print(f"  step {i+1:3d}: PROMOTE {report.winner_label} "
                      f"(acc={report.winner_score.accuracy:.3f}, "
                      f"prod_acc={report.production_score.accuracy:.3f}, "
                      f"Δ_CI={report.delta_acc_ci})")
                # Swap in the new shape.
                improver.production_harness = report.winner_harness
                # Remove from candidates so we don't re-promote.
                del improver.candidate_harnesses[report.winner_label]

    print(f"\n=== Final state ===")
    print(f"  total ingests:       {improver.history_size} (rolled-off cap)")
    print(f"  active searches ran: {searches}")
    print(f"  candidates admitted: {improver.admitted_count}")
    print(f"  promotions:          {promotions}")
    print(f"  candidate pool:      {list(improver.candidate_harnesses.keys())}")


if __name__ == "__main__":
    main()
