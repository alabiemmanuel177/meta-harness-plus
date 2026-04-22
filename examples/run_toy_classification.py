"""Run Meta-Harness++ search on the toy classification task.

Usage:
    python3 examples/run_toy_classification.py

Writes a run log to ``./runs/toy/`` — inspect ``frontier.json`` for the
discovered Pareto front and ``attribution_stats.json`` for per-component
value estimates.
"""

from __future__ import annotations

import os
import sys

# Allow running without `pip install -e .`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from meta_harness_plus.attribution import AttributionTracker
from meta_harness_plus.components import (
    BagOfWordsRetriever,
    MajorityVoter,
    MockLLMPredictor,
    NullFewShot,
    NullRetriever,
    NullVoter,
    SimpleFormatter,
    TopKFewShot,
    baseline_for,
)
from meta_harness_plus.harness import Harness
from meta_harness_plus.runner import SearchConfig, SearchRunner
from meta_harness_plus.scorer import Scorer
from meta_harness_plus.search.mock_proposer import AttributionGuidedMutationProposer
from meta_harness_plus.tasks import build_toy_task, mock_llm


def seed_harness(task, llm) -> Harness:
    return Harness(components=[
        NullRetriever(),
        NullFewShot(),
        SimpleFormatter(),
        MockLLMPredictor(llm_fn=llm, n_samples=1),
        NullVoter(),
    ])


def mutators(task, llm) -> dict:
    return {
        "retriever": [
            lambda: NullRetriever(),
            lambda: BagOfWordsRetriever(corpus=task.train, k=2),
            lambda: BagOfWordsRetriever(corpus=task.train, k=3),
            lambda: BagOfWordsRetriever(corpus=task.train, k=5),
        ],
        "fewshot": [
            lambda: NullFewShot(),
            lambda: TopKFewShot(k=1),
            lambda: TopKFewShot(k=2),
            lambda: TopKFewShot(k=3),
        ],
        "voter": [
            lambda: NullVoter(),
            lambda: MajorityVoter(),
        ],
        "predictor": [
            lambda: MockLLMPredictor(llm_fn=llm, n_samples=1),
            lambda: MockLLMPredictor(llm_fn=llm, n_samples=3),
            lambda: MockLLMPredictor(llm_fn=llm, n_samples=5),
            lambda: MockLLMPredictor(llm_fn=llm, n_samples=7),
        ],
    }


def main() -> None:
    task = build_toy_task(seed=0)
    llm = mock_llm()
    scorer = Scorer(task)

    attribution = AttributionTracker(scorer, baseline_for)
    proposer = AttributionGuidedMutationProposer(
        mutators=mutators(task, llm), seed=0, epsilon=0.25, temperature=0.5,
    )
    runner = SearchRunner(
        task=task, scorer=scorer, proposer=proposer, attribution=attribution,
        config=SearchConfig(
            n_iterations=10,
            proposals_per_iter=8,
            screen_size=6,
            halving_k0=3, halving_eta=2, halving_final_keep=2,
            run_dir="runs/toy",
        ),
        seed_harnesses=[seed_harness(task, llm)],
    )

    state = runner.run()

    print("\n=== Discovered Pareto frontier ===")
    for e in sorted(state.frontier.entries, key=lambda x: -x.score.accuracy):
        kinds = " / ".join(
            f"{c['kind']}:{c.get('name', '?')}{'(k=' + str(c['k']) + ')' if 'k' in c else ''}"
            f"{'(n=' + str(c['n_samples']) + ')' if 'n_samples' in c else ''}"
            for c in e.meta["describe"]
        )
        s = e.score
        print(f"  {e.candidate_id}  acc={s.accuracy:.2f}  tok={s.tokens:5.1f}  "
              f"lat={s.latency_ms:5.1f}  <-  {kinds}")

    print("\n=== Attribution ranking (mean accuracy delta from drop-one ablation) ===")
    for stats in attribution.ranking():
        print(f"  {stats.kind:12s}  mean_delta={stats.mean_delta:+.3f}  n={stats.n}")

    knee = state.frontier.knee()
    if knee:
        print(f"\n=== Knee-point recommendation (closest to utopia) ===")
        print(f"  {knee.candidate_id}  acc={knee.score.accuracy:.2f}  "
              f"tok={knee.score.tokens:.1f}  lat={knee.score.latency_ms:.1f}")

    print(f"\nRun log written to: runs/toy/")


if __name__ == "__main__":
    main()
