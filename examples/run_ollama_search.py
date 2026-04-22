"""Mini-search with real LLMPredictor (Ollama gpt-oss:20b) + MockProposer.

Small eval size (5 items) and iteration count so it finishes in a few
minutes. Demonstrates that the whole pipeline works against a real local
model: real token accounting, real latency, real Pareto tradeoffs.

Usage:
    python3 examples/run_ollama_search.py [--model gpt-oss:20b] [--eval-size 5]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from meta_harness_plus.attribution import AttributionTracker
from meta_harness_plus.components import (
    BagOfWordsRetriever,
    MajorityVoter,
    NullFewShot,
    NullRetriever,
    NullVoter,
    SimpleFormatter,
    TopKFewShot,
    baseline_for,
)
from meta_harness_plus.harness import Harness
from meta_harness_plus.llm.client import HTTPClient
from meta_harness_plus.llm.predictor import LLMPredictor
from meta_harness_plus.runner import SearchConfig, SearchRunner
from meta_harness_plus.scorer import Scorer
from meta_harness_plus.search.mock_proposer import AttributionGuidedMutationProposer
from meta_harness_plus.tasks import build_toy_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-oss:20b")
    ap.add_argument("--ollama-url", default="http://localhost:11434/api/chat")
    ap.add_argument("--eval-size", type=int, default=5,
                    help="eval subset size (small = fast, noisy)")
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--proposals", type=int, default=3)
    args = ap.parse_args()

    task = build_toy_task(seed=0)
    client = HTTPClient(api_url=args.ollama_url, model=args.model)

    def pred(n_samples: int) -> LLMPredictor:
        return LLMPredictor(
            client=client, classes=task.classes,
            n_samples=n_samples, temperature=0.3 if n_samples > 1 else 0.0,
        )

    mutators = {
        "retriever": [
            lambda: NullRetriever(),
            lambda: BagOfWordsRetriever(corpus=task.train, k=2),
            lambda: BagOfWordsRetriever(corpus=task.train, k=3),
        ],
        "fewshot": [
            lambda: NullFewShot(),
            lambda: TopKFewShot(k=1),
            lambda: TopKFewShot(k=2),
        ],
        "voter": [lambda: NullVoter(), lambda: MajorityVoter()],
        "predictor": [lambda: pred(1), lambda: pred(3)],
    }

    scorer = Scorer(task)
    attribution = AttributionTracker(scorer, baseline_for)
    proposer = AttributionGuidedMutationProposer(
        mutators=mutators, seed=0, epsilon=0.3, temperature=0.5,
    )

    seed = Harness(components=[
        NullRetriever(), NullFewShot(), SimpleFormatter(),
        pred(1), NullVoter(),
    ])

    runner = SearchRunner(
        task=task, scorer=scorer, proposer=proposer, attribution=attribution,
        config=SearchConfig(
            n_iterations=args.iterations,
            proposals_per_iter=args.proposals,
            screen_size=2,
            full_eval_size=args.eval_size,
            halving_k0=2, halving_eta=2, halving_final_keep=1,
            run_dir="runs/ollama",
        ),
        seed_harnesses=[seed],
    )

    print(f"Starting mini-search with {args.model}...")
    t0 = time.time()
    state = runner.run()
    elapsed = time.time() - t0
    print(f"\n--- Done in {elapsed:.1f}s ---\n")

    print("=== Discovered Pareto frontier ===")
    for e in sorted(state.frontier.entries, key=lambda x: -x.score.accuracy):
        s = e.score
        shape = " + ".join(
            f"{c['kind']}/{c.get('name','?')}"
            + (f"(k={c['k']})" if 'k' in c else "")
            + (f"(n={c['n_samples']})" if 'n_samples' in c else "")
            for c in e.meta["describe"]
        )
        print(f"  acc={s.accuracy:.2f}  tok={s.tokens:6.0f}  lat={s.latency_ms:6.0f}ms  :: {shape}")

    print("\n=== Attribution ranking ===")
    for stats in attribution.ranking():
        print(f"  {stats.kind:12s}  mean_delta={stats.mean_delta:+.3f}  n={stats.n}")


if __name__ == "__main__":
    main()
