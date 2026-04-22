"""Run Meta-Harness++ search on the bundled medical symptom task.

By default uses the offline ``symptom_mock_llm`` — no external API required.
Pass ``--backend ollama --model <tag>`` to use a local Ollama model as the
real predictor; ``--backend anthropic`` / ``--backend openai`` also supported.

Usage:
    # Offline (deterministic, no API):
    python3 examples/run_symptom_classification.py

    # Local Ollama:
    python3 examples/run_symptom_classification.py \\
        --backend ollama --model gpt-oss:20b

    # Anthropic:
    ANTHROPIC_API_KEY=sk-ant-... python3 examples/run_symptom_classification.py \\
        --backend anthropic --model claude-opus-4-7
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
from meta_harness_plus.tasks import build_symptom_task, symptom_mock_llm


def _make_predictor(args, classes, n_samples: int):
    """Return a Predictor component matching the requested backend."""
    if args.backend == "mock":
        llm_fn = symptom_mock_llm()
        return MockLLMPredictor(llm_fn=llm_fn, n_samples=n_samples)
    # Real-LLM backends live on the `llm-proposer` branch (merged into main).
    from meta_harness_plus.llm.client import HTTPClient
    from meta_harness_plus.llm.predictor import LLMPredictor
    if args.backend == "ollama":
        client = HTTPClient(
            api_url=args.ollama_url or "http://localhost:11434/api/chat",
            model=args.model or "gpt-oss:20b",
        )
    elif args.backend == "anthropic":
        client = HTTPClient(
            api_url="https://api.anthropic.com/v1/messages",
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            model=args.model or "claude-opus-4-7",
        )
    elif args.backend == "openai":
        client = HTTPClient(
            api_url=args.openai_url or "https://api.openai.com/v1/chat/completions",
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            model=args.model or "gpt-4o-mini",
        )
    else:
        raise SystemExit(f"unknown backend: {args.backend}")
    return LLMPredictor(client=client, classes=classes, n_samples=n_samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("mock", "ollama", "anthropic", "openai"),
                    default="mock")
    ap.add_argument("--model", default=None)
    ap.add_argument("--ollama-url", default=None)
    ap.add_argument("--openai-url", default=None)
    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--proposals", type=int, default=6)
    args = ap.parse_args()

    task = build_symptom_task()
    scorer = Scorer(task)

    def predictor(n_samples: int):
        return _make_predictor(args, task.classes, n_samples)

    mutators = {
        "retriever": [
            lambda: NullRetriever(),
            lambda: BagOfWordsRetriever(corpus=task.train, k=2),
            lambda: BagOfWordsRetriever(corpus=task.train, k=3),
            lambda: BagOfWordsRetriever(corpus=task.train, k=5),
        ],
        "fewshot": [lambda: NullFewShot(),
                    lambda: TopKFewShot(k=1),
                    lambda: TopKFewShot(k=2),
                    lambda: TopKFewShot(k=3)],
        "voter": [lambda: NullVoter(), lambda: MajorityVoter()],
        "predictor": [lambda: predictor(1), lambda: predictor(3),
                      lambda: predictor(5), lambda: predictor(7)],
    }

    attribution = AttributionTracker(scorer, baseline_for)
    proposer = AttributionGuidedMutationProposer(
        mutators=mutators, seed=3, epsilon=0.3, temperature=0.5,
        depth_weights={1: 0.2, 2: 0.5, 3: 0.3},
    )
    seed_harness = Harness(components=[
        NullRetriever(), NullFewShot(), SimpleFormatter(),
        predictor(1), NullVoter(),
    ])

    runner = SearchRunner(
        task=task, scorer=scorer, proposer=proposer, attribution=attribution,
        config=SearchConfig(
            n_iterations=args.iterations,
            proposals_per_iter=args.proposals,
            screen_size=6,
            halving_k0=3, halving_eta=2, halving_final_keep=2,
            run_dir="runs/symptom",
        ),
        seed_harnesses=[seed_harness],
    )

    t0 = time.time()
    state = runner.run()
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. backend={args.backend} model={args.model or 'n/a'}\n")

    print("=== Discovered Pareto frontier ===")
    for e in sorted(state.frontier.entries, key=lambda x: -x.score.accuracy):
        s = e.score
        shape = " + ".join(
            f"{c['kind']}/{c.get('name','?')}"
            + (f"(k={c['k']})" if 'k' in c else "")
            + (f"(n={c['n_samples']})" if 'n_samples' in c else "")
            for c in e.meta["describe"]
        )
        print(f"  acc={s.accuracy:.2f}  tok={s.tokens:6.1f}  lat={s.latency_ms:6.1f}ms  :: {shape}")

    print("\n=== Attribution ranking ===")
    for stats in attribution.ranking():
        print(f"  {stats.kind:12s}  mean_delta={stats.mean_delta:+.3f}  n={stats.n}")


if __name__ == "__main__":
    main()
