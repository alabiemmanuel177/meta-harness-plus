"""Bakeoff: MockProposer vs LLMProposer on the same search config.

Usage:
    # Use the mutation (mock) proposer — offline, no LLM needed:
    python3 examples/bakeoff.py --proposer mock

    # Use a local Ollama model:
    python3 examples/bakeoff.py --proposer llm \\
        --ollama-url http://localhost:11434/api/chat \\
        --model llama3.2

    # Use Anthropic:
    ANTHROPIC_API_KEY=... python3 examples/bakeoff.py --proposer llm \\
        --anthropic --model claude-opus-4-7

    # Use any OpenAI-compatible endpoint:
    OPENAI_API_KEY=... python3 examples/bakeoff.py --proposer llm \\
        --openai-url https://api.openai.com/v1/chat/completions \\
        --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import os
import sys

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
from meta_harness_plus.llm.client import HTTPClient
from meta_harness_plus.llm.proposer import LLMProposer
from meta_harness_plus.llm.registry import default_registry
from meta_harness_plus.runner import SearchConfig, SearchRunner
from meta_harness_plus.scorer import Scorer
from meta_harness_plus.search.mock_proposer import AttributionGuidedMutationProposer
from meta_harness_plus.tasks import build_toy_task, mock_llm


def seed_harness(task, llm) -> Harness:
    return Harness(components=[
        NullRetriever(), NullFewShot(), SimpleFormatter(),
        MockLLMPredictor(llm_fn=llm, n_samples=1), NullVoter(),
    ])


def mock_proposer_for(task, llm):
    mutators = {
        "retriever": [
            lambda: NullRetriever(),
            lambda: BagOfWordsRetriever(corpus=task.train, k=2),
            lambda: BagOfWordsRetriever(corpus=task.train, k=3),
            lambda: BagOfWordsRetriever(corpus=task.train, k=5),
        ],
        "fewshot": [lambda: NullFewShot(), lambda: TopKFewShot(k=1),
                    lambda: TopKFewShot(k=2), lambda: TopKFewShot(k=3)],
        "voter": [lambda: NullVoter(), lambda: MajorityVoter()],
        "predictor": [
            lambda: MockLLMPredictor(llm_fn=llm, n_samples=1),
            lambda: MockLLMPredictor(llm_fn=llm, n_samples=3),
            lambda: MockLLMPredictor(llm_fn=llm, n_samples=5),
            lambda: MockLLMPredictor(llm_fn=llm, n_samples=7),
        ],
    }
    return AttributionGuidedMutationProposer(mutators=mutators, seed=0,
                                             epsilon=0.25, temperature=0.5)


def build_client(args) -> HTTPClient:
    if args.anthropic:
        return HTTPClient(
            api_url="https://api.anthropic.com/v1/messages",
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            model=args.model or "claude-opus-4-7",
        )
    if args.ollama_url:
        return HTTPClient(
            api_url=args.ollama_url,
            model=args.model or "llama3.2",
        )
    if args.openai_url:
        return HTTPClient(
            api_url=args.openai_url,
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            model=args.model or "gpt-4o-mini",
        )
    raise SystemExit("--proposer llm requires one of --anthropic / --ollama-url / --openai-url")


def run_one(name: str, proposer, task, llm, run_dir: str):
    scorer = Scorer(task)
    attribution = AttributionTracker(scorer, baseline_for)
    runner = SearchRunner(
        task=task, scorer=scorer, proposer=proposer, attribution=attribution,
        config=SearchConfig(
            n_iterations=8, proposals_per_iter=6, screen_size=6,
            halving_k0=3, halving_eta=2, halving_final_keep=2,
            run_dir=run_dir,
        ),
        seed_harnesses=[seed_harness(task, llm)],
    )
    state = runner.run()
    print(f"\n=== {name} ===")
    for e in sorted(state.frontier.entries, key=lambda x: -x.score.accuracy):
        s = e.score
        shape = " + ".join(
            f"{c['kind']}/{c.get('name','?')}"
            + (f"(k={c['k']})" if 'k' in c else "")
            + (f"(n={c['n_samples']})" if 'n_samples' in c else "")
            for c in e.meta["describe"]
        )
        print(f"  acc={s.accuracy:.2f}  tok={s.tokens:5.1f}  lat={s.latency_ms:5.1f}  :: {shape}")
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposer", choices=("mock", "llm"), default="mock")
    ap.add_argument("--anthropic", action="store_true")
    ap.add_argument("--ollama-url")
    ap.add_argument("--openai-url")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    task = build_toy_task(seed=0)
    llm_fn = mock_llm()

    if args.proposer == "mock":
        proposer = mock_proposer_for(task, llm_fn)
        run_one("MockProposer", proposer, task, llm_fn, run_dir="runs/bakeoff_mock")
    else:
        client = build_client(args)
        registry = default_registry(task, llm_fn)
        proposer = LLMProposer(client=client, registry=registry,
                               run_dir="runs/bakeoff_llm", temperature=0.6)
        run_one("LLMProposer", proposer, task, llm_fn, run_dir="runs/bakeoff_llm")


if __name__ == "__main__":
    main()
