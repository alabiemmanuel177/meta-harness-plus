"""Score the MH++-discovered "alpha" shape directly.

The alpha shape is the seed-0 winner from the beat-cot experiment:
bm25(k=5) + llm_reranker(m=3) + topk_fewshot(k=2) +
compressed_cot(15w) + null_voter

Goal: confirm this shape reproducibly beats hand-tuned CoT-RAG (0.94)
on Gemini × news_hard_50. If it scores ≥ 0.96 across multiple repeats,
the paper can claim "MH++ search discovers harness shapes that
hand-tuned baselines miss" — even if the SEARCH itself doesn't always
find the shape.

This is the "alpha-shape reproducibility" experiment.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from meta_harness_plus.components import (
    BM25Retriever,
    CompressedCoTFormatter,
    NullVoter,
    TopKFewShot,
)
from meta_harness_plus.harness import Harness
from meta_harness_plus.llm.cache import CachedLLMClient, PromptCache
from meta_harness_plus.llm.client import HTTPClient
from meta_harness_plus.llm.predictor import LLMPredictor
from meta_harness_plus.llm.reranker import LLMReranker
from meta_harness_plus.scorer import Scorer
from meta_harness_plus.tasks import build_news_hard_50_task


def _build_client(model: str, api: str) -> HTTPClient:
    if api == "openai":
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise SystemExit("OPENAI_API_KEY required")
        return HTTPClient(api_url="https://api.openai.com/v1/chat/completions",
                          api_key=key, model=model, timeout_s=300.0)
    if api == "gemini":
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
        if not key:
            raise SystemExit("GEMINI_API_KEY/GOOGLE_API_KEY required")
        url = (f"https://generativelanguage.googleapis.com/v1beta/"
               f"models/{model}:generateContent")
        return HTTPClient(api_url=url, api_key=key, model=model, timeout_s=300.0)
    raise SystemExit(f"unknown api: {api}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", choices=("openai", "gemini"), default="gemini")
    ap.add_argument("--model", default="gemini-2.5-flash-lite")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--cache-path", default="runs/cache/alpha_shape_gemini_news_hard_50.jsonl")
    ap.add_argument("--output", default="runs/alpha_shape_gemini_news_hard_50.json")
    args = ap.parse_args()

    task = build_news_hard_50_task()
    raw_client = _build_client(args.model, args.api)
    cache = PromptCache(path=args.cache_path)
    client = CachedLLMClient(raw_client, cache, model_id=args.model)

    print(f"=== Alpha-shape scorer on {args.model} × news_hard_50 ===")
    print(f"  cache: {args.cache_path} ({len(cache)} preloaded keys)")
    print(f"  repeats: {args.repeats}")

    # The alpha shape, as discovered by MH++ on seed 0 of beat-cot v1:
    # bm25(k=5) + llm_reranker(m=3, t=0.2) + topk_fewshot(k=2) +
    # compressed_cot(max_words=15) + null_voter
    alpha = Harness(components=[
        BM25Retriever(corpus=task.train, k=5, k1=1.5, b=0.75),
        LLMReranker(client=client, m=3, temperature=0.2),
        TopKFewShot(k=2),
        CompressedCoTFormatter(max_reasoning_words=15),
        LLMPredictor(client=client, classes=task.classes,
                     n_samples=1, temperature=0.0, max_tokens=512),
        NullVoter(),
    ])

    scorer = Scorer(task)
    eval_examples = task.eval_set[:50]

    t0 = time.time()
    score = scorer.score(alpha, eval_examples,
                         n_repeats=args.repeats,
                         max_workers=8)
    elapsed = time.time() - t0

    print(f"\n  Alpha-shape (median over {args.repeats} repeats):")
    print(f"    accuracy:    {score.accuracy:.3f}")
    print(f"    tokens:      {score.tokens:.1f}")
    print(f"    latency_ms:  {score.latency_ms:.0f}")
    print(f"    spread:      {score.accuracy_spread:.3f}")
    print(f"    elapsed:     {elapsed:.0f}s")

    # Compare vs CoT-RAG (hand-tuned) — load from saved baselines if present.
    cot_baseline_path = "runs/hand_tuned_baselines_gemini_news_hard_50.json"
    if Path(cot_baseline_path).exists():
        baselines = json.load(open(cot_baseline_path))["baselines"]
        cot_acc = baselines["CoT-RAG"]["accuracy"]
        cot_tok = baselines["CoT-RAG"]["tokens"]
        delta_acc = score.accuracy - cot_acc
        delta_tok = score.tokens - cot_tok
        print(f"\n  vs CoT-RAG (hand-tuned) acc=0.940 tok=290:")
        print(f"    Δ acc: {delta_acc:+.3f}  ({'beats' if delta_acc > 0 else 'loses' if delta_acc < 0 else 'ties'} CoT-RAG)")
        print(f"    Δ tok: {delta_tok:+.1f}")

    out = {
        "model": args.model,
        "task": "news_hard_50",
        "repeats": args.repeats,
        "alpha_shape": {
            "components": [c.config() for c in alpha.components],
        },
        "score": {
            "accuracy": score.accuracy,
            "tokens": score.tokens,
            "latency_ms": score.latency_ms,
            "spread": score.accuracy_spread,
        },
        "cache_stats": cache.stats(),
        "elapsed_s": round(elapsed, 1),
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()
