"""Score variants of the alpha shape to find one that substantially beats CoT-RAG.

Alpha base: bm25(k=5) + llm_reranker(m=3) + topk_fewshot(k=2) +
            compressed_cot(15w) + null_voter  → 0.960 acc

Variants tested:
  V1: alpha + majority_voter (3 samples, t=0.4) — adds self-consistency
  V2: alpha but bm25(k=8) + topk_fewshot(k=4) — wider context
  V3: alpha but compressed_cot(40w) — more reasoning room
  V4: alpha + voting + wider context (V1 + V2 combined)
  V5: alpha + voting + more reasoning (V1 + V3 combined)
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
    MajorityVoter,
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


def _build_client(model: str) -> HTTPClient:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        raise SystemExit("GEMINI_API_KEY/GOOGLE_API_KEY required")
    url = (f"https://generativelanguage.googleapis.com/v1beta/"
           f"models/{model}:generateContent")
    return HTTPClient(api_url=url, api_key=key, model=model, timeout_s=300.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-2.5-flash-lite")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cache-path", default="runs/cache/alpha_variants_gemini_news_hard_50.jsonl")
    ap.add_argument("--output", default="runs/alpha_variants_gemini_news_hard_50.json")
    args = ap.parse_args()

    task = build_news_hard_50_task()
    raw_client = _build_client(args.model)
    cache = PromptCache(path=args.cache_path)
    client = CachedLLMClient(raw_client, cache, model_id=args.model)

    print(f"=== Alpha-variants scorer on {args.model} × news_hard_50 ===")
    print(f"  cache: {args.cache_path} ({len(cache)} preloaded keys)")
    print(f"  repeats per variant: {args.repeats}")

    def _pred(temp: float = 0.0, n_samples: int = 1) -> LLMPredictor:
        return LLMPredictor(client=client, classes=task.classes,
                            n_samples=n_samples, temperature=temp,
                            max_tokens=512)

    variants: dict[str, Harness] = {
        "alpha":                     Harness(components=[
            BM25Retriever(corpus=task.train, k=5),
            LLMReranker(client=client, m=3, temperature=0.2),
            TopKFewShot(k=2),
            CompressedCoTFormatter(max_reasoning_words=15),
            _pred(), NullVoter(),
        ]),
        "V1_alpha+voting":           Harness(components=[
            BM25Retriever(corpus=task.train, k=5),
            LLMReranker(client=client, m=3, temperature=0.2),
            TopKFewShot(k=2),
            CompressedCoTFormatter(max_reasoning_words=15),
            _pred(temp=0.4, n_samples=3), MajorityVoter(),
        ]),
        "V2_alpha_wider":            Harness(components=[
            BM25Retriever(corpus=task.train, k=8),
            LLMReranker(client=client, m=4, temperature=0.2),
            TopKFewShot(k=4),
            CompressedCoTFormatter(max_reasoning_words=15),
            _pred(), NullVoter(),
        ]),
        "V3_alpha+more_reasoning":   Harness(components=[
            BM25Retriever(corpus=task.train, k=5),
            LLMReranker(client=client, m=3, temperature=0.2),
            TopKFewShot(k=2),
            CompressedCoTFormatter(max_reasoning_words=40),
            _pred(), NullVoter(),
        ]),
        "V4_alpha+voting+wider":     Harness(components=[
            BM25Retriever(corpus=task.train, k=8),
            LLMReranker(client=client, m=4, temperature=0.2),
            TopKFewShot(k=4),
            CompressedCoTFormatter(max_reasoning_words=15),
            _pred(temp=0.4, n_samples=3), MajorityVoter(),
        ]),
        "V5_alpha+voting+reasoning": Harness(components=[
            BM25Retriever(corpus=task.train, k=5),
            LLMReranker(client=client, m=3, temperature=0.2),
            TopKFewShot(k=2),
            CompressedCoTFormatter(max_reasoning_words=40),
            _pred(temp=0.4, n_samples=3), MajorityVoter(),
        ]),
    }

    scorer = Scorer(task)
    eval_examples = task.eval_set[:50]

    results: dict[str, dict] = {}
    for label, h in variants.items():
        t0 = time.time()
        print(f"  scoring {label}...", flush=True)
        score = scorer.score(h, eval_examples,
                             n_repeats=args.repeats, max_workers=8)
        elapsed = time.time() - t0
        results[label] = {
            "accuracy": score.accuracy,
            "tokens": score.tokens,
            "latency_ms": score.latency_ms,
            "spread": score.accuracy_spread,
            "elapsed_s": round(elapsed, 1),
        }
        print(f"    acc={score.accuracy:.3f}  tok={score.tokens:.0f}  "
              f"lat={score.latency_ms:.0f}ms  spread={score.accuracy_spread:.3f}  "
              f"({elapsed:.0f}s)")

    print()
    cot_acc = 0.940
    cot_tok = 290
    print(f"{'Variant':<28} {'acc':>8} {'tok':>8} {'spread':>8} {'Δacc':>10} {'Δtok':>8}")
    print("-" * 76)
    for label, r in results.items():
        delta_acc = r["accuracy"] - cot_acc
        delta_tok = r["tokens"] - cot_tok
        print(f"{label:<28} {r['accuracy']:>8.3f} {r['tokens']:>8.0f} "
              f"{r['spread']:>8.3f} {delta_acc:>+10.3f} {delta_tok:>+8.0f}")

    # Crown the substantially-beating variant.
    print()
    big_winners = [label for label, r in results.items()
                   if r["accuracy"] >= 0.97]
    if big_winners:
        print(f"  Variants substantially beating CoT-RAG (acc ≥ 0.97):")
        for w in big_winners:
            r = results[w]
            print(f"    {w}: acc={r['accuracy']:.3f} (+{(r['accuracy']-cot_acc)*100:.1f}pt)")
    else:
        print(f"  No variant substantially exceeded CoT-RAG (≥ 0.97).")
        best = max(results, key=lambda k: results[k]["accuracy"])
        r = results[best]
        print(f"  Best: {best} at acc={r['accuracy']:.3f} (+{(r['accuracy']-cot_acc)*100:.1f}pt)")

    out = {
        "model": args.model,
        "task": "news_hard_50",
        "repeats": args.repeats,
        "cot_rag_baseline": {"accuracy": cot_acc, "tokens": cot_tok},
        "variants": results,
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()
