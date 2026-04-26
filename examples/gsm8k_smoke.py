"""GSM8K smoke runner — end-to-end test of the math wiring (Task 1 math half).

Runs BARE harness (just MathLLMPredictor) on a small GSM8K eval set
to validate the math task end-to-end. Saves accuracy + cost to JSON.

Usage:
    python3 examples/gsm8k_smoke.py --api gemini --model gemini-2.5-flash-lite \\
        --n-eval 20 --output runs/gsm8k_smoke.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from meta_harness_plus.components import NullFewShot, NullRetriever, NullVoter, SimpleFormatter
from meta_harness_plus.harness import Harness
from meta_harness_plus.llm.cache import CachedLLMClient, PromptCache
from meta_harness_plus.llm.client import HTTPClient
from meta_harness_plus.llm.predictor import MathLLMPredictor
from meta_harness_plus.scorer import Scorer
from meta_harness_plus.tasks import build_gsm8k_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True, choices=("openai", "gemini"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-train", type=int, default=20)
    ap.add_argument("--n-eval", type=int, default=20)
    ap.add_argument("--cache-path", default="runs/cache/gsm8k_smoke.jsonl")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.api == "openai":
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise SystemExit("OPENAI_API_KEY required")
        raw = HTTPClient(api_url="https://api.openai.com/v1/chat/completions",
                         api_key=key, model=args.model, timeout_s=300.0)
    else:
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
        if not key:
            raise SystemExit("GEMINI_API_KEY/GOOGLE_API_KEY required")
        raw = HTTPClient(
            api_url=f"https://generativelanguage.googleapis.com/v1beta/models/{args.model}:generateContent",
            api_key=key, model=args.model, timeout_s=300.0,
        )
    cache = PromptCache(path=args.cache_path)
    client = CachedLLMClient(raw, cache, model_id=args.model)

    print(f"=== GSM8K smoke ({args.api}/{args.model}) ===")
    print(f"  loading task: n_train={args.n_train}, n_eval={args.n_eval} ...")
    task = build_gsm8k_task(n_train=args.n_train, n_eval=args.n_eval)
    print(f"  loaded: {len(task.train)} train, {len(task.eval_set)} eval")

    predictor = MathLLMPredictor(client=client, n_samples=1, temperature=0.0,
                                 max_tokens=512)
    harness = Harness(components=[
        NullRetriever(), NullFewShot(), SimpleFormatter(),
        predictor, NullVoter(),
    ])

    print(f"  scoring BARE on {len(task.eval_set)} eval examples ...")
    scorer = Scorer(task)
    t0 = time.time()
    score = scorer.score(harness, task.eval_set, n_repeats=1, max_workers=4)
    elapsed = time.time() - t0

    correct = int(score.accuracy * len(task.eval_set))
    print(f"\n  Score: acc={score.accuracy:.3f} ({correct}/{len(task.eval_set)})")
    print(f"  Tokens: {score.tokens:.1f}/example, latency: {score.latency_ms:.0f}ms/example")
    print(f"  Wall time: {elapsed:.0f}s")

    out = {
        "api": args.api, "model": args.model, "task": "gsm8k",
        "n_train": args.n_train, "n_eval": args.n_eval,
        "score": {
            "accuracy": score.accuracy,
            "tokens": score.tokens,
            "latency_ms": score.latency_ms,
            "spread": score.accuracy_spread,
        },
        "elapsed_s": round(elapsed, 1),
        "cache_stats": cache.stats(),
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()
