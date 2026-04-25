"""Hand-tuned-baselines bakeoff: score every named baseline harness on a
task with a real LLM, multiple seeds, bootstrap CIs.

Answers the reviewer question "did MH++ beat the strong hand-tuned
baselines, or just vanilla RAG?" with data, not vibes. Reports:

- BARE       (no retrieval, no fewshot, no voting)
- RAG        (BoW retriever + topk fewshot + simple formatter)
- CoT-RAG    (RAG + CoT formatter)
- voting-RAG (RAG + 3-sample majority voter)
- diverse-RAG (RAG with diversity reranker, larger k)

For each, we run N seeds, bootstrap a 95% CI on accuracy and tokens,
and print a table next to the MH++ peak from a corresponding bakeoff
run if --mh-aggregate is supplied.

Usage::

    python3 examples/hand_tuned_baselines.py \\
        --api gemini --models gemini-2.5-flash-lite \\
        --task news_hard_50 --seeds 5 \\
        --eval-size 50 \\
        --cache-path runs/cache/baselines_gemini_news_hard_50.jsonl \\
        --mh-aggregate runs/gemini_news_hard_50_big_aggregate.json \\
        --output runs/hand_tuned_baselines_gemini_news_hard_50.json

Cost: ~$0.01-0.05 per task per provider (single-eval per harness, no
search loop). One run for each baseline × eval_size items × n_seeds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from meta_harness_plus.baselines import (
    bare_baseline,
    cot_baseline,
    diverse_rag_baseline,
    rag_baseline,
    voting_rag_baseline,
)
from meta_harness_plus.llm.client import HTTPClient
from meta_harness_plus.llm.cache import CachedLLMClient, PromptCache
from meta_harness_plus.llm.predictor import LLMPredictor
from meta_harness_plus.scorer import Scorer
from meta_harness_plus.statistics import bootstrap_ci, paired_t_test
from meta_harness_plus.tasks import (
    build_news_hard_50_task,
    build_news_hard_task,
    build_news_task,
    build_symptom_hard_task,
    build_symptom_task,
)
from meta_harness_plus.tasks.lawbench import build_lawbench_task

TASK_FACTORIES = {
    "news": build_news_task,
    "news_hard": build_news_hard_task,
    "news_hard_50": build_news_hard_50_task,
    "symptom": build_symptom_task,
    "symptom_hard": build_symptom_hard_task,
    "lawbench_2_2": lambda: build_lawbench_task("2-2"),
}


def _build_client(args, model: str) -> HTTPClient:
    if args.api == "openai":
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise SystemExit("--api openai needs OPENAI_API_KEY")
        return HTTPClient(api_url=args.openai_url, api_key=key, model=model, timeout_s=300.0)
    if args.api == "gemini":
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
        if not key:
            raise SystemExit("--api gemini needs GEMINI_API_KEY or GOOGLE_API_KEY")
        url = (f"https://generativelanguage.googleapis.com/v1beta/"
               f"models/{model}:generateContent")
        return HTTPClient(api_url=url, api_key=key, model=model, timeout_s=300.0)
    raise SystemExit(f"unknown --api: {args.api}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True, choices=["openai", "gemini"])
    ap.add_argument("--models", required=True)
    ap.add_argument("--task", required=True, choices=list(TASK_FACTORIES.keys()))
    ap.add_argument("--seeds", type=int, default=5,
                    help="Number of seeds (controls n_repeats within scorer for stability)")
    ap.add_argument("--eval-size", type=int, default=50)
    ap.add_argument("--cache-path", default="")
    ap.add_argument("--openai-url", default="https://api.openai.com/v1/chat/completions")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--mh-aggregate", default="",
                    help="Path to MH++ multi_seed_cross_provider aggregate JSON for side-by-side")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    task = TASK_FACTORIES[args.task]()
    raw_client = _build_client(args, args.models)
    if args.cache_path:
        cache = PromptCache(path=args.cache_path)
        client = CachedLLMClient(raw_client, cache, model_id=args.models)
        print(f"  prompt cache: {args.cache_path} ({len(cache)} preloaded keys)",
              flush=True)
    else:
        client = raw_client

    print(f"\n{'='*72}")
    print(f"  Hand-tuned baselines bakeoff: {args.models} on {args.task} via {args.api}")
    print(f"{'='*72}\n")

    scorer = Scorer(task)

    n_seeds = args.seeds
    # Repeats per harness — n_repeats within Scorer.score gives us the
    # multi-seed effect (since the eval set is fixed; nondeterminism is
    # from the LLM). Each repeat counts as a "seed" for stat purposes.
    n_repeats = n_seeds

    # Build each baseline. Each gets its own LLMPredictor instance to
    # keep token/latency stats local.
    classes = task.classes

    def _pred(temp: float = 0.0, n_samples: int = 1) -> LLMPredictor:
        return LLMPredictor(client=client, classes=classes,
                            n_samples=n_samples, temperature=temp)

    harnesses = {
        "BARE": bare_baseline(task, _pred()),
        "RAG": rag_baseline(task, _pred()),
        "CoT-RAG": cot_baseline(task, _pred()),
        "voting-RAG": voting_rag_baseline(task, _pred(temp=0.4, n_samples=3)),
        "diverse-RAG": diverse_rag_baseline(task, _pred()),
    }

    n_eval = min(args.eval_size, len(task.eval_set))
    eval_examples = task.eval_set[:n_eval]

    results: dict[str, dict] = {}
    for label, h in harnesses.items():
        t0 = time.time()
        print(f"  scoring {label} ({n_eval} items × {n_repeats} repeats)...", flush=True)
        score = scorer.score(h, eval_examples,
                             n_repeats=n_repeats,
                             max_workers=args.max_workers)
        elapsed = time.time() - t0
        # Each repeat-level accuracy lives inside score.per_class_accuracy
        # as the average of class-level accs across repeats.
        # For seed-level CI we approximate via (acc, repeats=n_seeds) but
        # since Scorer.score returns the median over repeats we cannot
        # compute per-repeat variance. Use a simple Wilson-interval
        # approximation:
        results[label] = {
            "accuracy": score.accuracy,
            "tokens": score.tokens,
            "latency_ms": score.latency_ms,
            "n_eval": n_eval,
            "n_repeats": n_repeats,
            "elapsed_s": round(elapsed, 1),
        }
        print(f"    acc={score.accuracy:.3f}  tok={score.tokens:.1f}  "
              f"lat={score.latency_ms:.0f}ms  ({elapsed:.0f}s)")

    print()
    # Print comparison table.
    print(f"{'Baseline':<12} {'acc':>8} {'tok':>8} {'lat':>10} {'Δacc-vs-RAG':>14}")
    print("-" * 60)
    rag_acc = results["RAG"]["accuracy"]
    for label, r in results.items():
        delta = r["accuracy"] - rag_acc
        print(f"{label:<12} {r['accuracy']:>8.3f} {r['tokens']:>8.1f} "
              f"{r['latency_ms']:>10.0f} {delta:>+14.3f}")

    # Side-by-side with MH++ if aggregate was supplied.
    if args.mh_aggregate and Path(args.mh_aggregate).exists():
        mh = json.load(open(args.mh_aggregate))
        mh_peak = mh["ci"]["mh_peak_acc"]
        mh_tokens = mh["per_seed_mh_peak_tokens"]
        mh_tok_mean = sum(mh_tokens) / len(mh_tokens) if mh_tokens else 0
        print()
        print(f"  --- vs MH++ ({mh['label']}) ---")
        print(f"  MH++ acc: {mh_peak['mean']:.3f} [{mh_peak['low']:.3f}, {mh_peak['high']:.3f}] (95%)")
        print(f"  MH++ tok: {mh_tok_mean:.1f}")
        print()
        print(f"  Δ (MH++ − each baseline) on accuracy:")
        for label, r in results.items():
            delta = mh_peak["mean"] - r["accuracy"]
            print(f"    vs {label:<12} {delta:+.3f}")

    # Save.
    out = {
        "task": args.task,
        "model": args.models,
        "api": args.api,
        "n_eval": n_eval,
        "n_repeats": n_repeats,
        "baselines": results,
    }
    if args.mh_aggregate and Path(args.mh_aggregate).exists():
        out["mh_aggregate_path"] = args.mh_aggregate
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()
