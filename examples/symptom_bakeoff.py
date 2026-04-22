"""Reproducibility-enabled bakeoff on the bundled symptom task.

Combines main's LLM integration with the reproducibility branch's
median-over-repeats aggregation and the real-dataset branch's bundled
medical symptom classification task. Runs N models sequentially, each
with ``eval_repeats`` and ``attribution_repeats`` > 1 so the resulting
Pareto frontiers are robust to backend non-determinism.

Usage:
    python3 examples/symptom_bakeoff.py \\
        --ollama-url http://localhost:11434/api/chat \\
        --models gpt-oss:20b gemma4:26b \\
        --iterations 3 --proposals 4 \\
        --eval-repeats 2 --attribution-repeats 2 \\
        --attribution-screen-size 12

Writes per-model artifacts to ``runs/symptom_bakeoff/<model>/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from meta_harness_plus.attribution import AttributionTracker
from meta_harness_plus.components import (
    NullFewShot, NullRetriever, NullVoter, SimpleFormatter, baseline_for,
)
from meta_harness_plus.harness import Harness
from meta_harness_plus.llm.client import HTTPClient
from meta_harness_plus.llm.predictor import LLMPredictor
from meta_harness_plus.llm.proposer import LLMProposer
from meta_harness_plus.llm.registry import llm_search_registry
from meta_harness_plus.runner import SearchConfig, SearchRunner
from meta_harness_plus.scorer import Scorer
from meta_harness_plus.tasks import build_symptom_task


def run_one_model(model: str, ollama_url: str, run_dir: Path, args) -> dict:
    print(f"\n{'=' * 70}\n  Symptom bakeoff (reproducibility): {model}\n{'=' * 70}", flush=True)
    t0 = time.time()

    task = build_symptom_task()
    client = HTTPClient(api_url=ollama_url, model=model, timeout_s=300.0)

    registry = llm_search_registry(task, client, predictor_max_tokens=args.predictor_max_tokens)
    proposer = LLMProposer(
        client=client, registry=registry, run_dir=str(run_dir),
        temperature=args.proposer_temperature, max_tokens=args.proposer_max_tokens,
    )
    scorer = Scorer(task)
    attribution = AttributionTracker(scorer, baseline_for)

    seed_harness = Harness(components=[
        NullRetriever(), NullFewShot(), SimpleFormatter(),
        LLMPredictor(client=client, classes=task.classes, n_samples=1,
                     max_tokens=args.predictor_max_tokens),
        NullVoter(),
    ])

    runner = SearchRunner(
        task=task, scorer=scorer, proposer=proposer, attribution=attribution,
        config=SearchConfig(
            n_iterations=args.iterations,
            proposals_per_iter=args.proposals,
            screen_size=args.screen_size,
            full_eval_size=args.eval_size,
            halving_k0=args.halving_k0, halving_eta=2,
            halving_final_keep=args.halving_final_keep,
            eval_repeats=args.eval_repeats,
            screen_repeats=args.screen_repeats,
            attribution_repeats=args.attribution_repeats,
            attribution_screen_size=args.attribution_screen_size,
            run_dir=str(run_dir),
        ),
        seed_harnesses=[seed_harness],
    )

    state = runner.run()
    elapsed = time.time() - t0

    frontier_rows = []
    for e in sorted(state.frontier.entries, key=lambda x: -x.score.accuracy):
        frontier_rows.append({
            "candidate_id": e.candidate_id,
            "accuracy": round(e.score.accuracy, 3),
            "accuracy_spread": round(e.score.accuracy_spread, 3),
            "n_repeats": e.score.n_repeats,
            "tokens": round(e.score.tokens, 1),
            "latency_ms": round(e.score.latency_ms, 1),
            "components": [
                {k: v for k, v in c.items() if k in {"kind", "name", "k", "n_samples"}}
                for c in e.meta.get("describe", [])
            ],
        })

    attr_rows = [
        {"kind": s.kind, "mean_delta": round(s.mean_delta, 3),
         "variance": round(s.variance, 3), "n": s.n}
        for s in attribution.ranking()
    ]

    summary = {
        "model": model,
        "task": task.name,
        "elapsed_s": round(elapsed, 1),
        "config": {
            "iterations": args.iterations,
            "proposals_per_iter": args.proposals,
            "eval_repeats": args.eval_repeats,
            "screen_repeats": args.screen_repeats,
            "attribution_repeats": args.attribution_repeats,
            "attribution_screen_size": args.attribution_screen_size,
            "eval_size": args.eval_size,
        },
        "frontier": frontier_rows,
        "attribution": attr_rows,
    }

    print(f"\n[{model}] wall={elapsed:.1f}s   frontier_size={len(frontier_rows)}")
    print(f"[{model}] Pareto frontier (acc reported as median over {args.eval_repeats} repeats):")
    for row in frontier_rows:
        shape = " + ".join(
            f"{c['kind']}/{c.get('name','?')}"
            + (f"(k={c['k']})" if 'k' in c else "")
            + (f"(n={c['n_samples']})" if 'n_samples' in c else "")
            for c in row["components"]
        )
        spread_note = f" spread={row['accuracy_spread']:.2f}" if row['accuracy_spread'] > 0 else ""
        print(f"  acc={row['accuracy']:.2f}{spread_note}  "
              f"tok={row['tokens']:7.0f}  lat={row['latency_ms']:8.0f}ms  :: {shape}")
    print(f"[{model}] Attribution ranking:")
    for row in attr_rows:
        print(f"  {row['kind']:12s}  mean_delta={row['mean_delta']:+.3f}  "
              f"var={row['variance']:.3f}  n={row['n']}")

    (run_dir / "bakeoff_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ollama-url", default="http://localhost:11434/api/chat")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--proposals", type=int, default=4)
    ap.add_argument("--screen-size", type=int, default=6)
    ap.add_argument("--eval-size", type=int, default=20)
    ap.add_argument("--halving-k0", type=int, default=3)
    ap.add_argument("--halving-final-keep", type=int, default=2)
    ap.add_argument("--proposer-temperature", type=float, default=0.5)
    ap.add_argument("--proposer-max-tokens", type=int, default=4096)
    ap.add_argument("--predictor-max-tokens", type=int, default=512)
    # Reproducibility knobs:
    ap.add_argument("--eval-repeats", type=int, default=2)
    ap.add_argument("--screen-repeats", type=int, default=1)
    ap.add_argument("--attribution-repeats", type=int, default=2)
    ap.add_argument("--attribution-screen-size", type=int, default=12)
    args = ap.parse_args()

    out_root = Path("runs/symptom_bakeoff")
    out_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for model in args.models:
        model_dir = out_root / model.replace(":", "_").replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)
        s = run_one_model(model, args.ollama_url, model_dir, args)
        summaries.append(s)

    print(f"\n{'=' * 70}\n  FINAL COMPARISON (symptom task, eval_repeats={args.eval_repeats})\n{'=' * 70}")
    for s in summaries:
        best = max(s["frontier"], key=lambda r: r["accuracy"]) if s["frontier"] else None
        cheapest = min((r["tokens"] for r in s["frontier"]), default=0.0)
        print(f"\n{s['model']}  (wall={s['elapsed_s']}s, iters={s['config']['iterations']}, "
              f"repeats={s['config']['eval_repeats']})")
        print(f"  frontier size:        {len(s['frontier'])}")
        if best:
            print(f"  best accuracy:        {best['accuracy']:.2f}  (spread={best['accuracy_spread']:.2f})")
            print(f"  best @ tokens:        {best['tokens']:.0f}")
        print(f"  cheapest on frontier: {cheapest:.0f} tokens")
        if s["attribution"]:
            top = max(s["attribution"], key=lambda r: r["mean_delta"])
            print(f"  most valuable kind:   {top['kind']} (mean_delta={top['mean_delta']:+.3f})")

    (out_root / "comparison.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nArtifacts written under: {out_root}")


if __name__ == "__main__":
    main()
