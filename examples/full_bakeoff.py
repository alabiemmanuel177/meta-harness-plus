"""Full bakeoff: LLMProposer + LLMPredictor end-to-end, across multiple models.

Both the proposer (reading the filesystem run log + emitting JSON harness
specs) and the predictor (classifying each eval item) hit the same Ollama
backend. Produces side-by-side Pareto frontiers and attribution rankings.

Usage:
    python3 examples/full_bakeoff.py \\
        --ollama-url http://localhost:11434/api/chat \\
        --models gpt-oss:20b gemma4:26b

Writes a run log per model to ``runs/full_bakeoff/<model>/``.
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
from meta_harness_plus.components import NullFewShot, NullRetriever, NullVoter, SimpleFormatter, baseline_for
from meta_harness_plus.harness import Harness
from meta_harness_plus.llm.client import HTTPClient
from meta_harness_plus.llm.predictor import LLMPredictor
from meta_harness_plus.llm.proposer import LLMProposer
from meta_harness_plus.llm.registry import llm_search_registry
from meta_harness_plus.runner import SearchConfig, SearchRunner
from meta_harness_plus.scorer import Scorer
from meta_harness_plus.tasks import build_toy_task


def run_one_model(model: str, ollama_url: str, run_dir: Path, args) -> dict:
    """Run the full bakeoff against ``model``. Returns a summary dict."""
    print(f"\n{'=' * 70}\n  Full bakeoff: {model}\n{'=' * 70}", flush=True)
    t0 = time.time()

    task = build_toy_task(seed=0)
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
            halving_k0=args.halving_k0, halving_eta=2, halving_final_keep=args.halving_final_keep,
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

    # Format frontier + attribution for the summary.
    entries = sorted(state.frontier.entries, key=lambda x: -x.score.accuracy)
    frontier_rows = []
    for e in entries:
        frontier_rows.append({
            "candidate_id": e.candidate_id,
            "accuracy": round(e.score.accuracy, 3),
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
        "elapsed_s": round(elapsed, 1),
        "frontier": frontier_rows,
        "attribution": attr_rows,
        "total_proposals_attempted": len(list(runner._id_counter)) if False else None,
    }

    # Print a readable summary.
    print(f"\n[{model}] wall={elapsed:.1f}s   frontier_size={len(frontier_rows)}")
    print(f"[{model}] Pareto frontier:")
    for row in frontier_rows:
        shape = " + ".join(
            f"{c['kind']}/{c.get('name','?')}"
            + (f"(k={c['k']})" if 'k' in c else "")
            + (f"(n={c['n_samples']})" if 'n_samples' in c else "")
            for c in row["components"]
        )
        print(f"  acc={row['accuracy']:.2f}  tok={row['tokens']:7.0f}  "
              f"lat={row['latency_ms']:8.0f}ms  :: {shape}")
    print(f"[{model}] Attribution ranking:")
    for row in attr_rows:
        print(f"  {row['kind']:12s}  mean_delta={row['mean_delta']:+.3f}  "
              f"var={row['variance']:.3f}  n={row['n']}")

    (run_dir / "bakeoff_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ollama-url", default="http://localhost:11434/api/chat")
    ap.add_argument("--models", nargs="+", required=True,
                    help="one or more Ollama model tags to bake off")
    ap.add_argument("--iterations", type=int, default=4)
    ap.add_argument("--proposals", type=int, default=4)
    ap.add_argument("--screen-size", type=int, default=6)
    ap.add_argument("--eval-size", type=int, default=20)
    ap.add_argument("--halving-k0", type=int, default=3)
    ap.add_argument("--halving-final-keep", type=int, default=2)
    ap.add_argument("--proposer-temperature", type=float, default=0.5)
    ap.add_argument("--proposer-max-tokens", type=int, default=4096)
    ap.add_argument("--predictor-max-tokens", type=int, default=512)
    # Reproducibility (reproducibility branch):
    ap.add_argument("--eval-repeats", type=int, default=1,
                    help="repeat each full eval N times; aggregate = median acc, mean cost")
    ap.add_argument("--screen-repeats", type=int, default=1,
                    help="repeat halving screen evals N times (lower, since cheaper)")
    ap.add_argument("--attribution-repeats", type=int, default=1,
                    help="repeat drop-one ablations N times")
    ap.add_argument("--attribution-screen-size", type=int, default=None,
                    help="separate larger subset for attribution; None = same as screen-size")
    args = ap.parse_args()

    out_root = Path("runs/full_bakeoff")
    out_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for model in args.models:
        model_dir = out_root / model.replace(":", "_").replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)
        s = run_one_model(model, args.ollama_url, model_dir, args)
        summaries.append(s)

    # Side-by-side final comparison.
    print(f"\n{'=' * 70}\n  FINAL COMPARISON\n{'=' * 70}")
    for s in summaries:
        best_acc = max((r["accuracy"] for r in s["frontier"]), default=0.0)
        cheapest = min((r["tokens"] for r in s["frontier"]), default=0.0)
        print(f"\n{s['model']}  (wall={s['elapsed_s']}s)")
        print(f"  frontier size:        {len(s['frontier'])}")
        print(f"  best accuracy:        {best_acc:.2f}")
        print(f"  cheapest on frontier: {cheapest:.0f} tokens")
        if s["attribution"]:
            top_kind = max(s["attribution"], key=lambda r: r["mean_delta"])
            print(f"  most valuable kind:   {top_kind['kind']}  "
                  f"(mean_delta={top_kind['mean_delta']:+.3f})")

    (out_root / "comparison.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nArtifacts written under: {out_root}")


if __name__ == "__main__":
    main()
