"""RAG vs MH++ head-to-head bakeoff on the adversarial symptom-hard task.

Every run seeds the Pareto frontier with two labeled reference points:
  - ``bare_baseline``: no retrieval, no fewshot, no voting (floor)
  - ``rag_baseline``: canonical RAG (BoW-retrieve + top-k few-shot + predictor)

Then MH++ search runs on top. At the end, the script reports:

  - Where RAG sits on the final frontier (always — we seeded it).
  - Which discovered harnesses, if any, strictly Pareto-dominate RAG (better
    on at least one axis, at-least-tied on every other).
  - Which discovered harnesses match RAG's accuracy at lower cost.

This is the actual research question: **does the search find shapes that
beat hand-tuned RAG on the frontier?** If yes, we win this benchmark;
if no, we honestly report RAG wins here.

Usage:
    python3 examples/rag_vs_mh_bakeoff.py \\
        --ollama-url http://localhost:11434/api/chat \\
        --models gpt-oss:20b gemma4:26b
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
from meta_harness_plus.baselines import bare_baseline, rag_baseline
from meta_harness_plus.components import baseline_for
from meta_harness_plus.llm.client import HTTPClient
from meta_harness_plus.llm.predictor import LLMPredictor
from meta_harness_plus.llm.proposer import LLMProposer
from meta_harness_plus.llm.registry import llm_search_registry
from meta_harness_plus.pareto import dominates
from meta_harness_plus.runner import SearchConfig, SearchRunner
from meta_harness_plus.scorer import Scorer
from meta_harness_plus.tasks import build_symptom_hard_task


def run_one_model(model: str, ollama_url: str, run_dir: Path, args) -> dict:
    print(f"\n{'=' * 72}\n  RAG vs MH++ bakeoff: {model}\n{'=' * 72}", flush=True)
    t0 = time.time()

    task = build_symptom_hard_task()
    client = HTTPClient(api_url=ollama_url, model=model, timeout_s=300.0)

    registry = llm_search_registry(task, client, predictor_max_tokens=args.predictor_max_tokens)
    proposer = LLMProposer(
        client=client, registry=registry, run_dir=str(run_dir),
        temperature=args.proposer_temperature, max_tokens=args.proposer_max_tokens,
    )

    scorer = Scorer(task)
    attribution = AttributionTracker(scorer, baseline_for)

    # Build the two labeled baselines. They share a predictor instance only
    # because LLMPredictor is stateless — the search doesn't assume this.
    predictor = LLMPredictor(client=client, classes=task.classes, n_samples=1,
                             max_tokens=args.predictor_max_tokens)
    bare = bare_baseline(task, predictor)
    rag  = rag_baseline(task, predictor, retriever_k=3, fewshot_k=2)

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
        # Seed order: bare first, then RAG. Both get full-evaluated and
        # added to the Pareto frontier before the search starts.
        seed_harnesses=[bare, rag],
    )

    state = runner.run()
    elapsed = time.time() - t0

    # Locate the seeded baselines on the final frontier (or note if they
    # were dominated off by discovered harnesses).
    entries = list(state.frontier.entries)
    # First two candidate_ids are the seeds — see runner's _next_id counter.
    bare_entry = next((e for e in entries if e.candidate_id == "cand_0001"), None)
    rag_entry = next((e for e in entries if e.candidate_id == "cand_0002"), None)

    # Even if RAG was evicted from the frontier, we want its standalone score
    # for the final report. Re-score deterministically if missing.
    if rag_entry is None:
        rag_score = scorer.score(rag, task.eval_set, n_repeats=args.eval_repeats)
    else:
        rag_score = rag_entry.score
    if bare_entry is None:
        bare_score = scorer.score(bare, task.eval_set, n_repeats=args.eval_repeats)
    else:
        bare_score = bare_entry.score

    # Compute: which discovered harnesses Pareto-dominate RAG?
    discovered = [e for e in entries if e.candidate_id not in {"cand_0001", "cand_0002"}]
    dominates_rag = [e for e in discovered if dominates(e.score, rag_score)]
    ties_rag_on_acc_cheaper = [
        e for e in discovered
        if e.score.accuracy >= rag_score.accuracy and e.score.tokens < rag_score.tokens
    ]

    frontier_rows = []
    for e in sorted(entries, key=lambda x: -x.score.accuracy):
        label = ""
        if e.candidate_id == "cand_0001":
            label = " [BARE]"
        elif e.candidate_id == "cand_0002":
            label = " [RAG]"
        frontier_rows.append({
            "candidate_id": e.candidate_id,
            "label": label.strip() or "discovered",
            "accuracy": round(e.score.accuracy, 3),
            "accuracy_spread": round(e.score.accuracy_spread, 3),
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
        "bare_baseline": {
            "accuracy": round(bare_score.accuracy, 3),
            "tokens": round(bare_score.tokens, 1),
            "latency_ms": round(bare_score.latency_ms, 1),
        },
        "rag_baseline": {
            "accuracy": round(rag_score.accuracy, 3),
            "tokens": round(rag_score.tokens, 1),
            "latency_ms": round(rag_score.latency_ms, 1),
        },
        "n_discovered_dominates_rag": len(dominates_rag),
        "n_discovered_ties_rag_acc_cheaper": len(ties_rag_on_acc_cheaper),
        "frontier": frontier_rows,
        "attribution": attr_rows,
    }

    # Readable print
    print(f"\n[{model}] wall={elapsed:.1f}s   frontier_size={len(entries)}")
    print(f"[{model}] seeded references:")
    print(f"  BARE:  acc={bare_score.accuracy:.2f}  tok={bare_score.tokens:.0f}  "
          f"lat={bare_score.latency_ms:.0f}ms")
    print(f"  RAG:   acc={rag_score.accuracy:.2f}  tok={rag_score.tokens:.0f}  "
          f"lat={rag_score.latency_ms:.0f}ms")
    print(f"\n[{model}] Final Pareto frontier (median over {args.eval_repeats} repeats):")
    for row in frontier_rows:
        shape = " + ".join(
            f"{c['kind']}/{c.get('name','?')}"
            + (f"(k={c['k']})" if 'k' in c else "")
            + (f"(n={c['n_samples']})" if 'n_samples' in c else "")
            for c in row["components"]
        )
        print(f"  [{row['label']:>10s}]  acc={row['accuracy']:.2f}  "
              f"tok={row['tokens']:6.0f}  lat={row['latency_ms']:7.0f}ms  :: {shape}")
    print(f"\n[{model}] Discovered harnesses that dominate RAG: {len(dominates_rag)}")
    print(f"[{model}] Discovered harnesses matching RAG acc at lower tokens: "
          f"{len(ties_rag_on_acc_cheaper)}")
    print(f"[{model}] Attribution:")
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
    ap.add_argument("--eval-size", type=int, default=15)
    ap.add_argument("--halving-k0", type=int, default=3)
    ap.add_argument("--halving-final-keep", type=int, default=2)
    ap.add_argument("--proposer-temperature", type=float, default=0.5)
    ap.add_argument("--proposer-max-tokens", type=int, default=4096)
    ap.add_argument("--predictor-max-tokens", type=int, default=1024)  # bigger for CoT
    ap.add_argument("--eval-repeats", type=int, default=2)
    ap.add_argument("--screen-repeats", type=int, default=1)
    ap.add_argument("--attribution-repeats", type=int, default=2)
    ap.add_argument("--attribution-screen-size", type=int, default=10)
    args = ap.parse_args()

    out_root = Path("runs/rag_vs_mh")
    out_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for model in args.models:
        model_dir = out_root / model.replace(":", "_").replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)
        s = run_one_model(model, args.ollama_url, model_dir, args)
        summaries.append(s)

    print(f"\n{'=' * 72}\n  FINAL COMPARISON (RAG vs MH++ on symptom_hard)\n{'=' * 72}")
    for s in summaries:
        print(f"\n{s['model']}  (wall={s['elapsed_s']}s)")
        print(f"  BARE:           acc={s['bare_baseline']['accuracy']:.2f}  "
              f"tok={s['bare_baseline']['tokens']:.0f}")
        print(f"  RAG (seeded):   acc={s['rag_baseline']['accuracy']:.2f}  "
              f"tok={s['rag_baseline']['tokens']:.0f}")
        print(f"  Discovered dominating RAG:          {s['n_discovered_dominates_rag']}")
        print(f"  Discovered matching RAG acc, cheaper: {s['n_discovered_ties_rag_acc_cheaper']}")
        if s["n_discovered_dominates_rag"] > 0:
            print(f"  ==> MH++ WIN on this model")
        elif s["n_discovered_ties_rag_acc_cheaper"] > 0:
            print(f"  ==> MH++ TIE on accuracy, cheaper in tokens")
        else:
            print(f"  ==> RAG holds this model")

    (out_root / "comparison.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nArtifacts written under: {out_root}")


if __name__ == "__main__":
    main()
