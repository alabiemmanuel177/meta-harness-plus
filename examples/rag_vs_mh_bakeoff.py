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

from meta_harness_plus.ablations import ScalarAccuracyFrontier
from meta_harness_plus.attribution import AttributionTracker
from meta_harness_plus.baselines import (
    bare_baseline, cot_baseline, diverse_rag_baseline, rag_baseline,
    voting_rag_baseline,
)
from meta_harness_plus.components import baseline_for
from meta_harness_plus.search.random_proposer import RandomProposer
from meta_harness_plus.llm.cache import CachedLLMClient, PromptCache
from meta_harness_plus.llm.client import HTTPClient
from meta_harness_plus.llm.predictor import LLMPredictor
from meta_harness_plus.llm.proposer import LLMProposer
from meta_harness_plus.llm.registry import llm_search_registry
from meta_harness_plus.pareto import dominates
from meta_harness_plus.runner import SearchConfig, SearchRunner
from meta_harness_plus.scorer import Scorer
from meta_harness_plus.search.ensemble_proposer import EnsembleProposer
from meta_harness_plus.tasks import (
    build_agnews_task,
    build_emotion_task,
    build_news_hard_50_task,
    build_news_hard_task,
    build_news_task,
    build_symptom_hard_task,
)
from meta_harness_plus.tasks.lawbench import build_lawbench_task


TASK_FACTORIES = {
    "symptom_hard": build_symptom_hard_task,
    "news": build_news_task,
    "news_hard": build_news_hard_task,
    "news_hard_50": build_news_hard_50_task,
    "lawbench_2_2": lambda: build_lawbench_task("2-2"),
    "agnews": build_agnews_task,
    "emotion": build_emotion_task,
}


def _build_client(model: str, args) -> HTTPClient:
    """Pick the right HTTPClient backend based on --api."""
    if args.api == "ollama":
        return HTTPClient(api_url=args.ollama_url, model=model, timeout_s=300.0)
    if args.api == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise SystemExit("--api anthropic needs ANTHROPIC_API_KEY env var")
        return HTTPClient(
            api_url="https://api.anthropic.com/v1/messages",
            api_key=key, model=model, timeout_s=300.0,
        )
    if args.api == "openai":
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise SystemExit("--api openai needs OPENAI_API_KEY env var")
        return HTTPClient(
            api_url=args.openai_url, api_key=key,
            model=model, timeout_s=300.0,
        )
    if args.api == "gemini":
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
        if not key:
            raise SystemExit("--api gemini needs GEMINI_API_KEY or GOOGLE_API_KEY env var")
        url = (f"https://generativelanguage.googleapis.com/v1beta/"
               f"models/{model}:generateContent")
        return HTTPClient(api_url=url, api_key=key, model=model, timeout_s=300.0)
    raise SystemExit(f"unknown --api: {args.api}")


def run_one_model(model: str, ollama_url: str, run_dir: Path, args) -> dict:
    task_name = args.task
    print(f"\n{'=' * 72}\n  RAG vs MH++ bakeoff: {model} on {task_name} via {args.api}\n{'=' * 72}", flush=True)
    t0 = time.time()

    task = TASK_FACTORIES[task_name]()
    raw_client = _build_client(model, args)
    if args.cache_path:
        cache = PromptCache(path=args.cache_path)
        client = CachedLLMClient(raw_client, cache, model_id=model)
        print(f"  prompt cache: {args.cache_path} ({len(cache)} preloaded keys)",
              flush=True)
    else:
        cache = None
        client = raw_client
    if args.max_workers > 1:
        print(f"  parallel scoring: {args.max_workers} threads", flush=True)

    registry = llm_search_registry(task, client, predictor_max_tokens=args.predictor_max_tokens)

    if args.ablation == "no-c3":
        # C3 ablation: random proposer instead of attribution-guided LLMProposer.
        # Build mutators dict from the registry by enumerating each kind's
        # available components and binding zero-arg factories.
        mutators: dict[str, list] = {}
        for kind, name in registry.available():
            entry = registry.entry(kind, name)
            if entry is None:
                continue
            # Use empty cfg dict — uses each factory's defaults.
            mutators.setdefault(kind, []).append(
                (lambda e=entry: e.factory({}))
            )
        proposer = RandomProposer(mutators=mutators, seed=args.screen_seed)
        print(f"  proposer: RandomProposer (no-c3 ablation, seed={args.screen_seed})",
              flush=True)
    elif args.proposer_mode == "ensemble":
        # Ensemble of K LLMProposers at different temperatures. Each reads
        # the same filesystem run log but explores from different
        # sampling-temperature regimes — diversity at no extra search-
        # iteration cost (each child gets n // K proposals per call).
        temps = [args.proposer_temperature * (1.0 + 0.3 * i)
                 for i in range(args.ensemble_size)]
        children = [
            LLMProposer(
                client=client, registry=registry, run_dir=str(run_dir),
                temperature=t, max_tokens=args.proposer_max_tokens,
            )
            for t in temps
        ]
        proposer = EnsembleProposer(proposers=children, dedup=True)
        print(f"  proposer: ensemble of {args.ensemble_size} at temps={temps}",
              flush=True)
    else:
        proposer = LLMProposer(
            client=client, registry=registry, run_dir=str(run_dir),
            temperature=args.proposer_temperature, max_tokens=args.proposer_max_tokens,
        )
        print(f"  proposer: single LLMProposer at temp={args.proposer_temperature}",
              flush=True)

    scorer = Scorer(task)
    attribution = AttributionTracker(scorer, baseline_for)

    # Build the two labeled baselines. They share a predictor instance only
    # because LLMPredictor is stateless — the search doesn't assume this.
    predictor = LLMPredictor(client=client, classes=task.classes, n_samples=1,
                             max_tokens=args.predictor_max_tokens)
    bare = bare_baseline(task, predictor)
    rag  = rag_baseline(task, predictor, retriever_k=3, fewshot_k=2)

    # Optional: seed additional hand-tuned baselines into the frontier so
    # the search starts with a stronger floor. Practitioners typically do
    # have hand-tuned baselines and would not want the search to throw
    # them away.
    extra_seeds: list = []
    if args.seed_extra_baselines:
        # Need a predictor with n_samples > 1 for voting-RAG to matter.
        voting_pred = LLMPredictor(
            client=client, classes=task.classes,
            n_samples=3, temperature=0.4,
            max_tokens=args.predictor_max_tokens,
        )
        extra_seeds = [
            cot_baseline(task, predictor),
            voting_rag_baseline(task, voting_pred),
            diverse_rag_baseline(task, predictor),
        ]
        print(f"  seeding extra baselines: cot, voting-RAG, diverse-RAG", flush=True)

    # Apply ablation knobs.
    if args.ablation == "no-c2":
        # No halving: full-eval every candidate, keep them all through to
        # the next iteration. Same total compute as full MH++ but no early
        # elimination.
        halving_k0 = args.eval_size
        halving_final_keep = args.proposals
        print(f"  ablation: no-c2 (full-eval every candidate, no halving)", flush=True)
    else:
        halving_k0 = args.halving_k0
        halving_final_keep = args.halving_final_keep

    if args.ablation == "no-c1":
        frontier_factory = lambda: ScalarAccuracyFrontier()
        print(f"  ablation: no-c1 (ScalarAccuracyFrontier — accuracy only)", flush=True)
    else:
        frontier_factory = None

    runner = SearchRunner(
        task=task, scorer=scorer, proposer=proposer, attribution=attribution,
        config=SearchConfig(
            n_iterations=args.iterations,
            proposals_per_iter=args.proposals,
            screen_size=args.screen_size,
            full_eval_size=args.eval_size,
            halving_k0=halving_k0, halving_eta=2,
            halving_final_keep=halving_final_keep,
            eval_repeats=args.eval_repeats,
            screen_repeats=args.screen_repeats,
            attribution_repeats=args.attribution_repeats,
            attribution_screen_size=args.attribution_screen_size,
            frontier_max_spread=args.frontier_max_spread,
            max_workers=args.max_workers,
            screen_seed=args.screen_seed,
            run_dir=str(run_dir),
        ),
        # Seed order: bare first, then RAG. Both get full-evaluated and
        # added to the Pareto frontier before the search starts.
        seed_harnesses=[bare, rag] + extra_seeds,
        frontier_factory=frontier_factory,
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

    # Build map of seeded candidate_ids → label so reporting & filtering
    # works whether we seeded just BARE+RAG or also extras.
    seed_labels: dict[str, str] = {"cand_0001": "[BARE]", "cand_0002": "[RAG]"}
    if args.seed_extra_baselines:
        seed_labels["cand_0003"] = "[CoT-RAG]"
        seed_labels["cand_0004"] = "[voting-RAG]"
        seed_labels["cand_0005"] = "[diverse-RAG]"

    # Compute: which discovered harnesses Pareto-dominate RAG?
    discovered = [e for e in entries if e.candidate_id not in seed_labels]
    dominates_rag = [e for e in discovered if dominates(e.score, rag_score)]
    ties_rag_on_acc_cheaper = [
        e for e in discovered
        if e.score.accuracy >= rag_score.accuracy and e.score.tokens < rag_score.tokens
    ]

    frontier_rows = []
    for e in sorted(entries, key=lambda x: -x.score.accuracy):
        label = seed_labels.get(e.candidate_id, "")
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
        "ablation": args.ablation,
        "elapsed_s": round(elapsed, 1),
        "cache_stats": cache.stats() if cache else None,
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
    if cache is not None:
        s = cache.stats()
        print(f"[{model}] prompt cache: hits={s['hits']}  misses={s['misses']}  "
              f"hit_rate={s['hit_rate']:.2%}  keys={s['cached_keys']}")
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
    ap.add_argument("--api", choices=("ollama", "anthropic", "openai", "gemini"),
                    default="ollama",
                    help="provider backend; reads {ANTHROPIC,OPENAI,GEMINI}_API_KEY "
                         "from env when not ollama")
    ap.add_argument("--ollama-url", default="http://localhost:11434/api/chat")
    ap.add_argument("--openai-url", default="https://api.openai.com/v1/chat/completions",
                    help="OpenAI-compatible endpoint URL (also works for vLLM, "
                         "LM Studio, etc.)")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--task", choices=sorted(TASK_FACTORIES.keys()), default="symptom_hard")
    ap.add_argument("--run-name", default=None,
                    help="override the runs/ subdir name (default: rag_vs_mh_<task>)")
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--proposals", type=int, default=4)
    ap.add_argument("--screen-size", type=int, default=6)
    ap.add_argument("--eval-size", type=int, default=15)
    ap.add_argument("--halving-k0", type=int, default=3)
    ap.add_argument("--halving-final-keep", type=int, default=2)
    ap.add_argument("--proposer-temperature", type=float, default=0.5)
    ap.add_argument("--proposer-max-tokens", type=int, default=4096)
    ap.add_argument("--predictor-max-tokens", type=int, default=1024)  # bigger for CoT
    ap.add_argument("--screen-seed", type=int, default=0,
                    help="seed for screening subset selection — vary across "
                         "multi-seed runs to get different held-in slices")
    # Tier 5.2 — ensemble proposer with diversity pressure
    ap.add_argument("--proposer-mode", choices=("single", "ensemble"), default="single",
                    help="single LLMProposer or N-member ensemble at varied temps")
    ap.add_argument("--ensemble-size", type=int, default=4,
                    help="how many LLMProposers in the ensemble")
    # Tier 4.2 — variance-gated frontier admission
    ap.add_argument("--frontier-max-spread", type=float, default=None,
                    help="reject frontier candidates with accuracy_spread > this "
                         "(only effective when eval_repeats > 1)")
    # ROADMAP option C — prompt cache for cheaper multi-seed / ablation runs.
    ap.add_argument("--cache-path", default=None,
                    help="JSONL prompt cache file. Cache hits skip the LLM "
                         "call entirely. Persistent across runs. Only caches "
                         "temperature=0 calls.")
    # parallel-scoring branch — fan out per-example LLM calls across threads
    ap.add_argument("--max-workers", type=int, default=1,
                    help="ThreadPoolExecutor concurrency for per-example LLM "
                         "calls during scoring. 1 = sequential. Useful with "
                         "cloud APIs (urllib releases GIL during I/O); skip "
                         "with ScriptedClient (shared-state race).")
    ap.add_argument("--eval-repeats", type=int, default=2)
    ap.add_argument("--screen-repeats", type=int, default=1)
    ap.add_argument("--attribution-repeats", type=int, default=2)
    ap.add_argument("--attribution-screen-size", type=int, default=10)
    ap.add_argument("--seed-extra-baselines", action="store_true",
                    help="Seed CoT-RAG, voting-RAG, diverse-RAG into the search "
                         "frontier in addition to BARE and RAG. Use when the search "
                         "should be guaranteed to retain strong hand-tuned baselines.")
    ap.add_argument("--ablation",
                    choices=("none", "no-c1", "no-c2", "no-c3"),
                    default="none",
                    help="Ablation condition: no-c1 = ScalarAccuracyFrontier (no Pareto), "
                         "no-c2 = full-eval per candidate (no halving), "
                         "no-c3 = random proposer (no attribution-guided LLM proposer).")
    args = ap.parse_args()

    out_root = Path(f"runs/{args.run_name}" if args.run_name else f"runs/rag_vs_mh_{args.task}")
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
