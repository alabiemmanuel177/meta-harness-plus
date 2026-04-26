"""Killer-demo CLI: end-to-end MH++ in 10 minutes.

Single command: load a dataset + API key → MH++ search → the discovered
harness, the per-component attribution, and a static HTML dashboard
showing the Pareto frontier.

Usage:
    # Set OPENAI_API_KEY (or GEMINI_API_KEY) in env, then:
    python3 examples/demo.py --task news_hard_50 --api openai \\
        --model gpt-4.1-nano --output-dir runs/demo

The output directory contains:
    runs/demo/
      ├── frontier.json        # the discovered Pareto frontier
      ├── attribution.json     # per-component drop-one + synergy stats
      ├── dashboard.html       # standalone visualization, opens in browser
      └── README.md            # one-line replication instructions

Total wall time: ~3-10 minutes depending on dataset and budget.
Total API cost: ~$0.05-0.30.
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
from meta_harness_plus.baselines import bare_baseline, cot_baseline, rag_baseline
from meta_harness_plus.components import baseline_for
from meta_harness_plus.llm.cache import CachedLLMClient, PromptCache
from meta_harness_plus.llm.client import HTTPClient
from meta_harness_plus.llm.predictor import LLMPredictor
from meta_harness_plus.llm.proposer import LLMProposer
from meta_harness_plus.llm.registry import llm_search_registry
from meta_harness_plus.runner import SearchConfig, SearchRunner
from meta_harness_plus.scorer import Scorer
from meta_harness_plus.synergy import SynergyTracker
from meta_harness_plus.tasks import (
    build_news_hard_50_task, build_symptom_hard_task,
    build_agnews_task, build_emotion_task,
)
from meta_harness_plus.tasks.lawbench import build_lawbench_task

TASKS = {
    "news_hard_50": build_news_hard_50_task,
    "symptom_hard": build_symptom_hard_task,
    "agnews": build_agnews_task,
    "emotion": build_emotion_task,
    "lawbench_2_2": lambda: build_lawbench_task("2-2"),
}


def _build_client(api: str, model: str) -> HTTPClient:
    if api == "openai":
        return HTTPClient(
            api_url="https://api.openai.com/v1/chat/completions",
            api_key=os.environ["OPENAI_API_KEY"],
            model=model, timeout_s=300.0,
        )
    if api == "gemini":
        key = os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]
        return HTTPClient(
            api_url=f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            api_key=key, model=model, timeout_s=300.0,
        )
    raise SystemExit(f"unknown api: {api}")


DASHBOARD_HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset='utf-8'>
<title>MH++ Discovered Frontier</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 800px; margin: 2em auto; padding: 0 1em; line-height: 1.5; }}
  h1 {{ font-size: 1.5em; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
  th, td {{ border: 1px solid #ddd; padding: 0.5em; text-align: left; }}
  th {{ background: #f5f5f5; }}
  .winner {{ background: #d4edda; }}
  .seed {{ background: #fff3cd; }}
  .metric {{ font-family: monospace; }}
  .pill {{ display: inline-block; padding: 2px 8px; border-radius: 8px; background: #e7f1ff; font-size: 0.9em; }}
</style></head><body>
<h1>MH++ on {task} via {api}/{model}</h1>
<p>Search ran <strong>{elapsed:.0f}s</strong> wall, evaluated <strong>{n_candidates}</strong> candidate harnesses,
admitted <strong>{n_frontier}</strong> to the Pareto frontier.</p>

<h2>Pareto frontier (sorted by accuracy)</h2>
<table>
<tr><th>Label</th><th>Accuracy</th><th>Tokens</th><th>Latency (ms)</th><th>Components</th></tr>
{frontier_rows}
</table>

<h2>Component attribution (drop-one)</h2>
<table>
<tr><th>Kind</th><th>Mean Δacc</th><th>EWMA Δacc</th><th>Variance</th><th>n</th></tr>
{attribution_rows}
</table>

<h2>Component synergy (drop-pair, top frontier candidate)</h2>
<table>
<tr><th>Pair</th><th>Synergy Δ</th><th>Interpretation</th></tr>
{synergy_rows}
</table>

<h2>Replicate</h2>
<pre><code>python3 examples/demo.py --task {task} --api {api} --model {model}</code></pre>
<p class='pill'>API spend: ~${cost:.2f} · Wall: ~{elapsed:.0f}s</p>

</body></html>
"""


def _interpret_synergy(d: float) -> str:
    if d > 0.05:
        return "super-additive (only work together)"
    if d < -0.05:
        return "redundant"
    return "approximately additive"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS.keys()))
    ap.add_argument("--api", required=True, choices=["openai", "gemini"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--iterations", type=int, default=4)
    ap.add_argument("--proposals", type=int, default=4)
    ap.add_argument("--output-dir", default="runs/demo")
    args = ap.parse_args()

    task = TASKS[args.task]()
    client_raw = _build_client(args.api, args.model)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = PromptCache(path=str(out_dir / "cache.jsonl"))
    client = CachedLLMClient(client_raw, cache, model_id=args.model)

    print(f"=== MH++ Demo: {args.task} on {args.api}/{args.model} ===")
    print(f"  search budget: {args.iterations} iterations × {args.proposals} proposals")
    print(f"  output: {out_dir}/")

    predictor = LLMPredictor(
        client=client, classes=task.classes,
        n_samples=1, temperature=0.0, max_tokens=512,
    )
    bare = bare_baseline(task, predictor)
    rag = rag_baseline(task, predictor)
    cot = cot_baseline(task, predictor)

    scorer = Scorer(task)
    attribution = AttributionTracker(scorer, baseline_for)
    registry = llm_search_registry(task, client)
    proposer = LLMProposer(
        client=client, registry=registry,
        run_dir=str(out_dir / "search"),
        temperature=0.5,
    )

    runner = SearchRunner(
        task=task, scorer=scorer, proposer=proposer, attribution=attribution,
        config=SearchConfig(
            n_iterations=args.iterations,
            proposals_per_iter=args.proposals,
            screen_size=8, full_eval_size=min(50, len(task.eval_set)),
            halving_k0=4, halving_eta=2, halving_final_keep=2,
            eval_repeats=2, attribution_repeats=2, attribution_screen_size=12,
            max_workers=8, screen_seed=0,
            run_dir=str(out_dir / "search"),
        ),
        seed_harnesses=[bare, rag, cot],
    )

    t0 = time.time()
    state = runner.run()
    elapsed = time.time() - t0
    print(f"  search done in {elapsed:.0f}s")

    # Synergy on the top-of-frontier candidate
    entries = list(state.frontier.entries)
    top = max(entries, key=lambda e: e.score.accuracy)
    print(f"  running synergy analysis on top candidate (acc={top.score.accuracy:.3f})...")
    synergy = SynergyTracker(scorer, baseline_for)
    synergy.analyze(
        candidate_id=top.candidate_id,
        harness=top.meta.get("harness"),
        examples=task.eval_set[:20],  # smaller subset for synergy (it's O(K²))
        full_score=top.score,
    )

    # Write outputs
    seed_label = {
        "cand_0001": "BARE", "cand_0002": "RAG", "cand_0003": "CoT-RAG",
    }
    frontier_rows_html: list[str] = []
    for e in sorted(entries, key=lambda x: -x.score.accuracy):
        label = seed_label.get(e.candidate_id, "discovered")
        css_class = "seed" if label != "discovered" else ("winner" if e == top else "")
        comps = e.meta.get("describe", [])
        comp_str = " + ".join(c.get("name", "?") for c in comps)
        frontier_rows_html.append(
            f"<tr class='{css_class}'><td>{label}</td>"
            f"<td class='metric'>{e.score.accuracy:.3f}</td>"
            f"<td class='metric'>{e.score.tokens:.0f}</td>"
            f"<td class='metric'>{e.score.latency_ms:.0f}</td>"
            f"<td>{comp_str}</td></tr>"
        )

    attribution_rows_html = [
        f"<tr><td>{s.kind}</td>"
        f"<td class='metric'>{s.mean_delta:+.3f}</td>"
        f"<td class='metric'>{s.ewma_delta:+.3f}</td>"
        f"<td class='metric'>{s.variance:.4f}</td>"
        f"<td>{s.n}</td></tr>"
        for s in attribution.ranking()
    ]

    synergy_rows_html = [
        f"<tr><td>{s.pair[0]} × {s.pair[1]}</td>"
        f"<td class='metric'>{s.mean_synergy:+.3f}</td>"
        f"<td>{_interpret_synergy(s.mean_synergy)}</td></tr>"
        for s in synergy.ranking()
    ]

    cost_estimate = cache.stats()["misses"] * 0.0001  # very rough
    html = DASHBOARD_HTML_TEMPLATE.format(
        task=args.task, api=args.api, model=args.model,
        elapsed=elapsed, n_candidates=len(state.frontier.entries) +
            sum(1 for _ in attribution.snapshots),
        n_frontier=len(entries),
        frontier_rows="\n".join(frontier_rows_html),
        attribution_rows="\n".join(attribution_rows_html),
        synergy_rows="\n".join(synergy_rows_html),
        cost=cost_estimate,
    )
    (out_dir / "dashboard.html").write_text(html)
    print(f"  dashboard written: {out_dir}/dashboard.html")

    # frontier.json
    frontier_json = [
        {"candidate_id": e.candidate_id,
         "label": seed_label.get(e.candidate_id, "discovered"),
         "accuracy": e.score.accuracy, "tokens": e.score.tokens,
         "latency_ms": e.score.latency_ms,
         "components": e.meta.get("describe", [])}
        for e in sorted(entries, key=lambda x: -x.score.accuracy)
    ]
    (out_dir / "frontier.json").write_text(json.dumps(frontier_json, indent=2))

    # attribution.json
    attr_json = {
        "drop_one": [
            {"kind": s.kind, "mean_delta": s.mean_delta, "ewma_delta": s.ewma_delta,
             "variance": s.variance, "n": s.n}
            for s in attribution.ranking()
        ],
        "synergy": [
            {"pair": [s.pair[0], s.pair[1]], "mean_synergy": s.mean_synergy,
             "interpretation": _interpret_synergy(s.mean_synergy)}
            for s in synergy.ranking()
        ],
    }
    (out_dir / "attribution.json").write_text(json.dumps(attr_json, indent=2))

    # README.md
    readme = f"""# MH++ Demo Output

Discovered harness for `{args.task}` on `{args.api}/{args.model}` after
{elapsed:.0f}s of search.

## Top of frontier

- **accuracy:** {top.score.accuracy:.3f}
- **tokens:** {top.score.tokens:.0f}
- **latency:** {top.score.latency_ms:.0f}ms

## To replicate

```bash
python3 examples/demo.py --task {args.task} --api {args.api} --model {args.model}
```

## Files

- `dashboard.html` — interactive visualization (open in browser)
- `frontier.json` — discovered Pareto frontier
- `attribution.json` — per-component contribution + pair synergy
- `cache.jsonl` — every LLM call (replay-safe)
- `search/` — per-iteration search log
"""
    (out_dir / "README.md").write_text(readme)
    print(f"  README + JSON written under {out_dir}/")
    print(f"\n  Top discovered harness: acc={top.score.accuracy:.3f}, tok={top.score.tokens:.0f}")
    print(f"\n  Open {out_dir}/dashboard.html in your browser to see the frontier.")


if __name__ == "__main__":
    main()
