# RAG vs MH++ — News-Headline Benchmark

**Run:** 2026-04-23
**Branch:** `news-benchmark`
**Task:** `news_classification` — 4-class AG-News-style (world, sports, business, tech), 40 train / 20 eval, hand-curated
**Search config:** 3 iter × 4 props, `eval_repeats=2`, `attribution_repeats=2`, `attribution_screen_size=10`
**Baselines seeded:** BARE (null/null/simple/null) + RAG (bow k=3 + topk k=2 / simple / null)

## Headline verdict

| Model | BARE acc @ tok | RAG acc @ tok | MH++ top acc @ tok | Dominates RAG? |
|---|---|---|---|---|
| `gpt-oss:20b` | 1.00 @ **212** | 1.00 @ 316 | 1.00 @ **210** | **Yes** (1 harness) |
| `gemma4:26b`  | 1.00 @ 336 | 1.00 @ 437 | 1.00 @ **324** | **Yes** (1 harness) |

**Both models: MH++ strictly Pareto-dominates the seeded RAG baseline.**

## What actually happened (honest)

This is a *ceiling-effect* task. All three reference points — BARE, RAG, MH++-discovered — achieve **1.00 accuracy on the 20-item eval**. No accuracy axis to climb.

The Pareto win comes entirely from the cost axes:

- On both models, **RAG adds tokens and latency without adding accuracy**. It's pure overhead on this task.
- The search correctly rediscovered that the bare harness (no retrieval, no few-shot) already hits 1.00, and therefore **any harness that strips RAG's overhead dominates it**.
- gpt-oss's discovered top adds `cot_formatter` and `majority_voter` on top of the bare shape — net-zero cost vs plain BARE (majority_voter with `n_samples=1` is a no-op; `cot_formatter`'s extra output tokens here were trivial).
- gemma4's discovered top *is* the bare shape (null/null/simple/null/null) — the search found it directly by evicting retrieval/fewshot.

Attribution mean_delta = 0.000 on every component kind (and variance=0) — exactly zero, same signal as on the easy `symptom_classification` task. The search's drop-one ablations correctly report "nothing here contributes to accuracy because nothing can."

## Claim: MH++ earns the Pareto win, but not the cleverness win

Reading the result carefully:

- **MH++ dominates RAG because RAG wastes cost on this task.** That's a legitimate use of the search — you generally don't know in advance whether RAG helps, and running MH++ tells you. For a production decision, "MH++ said don't bother retrieving here" is actionable.
- **MH++ does NOT find a non-obvious harness shape RAG couldn't reach.** The discovered top is (roughly) the bare harness. No CoT-lifted-past-ceiling story like on `symptom_hard`.

The contrast with `symptom_hard` is the real data point:

| Task | Base ceiling | RAG vs BARE | MH++ top | Mechanism |
|---|---|---|---|---|
| `symptom_hard` | ~0.80 | RAG lifts +0.13pt | +0.04–0.07pt above RAG via cot_formatter + majority_voter + bigger retrieval | **MH++ extends the curve** |
| `news` (this) | 1.00 | RAG adds cost, not accuracy | Bare shape, cheaper than RAG | **MH++ strips overhead** |

Both are wins in the narrow Pareto sense. The first is a richer research story (novel shape discovery); the second is a cost-recovery story.

## So what is this benchmark worth?

As a cross-domain replication: **useful but qualified.**
- ✅ MH++ dominated RAG on both gpt-oss and gemma4, on a different-domain task than medical.
- ⚠️ Dominance was cost-only. The task didn't let MH++ show off accuracy extension.
- ⚠️ 20 eval items is too small to trust a 1.00 ceiling — maybe one or two items away from RAG showing accuracy wins on a bigger eval.

For a genuine "MH++ wins via shape discovery" cross-domain claim, we need a news variant where base LLM accuracy is < 1.00. The natural next step is `news_hard`: ambiguous or mixed-domain headlines designed to suppress the base model's ceiling.

## Artifact layout

```
runs/rag_vs_mh_news/
├── comparison.json
├── gpt-oss_20b/
│   ├── bakeoff_summary.json
│   ├── frontier.json
│   ├── attribution_stats.json
│   ├── history.jsonl
│   └── candidates/
└── gemma4_26b/
    └── (same layout)
```

## What I'd change for the next iteration

1. **`news_hard` variant** — headlines that fail clear keyword classification. E.g. "Quantum computing company files IPO with record valuation" (tech? business?), "Tennis star invests in pickleball league backed by Saudi fund" (sports? business? world?).
2. **Longer eval** — 50+ items. At 20 items, one misclassification is 5pt of accuracy, so "0.95 vs 1.00" can be a single example.
3. **Accuracy-weighted Pareto gating** — right now the frontier admits any non-dominated point; for benchmarks where 1.00 is attainable, we might want a rule "don't admit ties on accuracy if existing point is strictly cheaper." The gemma4 frontier only has 1 discovered harness because the Pareto eviction logic already collapses ties — but a minor reshuffle of scores could leave multiple tied points. Worth testing.
