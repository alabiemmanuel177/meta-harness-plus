# Strict Pareto Dominance over RAG — Achieved

**Date:** 2026-04-25
**Branch:** `main` (post all framework-upgrade merges)
**Task suite:** news_hard_50 + symptom_hard
**Search budget:** 6 iterations × 8 proposals (4× larger than the prior cross-provider runs)
**Repeats:** eval_repeats=2, attribution_repeats=2, attribution_screen_size=15
**Seeds:** 5 per (provider × task) cell
**Statistical method:** paired bootstrap CI + paired t-test + Cohen's d

## Summary

Across 10 seeds × 2 tasks, **MH++-discovered harnesses strictly Pareto-dominated the seeded RAG baseline on 7 of 10 Gemini runs.** This is the first time in the entire project's history that any discovered harness has been *strictly better* than RAG on every axis simultaneously — better-or-equal on accuracy AND strictly fewer tokens AND strictly lower-or-equal latency.

The OpenAI half of the experiment has now completed (post-bug-fix). It shows even larger absolute accuracy gains than Gemini, though structurally fewer "strict Pareto dominance" wins because MH++ on OpenAI uses more tokens than RAG (RAG is very cheap on these tasks for gpt-4.1-nano). See bottom of file for OpenAI section.

## Gemini × news_hard_50 (6×8 budget, 5 seeds)

| Per-seed | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| BARE | 0.84 | 0.84 | 0.84 | 0.84 | 0.84 |
| RAG | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 |
| **MH++** | **0.92** | **0.92** | 0.90 | 0.90 | 0.90 |
| RAG tokens | 174 | 174 | 174 | 174 | 174 |
| MH++ tokens | 420 | 458 | 250 | **172** | 231 |
| **Strictly dominates RAG?** | **YES** (1 harness) | no | no | **YES** (1 harness) | no |

**Aggregate:**
- RAG acc 95% CI: 0.880 [0.880, 0.880]
- MH++ acc 95% CI: 0.908 [0.900, 0.916]
- **Δ (MH++ − RAG) 95% CI: +0.028 [+0.020, +0.036] — significant**
- Paired t-test: t=+5.72, p=0.0046, Cohen's d=+2.56 (large effect)

**2 of 5 seeds achieved strict Pareto dominance over RAG.**

## Gemini × symptom_hard (6×8 budget, 5 seeds) — every seed dominates

| Per-seed | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| BARE | 0.667 | 0.667 | 0.667 | 0.667 | 0.667 |
| RAG | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| MH++ | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| RAG tokens | 184 | 184 | 184 | 184 | 184 |
| **MH++ tokens** | **124** | **147** | 332 | **124** | **124** |
| **Strict dominance count** | **1** | **2** | **1** | **1** | **1** |

**Aggregate:** Δ = 0 ± 0 (both methods saturate at 1.0 accuracy).

But per-seed strict-dominance count = 1+2+1+1+1 = **6 strict-dominant discovered harnesses across 5 seeds**. Both methods perfect on accuracy; MH++ strictly cheaper on tokens 4 of 5 times.

## Why this happened

The previous 5-seed cross-provider run used a **3 iter × 4 proposal** search budget — at most 12 candidates evaluated per seed. The Pareto frontier never had time to find shapes both more-accurate-than-RAG AND cheaper-than-RAG. Every accuracy lift came at +tokens.

This run quadrupled the budget to **6 iter × 8 proposals** = up to 48 candidates per seed. The search finally had enough trajectories to find, by chance and then by attribution, harness shapes that:
- Match RAG's accuracy on the saturated symptom_hard task while using less retrieval (smaller k or different retriever picks)
- Beat RAG's accuracy on news_hard_50 with fewer or differently-arranged components

The discovered shapes vary across seeds, but the dominant pattern: **stripped-down retrieval + cheaper-formatter combinations** that the smaller-budget search never visited.

## Combined Gemini scorecard

**7 of 10 Gemini seeds (across both tasks) achieved strict Pareto dominance over hand-tuned RAG.** Two on news_hard_50, five on symptom_hard.

This is the qualitative jump the project was chasing: not just "MH++ extends RAG's accuracy curve at higher cost" but **"MH++ finds harness shapes that beat RAG on every axis simultaneously."**

## What this means for the paper claim

Previous best claim (from the cross-provider 5-seed run earlier today):

> *MH++-discovered harnesses extend RAG's accuracy by +4.0±1.2pt (95% CI) on news_hard_50, replicated across two providers.*

Updated claim (preliminary, Gemini-only):

> *On Gemini 2.5-flash-lite at 6×8 search budget, MH++-discovered harnesses **strictly Pareto-dominate** hand-tuned RAG on 70% of seeds across two adversarial classification tasks — same-or-higher accuracy at strictly fewer tokens and not-worse latency, on every axis simultaneously.*

That's a meaningfully stronger claim. The OpenAI half of the experiment is in flight; if it replicates, the cross-provider strict-dominance claim becomes paper-grade.

## What's next

1. **OpenAI 5-seed × 2-task** (running now) — replicates the strict-dominance claim across providers if results hold.
2. **Re-run with framework upgrades that landed mid-experiment**: the Gemini run used the framework state at 23:02:56, which predates today's CompressedCoTFormatter, token-budget hint, per-class accuracy in proposer prompt, and EWMA attribution. Re-running with the merged main framework should produce *more* strict-dominance seeds (token-budget hint in particular biases toward cheaper proposals).
3. **Public dataset replication** — a benchmark not curated by the same person who designed the baselines. Tier 1.1 from ROADMAP.

## OpenAI × news_hard_50 (6×8 budget, 5 seeds) — post-bugfix

| Per-seed | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| BARE | 0.72 | 0.72 | 0.72 | 0.72 | 0.72 |
| RAG | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 |
| **MH++** | **0.94** | **0.92** | **0.92** | **0.92** | **0.92** |
| RAG tokens | 182 | 182 | 182 | 182 | 182 |
| MH++ tokens | 183 | 235 | 235 | 185 | 179 |
| match_cheaper | 2 | 0 | 0 | 1 | 1 |

**Aggregate:**
- Δ (MH++ − RAG) 95% CI: +0.044 [+0.040, +0.052]
- Paired t=+11.0, p=0.0004, Cohen's d=+4.92 (very large)
- 0/5 strict-dominance (latency on discovered tops marginally above RAG, breaking the strict ≥-on-every-axis test)
- 4/5 match-cheaper (tied accuracy at fewer tokens)

This is the largest news_hard_50 effect size of the project.

## OpenAI × symptom_hard (6×8 budget, 5 seeds) — post-bugfix

| Per-seed | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| BARE | 0.667 | 0.667 | 0.667 | 0.667 | 0.667 |
| RAG | 0.667 | 0.667 | 0.667 | 0.667 | 0.667 |
| **MH++** | **1.000** | **0.933** | **0.933** | **0.933** | **1.000** |
| RAG tokens | 200 | 200 | 200 | 200 | 200 |
| MH++ tokens | 670 | 446 | 422 | 346 | 388 |

**Aggregate:**
- Δ (MH++ − RAG) 95% CI: **+0.293 [+0.266, +0.320]** ← largest single Δ in entire project
- Paired t=+17.8, p=5.8×10⁻⁵, Cohen's d=+7.98 (huge)
- 0/5 strict-dominance, 0/5 match-cheaper — MH++ uses 1.7×–3.4× more tokens than RAG. **The accuracy gain comes at a real cost cost.**

gpt-4.1-nano on symptom_hard's hand-tuned RAG sits at 0.667 (saturates with the RAG-shape on Gemini). MH++ search finds harness shapes that nearly perfect the task — at 2-3× token cost. The accuracy lift is overwhelming, but this is the cell where MH++ trades cost for accuracy rather than dominating on both.

## Combined cross-provider scorecard

**Gemini:** 7 of 10 seeds achieve strict Pareto dominance over RAG.
**OpenAI:** 0 of 10 seeds achieve strict dominance, but every cell has CI-excluding-zero accuracy lift (avg +0.044 on news, +0.293 on symptom).

Two interpretations:
1. **Cross-provider replication of the *direction* of effect** — both providers show CI-excluding-zero accuracy gains at the same search budget.
2. **Provider-dependent Pareto strategy** — Gemini's discovered tops are token-efficient (matching or beating RAG on cost); OpenAI's discovered tops are accuracy-pushy (significant +acc, but at +tok). The search adapts to the model's strengths and weaknesses on these tasks differently.

## Artifacts

```
runs/gemini_news_hard_50_seed{0..4}_big/      # 5 per-seed Gemini news_hard_50 (6×8)
runs/gemini_symptom_hard_seed{0..4}_big/      # 5 per-seed Gemini symptom_hard (6×8)
runs/openai_news_hard_50_seed{0..4}_big/      # 5 per-seed OpenAI news_hard_50 (6×8)
runs/openai_symptom_hard_seed{0..4}_big/      # 5 per-seed OpenAI symptom_hard (6×8)
runs/gemini_news_hard_50_big_aggregate.json
runs/gemini_symptom_hard_big_aggregate.json
runs/openai_news_hard_50_big_aggregate.json
runs/openai_symptom_hard_big_aggregate.json
```

`runs/cache/{gemini,openai}_*_big.jsonl` — the prompt caches that made this <$1 experiment.
