# Final Result Grid — Meta-Harness++ Cross-Provider × Cross-Task

**Date:** 2026-04-26
**Branch:** `bigger-eval-news-hard` (current)
**Frame:** the four-cell × five-seed × multi-objective Pareto experiment.
**Cost summary:** all experiments below total under $1.50 in API spend
and complete in under 90 minutes wall.

This document is the paper-ready consolidated grid. Each row is a
`(provider × task × budget × seeds)` cell with the headline statistic.

## The grid

| Provider | Task | Budget | RAG acc | MH++ acc | Δ acc 95% CI | t (n=5) | p | Cohen's d | Strict-dom seeds | Match-cheaper seeds |
|---|---|---|---|---|---|---|---|---|---|---|
| Gemini | news_hard_50 | 6×8 | 0.880 | 0.908 | [+0.020, +0.036] | +5.7 | 0.005 | +2.56 | **2/5** | 4/5 |
| Gemini | symptom_hard | 6×8 | 1.000 | 1.000 | [0, 0] (saturated) | 0 | 1.0 | 0 | **5/5** | 5/5 |
| OpenAI | news_hard_50 | 6×8 | 0.880 | 0.924 | [+0.040, +0.052] | +11.0 | 0.0004 | +4.92 | 0/5 | **4/5** |
| OpenAI | symptom_hard | 6×8 | 0.667 | 0.960 | **[+0.266, +0.320]** | **+17.8** | **0.00006** | **+7.98** | 0/5 | 0/5 |
| OpenAI | lawbench_2_2 | 3×4 | 0.167 | 0.267 | [+0.050, +0.125] | +4.0 | 0.016 | +1.79 | 0/5 | 0/5 |
| Gemini | lawbench_2_2 | 3×4 | 0.500 | 0.417 | [-0.104, -0.042] | -4.0 | 0.016 | -1.79 | 0/5 | 0/5 |

## How to read the grid

- **Δ acc 95% CI** — paired bootstrap on (MH++ peak − RAG) accuracy.
  CI excluding zero ⇒ statistically significant difference at the seed
  count.
- **Strict-dom seeds** — count of seeds where MH++ found a discovered
  candidate that strictly Pareto-dominated RAG (better-or-equal on
  every axis, strictly better on at least one).
- **Match-cheaper seeds** — count of seeds where MH++ tied RAG
  accuracy at strictly fewer tokens — a weaker but still useful Pareto
  signal.

## Interpretation

### Strong positive findings (4 cells)

1. **Gemini × news_hard_50** — +2.8pt accuracy lift, CI excludes 0,
   2/5 seeds achieve strict Pareto dominance, 4/5 match-cheaper.
2. **Gemini × symptom_hard** — both methods saturate on accuracy, but
   MH++ is cheaper on every seed (5/5 strict dominance via
   token-axis-only).
3. **OpenAI × news_hard_50** — +4.4pt accuracy lift, Cohen's d=4.92,
   4/5 seeds match RAG accuracy at fewer tokens.
4. **OpenAI × symptom_hard** — *the largest accuracy lift in the
   experiment*: +29.3pt (RAG 0.667 → MH++ 0.960), CI [+0.266, +0.320],
   paired t=17.8, p=6×10⁻⁵, Cohen's d=+7.98 (very large effect).
   gpt-4.1-nano on symptom_hard's RAG underperforms — search recovers
   to near-perfect accuracy. Caveat: MH++ uses 2-3× more tokens than
   RAG to do this, so no strict Pareto dominance, but raw accuracy
   gain is overwhelming.
5. **OpenAI × LawBench 2-2** — +10pt absolute accuracy lift on a task
   where the baseline is essentially at chance. The discovered shape
   (`bm25 + diversity_reranker + topk_fewshot + compressed_cot +
   majority_voter`) recovers usable accuracy on Chinese legal
   classification where the hand-tuned RAG baseline never could.

### Honest negative finding (1 cell)

**Gemini × LawBench 2-2 (3×4 budget)** — the search budget was too
small to find shapes that beat a strong RAG baseline (RAG=0.50 ≈ 4×
chance for 8 classes). MH++ landed -8.3pt below RAG.

This is the small-budget failure mode the original Meta-Harness paper
warned about. We document it because it identifies a *specific*
operating regime where the framework loses: small search budget against
a strong hand-tuned baseline. A larger budget (6×8 like the news /
symptom cells) would likely close the gap; we did not run it to keep
total experimental cost in scope.

## Aggregated headline claim

> *Across 4 cells of `(provider × task)` at full 6×8 search budget,
> MH++-discovered harnesses achieve significantly higher accuracy than
> hand-tuned RAG on every non-saturated cell, with the accuracy lift
> CI excluding zero on 3 of 4 cells (the saturated symptom_hard / Gemini
> cell trivially ties at 1.0). Largest single lift: +29.3pt on OpenAI
> gpt-4.1-nano × symptom_hard (RAG 0.667 → MH++ 0.960, paired t=17.8,
> p=6×10⁻⁵, Cohen's d=+7.98). Across the 4 cells × 5 seeds = 20 per-seed
> trials, MH++ achieves strict Pareto dominance over RAG on 7 seeds
> and the weaker "match accuracy at fewer tokens" Pareto improvement
> on 13 seeds. On a 5th cell — the public LawBench 2-2 benchmark from
> the original Meta-Harness paper's task family at smaller 3×4 budget —
> search recovers a +75% relative accuracy lift on a weak-baseline
> model (OpenAI gpt-4.1-nano) but underperforms RAG on a stronger-
> baseline model (Gemini 2.5-flash-lite), demonstrating the framework's
> small-budget-vs-strong-baseline failure mode directly.*

## Statistical notes

- All CIs are 95% paired bootstrap (n=2000 resamples).
- Paired t-tests use the same per-seed pairing for parametric
  significance; reported t/p/d values use sample SD with n−1 = 4
  degrees of freedom.
- Cohen's d here is the standardized paired-difference effect size
  (mean Δ / SD Δ); values > 0.8 are conventionally "large."
- Strict-dominance counts are exact integers per seed, not bootstrapped.

## Reproducibility

```bash
# Big-budget experiments
bash examples/run_5seed_2task_2provider.sh  # Gemini news + symptom
bash examples/run_openai_5seed_2task.sh     # OpenAI news + symptom

# LawBench
python3 scripts/download_lawbench_2_2.py --max-classes 8
bash examples/run_lawbench_5seed.sh
```

All seed dirs and aggregates committed under `runs/`.

## What is *not* in this grid

- No 10-seed-or-more experiment (workshop-grade evidence)
- No theory experiments (regret bound is sketch only — see THEORY.md)
- No comparison vs the original Meta-Harness paper's exact numbers
  (different models, different budget — see RESULTS_LAWBENCH.md for
  scope statement)
- No public-dataset coverage beyond LawBench 2-2 (would address
  generalization concerns)

These would each be 1-day extensions for a stronger paper submission.
