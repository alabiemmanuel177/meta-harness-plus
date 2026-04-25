# LawBench 2-2 Cross-Provider Replication

**Date:** 2026-04-26
**Branch:** `lawbench-replication` (now merged)
**Task:** `lawbench_2_2` — Chinese legal dispute-focus classification
sourced directly from `open-compass/LawBench`'s public GitHub
(zero_shot/2-2.json), 8 most-common classes (top-N selection from 16
total) balanced 80 train / 48 eval items.
**Significance:** This is a public benchmark from **the original
Meta-Harness paper's task family** — running our framework on it
addresses the "Beat MH on its bed" cell of the four-paper-gaps matrix.
**Search config:** 3 iter × 4 proposals, eval_repeats=2, attribution_
repeats=2, attribution_screen_size=12 (matches the cross-provider
news_hard_50 protocol that produced the +0.04pt CI-excluding-zero
result).
**Seeds:** 5 per (provider × task) cell

## Headline numbers

| Provider | BARE acc | RAG acc | MH++ peak (5-seed mean) | Δ 95% CI | Strict-dominant seeds | Cohen's d |
|---|---|---|---|---|---|---|
| OpenAI gpt-4.1-nano | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemini 2.5-flash-lite | TBD | TBD | TBD | TBD | TBD | TBD |

**[Filled when 5-seed bakeoff completes]**

## Smoke-test pre-context

Before launching the full multi-seed, a 1-seed × 2 iter × 3 proposal
budget run on Gemini gave:

- BARE: 0.48
- RAG: 0.50
- MH++ discovered top: 0.42 (worse — small budget couldn't find good shapes)

This showed the pipeline runs end-to-end on Chinese legal text, but
the search needs more iterations to outperform RAG. The full 5-seed
× 3×4-budget run targets that.

## Why LawBench 2-2 specifically

The original Meta-Harness paper (Lee et al. 2026, arXiv 2603.28052)
reports +7.7pt accuracy with 4× fewer tokens on label-intensive
classification — LawBench is the family. Subtask 2-2 is dispute-focus
classification of Chinese legal-case text. We capped to the top-8
most-common classes for tractable search budget; the original paper
ran on the full label set.

Direct comparability with the original paper is partial:
- ✅ Same task family (LawBench 2-2)
- ✅ Same evaluation protocol (held-out test items)
- ⚠️ Different subset: top-8 classes vs full 16 (we picked the most
  common to keep eval balanced; full LawBench has heavy class
  imbalance with some classes at 2-4 items)
- ⚠️ Different LLMs (we tested gpt-4.1-nano + Gemini Flash Lite;
  paper used Claude Opus 4.6 + Haiku 4.5)
- ⚠️ Different budget (we ran 3 iter × 4 props for cost; paper ran
  longer search)

## Honest scope of the claim

If MH++ shows **CI-excluding-zero accuracy gain over RAG** on at least
one provider with multi-seed evidence, the claim becomes:

> *MH++'s multi-objective search + budget-aware halving + attribution-
> guided proposer extends the canonical RAG harness shape on LawBench
> 2-2's Chinese legal-classification task, replicating the original
> Meta-Harness framework's improvement on its own task family using
> a fraction of the original paper's compute budget.*

If the result is null or negative on this small budget:
> *On a smaller compute budget than the original Meta-Harness paper,
> MH++ matches but does not statistically distinguish from hand-tuned
> RAG on LawBench 2-2. We expect (and the original paper confirms)
> that larger search budgets close the gap, but document this honest
> small-budget result for reproducibility.*

Either way the experiment produces evidence that goes into the paper.

## Comparison vs prior runs

| Task | Difficulty (BARE acc) | Where MH++ has shown gains |
|---|---|---|
| toy_classification (5 keyword classes) | 0.45 baseline | small (ablation tested) |
| symptom_hard (5 medical specialties) | 0.66-0.87 | extends RAG, sometimes strict dominance |
| news_hard_50 (4 news classes, English) | 0.72-0.94 | +4pt CI excludes zero, 7/10 strict dominance @ 6×8 |
| **lawbench_2_2** (8 Chinese legal classes) | **TBD (smoke 0.48)** | **TBD** |

## Cost & wall-time

| | Smoke (1 seed × 2×3) | Full (5 seeds × 3×4) |
|---|---|---|
| API calls (cache-aware) | ~100-200 misses × 700 tokens | ~600-800 misses × 700 tokens |
| Wall time | 95 sec on Gemini | ~30-50 min for both providers in parallel |
| Cost | < $0.01 | ~$0.50-1.00 |

## Artifact layout

```
runs/openai_lawbench_2_2_seed{0..4}/    # 5 OpenAI per-seed runs
runs/gemini_lawbench_2_2_seed{0..4}/    # 5 Gemini per-seed runs
runs/lawbench_2_2_openai_aggregate.json
runs/lawbench_2_2_gemini_aggregate.json
runs/cache/{openai,gemini}_lawbench_2_2.jsonl  # persistent prompt cache
meta_harness_plus/tasks/data/lawbench/lawbench_2-2_*.jsonl  # bundled data
```

## Replication command

```bash
# One-time download (already committed under tasks/data/lawbench/):
python3 scripts/download_lawbench_2_2.py --max-classes 8

# Full 5-seed cross-provider run:
bash examples/run_lawbench_5seed.sh
```
