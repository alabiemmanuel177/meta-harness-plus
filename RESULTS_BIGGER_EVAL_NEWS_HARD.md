# Bigger-Eval news_hard_50 Bakeoff

**Run:** 2026-04-25 (gpt-oss completed pre-power-outage; gemma4 re-run after recovery)
**Branch:** `bigger-eval-news-hard` (now merged through main + ROADMAP-driven follow-on branches)
**Task:** `news_hard_50` — 50 adversarial news-headline eval items (vs 15 on `news_hard`); same 56 train items; 13/12/13/12 class balance
**Search config:** 3 iter × 4 proposals, eval_repeats=2, attribution_repeats=2, attribution_screen_size=15
**Models:** `gpt-oss:20b` (55 min wall) + `gemma4:26b` (96 min wall, cold-cache after reboot)

## Headline numbers

### gpt-oss:20b (55 min)

```
BARE  (seed)  acc=0.940  tok=255   lat=1388ms   ← HIGHEST acc on frontier
RAG   (seed)  acc=0.920  tok=362   lat=1350ms
MH++  top     acc=0.930  tok=512   lat=1370ms
MH++  cheap   acc=0.890  tok=251   lat=1337ms
```

**4-point Pareto frontier; no harness strictly dominates RAG.** RAG underperforms BARE on accuracy (-0.020pt) at +107 more tokens — RAG overhead without payoff on this eval mix. Attribution: retriever −0.017, fewshot −0.017, formatter −0.004, reranker +0.000, voter +0.008.

### gemma4:26b (96 min)

```
BARE  (seed)  acc=0.82  tok=485   lat=5290ms
RAG   (seed)  acc=0.88  tok=571   lat=4876ms   ← lifts +0.06pt vs BARE
MH++  top     acc=0.90  tok=607   lat=4701ms   ← lifts +0.02pt vs RAG
MH++  alt     acc=0.80  tok=498   lat=5219ms
```

Discovered top: `bow(k=5) + null_reranker + topk(k=3) + simple_formatter + null_voter`. Attribution: retriever +0.033, fewshot +0.033, formatter +0.017, voter +0.000, reranker +0.000.

**Verdict on both models:** RAG holds (no strict Pareto dominance). MH++ extends accuracy curve on gemma4 by +0.02pt; on gpt-oss BARE actually beats RAG, so the search has nothing to extend.

## The methodologically-important comparison: 15-item vs 50-item news_hard

| | gpt-oss:20b (15-item) | gpt-oss:20b (**50-item**) | gemma4:26b (15-item) | gemma4:26b (**50-item**) |
|---|---|---|---|---|
| BARE | 0.87 | **0.94** | 0.87 | **0.82** |
| RAG | 0.93 | **0.92** | 0.93 | **0.88** |
| MH++ peak | 0.93 | **0.93** | 1.00 | **0.90** |
| Δ MH++ vs RAG | +0.00pt | **+0.01pt** | +0.07pt | **+0.02pt** |
| RAG dominated by BARE? | No | **Yes** | No | No |

**Two distinct findings emerge from the bigger eval:**

### Finding 1 (gpt-oss): the 35 added items are *easier* than the original 15

gpt-oss's BARE went from 0.87 → 0.94, meaning it got 47/50 right total (47 = 13 + 34 of the new 35). On the easier-on-average expanded set, RAG's retrieval overhead doesn't pay off — BARE strictly dominates RAG. This is **selection bias** in the original 15-item adversarial curation: a smaller set let me concentrate hard items, but harder ≠ representative.

### Finding 2 (gemma4): same eval, different outcome — base model matters

gemma4's BARE moved from 0.87 (15-item) to **0.82** (50-item). Translation: gemma4 found the original 15 items *easier* than the 35 new ones. Different model, different sample-difficulty perception. With BARE at 0.82, RAG and MH++ both retain non-trivial headroom (+0.06 / +0.08pt respectively).

**Cross-model on news_hard_50:** gpt-oss is genuinely stronger at news classification — it aces 47/50 even with no harness scaffolding. Gemma4 needs structure (retrieval, few-shot) to cross 0.85.

## Honest accounting against ROADMAP's research bar

This is the run that most clearly demonstrates **why selection-bias-controlled benchmarks matter** — exactly the point Tier 1.2 (multi-seed + held-out + bootstrap CI) was added for:

- The +0.13pt MH++ "win" on the 15-item news_hard (richer-components branch results) was **partially a small-sample artifact**.
- On 50 items: +0.02pt on gemma4, **−0.01pt on gpt-oss** (where MH++ doesn't reach BARE).
- 15-item-eval × 2-repeat moves accuracy in 3.3pt steps, so any "win" of < 1 example is statistical noise.
- The 50-item-eval × 2-repeat moves accuracy in 1.0pt steps, exposing the true gap.

This is the kind of finding a reviewer would ask for and we now have data on. It also justifies (loudly) the Tier 1.2 multi-seed + held-out infrastructure that's now in the framework — which is exactly what would have caught this earlier.

## Cost-efficiency cross-model

At matched model accuracy:

| | gpt-oss MH++ top (acc=0.93) | gemma4 MH++ top (acc=0.90) |
|---|---|---|
| Tokens | 512 | 607 |
| Latency/item | 1370ms | 4701ms |

gpt-oss is cheaper *per-eval* but its peak-discoverable accuracy is below gemma4's at this task. There's no free lunch — you pay for accuracy with either model, just in different ways.

## What this run does NOT settle (and why)

- **Strict Pareto dominance** — still doesn't happen on either model. MH++ extends the accuracy curve, doesn't push RAG off the frontier on tokens.
- **Statistical significance** — 1 seed per cell. The Tier 1.2 `MultiSeedRunner` infrastructure shipped this session would let us run 5 seeds for proper bootstrap CIs, but each seed is ~hour-of-LLM-time and we ran 1 to fit the session budget.
- **Cross-domain replication** — symptom_hard hasn't been re-run on the bigger 50-item-equivalent. Would replicate-or-falsify the medical-domain finding.

## Path to the strongest defensible claim

With the infrastructure now in place (post-session shipping):

1. **Re-run news_hard_50 with `MultiSeedRunner.run([0,1,2,3,4])`** — 5 seeds × 1.5 hours each = ~7.5 hours per model = 15 hours total. Output: bootstrap 95% CI on `MH++ vs RAG` accuracy difference. If CI excludes zero, defensible "MH++ > RAG" claim on this benchmark.
2. **Add `--proposer-mode ensemble` to one of those runs** — Tier 5.2 single-vs-ensemble head-to-head at matched compute. The CLI flag was wired this session.
3. **Run `cot_baseline` + `voting_rag_baseline` + `diverse_rag_baseline`** as additional seed harnesses on the same task, so the comparison isn't just MH++ vs the simplest RAG.

That sequence — multi-seed + ensemble + richer baselines — would produce the first publishable-quality result from this codebase.

## Run artifacts

```
runs/bigger_eval_news_hard/
├── gpt-oss_20b/  (committed earlier on bigger-eval-news-hard branch)
└── gemma4_26b/   (committed in the gemma4-rerun commit)
```

Per-candidate harness specs, scores, attribution snapshots — grep-able + reusable as input to a future LLM-proposer run pointed at the same path.
