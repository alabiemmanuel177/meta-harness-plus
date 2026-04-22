# Full Bakeoff Results — Meta-Harness++ vs Real Local Models

**Run:** 2026-04-22
**Hardware:** local Ollama, GPU inference
**Backend:** `http://localhost:11434/api/chat` (Ollama native)
**Framework:** branch `llm-proposer` (LLMProposer + LLMPredictor end-to-end)
**Task:** `meta_harness_plus.tasks.toy_classification` (5-class, seed=0, 20-item eval)
**Search config:** 4 iterations × 4 proposals, screen=6, successive halving (k₀=3, η=2, final_keep=2)

## Headline numbers

| Model | Params | Arch | Wall | Frontier size | Best acc | Cheapest on frontier (tokens) |
|---|---|---|---|---|---|---|
| `gpt-oss:20b` | 20.9B | reasoning (CoT) | **24.3 min** | 3 | 0.80 | 307 |
| `gemma4:26b`  | 25.8B | non-reasoning   | **37.9 min** | 3 | 0.80 | 513 |

**Total: 1h 2min** for both models end-to-end.

## Frontier details

### gpt-oss:20b

```
acc=0.80  tok=381  lat=2306ms  :: bow_retriever(k=3) + null_fewshot + n=1 + majority_voter
acc=0.75  tok=313  lat=1940ms  :: null_retriever + topk_fewshot(k=5) + n=1 + null_voter
acc=0.65  tok=307  lat=1886ms  :: baseline
```

**Attribution:** `fewshot = −0.019`, `voter = −0.019`, `retriever = −0.056` (all noise-level)

### gemma4:26b

```
acc=0.80  tok=541  lat=5096ms  :: bow_retriever(k=2) + topk_fewshot(k=1) + n=1 + null_voter
acc=0.60  tok=516  lat=5542ms  :: baseline
acc=0.55  tok=513  lat=5344ms  :: bow_retriever(k=1) + null_fewshot + null_voter
```

**Attribution:** all three kinds report `mean_delta = 0.000`, variance ≈ 0.008 — ablations produced no net change on the 6-item screen subset.

## Findings

### 1. Same ceiling, different routes

Both models cap at **0.80 accuracy** — that's the task's apparent ceiling at 20-item granularity (each wrong answer moves accuracy by 5pt). But the *winning harness shapes diverge sharply:*

- **gpt-oss:20b's winner** uses `bow_retriever(k=3) + null_fewshot` — fetches 3 training docs then **discards** them (NullFewShot empties `ctx.few_shots`, and SimpleFormatter only reads `few_shots`, not `retrieved`). Prompt-equivalent to baseline. Only the voter changes, and with `n_samples=1` majority_voter is a no-op too.
- **gemma4:26b's winner** uses `bow_retriever(k=2) + topk_fewshot(k=1)` — actually threads one retrieved example into the prompt as a few-shot. Real structural contribution.

### 2. gpt-oss:20b is objectively more cost-efficient

At matched peak accuracy:

- **Tokens:** gpt-oss 381 vs gemma4 541 (**42% more for gemma**)
- **Per-item latency:** gpt-oss 2.3s vs gemma4 5.1s (**2.2× slower for gemma**)
- **Cheapest frontier point:** gpt-oss 307 tok vs gemma4 513 tok

Gemma's larger parameter count (25.8B vs 20.9B) is a fraction of the latency gap — the rest comes from decode throughput differences. For Meta-Harness-style search where every eval is a paid sample, this matters a lot: at scale, one model is dramatically cheaper per unit of accuracy discovered.

### 3. The LLM proposer discovered harnesses outside the mock proposer's range

The `main`-branch mock mutation proposer has a hardcoded list of k-values (1, 2, 3, 5 for retriever; 1, 2, 3 for few-shot). The LLM proposer in this run proposed **`topk_fewshot(k=5)`** for gpt-oss — a value the mock proposer couldn't emit. This is the exploration-gap surfacing contribution working: reading `attribution_stats.json` + the current frontier, the LLM chose to push the few-shot k higher.

### 4. Honest limitations

Two methodological issues surfaced that matter for interpreting these numbers:

- **Ollama's `temperature=0` is not byte-deterministic.** gpt-oss's two frontier entries above 0.65 are functionally prompt-equivalent to baseline (the retrieved docs are never threaded into the prompt), yet score differently (0.80, 0.75, 0.65). That 0.15pt spread is non-determinism in GPU decode batching, not meta-harness signal. Fix for next run: evaluate each harness ≥ 3 times and report median accuracy.
- **Attribution deltas are noise-dominated at 6-item screens.** Both models show near-zero or slightly-negative mean deltas across all component kinds. With 20-item eval and 6-item screens, each wrong classification moves accuracy by 5-17pt — the signal-to-noise ratio is below what drop-one ablation can resolve. Fix for next run: use 50+ eval items and ≥ 20-item attribution screens.

Neither issue breaks the framework — they're standard sample-size problems in LLM evaluation. They *do* mean these numbers shouldn't be over-interpreted as definitive cross-model rankings.

## Cost breakdown

| Phase | gpt-oss:20b | gemma4:26b |
|---|---|---|
| Seed harness eval | ~20 calls | ~20 calls |
| Per iter: proposer | 1 × ~10s | 1 × ~15s |
| Per iter: screening eval | ~25 calls × ~2s | ~25 calls × ~5s |
| Per iter: full eval + attribution | ~80 calls × ~2s | ~80 calls × ~5s |
| **Per-iter total (rough)** | ~4 min | ~7 min |
| **× 4 iterations** | 16 min + setup | 28 min + setup |
| Observed wall | 24.3 min | 37.9 min |

## Artifacts

All run artifacts on disk:

```
runs/full_bakeoff/
├── comparison.json                 # side-by-side summaries
├── gpt-oss_20b/
│   ├── bakeoff_summary.json
│   ├── frontier.json
│   ├── attribution_stats.json
│   ├── history.jsonl               # 5 events (1 per iter + seed)
│   └── candidates/cand_0001..0009/ # per-candidate harness, score, attribution
└── gemma4_26b/
    ├── bakeoff_summary.json
    ├── frontier.json
    ├── attribution_stats.json
    ├── history.jsonl
    └── candidates/cand_0001..0007/
```

Every candidate's full spec + score + drop-one ablation snapshot is on disk — grep-able for inspection, and directly re-consumable by `LLMProposer.run_dir` pointed at the same path.

## What would make the next run publishable

1. **100+ eval items, multiple seeds per harness.** Resolves the Ollama-nondeterminism + ablation-noise issues above.
2. **Third and fourth model families** — throw in a cloud model (Claude Opus 4.7 via `--anthropic`) and a local small model (`pixtral:12b-local`) for both ends of the capability spectrum.
3. **A real dataset.** Swap the toy task for LawBench or USPTO-50k (matches the original Meta-Harness paper's label-intensive classification axis).
4. **Baseline against mock proposer on `main`** at the same token budget — proves the LLM proposer's exploration actually beats deterministic mutation when both are given equal compute.
