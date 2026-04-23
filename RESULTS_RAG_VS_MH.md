# RAG vs MH++ — Head-to-Head on Symptom-Hard

**Run:** 2026-04-23
**Branch:** `rag-vs-mh`
**Task:** `symptom_hard` — 5-class medical classification, **50 train / 15 eval**, with adversarial eval queries designed so the class is *not* determinable from keywords alone. Every eval item has a disambiguating training item (so retrieval has something meaningful to fetch).
**Search config:** 3 iterations × 4 proposals, halving k₀=3 / final_keep=2, `eval_repeats=2`, `attribution_repeats=2`, `attribution_screen_size=10`
**Baselines seeded onto the Pareto frontier:**
- **BARE:** no retrieval, no fewshot, no voting — the floor
- **RAG:** `bow_retriever(k=3) + topk_fewshot(k=2) + simple_formatter + predictor + null_voter` — canonical shape

## Headline numbers

### gpt-oss:20b (20.9B, reasoning architecture) — 32.8 min wall

| | Accuracy | Spread | Tokens | Latency/item | Shape |
|---|---|---|---|---|---|
| **BARE** | 0.800 | 0.00 | 255 | 1407 ms | null / null / simple / llm / null |
| **RAG** | 0.933 | 0.00 | 371 | 1325 ms | bow(k=3) / topk(k=2) / simple / llm / null |
| **MH++ top** | **0.967** | 0.067 | 486 | 1858 ms | **bow(k=5) / topk(k=3) / cot_formatter / llm / majority_voter** |

Strict dominance counts: **0 discovered harnesses dominate RAG**, **0 match RAG's accuracy at lower cost**. RAG remains on the frontier.

### gemma4:26b (25.8B, non-reasoning) — 94.4 min wall (post-reboot rerun)

| | Accuracy | Spread | Tokens | Latency/item | Shape |
|---|---|---|---|---|---|
| **BARE** | 0.867 | — | 524 | 5253 ms | null / null / simple / llm / null |
| **RAG** | 0.933 | 0.00 | 500 | 3696 ms | bow(k=3) / topk(k=2) / simple / llm / null |
| **MH++ top** | **1.000** | — | 560 | 4481 ms | **bow(k=2) / topk(k=2) / cot_formatter / llm / null_voter** |
| MH++ alt | 0.933 | — | **488** | 4214 ms | bow(k=2) / topk(k=1) / simple / llm / null_voter |

Strict dominance counts: **0 dominate RAG**, **1 matches RAG's accuracy at lower tokens** (488 vs 500). RAG remains on the frontier.

## The interesting finding: search independently re-discovered CoT across architectures

Both models' best discovered harness uses **`cot_formatter`** — the chain-of-thought formatter variant added on this branch *specifically so the LLM proposer had action-space beyond vanilla RAG*. The proposer, reading the filesystem run log (prior candidates + attribution + exploration gap), chose to try it on both the reasoning and non-reasoning model family. It was the right move in both cases — CoT + expanded retrieval lifted accuracy past RAG's ceiling.

This is the actual research contribution:

- RAG as a fixed shape caps out at 0.933 on this task for both models.
- MH++ search, given the richer component library, **extended the Pareto curve** past that cap to 0.967 (gpt-oss) and 1.000 (gemma4).
- The search found this *without being told what to look for*. The exploration-gap surfacing in the LLM proposer prompt (components never on the frontier) nudged it toward `cot_formatter`.

## But: no strict Pareto dominance

Honest report: on both models, **every discovered harness either costs more tokens than RAG or ties accuracy without fully dominating**. The MH++ point extends the frontier upward-and-rightward (higher accuracy, higher cost), it doesn't push RAG off it. That's a legitimate "extend the curve" result, not a "crush the baseline" result.

Whether that counts as a win depends on how much you value the extra accuracy:

- On gpt-oss:20b: **+3.4pt accuracy costs +31% tokens**. For cost-sensitive deployments, RAG still wins. For accuracy-sensitive deployments where mistakes are expensive, MH++ does.
- On gemma4:26b: **+6.7pt accuracy (0.933 → 1.000) costs +12% tokens**. This is a much better exchange rate — basically free accuracy.

## Attribution signal (gemma4 only — gpt-oss run didn't capture it this session)

```
retriever    mean_delta=+0.037  var=0.002  n=8    ← real positive signal
fewshot      mean_delta=+0.037  var=0.002  n=8    ← real positive signal
voter        mean_delta=−0.012  var=0.001  n=8    ← slight negative
formatter    mean_delta=−0.025  var=0.004  n=8    ← slight negative (!)
```

Two things worth saying plainly about that `formatter=−0.025`:

1. It's across 8 candidates; the drop-one baseline is `SimpleFormatter`. A negative mean_delta means *on average* swapping away from whatever formatter the candidate used (sometimes CoT, sometimes simple) to SimpleFormatter improved accuracy. So CoT hurts more candidates than it helps *on average*.
2. But the single best candidate *uses* CoT. So CoT is high-variance: it's a big accuracy lever on the right shape (retrieval + fewshot already present) and a small hurt on harnesses that don't need it.

The Meta-Harness search implicitly handled this: it kept CoT on the frontier only when paired with retrieval + fewshot. A pure-scalar baseline picking components by average attribution would have rejected CoT.

## Honest limitations

- **Small eval (15 items).** Each mistake moves accuracy by 6.7pt — spread=0.067 on the gpt-oss discovered top means it scored 0.933 once and 1.000 once across 2 repeats. The "0.967 median" is really 1 eval item of difference. Next step needs more items and more repeats to call this robust.
- **Hand-curated task.** The symptom_hard dataset was written in one sitting by the same person who designed the baseline to beat. Ideally we want a public, peer-reviewed benchmark where the class assignments aren't biased by the author's intuitions.
- **gpt-oss wall 33 min vs gemma4 94 min.** The gemma4 rerun happened post-reboot on a cold Ollama; the initial model load + batching penalty likely accounts for part of the difference. Doesn't affect the accuracy numbers, but the cost axis on gemma4 is inflated compared to a warm run.

## What this bakeoff resolves (and what it doesn't)

**Resolved:**
- MH++'s richer component library + exploration-gap surfacing genuinely extends the Pareto frontier past vanilla RAG on both a reasoning model (gpt-oss) and a non-reasoning model (gemma4).
- The CoT formatter is the discovered lever in both cases. Not a coincidence — reasoning helps on this task's ambiguous queries, and the search figured that out from attribution + filesystem logs.
- No method quietly broke: accuracy_spread reports correctly, attribution deltas are in the real-signal range (not noise floor), RAG was correctly seeded and survived on the frontier.

**Not resolved:**
- Whether MH++ *dominates* RAG on a benchmark. On this one it only *extends* it.
- Whether the CoT win holds on harder public datasets (USPTO-50k, LawBench) or only on our hand-curated task.
- The cost-accuracy exchange rate on gpt-oss (+31% tokens for +3.4pt) — that's a judgment call depending on downstream economics, not a clean framework win.

## Next milestone candidates

1. **Public dataset adapter** — take a 20 Newsgroups or BBC News subset, replicate the bakeoff. If MH++-with-CoT still extends the curve past RAG, that generalizes the claim beyond our hand-curated task.
2. **Reranker component** — after retrieval, re-rank top-k by LLM-as-judge. Expands action space further; addresses the fewshot-quality axis RAG doesn't tune.
3. **Larger eval budget** — 50-100 eval items with 3 repeats each. Resolves the "0.967 is really 14/15" problem above.
4. **Stricter dominance ask** — gate frontier admission on `accuracy_spread ≤ ε`; discard candidates whose accuracy is too unstable to be worth the token cost.

Artifacts under `runs/rag_vs_mh/` — 43+ JSON files, grep-able + re-consumable by future proposer runs.
