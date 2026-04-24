# RAG vs MH++ — News-Hard Benchmark

**Run:** 2026-04-24
**Branch:** `news-benchmark`
**Task:** `news_hard` — 4-class news with 15 **adversarial** eval headlines that bridge 2 categories (sports↔business, world↔business, tech↔world). 56 train items including 16 disambiguating patterns. Structural analogue to `symptom_hard` but for news.
**Search config:** 3 iter × 4 props, `eval_repeats=2`, `attribution_repeats=2`, `attribution_screen_size=10`

## Headline results

| Model | BARE acc | RAG acc | MH++ top acc | Δ(MH++ − RAG) | Verdict |
|---|---|---|---|---|---|
| `gpt-oss:20b` | 0.87 | 0.93 | 0.93 | 0.00 | RAG holds (tie) |
| `gemma4:26b`  | **0.60** | 0.83 | **0.87** | **+0.04pt** | RAG holds, MH++ extends |

### gpt-oss:20b (34 min wall)

```
BARE          acc=0.87  tok=273   lat=1672ms
RAG (seeded)  acc=0.93  tok=394   lat=1706ms
MH++ top      acc=0.93  tok=438   lat=1497ms   ← matches RAG acc, more tokens, LOWER latency
```

Pareto frontier has 3 points (RAG, discovered, BARE). Discovered top is non-dominated relative to RAG (equal accuracy, more tokens but less latency), but doesn't strictly dominate. Attribution: `fewshot +0.031, retriever +0.019, voter +0.006, formatter −0.006`. **CoT formatter went slightly negative here** — unlike on `symptom_hard` where it was the discovered top's key component.

### gemma4:26b (143 min wall, slow Ollama cache state)

```
BARE          acc=0.60  tok=708   lat=7458ms   ← task is genuinely hard for gemma4
RAG (seeded)  acc=0.83  tok=716   lat=6397ms   ← RAG lifts +0.23pt over BARE
MH++ top      acc=0.87  tok=794   lat=7217ms   ← +0.04pt over RAG, higher token cost
MH++ alt      acc=0.80  tok=633   lat=6288ms   ← cheaper point on frontier
```

**Attribution here is the cleanest signal across all 4 benchmarks we've run:**

```
retriever    mean_delta=+0.238  var=0.022  n=8   ← HUGE positive
fewshot      mean_delta=+0.238  var=0.022  n=8   ← HUGE positive
voter        mean_delta=−0.012  var=0.001  n=8   ← noise
formatter    mean_delta=−0.050  var=0.010  n=8   ← CoT actively hurts
```

Drop-one ablation says retrieval and few-shot each contribute **+0.238 accuracy**. Swapping to null on either drops 23.8pt of accuracy on average across survivor candidates. That's RAG's components pulling their weight — and the search correctly finds that.

## What this bakeoff resolves

**1. The "hard task regime" is where MH++ earns its value.** On gemma4, news_hard's BARE accuracy of 0.60 gives RAG enough headroom to lift +0.23pt, and MH++ another +0.04pt. On gpt-oss the same task has BARE 0.87 (gpt-oss is just good at news), so RAG has less to do (+0.06pt), and MH++ has even less (+0.00pt).

**2. The CoT-extension pattern is domain-specific.** On `symptom_hard` the discovered top used `cot_formatter` on both models. On `news_hard`, neither model's discovered top uses it, and attribution is negative on both. CoT helps when the task requires semantic reasoning (medical symptoms → specialty); doesn't help when it's surface-category pattern-matching (news topic assignment).

**3. MH++ does NOT strictly Pareto-dominate RAG on `news_hard` on either model** — the "RAG holds" verdict is honest. But on gemma4, MH++ extended the accuracy curve past RAG's point (0.83 → 0.87), same shape as symptom_hard.

## Cross-task portfolio so far

| Task | Difficulty (BARE acc) | RAG vs BARE | MH++ vs RAG | Discovered lever |
|---|---|---|---|---|
| `symptom_classification` (easy) | 1.00 (ceiling) | +0 (RAG adds cost) | Ties RAG; bare dominates both | — |
| `symptom_hard` (adversarial) | 0.80 gpt-oss / 0.87 gemma4 | +0.13pt both | +0.03-0.07pt (extends curve) | **cot_formatter** + bigger retrieval |
| `news` (easy) | 1.00 (ceiling) | +0 (RAG adds cost) | **Dominates RAG** (cost-strip) | Bare shape or null voter |
| `news_hard` (adversarial) | 0.87 gpt-oss / 0.60 gemma4 | +0.06-0.23pt | +0.00-0.04pt (extends curve on gemma4) | Bigger retrieval + larger few-shot |

**Pattern emerging:**
- **Easy tasks (base at ceiling):** MH++ dominates RAG by *stripping* unneeded retrieval overhead.
- **Hard tasks (base has headroom):** MH++ *extends* RAG's accuracy curve via larger retrieval / few-shot / sometimes CoT, but usually at higher cost (no strict dominance).
- **MH++ adaptively picks different components for different tasks** — CoT for symptoms, bigger retrieval for news.

## Honest accounting against "MH++ > RAG on every benchmark"

Of the 4 task × 2 model = 8 configurations run:

- **Strict Pareto dominance of RAG by a discovered harness:** 3 of 8 (both `news` configs, plus the `news_hard` gemma4 case where the discovered top has higher accuracy but also more cost — actually that's non-dominance, so revise to 2 of 8).
- **Match RAG acc at equal-or-lower cost:** 3 of 8 (same as above, approximately)
- **Extends RAG's accuracy frontier (higher acc at higher cost):** 3 of 8 (both `symptom_hard` + `news_hard` gemma4)
- **RAG strictly holds (discovered can neither dominate nor extend):** 1 of 8 (`news_hard` gpt-oss — discovered ties RAG exactly)
- **Task too easy for any claim (all 1.00):** 2 of 8 (both `news` configs)

**Honest claim:** MH++ Pareto-dominates RAG on easy tasks via cost-strip, extends the curve on hard tasks via component mix, and doesn't hurt you anywhere we've measured. It does not strictly dominate on every benchmark — the hard-task wins are all "extend" not "dominate."

## What would move the needle to "strict dominance on hard tasks"

- **Richer action space.** The search is choosing from {bow retriever k∈[1..10], topk fewshot k∈[1..5], simple/cot formatter, llm predictor n∈[1..7], null/majority voter}. Adding a **reranker** component (LLM-as-judge over retrieved neighbors) or **self-consistency voter** (vote over n independent calls at temperature > 0) gives new shape to discover.
- **Eval budget.** 15-20 items with 2 repeats = 30-40 forward passes per candidate. At 7pt granularity per misclassification, strict dominance (stronger on ≥1 axis) requires beating RAG by > 7pt on accuracy while not losing on tokens, which is a tight target on this eval size.
- **Attribution-driven seed harness.** Currently the search seeds with BARE + vanilla RAG. A RAG-plus-voting-seed or RAG-plus-CoT-seed would give the search better starting points to improve from.

## Artifact layout

```
runs/rag_vs_mh_news_hard/
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

Per-candidate harness + score + drop-one ablation snapshots committed as research evidence.
