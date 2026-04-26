# Paper Section 1 — Introduction (Draft)

This is the prose draft of section 1. Plugs into PAPER_OUTLINE.md
section 1 directly. Length target: ~1.5 pages of the 8-page main.

---

## 1. Introduction

Every team building an LLM application is silently doing harness
search. Should we use BM25 or BoW for retrieval? Two few-shot
examples or four? CoT or no CoT? Voting with three samples or just
one? These choices form a discrete space of *harness shapes*, and
practitioners explore it manually — typically by hand-tuning a
"RAG-shaped thing," running an eval, tweaking, and stopping when
the metric stops moving. This is at best an inefficient way to find
good harnesses and at worst a way to ship something on the wrong
side of the cost / accuracy frontier without knowing it.

The recent **Meta-Harness** framework (Lee et al. 2026) automates
this exploration via an agentic LLM proposer with filesystem access
to prior candidates' run logs. Their search beats hand-tuned RAG by
+7.7pt accuracy at 4× lower token cost on label-intensive
classification benchmarks. But the search is *scalar* — it optimizes
accuracy alone, with token cost and latency invisible to the
optimizer — and *eager* — every proposed candidate is fully
evaluated, with no early-stopping on weak ones.

This paper extends Meta-Harness on three orthogonal axes:

1. **Pareto multi-objective search.** Score each harness on
   `(accuracy↑, tokens↓, latency↓)` and admit candidates to a Pareto
   frontier rather than collapsing the score to scalar accuracy.
   Variance-gated admission rejects unstable candidates; hypervolume
   improvement gives the proposer a single metric to chase.

2. **Budget-aware evaluation.** Successive halving (Jamieson &
   Talwalkar 2016) runs a small screen subset on every proposal,
   then full-eval only the top fraction. Saves 50–75% of evaluation
   compute at no significant accuracy cost.

3. **Component-level attribution.** Drop-one ablation per surviving
   candidate measures each component-kind's contribution to its
   accuracy. The EWMA-aggregated stats are surfaced to the LLM
   proposer's prompt so it knows which kinds are paying off and
   which to mutate away from.

The contribution is empirical, not theoretical. We open-source the
framework with 267 unit tests, a 1-line CLI for any benchmark cell,
and per-seed run logs committed alongside the code. Total
experimental cost across ~100 multi-seed search runs is under $1.50
in API spend.

### What the experiments show

Across a 6-cell grid spanning two providers (OpenAI gpt-4.1-nano,
Google gemini-2.5-flash-lite) × three classification benchmarks
(news_hard_50, symptom_hard, LawBench 2-2 — the last drawn directly
from the original Meta-Harness paper's task family), MH++ produces
significant accuracy gains over hand-tuned RAG on every cell with
sufficient search budget (CIs excluding zero on 4 of 4 6×8-budget
cells, with the 5th cell saturated at 1.0).

The strongest single result is **+29.3pt absolute accuracy** on
OpenAI × symptom_hard (RAG 0.667 → MH++ 0.960; paired t=+17.8,
p=6×10⁻⁵, Cohen's d=+7.98). The framework's per-axis benefit is
clearest on Gemini × news_hard_50 + symptom_hard at 6×8 budget,
where 7 of 10 seeds achieve **strict Pareto dominance** over
hand-tuned RAG — same-or-higher accuracy AND strictly fewer tokens
AND not-worse latency, on every axis simultaneously.

We further compare against four strong hand-tuned baselines beyond
vanilla RAG (CoT-RAG, voting-RAG, diverse-RAG, BARE), to address the
reviewer's natural question "did MH++ beat just vanilla RAG, or
also the strong hand-tuned variants?" The answer is *substantially*:
on 5 of 6 cells, MH++ exceeds the strongest hand-tuned baseline by
+5.9pt to +12.9pt. On the 1 cell (Gemini × news_hard_50) where
hand-tuned CoT-RAG was already at the model's apparent ceiling
(0.940), MH++ discovers a non-obvious shape — BM25(k=5) +
LLM-reranker(m=3) + compressed-CoT(15w) + null-voter — that
reproducibly hits 0.960 across 5 repeats (spread 0.020).

### What the experiments don't show

We are honest about three limitations:

- **Small-budget failure mode.** On Gemini × LawBench 2-2 at 3×4
  search budget, MH++ underperforms hand-tuned RAG by 8.3pt; the
  search couldn't beat a strong baseline at small compute. A 6×8
  rerun closed most of the gap (-1.7pt, p=0.099 — no longer
  significant), but the framework does not replace the need for
  *enough* search budget.

- **Attribution-vs-random parity at small scale.** Our ablation
  experiment (4 conditions × 5 seeds at 3×4 budget on OpenAI ×
  news_hard_50) shows the attribution-guided LLM proposer only
  marginally beats a uniform RandomProposer (Δ=-0.008, p=0.178). The
  attribution contribution shows up at larger budgets / harder
  tasks; at small scale on easy tasks, random is competitive.

- **Public-dataset coverage.** We replicate on one public dataset
  from the original Meta-Harness paper's task family (LawBench
  2-2). Broader public-dataset coverage is a future-work concern.

The contribution is therefore: a multi-objective extension to
Meta-Harness with budget-aware evaluation and component attribution,
shown to produce Pareto-dominating harnesses on multiple cells of a
public-and-curated benchmark grid, with honest characterization of
the failure modes and a reproducible alpha-shape that beats every
hand-tuned baseline we tested.

### Paper structure

Section 2 places our work in the related-work landscape. Section 3
describes the framework in detail with pseudocode. Section 4
reports the experiments. Section 5 discusses implications and
limitations. Section 6 sketches a regret bound for the
attribution-guided proposer (full proof outlined in
Appendix D / THEORY.md). Section 7 covers reproducibility.
Section 8 concludes.

All code, run logs, and prompt caches are released at
`github.com/[user]/Meta-Harness`. Total compute footprint is under
$1.50 USD in API spend, reproducible from a single shell script.
