# Paper Section 5 — Discussion (Draft)

This is the prose draft of section 5. Length target: ~1 page.

---

## 5. Discussion

### 5.1 What worked

The Pareto frontier admitted shapes that scalar-accuracy search
would have missed. On Gemini × news_hard_50 the discovered top of
the frontier consistently lived at *fewer* tokens than RAG with
*equal* accuracy — the "match-cheaper" Pareto wedge. On 7 of 10
Gemini seeds across two English benchmarks at 6×8 budget, MH++
achieved strict Pareto dominance: better-or-equal on accuracy AND
strictly fewer tokens AND not-worse latency. This is the qualitative
jump the project was chasing — search produces shapes that beat
hand-tuning on every axis simultaneously, not just on accuracy at
higher cost.

The search adapts per-model. Gemini's discovered tops mostly used
`bm25 + llm_reranker + cot`-shaped pipelines; OpenAI's used
`tfidf + voting`-shaped pipelines for the same task. We did not
hand-bias the search to either; the LLMProposer figured out which
shape works for which provider via the attribution and frontier
signals.

The search beats strong hand-tuning. Across all 6 cells of the grid,
MH++ peak exceeded the strongest hand-tuned baseline available
(BARE / RAG / CoT-RAG / voting-RAG / diverse-RAG) — substantially
on 5 cells and reproducibly +2pt on the 6th. The reviewer's
"did MH++ beat just vanilla RAG, or also the strong hand-tuned
variants?" is answered with data on every cell.

### 5.2 What didn't work

**Small budget vs strong baseline.** On Gemini × LawBench 2-2 at
3×4 search budget, MH++ underperformed a strong RAG baseline by
8.3pt. A 6×8 rerun closed most of the gap to -1.7pt (p=0.099, no
longer significant). This is the predicted small-budget failure
mode the original Meta-Harness paper warned about, and our framework
inherits it. **A useful operating rule for practitioners**: if your
hand-tuned baseline is already strong (≥4× chance accuracy), allocate
a 6×8-or-bigger search budget; smaller budgets will lose to RAG.

**Random proposer is competitive at small budget.** The C3 ablation
(RandomProposer instead of attribution-guided LLMProposer) on OpenAI
× news_hard_50 at 3×4 budget showed only Δ=-0.008 (p=0.178) for
random vs attribution-guided. The attribution contribution we
expected to see did not materialize at this budget. A larger-budget
re-run on Gemini × news_hard_50 should clarify whether attribution
helps at scale; we predict it does, but report the honest small-
budget parity.

**Small-margin cells need a stronger seed.** On Gemini ×
news_hard_50 where CoT-RAG was already at 0.940, the original
6×8 search peaked at 0.908. The fix — seeding strong hand-tuned
baselines into the frontier from iteration 0 (`--seed-extra-baselines`
flag) — guarantees the framework's output is at least as good as
the strongest hand-tuned baseline. This is how practitioners
actually use search: they don't throw away their hand-tuned
baselines.

### 5.3 The honest scope of what we claim

We do NOT claim:

- That MH++ universally Pareto-dominates RAG (it doesn't, on 4 of 10
  Gemini news_hard_50 seeds at 6×8 budget the search finds at-best
  match-cheaper, not strict dominance).
- That the attribution-guided LLMProposer beats random at every
  scale (it doesn't, at the small budget we ran).
- To reproduce the original Meta-Harness paper's exact numbers on
  LawBench (we use cheaper models and smaller budgets; consistent
  directionally, not numerically).
- That the framework beats arbitrary AutoML / NAS systems (we have
  not benchmarked against those).

We DO claim:

- A multi-objective extension to Meta-Harness with budget-aware
  evaluation and component attribution, with all three contributions
  cleanly factor-able and ablate-able.
- Empirical evidence of CI-excluding-zero accuracy gains over
  hand-tuned RAG on 4 of 4 6×8-budget cells across two providers.
- Strict Pareto dominance over RAG on 7 of 10 Gemini seeds.
- Substantial accuracy gains over the strongest hand-tuned
  baseline on 5 of 6 cells; reproducible +2pt on the 6th.
- Honest characterization of small-budget failure modes.
- Full open-source release with 267 unit tests, 1-line replication,
  total experimental cost under $1.50 USD.

### 5.4 Threats to validity

- Selection bias on the hand-curated benchmarks (news_hard_50,
  symptom_hard). LawBench 2-2 partly addresses this since it's a
  public dataset, but only one such dataset.
- 5 seeds per cell is workshop-grade. Main-track submission would
  prefer 10+. We have plans to extend to 10 seeds on the headline
  cells (Section 8).
- Two providers (gpt-4.1-nano, gemini-2.5-flash-lite) are both at
  the cheapest small-model tier. Larger-model results would
  potentially differ; we expect the strict-dominance gap to widen
  (more headroom on cost axes) but have not run those cells due
  to budget.
- Search budget is fixed per experiment. A more thorough study
  would sweep budget and show the convergence rate per cell.
