# Out-of-Scope Future Work — Honest Disclosure

**Date:** 2026-04-26
**Branch:** `main`

This document tracks Tasks-roadmap items that are *not* fully addressed
by the current paper, with the honest reason why.

## The corrected scorecard

After user-audit on 2026-04-26:

| # | Task | Status | Honest reason |
|---|---|---|---|
| 1 | Beat MH directly | **Partial** | LawBench classification covered (+10pt). IMO/math + TerminalBench-style agent tasks are explicitly not implemented — they need framework extensions our classification-only architecture cannot provide. |
| 2 | Public benchmarks | **Partial** | 4 of 6 listed datasets wired and run (AG News, emotion, LawBench 2-2 plus 20 Newsgroups + Symptom2Disease loaders + 5-seed run in flight). USPTO-50k + MASSIVE not done. Full LawBench (15 more subtasks beyond 2-2) not done. Most cells at 5 seeds, only 4 cells at 10 seeds. |
| 3 | Strict Pareto headline | **Met** | RESULTS_FINAL_GRID.md leads with 17/40 strict-dominance trials. |
| 4 | Brutal baselines | **Partial** | All 9 baselines implemented and reported. **But MH++ does NOT beat all of them**: loses to OPRO on AG News (-4.4pt), loses to DSPy on LawBench (-3.7pt). The roadmap implies "if MH++ still wins, story is hard to ignore" — we have 3 wins / 2 losses on OpenAI cells. Honest, not full. |
| 5 | Killer demo | **Met** | examples/demo.py + examples/demo_notebook.ipynb + HTML dashboard. |
| 6 | Synergy discovery | **Met** | meta_harness_plus/synergy.py drop-pair attribution + 3 unit tests. |
| 7 | Online/continual | **Partial** | meta_harness_plus/online.py exists but is **passive** by design — it ingests + rescores + emits PromoteReport. There is no background search loop that proposes new candidates from production data. The online module's docstring explicitly says: "No background search loop yet." Real continual improvement (auto-tuning layer for all LLM apps) requires the active search loop, which is implemented in this commit but not battle-tested. |
| 8 | Sharp thesis | **Met** | THESIS.md with the "Pareto harness search is a new layer of AI infrastructure" claim. |

**Final: 4 fully met (3, 5, 6, 8), 4 partial (1, 2, 4, 7).**

## What partial means for each

### Task 1 (beat MH on math/agent) — what's missing

The framework's `Task` abstraction is **classification with discrete
labels** — a fixed `classes: list[str]` + case-insensitive label
extraction. Math benchmarks require:
- Open-ended generation (no fixed class list).
- Numerical/symbolic answer extraction (regex / equivalence).
- Different scoring (`MathScorer` ~hundreds of LoC).

Agent benchmarks require:
- Sandboxed shell environment.
- Multi-turn observation→action loop.
- Per-turn cost accounting.
- Different success function (task completion vs final-answer).

These are **substantive framework rearchitectures**, not one-line task
additions. Out of scope for the current paper's framework.

### Task 2 (public benchmarks) — what's missing

- ✅ AG News, emotion: 5-seed × 2-provider, 6×8 budget
- ✅ LawBench 2-2: 5-seed × 2-provider, 3×4 + 6×8 budgets
- 🔄 20 Newsgroups + Symptom2Disease: 5-seed × 2-provider in flight
  as of 2026-04-26 17:30
- ❌ Full LawBench (15 more subtasks beyond 2-2): each subtask is a
  custom loader; some are not classification (QA, summarization)
  and would need framework extensions.
- ❌ USPTO-50k: hundreds of fine-grained CPC class codes, exceeds our
  8-class budget; would need multi-label or hierarchy-aware scoring.
- ❌ MASSIVE: 60+ classes × 51 languages; exceeds our 8-class
  budget and adds multilingual evaluation complexity.

10+ seeds: only on 4 headline cells (Gemini × news_hard_50, Gemini ×
symptom_hard, OpenAI × news_hard_50, OpenAI × symptom_hard). Other
cells at 5 seeds.

### Task 4 (brutal baselines) — what's missing

| Cell                | Best non-MH baseline | MH++   | Δ vs best |
|---------------------|----------------------|--------|-----------|
| OpenAI × news_hard_50 | voting-RAG (0.900) | 0.928 | **+2.8pt MH++** |
| OpenAI × symptom_hard | TextGrad (0.933)   | 0.960 | **+2.7pt MH++** |
| OpenAI × emotion      | OPRO (0.583)       | 0.592 | **+0.9pt MH++** |
| OpenAI × agnews       | OPRO (0.896)       | 0.852 | **-4.4pt loss** |
| OpenAI × lawbench_2_2 | DSPy (0.333)       | 0.296 | **-3.7pt loss** |

We win 3 of 5 OpenAI cells. The 2 losses are on cells where the
specific baseline's narrow optimization hits its sweet spot:
- OPRO's instruction-string search beats MH++ on AG News's clean topic
  boundaries.
- DSPy's bootstrap-fewshot demo selection beats MH++ on Chinese legal
  classification at our 6×8 search budget.

To convert partial → met, MH++ would need either:
- A specific demo-bootstrap component (DSPy-style) added to the
  search space, or
- Even larger search budget (MH++ at 12×12+ may eventually find
  the OPRO/DSPy shapes itself), or
- An honest acknowledgment that broader search is sometimes weaker
  than narrower specialty optimization.

We chose option 3 in this paper.

### Task 7 (online/continual) — what's missing in the *passive* version

The `OnlineHarnessImprover` we shipped:
- ✅ Ingests labelled production examples.
- ✅ Periodically rescores frontier on new data.
- ✅ Emits PromoteReport with paired-bootstrap CI for safe promotion.

What it does NOT do (yet):
- ❌ Propose new candidates from production data (no background
  search loop).
- ❌ Auto-mutate the frontier when production-distribution drift is
  detected.
- ❌ Run the LLMProposer as a long-lived service.

This commit adds an `propose_with_search` extension to bring the
module closer to active continual improvement (see new code), but
it is not yet stress-tested in production.

## Path to "fully met" for each partial

| # | Task | Estimated work to fully meet |
|---|---|---|
| 1 | Beat MH math/agent | 2-4 weeks: math task abstraction + agent runtime |
| 2 | Public benchmarks | 1-2 days: USPTO/MASSIVE loaders + 10-seed extensions |
| 4 | Brutal baselines | 4-8 hours: DSPy-component integration + agnews retry at larger budget |
| 7 | Online/continual | 1-2 days: production-grade search loop + safety harness |

Totals: ~3-5 weeks of focused work to convert all 4 partials to met.

## Scope of the current paper

We claim:
- ✅ Workshop-grade evidence on classification with strict Pareto
  dominance (Task 3) and substantial accuracy gains over hand-tuned
  baselines (Task 4 partial).
- ✅ Open-source framework with notebook/CLI/dashboard demo (Task 5).
- ✅ Synergy discovery + sharp thesis (Tasks 6, 8).
- ⚠️ Direct replication of MH paper's classification-cell finding at
  <1/100th compute (Task 1 partial — only one of three task families).
- ⚠️ Public benchmarks (Task 2 partial — 5 of 6 datasets, 4 of those
  finished, mostly 5-seed not 10-seed).
- ⚠️ All 9 brutal baselines compared (Task 4 partial — 3 wins, 2
  losses).
- ⚠️ Passive online/continual mechanism (Task 7 partial — no active
  search loop yet).

This is the **honest** scope. The Tasks roadmap goalpost is "wow the
world"; what we have is "workshop-grade with some honest losses." The
paper should claim that, not more.
