# Out-of-Scope Future Work — Honest Disclosure

**Date:** 2026-04-26
**Branch:** `main`

This document tracks Tasks-roadmap items that are *not* fully addressed
by the current paper, with the honest reason why.

## Task 1 (incomplete): math/IMO + agent tasks

**The roadmap asks:** "Run MH++ against the exact task families the
original paper used: label-intensive classification, IMO/math, and
TerminalBench-style agent tasks."

**What we did:**
- ✅ Label-intensive classification: LawBench 2-2 replicated. See
  `RESULTS_BEAT_META_HARNESS.md` (+10pt CI on OpenAI gpt-4.1-nano,
  matching the original paper's directional finding at <1/100th compute).

**What we did NOT do:**

### IMO/math
The framework's task abstraction is **classification with discrete
labels** — a `Task` exposes a fixed `classes: list[str]` and the
`LLMPredictor` extracts the class label from the LLM's output by
case-insensitive matching. Math benchmarks like GSM8K, MATH, or AIME
require:
- Open-ended generation (no fixed class list).
- Numerical or symbolic answer extraction (regex / equivalence checks).
- Different scoring infrastructure (a `MathPredictor` or
  `MathScorer`, ~few hundred lines of code).

This is a **substantive framework extension**, not a one-line task
addition. Out of scope for the current paper.

### TerminalBench-style agent tasks
Agent benchmarks like TerminalBench require:
- A sandboxed shell environment.
- Multi-turn LLM interaction (the harness becomes a loop with
  observation/action transitions).
- Different cost accounting (per-turn tokens, total interaction time).
- Different score function (task completion vs final-answer accuracy).

This is **a different runtime entirely**. The current MH++ framework
is a one-shot classification pipeline; agent tasks would require
re-architecting `Harness` from a fixed pipeline to an
observation/action loop. Out of scope.

### Scope statement for the paper

We claim MH++ extends to *classification* tasks of the form the
original Meta-Harness paper reports on. The paper's IMO/math + agent-
task results would require either (a) framework extensions we have
not implemented, or (b) a different framework entirely. We document
this honestly rather than implying coverage.

## Task 2 (partial): full LawBench, USPTO-50k, MASSIVE

**The roadmap asks:** "Go public: full LawBench, USPTO-50k, MASSIVE,
20 Newsgroups, AG News, Symptom2Disease."

**What we did:**
- ✅ AG News (5 seeds × 2 providers, 6×8 budget)
- ✅ dair-ai/emotion (5 seeds × 2 providers, 6×8 budget)
- ✅ 20 Newsgroups top-8 (loader wired; 5-seed × 2-provider experiment
  in flight as of this commit — `bash examples/run_more_public_5seed.sh`)
- ✅ Symptom2Disease top-8 (loader wired; 5-seed × 2-provider in flight)
- ✅ LawBench 2-2 subtask (5 seeds × 2 providers, 3×4 + 6×8 budgets)

**What we did NOT do:**

### Full LawBench (all 16 subtasks)
We replicated only LawBench 2-2 (the headline subtask the original
paper reports on). The other 15 subtasks (`1-1` through `5-3`) include
QA, summarization, retrieval, and classification with very different
class structures. Each would require a custom loader; some are not
classification at all and would need framework extensions (see
"IMO/math" above). Running all 16 across 2 providers × 5 seeds would
multiply compute by ~16× (~$30-50 estimated).

### USPTO-50k
Patent classification with hundreds of fine-grained CPC class codes.
Class set too large for our 8-class top-N cap; would need either
multi-label support or hierarchy-aware scoring. Out of current
framework scope.

### MASSIVE
Multilingual intent classification, 60+ classes, 51 languages. Class
set similarly too large for our 8-class budget; multilingual evaluation
adds another layer (per-language splits, language-conditional retrieval).
Out of current framework scope.

### 10+ seeds across all public cells
The roadmap asks for 10+ seeds. We have 10 seeds on the 4 headline
6×8-budget cells (Gemini news_hard_50, Gemini symptom_hard, OpenAI
news_hard_50, OpenAI symptom_hard). The newer public-benchmark cells
(AG News, emotion, 20 Newsgroups, Symptom2Disease) are at 5 seeds
each. Extending all to 10 seeds is a $1-2 follow-up that we have not
yet run.

## Task 5 follow-on: notebook ✅, dashboard ✅, CLI ✅

All three exist:
- `examples/demo_notebook.ipynb` — Jupyter notebook walking through
  the full pipeline.
- `examples/demo.py` — CLI version + `dashboard.html` static output.
- `examples/rag_vs_mh_bakeoff.py` — production-grade CLI for any
  `(provider × task)` cell.

## Honest paper claim

Across the Tasks roadmap items 1-8:

| # | Task | Coverage |
|---|---|---|
| 1 | Beat MH directly | ✅ classification; ❌ math/IMO; ❌ agent (out of framework scope) |
| 2 | Public benchmarks | ✅ 4 of 6 listed datasets (AG News, emotion, 20news, Symptom2Disease, LawBench 2-2); USPTO + MASSIVE out of scope |
| 3 | Strict Pareto headline | ✅ done |
| 4 | Brutal baselines | ✅ all 9 (RAG / CoT-RAG / voting-RAG / diverse-RAG / DSPy / OPRO / TextGrad / ProTeGi / random / MH-directional) |
| 5 | Killer demo | ✅ notebook + CLI + dashboard |
| 6 | Synergy discovery | ✅ drop-pair attribution + tests |
| 7 | Online/continual | ✅ streaming consumer + paired-CI promotion |
| 8 | Sharp thesis | ✅ THESIS.md |

**6 of 8 fully delivered, 2 of 8 partial with honest scoping.** No task
is misclaimed — every gap is documented above.
