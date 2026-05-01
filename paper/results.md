# V10 — Results

This file is the running log of empirical results for V10. Numbers
land here as each split is touched (dev_50 freely, dev_100 once,
test_500 once at the end). It is the source of truth for the paper's
results section.

## Phase 1 — Hierarchical localization (oracle-free retrieval)

### Stage 1b + 1c + 1g — frozen on dev_100

| Split | Instances | Top-1 | Top-5 | Top-10 | Wall-clock |
|---|---|---|---|---|---|
| dev_50 (commit 5c) | 50 | 74.0% | 96.0% | 98.0% | 2197s |
| dev_100 (commit 6b) | 100 | **84.0%** | **97.0%** | **98.0%** | 13267s |

Acceptance gates (per V10_DESIGN.md §3.2 Phase 1 spec):
- Top-10 ≥ 90% → **PASS** (98.0%)
- Top-1 ≥ 60% → **PASS** (84.0%)

Hardware: AMD Radeon AI PRO R9700 (gfx1201, 32 GB VRAM) + ROCm 6.4
+ PyTorch 2.9.1+rocm6.4. Workers=1 (see V10_DESIGN.md §9 for the
multi-process GPU-Hang verdict on this AMD card).

Embedding model: BAAI/bge-large-en-v1.5 (batch 256). Reranker:
DeepSeek-chat at temperature 0 (per-strategy-rank features visible
in the prompt). Per-instance cost calibrated from actual checkpoint
totals: **$0.0024/inst** (the earlier ~$0.05/inst estimate was the
ceiling; observed cost is much lower because the rerank prompt is
~3-5K input / ~200 output tokens, well below the conservative
estimate).

**Match-strictness audit (commit 8b, `docs/audits/gold_match_strictness_dev_100.md`):**
all 98 top-10 hits and all 97 top-5 hits are exact-or-normalized matches under
`harness.eval._normalize_path` — 100% strict. At top-1, 84/86 of the audit's
lenient-classifier hits are strict; the 2 non-strict are basename matches that
the production matcher correctly rejects, so the published 84% top-1 is the
*strict* number, not inflated by lenient matching.

**Reproducibility (commit 12a, `docs/audits/rerank_variance_dev100.md`):**
two independent dev_100 runs of the same DeepSeek-chat reranker at
temperature 0 produce **bit-identical headline numbers** (84/97/98 top-1/5/10)
with **zero hit/miss flips** at any K across 100 instances. The reranker DOES
produce different top-10 orderings between runs (~40% of instances; first
documented in commit 11d), but the reordering is purely WITHIN the top-K window
and never crosses the K boundary on this dev split. The published 84/97/98 is
reproducible to **±0pp** under DeepSeek; the earlier 11d footnote that the
numbers carry "implicit ±1-2pp" was overstated and is superseded.

**Reranker model substitution (commit 12b,
`docs/audits/dev_100_retrieval_eval_sonnet.md`):**
swapping DeepSeek-chat for Sonnet-4.5 (two runs each, temperature 0) lifts
top-1 from 84% to **86-87%** (+2 to +3pp), top-5 from 97% to 98% (+1pp),
ties top-10 at 98%. The lift is robust: the same six instances flip from
DeepSeek-miss to Sonnet-hit at top-1 across both DeepSeek runs and both
Sonnet runs. Sonnet has slight self-variance (±1pp top-1); DeepSeek has none.
Sonnet costs 16-17× more per instance ($0.04 vs $0.0024). For test_500 we
ship DeepSeek-chat as the published headline because the +2-3pp top-1 lift
may not propagate to Phase 3 pipeline pass rate (Phase 3 hasn't been built
yet) and DeepSeek's bit-stable result is more reproducible. The Sonnet
ablation runs as a separate row alongside the headline. Decision rationale
in `docs/audits/test500_reranker_decision.md` (commit 13a).

### Per-strategy ablation on dev_100

| Strategy | Top-1 | Top-5 | Top-10 |
|---|---|---|---|
| `bm25_extracted_symbols` | 23.0% | 56.0% | 67.0% |
| `bm25_first_paragraph` | 28.0% | 62.0% | 74.0% |
| `bm25_full_issue` | 34.0% | 59.0% | 69.0% |
| `embedding` (bge-large) | 15.0% | 41.0% | 53.0% |
| union-aggregated | 34.0% | 67.0% | 80.0% |
| **reranked (Stage 1g)** | **84.0%** | **97.0%** | **98.0%** |

Stage 1g rerank lift over best individual upstream strategy:
- Top-1: 34% → 84% (+50pp) — matches the dev_50 commit-5d magnitude.
- Top-10: 80% → 98% (+18pp).

The rerank lift is the load-bearing finding of Phase 1. Score-
normalized union-aggregation actively destroys top-1 signal; the
rerank prompt sees per-strategy ranks (`upstream_best_rank`) as
features, not a flat aggregated list. See V10_DESIGN.md §9 working
agreement on aggregation discipline.

### Per-repo breakdown on dev_100

| Repo | n | Top-1 | Top-5 | Top-10 |
|---|---|---|---|---|
| astropy/astropy | 8 | 75.0% | 100.0% | 100.0% |
| django/django | 27 | 81.5% | 92.6% | 92.6% |
| matplotlib/matplotlib | 10 | 80.0% | 100.0% | 100.0% |
| mwaskom/seaborn | 1 | 100.0% | 100.0% | 100.0% |
| psf/requests | 4 | 100.0% | 100.0% | 100.0% |
| pydata/xarray | 6 | 66.7% | 100.0% | 100.0% |
| pylint-dev/pylint | 4 | 100.0% | 100.0% | 100.0% |
| pytest-dev/pytest | 6 | 66.7% | 100.0% | 100.0% |
| scikit-learn/scikit-learn | 10 | 90.0% | 100.0% | 100.0% |
| sphinx-doc/sphinx | 10 | 90.0% | 90.0% | 100.0% |
| sympy/sympy | 14 | 92.9% | 100.0% | 100.0% |

The two top-10 misses are both django:
- `django__django-11211` — gold at rank 17 (`bm25_first_paragraph`).
- `django__django-11299` — gold at rank 26 (`bm25_full_issue`).

Both are recall-rescuable by widening the rerank candidate pool from
top-10 to top-20/30. Deferred — at 98% the marginal cost of widening
exceeds the marginal lift, and the top-1 is already comfortably above
the bar.

### Stages 1d / 1e / 1f — skipped

Per commit-5e decision (`docs/audits/phase1_stages_1d_1e_1f_decision.md`),
stages 1d (grep), 1e (archaeology), 1f (dep-graph) are skipped: with
top-10 at 98%, residual misses are ordering problems (rank 17, 26),
not signal-source problems. The three skipped stages were designed
to add new signal sources for instances where no upstream strategy
named the gold file — that population is empty on dev_100.

If test-500 reveals a different miss profile (gold file in NO upstream
strategy), 1c is the natural unlock first; 1d-1f remain available
behind the `--include-…` flags but are not on the default path.

---

## Phase 2 — Reproduction oracle

Design doc: `docs/V10_DESIGN_PHASE2.md` (commit 7a). No implementation
yet — awaiting ack on four open questions (see Phase 2 doc §8).
