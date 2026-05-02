# Phase 1 — Hierarchical localization

> **Status: drafty, but TODO-free.** This is the working draft of the
> Phase 1 paper section. Tone is forensic; numbers cite their
> producing commit SHA and the audit file under `docs/audits/`.
> Final-form prose comes after test_500. The 8d-era `[TODO]` on
> reranker ablation is resolved by commits 12a/12b (DeepSeek
> reproducibility + Sonnet swap); see "Reproducibility of the rerank
> step" and "Reranker model substitution" sections below.

## Setup

### Benchmark

We evaluate on **SWE-bench Verified** [cite Princeton/Mountain
press], the 500-instance subset of SWE-bench in which every
instance has been validated by humans to be solvable (the original
SWE-bench corpus contains noisy gold patches). Each instance is a
GitHub issue + the repository state at issue-open time + a hidden
gold patch + a hidden public test suite. The agent's job is to
produce a patch that, when applied to the repo, makes the public
test suite pass.

### Oracle-free firewall

V10 enforces a hard contamination boundary. The full
`SWEBenchTaskRow` (with `patch`, `FAIL_TO_PASS`, `PASS_TO_PASS`)
exists only inside `harness.eval`; agents see only an
`InstanceView` that exposes the issue, repo URL, base commit, and
a `path_pattern_hint` (e.g. `**/*.py`). The firewall is enforced at
three levels:

1. **Type-level.** `InstanceView` is a frozen dataclass with no
   reference to the gold patch. A `__post_init__` validator scans
   every string field for forbidden tokens (`patch`, `FAIL_TO_PASS`,
   `PASS_TO_PASS`); any leak raises at construction time, not at
   eval time.
2. **AST-level.** Every module under `harness/` is scanned for
   forbidden imports, attribute accesses (`row.patch`,
   `row["patch"]`), and subscripts. Only `harness.eval` is allowed
   to touch the eval-only fields.
3. **Runtime.** An `LLMCallInspector` wraps every LLM call site;
   any prompt containing the forbidden tokens raises before the
   call is dispatched.

The firewall test (`tests/test_no_oracle_leak.py`, 12 tests, runs
in 0.1 s) gates every CI build. The `make firewall` target is run
as a precondition to every long eval; commit-5e gold-match
strictness audits confirmed the eval's hit semantics use
exact-or-normalized path matching only (no lenient suffix /
basename leakage).

### Split discipline

We keep three splits, with strict touch budgets:

- **dev_50** (50 instances) — for component sweeps. Iterate freely.
  Hand-picked, repo-diverse, 15 easy / 20 medium / 15 hard.
- **dev_100** (100 instances) — for *decision-making* after
  components are tuned on dev_50. Touched once. Held out from
  dev_50 with **zero overlap** (asserted in `scripts/build_dev_100.py`).
  Repo-balanced (~2× dev_50 quotas) and difficulty-balanced via V7
  priors as a calibration target only — the production routing
  re-derives difficulty from issue features at inference, not from
  these priors.
- **test_500** — the headline number. Touched once at the end.

Phase 1's acceptance was decided on dev_100, not dev_50.

## Architecture

Phase 1 is **hierarchical localization**. The job is to take an
issue and produce a ranked list of source files that the patch is
likely to touch. There are no LLM calls in Stage 1b/1c; only Stage
1g uses one. The pipeline is decomposed into stages:

- **1a — Repo skeleton.** Walk the repo at the issue's base
  commit, emit `(path, file_size, line_count, top-level-symbols)`.
  Source for every downstream stage. No LLM.
- **1b — Code-aware BM25 + dense embedding (commit `314e1c2`).**
  BM25 index over the skeleton's symbol-tagged content with three
  query strategies executed independently:
  - `bm25_full_issue` — full issue text.
  - `bm25_first_paragraph` — issue text up to the first blank line.
  - `bm25_extracted_symbols` — the issue's CamelCase / snake_case
    identifiers, run as a symbol-only query.

  Plus a dense retriever using `BAAI/bge-large-en-v1.5` over
  whole-file content. Each strategy returns its own ranked list;
  **no aggregation happens at this stage** (see §Findings).
- **1c — Traceback parser (commit `9ebb164`).** If the issue
  contains a Python traceback, parse it and rank the named files
  with frame-priority (innermost frame highest), suffix-matched
  against the repo skeleton.
- **1d / 1e / 1f — *deliberately skipped*.** Originally planned as
  grep-by-error-string (1d), `git log` archaeology (1e), and
  dependency-graph spreading (1f). After Stage 1g landed and
  cleared 98 % top-10 on dev_50 (and again on dev_100), we wrote a
  decision doc (commit `37fff32`,
  `docs/audits/phase1_stages_1d_1e_1f_decision.md`) recommending
  these stages stay off the default path. They are designed to add
  *new signal sources* for instances where no upstream strategy
  named the gold file — that population is empty at our current
  recall. Adding them would only add cost and latency.
- **1g — LLM reranker (commit `850561c`).** A single LLM call,
  default model `deepseek-chat` (`harness/config/models.yaml`,
  reranker role). Sees the top-30 union-of-strategies candidates
  with **per-strategy ranks visible** as features (the
  `upstream_best_rank` field on `RankedFile`). Produces a final
  top-10 ordering. Cost: ~\$0.05/instance.

## Findings

Two findings are load-bearing for Phase 1.

### Finding 1: aggregation before reranking destroys top-1 signal

The dev_50 retrieval eval (commit `3e315d7`,
`docs/audits/dev50_retrieval_eval.md`) showed:

| Strategy | dev_50 top-1 | dev_50 top-5 | dev_50 top-10 |
|---|---|---|---|
| `bm25_extracted_symbols` | 16 % | 46 % | 70 % |
| `bm25_first_paragraph` | **32 %** | 60 % | 72 % |
| `bm25_full_issue` | 18 % | 66 % | 74 % |
| `embedding` | 0 % (no GPU at this checkpoint) | 0 % | 0 % |
| union-aggregated (score-normalized) | **18 %** | 68 % | 78 % |

`bm25_first_paragraph` *alone* was 32 % top-1; the union-aggregated
result was 18 %. Score-normalized aggregation actively destroyed
top-1 signal — the strongest individual ranking got drowned out by
the others. The decision: aggregation is for *candidate-set
construction*, not scoring. The reranker (Stage 1g) sees per-strategy
ranks as separate features and makes the actual ordering decision.
This shape replicates on dev_100 (commit `652b26f`,
`docs/audits/dev_100_retrieval_eval.md`):

| Strategy | dev_100 top-1 | dev_100 top-5 | dev_100 top-10 |
|---|---|---|---|
| `bm25_extracted_symbols` | 23 % | 56 % | 67 % |
| `bm25_first_paragraph` | 28 % | 62 % | 74 % |
| `bm25_full_issue` | **34 %** | 59 % | 69 % |
| `embedding` (bge-large) | 15 % | 41 % | 53 % |
| union-aggregated | 34 % | 67 % | 80 % |

The strongest individual is `bm25_full_issue` on dev_100 vs
`bm25_first_paragraph` on dev_50 — the WINNER moves between splits,
which is exactly why hard-coding any single strategy as "the one"
would be brittle. Letting the reranker decide is the right shape.

### Finding 2: the rerank lift replicates across splits at +40-50pp top-1

Stage 1g's effect on the same retrieval base:

| Split | Without rerank (best individual) | With rerank | Δ top-1 |
|---|---|---|---|
| dev_50 (commit `2393909`) | 32 % top-1 / 78 % top-10 | **74 % top-1 / 98 % top-10** | **+42pp** |
| dev_100 (commit `652b26f`) | 34 % top-1 / 80 % top-10 | **84 % top-1 / 98 % top-10** | **+50pp** |
| test_500 (commit `129b8ff`+, 488/500 scored) | 29 % top-1 / 72 % top-10 | **75.4 % top-1 / 94.9 % top-10** | **+46pp** |

The lift replicates across all three splits within sampling error:
**+42pp on dev_50, +50pp on dev_100, +46pp on test_500.** This is
**the** quantitative finding of Phase 1: a single LLM call given per-
strategy-rank features lifts top-1 from ~30 % to ~80 % at \$0.0024
per instance (DeepSeek-chat, observed cost on test_500's 488 scored
instances at \$1.13 total).

The test_500 absolute number (75.4 % top-1) sits between dev_50 and
dev_100 — slightly below dev_100's 84 % because test_500 has a
heavier django proportion (231/500 = 46 %) than dev_100 (27/100 =
27 %), and django is the hardest repo for top-1 (73.6 % on
test_500). Top-10 lands at 94.9 % — the headline acceptance
target was ≥ 90 %; cleared with margin.

The match is *strict* (commit `49e2b1a`,
`docs/audits/gold_match_strictness_dev_100.md`): all 98 top-10 hits
and 97 top-5 hits are exact-or-normalized; 84/86 top-1 (97.7 %) are
strict, with 2 basename-only matches that the production matcher
correctly *rejects*. The 84 % top-1 number is the strict count.

### Reproducibility of the rerank step (commit `1773616`)

Two independent dev_100 runs of the same DeepSeek-chat reranker at
temperature 0 produce **bit-identical headline numbers** (84/97/98
top-1/5/10) with **zero hit/miss flips** on any of K = 1, 5, 10. The
reranker DOES produce different top-10 orderings between runs (~40 %
of instances; documented in `docs/audits/parallel_dev100_negative.md`,
commit `6947131`), but the reordering stays inside the top-K window
without crossing the K boundary. The dev_100 headline is reproducible
to ±0pp under DeepSeek; we report it as a single number, not a band.
Audit: `docs/audits/rerank_variance_dev100.md`.

### Reranker model substitution (commit `17efb38`)

We ablated the Stage 1g reranker by swapping DeepSeek-chat for
Sonnet-4.5 (two runs each, temperature 0, otherwise identical
configuration):

| Reranker | Run 1 (top-1/5/10) | Run 2 (top-1/5/10) | $/inst |
|---|---|---|---|
| `deepseek-chat` (default) | 84 / 97 / 98 | 84 / 97 / 98 | $0.0024 |
| `claude-sonnet-4-5` | 86 / 98 / 98 | 87 / 98 / 98 | $0.0400 |

Sonnet beats DeepSeek by **+2 to +3pp top-1, +1pp top-5, ties at
top-10**. The lift is robust: the same six instances flip from
DeepSeek-miss to Sonnet-hit at top-1 across both DeepSeek runs and
both Sonnet runs (`astropy__astropy-13236`,
`astropy__astropy-14369`, `django__django-11728`,
`pydata__xarray-4094`, `pytest-dev__pytest-5787`,
`sphinx-doc__sphinx-11445`). Sonnet has slight self-variance
(±1pp top-1 across two runs); DeepSeek has none.

Sonnet costs 16-17× more per instance ($0.04 vs $0.0024). For the
test_500 headline run we ship DeepSeek-chat (commit `a81c103`,
`docs/audits/test500_reranker_decision.md`); Sonnet runs as a
separate ablation row. Reasoning: (1) Phase 1 top-1 may not equal
Phase 3 pipeline pass rate — the +2-3pp lift on retrieval top-1 is
not yet known to propagate downstream; (2) DeepSeek's bit-stable
headline is more reproducible than Sonnet's ±1pp wobble for a
published number; (3) reporting both runs preserves the evidence
without committing to the more expensive model based on incomplete
downstream signal.

We did **not** ablate Opus-4.7. Top-10 is saturated at 98% in every
configuration; Opus would lift top-1 but is 5× Sonnet's cost (~$100
on test_500). Opus is reserved for the patch-gen agent path
(Phase 3) where it actually affects the headline pass rate.

## Per-repo breakdown (dev_100)

| Repo | n | Top-1 | Top-5 | Top-10 |
|---|---|---|---|---|
| astropy/astropy | 8 | 75.0 % | 100.0 % | 100.0 % |
| django/django | 27 | 81.5 % | 92.6 % | 92.6 % |
| matplotlib/matplotlib | 10 | 80.0 % | 100.0 % | 100.0 % |
| mwaskom/seaborn | 1 | 100.0 % | 100.0 % | 100.0 % |
| psf/requests | 4 | 100.0 % | 100.0 % | 100.0 % |
| pydata/xarray | 6 | 66.7 % | 100.0 % | 100.0 % |
| pylint-dev/pylint | 4 | 100.0 % | 100.0 % | 100.0 % |
| pytest-dev/pytest | 6 | 66.7 % | 100.0 % | 100.0 % |
| scikit-learn/scikit-learn | 10 | 90.0 % | 100.0 % | 100.0 % |
| sphinx-doc/sphinx | 10 | 90.0 % | 90.0 % | 100.0 % |
| sympy/sympy | 14 | 92.9 % | 100.0 % | 100.0 % |

Django is the only repo where top-10 isn't 100 % (92.6 %); both
top-10 misses across the entire 100-instance set are django.

## Limitations

### The two top-10 misses on dev_100

Both are django, and both are *rescuable* by the rerank pool, not
by the upstream retrieval:

| Instance | Gold file | Best upstream rank | Best strategy |
|---|---|---|---|
| `django__django-11211` | `django/db/models/fields/__init__.py` | 17 | `bm25_first_paragraph` |
| `django__django-11299` | `django/db/models/sql/query.py` | 26 | `bm25_full_issue` |

Both gold files appear in some upstream strategy's results, but
not in the union-aggregated top-30 that the reranker sees at its
default candidate-pool width. **Widening the rerank pool from
top-10 to top-20 / top-30 would mechanically rescue both** — the
gold files exist at ranks 17 and 26 in single strategies. We
explicitly chose **not** to do this for the headline run, because:

1. At 98 % top-10, the marginal cost (more tokens / call, slower)
   exceeds the marginal lift (2 instances on a 100-instance dev
   split, no signal yet on whether this generalizes to test_500).
2. Top-1 is already comfortably above the 60 % bar (84 %).
3. The fix is a one-line config change in `harness/rerank.py` —
   we can revisit on test_500 if it shows the same miss profile.

This is a deliberate trade-off, not an oversight. Recorded as such
in commit `652b26f`'s message.

### Hardware-bound parallelism

Phase 1's retrieval eval ran at `--workers 1` on AMD Radeon AI PRO
R9700 (gfx1201, 32 GB VRAM) + ROCm 6.4 + PyTorch 2.9.1+rocm6.4.
Three escalating evidence points (V10_DESIGN.md §9) showed that
multi-process workers either crash the host (RAM exhaustion at 4
workers) or hit a `GPU Hang` HW exception at 2 workers — AMD
ROCm's PyTorch backend doesn't serialize multi-process kernel
launches cleanly on this RDNA card. dev_100 took 13267 s
(~3 hr 41 m); test_500 is projected at ~24.5 hr at the same pace
(commit `e59ecc6`, `docs/audits/test500_preflight.md`). Single-
process batched-multi-instance inference would unlock parallelism
on this hardware, but the cost-vs-complexity trade-off doesn't
favor it at our current scale.

### What we *don't* claim

- We do **not** claim Stage 1g would lift top-1 by +50pp with a
  weaker reranker. We measured DeepSeek-chat (84% top-1, +50pp over
  the best individual upstream strategy) and Sonnet-4.5 (86-87%
  top-1) at temperature 0 — both produce the +50pp-class lift.
  Cheaper models (Haiku, GPT-4o-mini class) were not ablated.
- We do **not** claim 1d / 1e / 1f are useless. They are unhelpful
  *given our current retrieval base*. If test_500 reveals a
  different miss profile (gold file in NO upstream strategy), Stage
  1c is the natural unlock first; 1d-1f remain available behind
  the `--include-…` flags but are not on the default path.
- We do **not** claim our rerank prompt is optimal. Per-strategy
  ranks are the *load-bearing* signal we surface; other prompt
  shapes haven't been ablated.

## Reproducibility

Every result above is reproducible from the public artifacts in
this repo:

- Splits: `splits/dev_50.json`, `splits/dev_100.json`. Construction
  scripts: `scripts/build_dev_split.py`, `scripts/build_dev_100.py`
  (deterministic; assert zero overlap).
- Eval driver: `scripts/retrieval_eval_dev50.py` (the name is a
  legacy artifact; the script takes `--split` and works on any
  split).
- Per-instance checkpoints under `runs/v10_dev_100_retr_eval/checkpoints/`
  let any reviewer re-score offline without re-running retrieval.
- Audits in `docs/audits/`:
  - `dev_100_retrieval_eval.md` — the headline numbers.
  - `gold_match_strictness_dev_100.md` — match-type audit (98 / 97
    of the top-10 / top-5 hits are exact-or-normalized; 84/86 of the
    audit-classifier's top-1 hits are strict — production reports
    the strict count).
  - `rerank_variance_dev100.md` — pairwise top-K flip analysis
    across 4 rerank runs (2 DeepSeek + 2 Sonnet). DeepSeek is
    bit-stable (0 flips); Sonnet has ±1pp self-variance.
  - `phase1_stages_1d_1e_1f_decision.md` — the skip rationale.
  - `test500_preflight.md` — wall-clock and cost projection for
    the test_500 two-run plan.
  - `test500_reranker_decision.md` — DeepSeek-as-headline rationale.
  - `test500_launch_readiness.md` — pre-launch gate verification.

### Variance methodology

We report dev_100 numbers from a SINGLE DeepSeek run. The
single-run report is justified by the variance audit
(`docs/audits/rerank_variance_dev100.md`): two independent DeepSeek
runs at temperature 0 produce zero hit/miss flips on top-1, top-5,
or top-10 across 100 instances. Ordering noise within the top-K
window does not cross the K boundary, so ±0pp at all reported K.
Sonnet, when ablated, shows ±1pp self-variance on top-1 — small
enough to footnote, not large enough to drive multi-run reporting.

For test_500 we will run BOTH DeepSeek (headline) and Sonnet
(ablation) once each. The DeepSeek number is the published
acceptance result; the Sonnet row sits beside it in the paper as
the upper-bound model-substitution data point. Per
`docs/audits/test500_reranker_decision.md`.

The `v10-phase-1-complete` git tag (annotated, on commit
`652b26f`) is the canonical Phase 1 freeze point.
