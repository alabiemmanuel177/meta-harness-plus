# Test_500 reranker decision (commit 12c)

This document analyzes the dev_100 evidence from commits 12a (DeepSeek
variance) and 12b (Sonnet ablation) and recommends which Stage 1g
reranker model to ship in the test_500 headline run, and whether to
multi-run test_500 for variance bars.

## Evidence summary

All four runs are on the same 100 instances of `splits/dev_100.json`
with the same upstream retrieval (BM25 across 3 query strategies +
bge-large-en-v1.5 embedding + Stage 1c traceback). The reranker is
the only thing that varies. All runs use `temperature = 0`.

| Run            | top-1  | top-5  | top-10 | $/inst   | $/test_500 (proj.) |
|----------------|--------|--------|--------|----------|---------------------|
| deepseek_run1  | 84.0%  | 97.0%  | 98.0%  | $0.0024  | $1.20               |
| deepseek_run2  | 84.0%  | 97.0%  | 98.0%  | $0.0024  | $1.20               |
| sonnet_run1    | 86.0%  | 98.0%  | 98.0%  | $0.0400  | $20.00              |
| sonnet_run2    | 87.0%  | 98.0%  | 98.0%  | $0.0397  | $19.85              |

(Detailed audits per run: `docs/audits/dev_100_retrieval_eval.md` and
its `_deepseek_run2` / `_sonnet` / `_sonnet_run2` siblings. Pairwise
flip table at `docs/audits/rerank_variance_dev100.md`.)

### Pairwise top-K flip rates (on common-instance set)

| Pair                       | n=  | top-1 | top-5 | top-10 |
|----------------------------|-----|-------|-------|--------|
| deepseek_run1 vs run2      | 100 | **0** | **0** | **0**  |
| sonnet_run1   vs run2      | 100 | 1     | **0** | **0**  |
| deepseek_run1 vs sonnet_r1 | 100 | 6     | 1     | **0**  |
| deepseek_run1 vs sonnet_r2 | 100 | 5     | 1     | **0**  |
| deepseek_run2 vs sonnet_r1 | 100 | 6     | 1     | **0**  |
| deepseek_run2 vs sonnet_r2 | 100 | 5     | 1     | **0**  |

## Findings

1. **DeepSeek-chat at temp 0 is bit-stable on dev_100.** Two
   independent runs produce 0 flips at every K. The earlier 11d
   worry that the published headline carries an implicit ±1-2pp
   variance was overstated for the K = 1 / 5 / 10 metrics: ordering
   noise WITHIN the top-K window does not translate to recall noise.

2. **Sonnet-4.5 at temp 0 has ±1pp self-variance on dev_100.** Two
   Sonnet runs produced 86 vs 87 top-1, with 1 instance flipping
   between hit and miss. Top-5 and top-10 are fully stable across
   the Sonnet pair.

3. **Sonnet is +2 to +3pp better than DeepSeek on top-1.** The lift
   comes from 6 instances (top-1 flips) where Sonnet picks the gold
   file at rank #1 and DeepSeek doesn't:
     - `astropy__astropy-13236`
     - `astropy__astropy-14369`
     - `django__django-11728`
     - `pydata__xarray-4094`
     - `pytest-dev__pytest-5787`
     - `sphinx-doc__sphinx-11445`
   Of these 6, Sonnet wins 4 in run-1 and 5 in run-2 (one flips
   between Sonnet runs). DeepSeek wins 2 in run-1 and 1 in run-2 vs
   Sonnet — the 2-3pp Sonnet net lift comes from those 4-5 wins
   minus 1-2 losses.

4. **Top-10 is identical across all four runs (98%).** The 2 misses
   are the same instances (`django__django-11211` and
   `django__django-11299`, both rescuable by widening the rerank
   pool past 10) regardless of model. Whatever we ship at top-10
   stays at 98%.

5. **Sonnet costs 16-17× more per instance.** test_500 projection:
   DeepSeek ≈ $1.20, Sonnet ≈ $20. Net delta is ~$19 for the +2-3pp
   top-1 lift.

## Recommendation (revised in commit 13a)

**Run test_500 twice — DeepSeek as the headline, Sonnet as the
ablation.** Two runs, sequenced:

  1. **test_500 headline run with DeepSeek-chat.**
       - Reproducibility: 12a showed 0 flips on dev_100 across two
         reruns. The headline is a single number with no error bars
         needed.
       - Cost: ~$1.20.
       - Wall-clock: ~24 hr at workers=1 (per the 8c preflight, scaled
         from dev_100's 24 min × 5).
       - This is the number we publish as the V10 V1 result.

  2. **test_500 ablation run with Sonnet-4.5.**
       - Lift: +2-3pp on dev_100 top-1, replicable across two Sonnet
         runs and the same instances rescued in both DeepSeek runs.
       - Cost: ~$20.
       - Wall-clock: ~48 hr at workers=1 (Sonnet API latency was 1.7×
         DeepSeek's on dev_100; 24 hr × 1.7 ≈ 40 hr; 48 hr is the
         rounded ceiling).
       - This becomes a paper-section table row: "DeepSeek baseline
         vs Sonnet ablation" so reviewers see the full picture.

### Why two runs and not one Sonnet run

The 12c original recommendation (commit `4cdb097`) was "ship Sonnet
as the headline." Revised on review because:

  - Phase 1 top-1 doesn't directly equal Phase 3 pipeline pass rate.
    Phase 3 will pick from the ranked file list and try to generate a
    patch; whether it picks rank 1 vs rank 5 depends on patch-gen
    behavior we haven't measured yet. The +2-3pp Sonnet lift on
    *retrieval top-1* may or may not translate to +2-3pp on
    *pipeline pass rate* downstream.
  - Reporting both runs lets reviewers see the full evidence chain.
    If Phase 3 picks rank-1 aggressively, Sonnet wins; if it widens
    the candidate set, DeepSeek's rank-5 lifts more (97% top-5 vs
    Sonnet's 98% top-5 — only +1pp gap).
  - Cost is bounded: $1.20 + $20 = $21.20 total. Well under the
    §11 budget for the test_500 run.
  - Wall-clock allows it: total ~72 hr serial, fits in a long
    weekend; can interleave the two runs (they don't share GPU
    state — DeepSeek + retrieval cache from headline run feeds
    directly into Sonnet ablation by signature swap).

### Hedge plan (unchanged from the original 12c recommendation)

If either run rate-limits or fails mid-flight: the
`--reranker-model` flag swaps the model with no other changes.
Per-instance checkpoints under
`runs/v10_test_500_*/checkpoints/embed_…_rerank_<model>` mean a
mixed-model run is recoverable — restart with the other model and
the previously-completed rerank checkpoints carry over (they're
keyed on signature, which includes the model suffix).

### Considered alternative: Sonnet-only headline

The original 12c recommendation was "ship test_500 with Sonnet
as the headline, single run, $20." Three things tilted the revision
back toward DeepSeek as the published number:

  - The +2-3pp Sonnet lift is on top-1 ONLY; top-10 is identical.
    If Phase 3 ends up using top-K with K > 1, the lift is much
    smaller or zero.
  - Sonnet has ±1pp self-variance (12b's 86 vs 87 top-1 across two
    runs). DeepSeek has 0pp self-variance. The published headline
    is more reproducible if it's the deterministic option.
  - The Sonnet number is still useful as the upper-bound ablation
    in the paper, just not as the published headline. Running both
    captures the full picture without committing to the more
    expensive model on incomplete downstream evidence.

The Sonnet-only path remains cheaper ($20 single run) and is the
right call IF subsequent evidence shows Phase 3 picks rank-1 hard.
That evidence does not yet exist.

## What we are NOT recommending

- We are NOT recommending Opus 4.7 for the rerank role. Top-10 is
  saturated at 98%; Opus is unlikely to lift it. Opus would lift top-1,
  but Opus is 5× more expensive than Sonnet (~$100 for test_500) and
  the Phase 1 acceptance gate doesn't require >87% top-1. Opus is
  reserved for the patch-gen agent path (Phase 3) where it actually
  affects the headline pass rate.
- We are NOT recommending widening the rerank candidate pool from
  top-10 to top-30 for test_500. Both django misses are at ranks 17
  and 26, so widening the pool COULD rescue them, but that's a
  Phase 1 follow-up, not a reranker-model decision. Tracked as a TODO
  in `paper/section_phase1.md` "Limitations".

## Operational notes

- The reranker model swap is config-only, no source edits:
  `make eval` or any `scripts/retrieval_eval_dev50.py` invocation
  takes `--reranker-model claude-sonnet-4-5`. Set the env var
  `V10_RERANKER_MODEL=claude-sonnet-4-5` to make Sonnet the default
  for a session without code changes.
- ANTHROPIC_API_KEY must be set in `.env` (already present).
- The dev_100 retrieval cache (`runs/v10_dev_100_retr_eval/checkpoints/embed_noshortlist_tb_bs256/`)
  is reused — Sonnet on test_500 needs only the rerank step; the
  retrieval would be rebuilt fresh for the test_500 corpus.

## Cumulative cost on commit 12 batch

  - 12a deepseek_run2:  $0.24
  - 12b sonnet_run1:    $4.00
  - 12b sonnet_run2:    $3.97
  - 12c (this doc):     $0.00 (pure prose)
  - 13a (this revision): $0.00 (pure prose)
  - **Total: $8.21** of the spec's $50 cap.

## Projected cost on test_500 (with revised plan)

  - DeepSeek headline run: ~$1.20 (500 × $0.0024)
  - Sonnet ablation run:   ~$20.00 (500 × $0.04)
  - **Total: ~$21.20** for a complete test_500 evidence package.

## Next gates (unchanged by this batch)

  1. **Phase 2 design ack** on `docs/V10_DESIGN_PHASE2.md` — still
     the next thing blocking Phase 2 code.
  2. **Test_500 launch** — gated on Phase 2 design ack landing AND
     a sign-off on this decision doc (revised in 13a).
