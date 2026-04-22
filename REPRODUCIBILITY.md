# Reproducibility Branch

**Branch:** `reproducibility`
**Problem:** the main-branch `llm-proposer` bakeoff ([RESULTS.md](RESULTS.md)) revealed two methodological issues that made the search numbers noisy, not wrong:

1. **Ollama at `temperature=0` isn't byte-deterministic** — GPU batching non-determinism makes the same harness score differently across evaluations.
2. **6-item attribution screens are too small** to resolve drop-one accuracy deltas. Both models reported mean_deltas ≈ 0 because the signal was below the noise floor.

This branch addresses both without breaking the `n_repeats=1` fast path.

## Changes

### `Scorer.score(harness, examples, *, n_repeats=1)`

When `n_repeats > 1`, the harness is evaluated the full sweep `n_repeats` times. Aggregation is:

- **Median accuracy** — robust to the one-in-N outlier runs from GPU non-determinism. Mean would drag toward those outliers.
- **Mean tokens / latency** — cost is approximately Gaussian across repeats, so mean is fine.
- **`accuracy_spread`** — max(acc) − min(acc) across repeats. Surfaced on the `ScoreVector` so downstream code can see how noisy a candidate is.

`n_repeats=1` takes the original fast path exactly — all 49 pre-branch tests still pass byte-identically.

### `SearchConfig` reproducibility knobs

```python
SearchConfig(
    eval_repeats=3,                 # full-eval median over 3 repeats
    screen_repeats=1,               # halving-screen: usually not needed, cheap path
    attribution_repeats=3,          # drop-one ablation median over 3 repeats
    attribution_screen_size=20,     # bigger subset *just for* drop-one ablation
)
```

`attribution_screen_size` is the second fix: attribution now gets its own, typically larger, subset of the eval set. It's seeded differently from the halving screen so the two subsets overlap minimally — drop-one doesn't leak the halving's held-in items.

Defaults are all `1` / `None`, so existing configs behave identically.

### `AttributionTracker.analyze(..., n_repeats=1)`

Drop-one ablation now takes `n_repeats` and threads it through. The cache check is stricter: a passed-in `full_score` is only reused if it matches `len(examples)` *and* `n_repeats`. Otherwise we re-score (fresh compute) so the delta is apples-to-apples.

### `examples/full_bakeoff.py` CLI flags

```bash
python3 examples/full_bakeoff.py \
    --ollama-url http://localhost:11434/api/chat \
    --models gpt-oss:20b gemma4:26b \
    --eval-repeats 3 \
    --attribution-repeats 3 \
    --attribution-screen-size 20
```

## Tests

14 new tests in `tests/test_reproducibility.py`:

- `TestMedianHelper` — median of odd / even / out-of-order / empty lists.
- `TestSingleRepeatUnchanged` — `n_repeats=1` sets metadata correctly and preserves behaviour.
- `TestMedianAggregation` — noisy-LLM produces single-shot variance; `accuracy_spread` reports it; deterministic-LLM median equals single-shot; `n_repeats=0` raises.
- `TestAttributionWithRepeats` — cache invalidated when repeats mismatch; cache hit when shape matches.
- `TestRunnerConfigWiring` — defaults are backward-compatible; `attribution_screen_size` threads through correctly.

Full suite: **63/63 tests pass**.

## Expected impact

At the cost of `eval_repeats × attribution_repeats` more LLM calls, the reproducibility fix should:

- Eliminate the spurious 0.15pt spread on prompt-equivalent harnesses seen in the gpt-oss:20b bakeoff.
- Move attribution `mean_delta` readings out of the noise floor at the same iteration count.
- Leave `accuracy_spread` as a visible signal for "this candidate is unreliable, don't admit it to the frontier without more eval budget" — a follow-up could gate frontier admission on spread ≤ ε.

We do **not** re-run the bakeoff on this branch; that's where the `real-dataset` branch comes in (phase C).

## Things this branch explicitly does NOT do (scope control)

- No change to the mock proposer or LLM proposer — both still see aggregated `ScoreVector`s with the same API.
- No change to the Pareto domination logic — `accuracy_spread` is reported but not yet used as a frontier gate.
- No change to the default values (all repeats default to 1). This branch is infrastructure; turning the knobs is per-experiment.
