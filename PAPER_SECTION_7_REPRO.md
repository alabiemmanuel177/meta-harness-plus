# Paper Section 7 — Reproducibility (Draft)

This is the prose draft of section 7. Length target: ~½ page.

---

## 7. Reproducibility

The framework, the experiments, the run logs, and the prompt caches
are all open-source.

### 7.1 Code

`https://github.com/[user]/Meta-Harness` (released at submission).

- Single Python package `meta_harness_plus/` (~3500 lines).
- 267 unit tests, all passing. Run with
  `python3 -m unittest discover -s tests`.
- Standard library only — no torch, no numpy, no scipy. urllib for
  HTTP. Runs on any Python 3.10+ host without environment
  management.

### 7.2 One-line replication

Every cell of the experiment grid is a one-line shell command. The
6×8-budget Gemini × news_hard_50 cell, for example:

```bash
python3 examples/rag_vs_mh_bakeoff.py \
    --api gemini --models gemini-2.5-flash-lite \
    --task news_hard_50 \
    --run-name gemini_news_hard_50_seed0_big \
    --iterations 6 --proposals 8 \
    --eval-size 50 --screen-size 8 \
    --halving-k0 4 --halving-final-keep 2 \
    --eval-repeats 2 --attribution-repeats 2 \
    --attribution-screen-size 15 \
    --cache-path runs/cache/gemini_news_hard_50_big.jsonl \
    --max-workers 8 --screen-seed 0
```

The full multi-seed cross-provider grid is one shell script:
`bash examples/run_5seed_2task_2provider.sh`. The LawBench cell:
`bash examples/run_lawbench_5seed.sh`. The ablation:
`bash examples/run_ablation_full_scale.sh`. The beat-CoT
experiment: `bash examples/run_gemini_news_beat_cot.sh`.

### 7.3 Run logs and aggregates committed

Every per-seed run dir under `runs/<run-name>/` contains:
`comparison.json` (per-cell summary), `<model>/bakeoff_summary.json`
(framework state), `<model>/frontier.json` (final Pareto frontier),
`<model>/history.jsonl` (every search iteration's diagnostic),
`<model>/attribution_stats.json` (per-kind EWMA stats),
`<model>/candidates/<id>/{harness, score, attribution}.json` (every
candidate's full state).

All aggregate JSON files (multi-seed CIs, paired t-tests, per-class
deltas) are committed under `runs/*_aggregate.json` at the repo
root. Commits are tagged with the experiment name. Full git log is
the experimental timeline.

### 7.4 Prompt caches

`runs/cache/*.jsonl` are SHA-256-keyed JSONL append-only caches of
every LLM call's response. Re-running an experiment with the same
cache file produces 80–93% cache hit rates on follow-on seeds, so
the multi-seed × multi-provider grid total cost is under $1.50
USD.

The caches are committed to git so a reviewer can replay the exact
LLM calls we issued (verifying our scores) without paying for any
API calls themselves. **A reviewer who wants to verify a single
seed's number can do so for $0 in API spend** — just point the run
at the committed cache file.

### 7.5 Statistical aggregation reproducibility

`examples/multi_seed_cross_provider.py` accepts a list of seed run
dirs, computes the paired bootstrap CI, paired t-test, and Cohen's
d, and writes the aggregate JSON. The script is deterministic
given the seed (n=2000 bootstrap resamples with seed=0 fixed).
Running this script reproduces every CI/p/d in the paper exactly.

### 7.6 Compute footprint

Total experimental cost (API spend, all multi-seed × multi-provider
× multi-budget runs combined): **under $1.50 USD**. Total wall
time on a single 8-core machine: ~6 hours. The framework was
designed for reproducibility on commodity hardware with no GPU
required and no large-model API access required.
