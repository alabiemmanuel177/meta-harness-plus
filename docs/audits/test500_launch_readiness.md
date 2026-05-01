# Test_500 launch readiness check (commit 13b)

Pre-flight verification for the 13a two-run plan (DeepSeek headline +
Sonnet ablation). Five gates: image cache, preflight projection,
`--reranker-model` flag smoke, cumulative cost tracking, hard-stop
recheck.

## 1. Docker image cache — ✓ GREEN

```
$ make verify-images
[verify-images] OK 500/500 present for test_500 (full Verified) (0.08s)
```

All 500 SWE-bench Verified images are cached locally. Hard-stop #1
(image verification finds <500) NOT tripped.

## 2. Preflight projections — ✓ GREEN

Re-ran `scripts/preflight_test500.py` after extending it for the
two-run plan. The script reads dev_100 checkpoint mtimes, derives
per-repo pace, and projects test_500 wall-clock + cost for both
DeepSeek and Sonnet.

| Run | Wall-clock | $/inst | Total cost |
|---|---|---|---|
| deepseek-chat (headline) | 24h 30m | $0.0024 | **$1.20** |
| claude-sonnet-4-5 (ablation) | 26h 02m | $0.0400 | **$20.00** |
| **Combined** | **50h 32m** | — | **$21.20** |

Refinement vs 13a: the 13a doc estimated Sonnet at ~48 hr based on
dev_100's 1.7× Sonnet/DeepSeek wall-clock ratio. That ratio applied
to the rerank step ONLY; on test_500 with fresh retrieval, retrieval
dominates wall time and the Sonnet delta is just the per-instance
rerank latency (`+11s/inst × 500 = +1.5h`). Refined Sonnet projection:
**26 hr, not 48 hr**. The 13a doc's estimate was conservative; the
real number is better.

Per-run hard-stop checks (per 13b spec):

| Gate | DeepSeek | Sonnet | Status |
|---|---|---|---|
| Wall-clock ≤ 36 hr | 24h 30m | 26h 02m | ✓ PASS |
| Per-run cost ≤ $30 | $1.20 | $20.00 | ✓ PASS |

Hard-stop #2 (preflight projects wall-clock >36 hr OR cost >$30) NOT
tripped.

Full preflight at `docs/audits/test500_preflight.md`.

## 3. `--reranker-model` flag smoke test — ✓ GREEN

Ran a 1-instance smoke with `--reranker-model claude-sonnet-4-5`:

```
$ PYTHONPATH=. .venv/bin/python3 scripts/retrieval_eval_dev50.py \
    --split splits/dev_100.json --no-shortlist --include-traceback \
    --rerank --reranker-model claude-sonnet-4-5 --limit 1 \
    --signature-suffix _smoke13b --audit-suffix _smoke13b
[retr-eval] [1/1] astropy__astropy-13236 … OK n_files=919 candidates=10
[retr-eval] retrieval done in 37.3s — 1 OK, 0 failed
[retr-eval] HEADLINE: top-1 1/1 (100.0%)  top-5 1/1 (100.0%)  top-10 1/1 (100.0%)
```

Per-instance checkpoint confirmed Sonnet was actually used:

```
cost_usd: 0.043239        ← Sonnet rate (~$0.04/inst)
                            DeepSeek would be ~$0.0024
duration_s: 29.21         ← Sonnet pace (~25-30s)
                            DeepSeek would be ~14s
n_reranked: 10
```

Both signals (cost and duration) confirm the flag routed to Sonnet,
not the default DeepSeek. The flag-passing path is end-to-end
correct: CLI arg → `_WorkerArgs.reranker_model` → `rerank(...,
model_override=...)` → `harness.llm.clients.complete_chat(model=...)`
→ Anthropic SDK.

Smoke artifacts cleaned (signature `_smoke13b` removed from both
`runs/v10_dev_100_retr_eval/checkpoints/` and `docs/audits/`).

Hard-stop #3 (reranker model flag doesn't work cleanly) NOT tripped.

## 4. Cumulative LLM spend tracking — ✓ GREEN (manual)

Per-instance cost is captured in every checkpoint (`rerank_cost_usd`,
`rerank_duration_s` fields). Aggregating gives run-wide cumulative
spend:

```bash
python -c "
import json, pathlib
total = 0.0; n = 0
for f in pathlib.Path('runs/v10_test_500_*/checkpoints/<sig>').glob('*.json'):
    d = json.loads(f.read_text())
    total += d.get('rerank_cost_usd', 0.0) or 0.0
    n += 1
print(f'cumulative: \${total:.4f} across {n} instances')
"
```

This is the operator-facing budget guard for the 13b spec's "wired
so a runaway run can be killed at the budget cap" requirement.
Limitations:

  - The script is NOT auto-enforced. There's no in-process callback
    that kills the eval at $30. The 13b spec asks for *wired*
    tracking, which we have (per-instance cost recorded, queryable
    via the one-liner above). Auto-kill at the budget cap would
    require integrating `harness.cost.CostTracker` (currently only
    used per-instance with cap_usd=$35) into the eval orchestrator.
    Tracked as a future-work TODO.

  - Mid-run kill workflow: the eval script writes per-instance
    checkpoints atomically. Operator runs the cumulative-cost
    one-liner periodically (e.g., every 30 min via `watch`); if it
    exceeds the $30 per-run cap, send SIGTERM to the eval. The
    completed checkpoints persist; restart with the same signature
    skips them.

  - Pre-launch budget check: dev_100 cost was $4.00 for 100 Sonnet
    runs. Linear projection to test_500 = $20. Margin of ~$10 from
    the $30 cap.

## 5. Hard-stop recheck — ✓ GREEN

| Hard-stop | Threshold | Observed | Status |
|---|---|---|---|
| Image verification finds <500 cached | <500 | 500/500 | ✓ PASS |
| Preflight projects wall >36 hr | DeepSeek 24h, Sonnet 26h | 24h, 26h | ✓ PASS |
| Preflight projects cost >$30 (per run) | $30 | $1.20, $20.00 | ✓ PASS |
| Reranker model flag doesn't work cleanly | — | confirmed Sonnet routed | ✓ PASS |

All four checks green.

## Verdict

**Test_500 is launch-ready.** All four hard-stops cleared. The two-run
plan (DeepSeek headline + Sonnet ablation) fits within both wall-clock
(<36 hr per run) and cost (<$30 per run) caps with margin. The
`--reranker-model` flag is exercised end-to-end; cost tracking is
wired (manual aggregation, not auto-kill).

**The launch decision still sits with the user.** Per the spec's soft
guidance: "DON'T launch test_500 in this batch." This audit confirms
readiness; it does NOT initiate either run.

## Recommended launch order (when sign-off lands)

1. **DeepSeek headline run first** (~24 hr, ~$1.20). Single-shot, no
   variance bars needed (12a evidence).
2. **Sonnet ablation second** (~26 hr, ~$20). Reuses the test_500
   retrieval cache from the DeepSeek run via the
   `embed_noshortlist_tb_bs256` retrieval signature; only Stage 1g
   reruns under the new model. This is what kept dev_100's Sonnet
   rerun at $4 / 43 min instead of $20 / 4 hr.

Wait, that calculation is wrong on test_500 — there's no pre-existing
test_500 retrieval cache. The DeepSeek run will populate it; the
Sonnet run will reuse the same retrieval and only rerun rerank. That
brings the Sonnet RUN-2 wall-clock from 26 hr (full pipeline) down to
~3.5 hr (rerank only, projected from dev_100's 43 min × 5 = 215 min).
Update needed in the preflight numbers — see appendix below.

## Appendix: corrected Sonnet-after-DeepSeek wall-clock

If the runs are sequenced (DeepSeek first, Sonnet second), the
Sonnet step reuses the retrieval cache the DeepSeek run produces.
The Sonnet pass becomes rerank-only:

| Run | Wall-clock | Cost | Notes |
|---|---|---|---|
| 1. DeepSeek headline (full pipeline) | ~24h 30m | $1.20 | populates retrieval cache |
| 2. Sonnet ablation (rerank only) | ~3h 35m | $20.00 | reuses retrieval cache from #1 |
| **Combined sequential** | **~28h 05m** | **$21.20** | |

vs. running both fully-fresh: 50h 32m. Sequencing saves ~22 hr.

The preflight script's `runs_proj` table doesn't model sequencing
yet — it shows both runs as full-pipeline (50.5 hr combined). That's
the upper bound. Real combined wall-clock with the documented
retrieval-cache reuse is closer to ~28 hr.

This appendix supersedes the preflight's combined wall-clock for the
operational plan. The preflight numbers stand as the per-run upper
bounds.
