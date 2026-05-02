# Model swap plan for V10 (commit 16b)

This is the plan-of-record for which model serves which role across
V10 development and the final test_500 publication runs. Created in
response to the strategy revision: skip per-phase Sonnet/Opus
ablations during Phase 2/3/4/5 development; develop everything on
DeepSeek; do TWO final test_500 runs (cheap baseline + leaderboard
mixed-model). Saves ~$300 in dev-time API costs and simplifies the
plan.

## TL;DR

  - **Dev-time YAML config (`harness/config/models.yaml`):** locked
    at `deepseek-chat` for every role, including
    `selection_escalation_reviewer` (was `claude-opus-4-7`; updated in
    this commit).
  - **Per-phase sanity checks on dev_50:** DeepSeek throughout. Not
    model ablations — just "does this phase produce sensible numbers"
    verifications at each phase boundary.
  - **Final test_500: TWO runs.** DeepSeek end-to-end (the
    reproducibility-headline cheap baseline), and a mixed-model
    Claude config (the leaderboard headline). Both are completed
    AFTER all phases ship. No mid-stream swaps.

## What changed vs the prior plan

The original plan (commit 12c, then revised in 13a) was:

  1. Test_500 DeepSeek headline (~$1.20, ~24h)
  2. Test_500 Sonnet ablation on the rerank step alone (~$20, ~4h
     using cached retrieval)

That was the **Phase 1 evidence package.** It still runs; the
Phase 1 layer's reranker model swap stays as-is.

What the strategy revision adds: when Phase 2/3/4/5 ship, we do NOT
re-ablate Sonnet/Opus per phase. Instead, **all of Phase 2-5 develops
on DeepSeek**, and the final test_500 leaderboard run swaps Claude
into the right roles all at once.

Why this is cheaper:

  - Per-phase Sonnet ablation on dev_100: ~$20 each phase × 4 phases
    = ~$80
  - Per-phase Opus ablation on dev_100: ~$50 each phase × 4 phases
    = ~$200
  - Skipping these saves ~$280 in dev-time API costs.
  - The penalty is we don't know per-phase whether Claude beats
    DeepSeek BEFORE the leaderboard run. Acceptable — dev_100 lift on
    Phase 1 (12b: +2-3pp top-1) is informative; we don't need the
    same evidence at every phase boundary.

## Final test_500 — Run 1 (DeepSeek end-to-end, cheap baseline)

Reproducibility headline. Single run, no variance bars (DeepSeek is
0pp self-variance per 12a).

| Role | Model | Source |
|---|---|---|
| `reranker` | `deepseek-chat` | `models.yaml` default |
| `repro_generator` | `deepseek-chat` | YAML default |
| `repro_verifier` | `deepseek-chat` | YAML default |
| `patch_generator_pipeline` | `deepseek-chat` | YAML default |
| `patch_generator_agent` | `deepseek-chat` | YAML default |
| `patch_minimizer` | `deepseek-chat` | YAML default |
| `selection_reviewer` | `deepseek-chat` | YAML default |
| `selection_escalation_reviewer` | `deepseek-chat` | YAML default |

**Invocation:** no env-var overrides; the YAML defaults serve.

```bash
PYTHONPATH=. .venv/bin/python3 scripts/run_pipeline_test500.py \
    --split splits/test_500.json \
    --workers 1 \
    > /tmp/test500_deepseek_e2e.log 2>&1
```

(`scripts/run_pipeline_test500.py` lands when Phase 3 / Phase 4 ship;
flags TBD.)

**Cost projection** (rough, will be refined by a preflight commit
when Phase 3 lands):
  - Phase 1 retrieval+rerank: $1.20 (measured)
  - Phase 2 repro generation: ~$50-75 ($0.10-0.15/inst × 500 with
    the §8.6 $0.30 cap; expect most below $0.10)
  - Phase 3 patch generation: TBD — heavily depends on agent-path
    fraction; bounded above at $35/inst × 500 = $17,500 worst case.
    Realistic with hybrid pipeline+agent: ~$5-10/inst → $2.5K-5K
    total.
  - Phase 5 selection: ~$5-10 (single reviewer call per instance,
    DeepSeek is cheap).

## Final test_500 — Run 2 (mixed-model Claude, leaderboard headline)

> **Update (2026-05-03):** the **leaderboard run targets SWE-bench
> Pro, not Verified.** The Pro public test set is the externally-
> recognized leaderboard surface (731 instances, more recent and
> stricter than Verified). Verified Run 1 (DeepSeek end-to-end on
> `splits/test_500.json`) remains the reproducibility-headline cheap
> baseline; Run 2 swaps to `splits/test_pro.json` *and* to the mixed-
> model Claude config below. The role/env table is unchanged. The
> per-instance pipeline (retrieval → repro → patchgen → selection)
> runs identically; the dataset adapter (`harness.dataset` +
> `Sandbox` Pro detection, see `V10_DESIGN.md` §13.2.1) is the only
> machinery that needed Pro support, and it has shipped (see commit
> log around 2026-05-03 and `tests/test_dataset_pro.py` for the
> firewall + projection coverage).

Leaderboard headline. Reuses the retrieval cache from Run 1's
**Verified** instances where possible (the cache is keyed by
instance_id — Pro instance_ids do not overlap, so this is a clean
fresh cache for Pro). Per-instance checkpoint signatures include
the role's model name so Run 2's checkpoints don't collide with
Run 1's.

| Role | Run 1 (DeepSeek) | Run 2 (Mixed) | Env override for Run 2 |
|---|---|---|---|
| `reranker` | `deepseek-chat` | `deepseek-chat` | (none — DeepSeek elsewhere) |
| `repro_generator` | `deepseek-chat` | `deepseek-chat` | (none) |
| `repro_verifier` | `deepseek-chat` | `deepseek-chat` | (none) |
| `patch_generator_pipeline` | `deepseek-chat` | **`claude-opus-4-7`** | `V10_PATCH_GENERATOR_PIPELINE_MODEL` |
| `patch_generator_agent` | `deepseek-chat` | **`claude-opus-4-7`** | `V10_PATCH_GENERATOR_AGENT_MODEL` |
| `patch_minimizer` | `deepseek-chat` | `deepseek-chat` | (none) |
| `selection_reviewer` | `deepseek-chat` | **`claude-sonnet-4-5`** | `V10_SELECTION_REVIEWER_MODEL` |
| `selection_escalation_reviewer` | `deepseek-chat` | `deepseek-chat` | (none) |

**Invocation** (env-var-driven; YAML untouched):

```bash
V10_SPLIT=pro \
V10_PATCH_GENERATOR_PIPELINE_MODEL=claude-opus-4-7 \
V10_PATCH_GENERATOR_AGENT_MODEL=claude-opus-4-7 \
V10_SELECTION_REVIEWER_MODEL=claude-sonnet-4-5 \
PYTHONPATH=. .venv/bin/python3 scripts/run_pipeline_test500.py \
    --split splits/test_pro.json \
    --workers 1 \
    --signature-suffix _leaderboard_pro_mixed \
    > /tmp/test_pro_mixed_leaderboard.log 2>&1
```

**Cost projection** (rough):
  - Patch-gen pipeline + agent at Opus rates: ~3-5× DeepSeek's
    patch-gen line. If DeepSeek patch-gen is $5-10/inst → Opus is
    $15-50/inst → $7.5K-25K for 500 (assuming Phase 3 actually
    fires the agent path on most instances).
  - Sonnet selection_reviewer: ~$0.04/inst × 500 = $20.
  - Total Run 2: dominated by patch-gen Opus cost. Will need a
    proper preflight before launch.

**Cost cap** (per instance): $35 (V10_DESIGN.md §10) still applies.
Even on Opus, the per-turn budget guard fires at 80% of cap and the
agent finalizes its best partial. The $35 hard ceiling × 500 = $17.5K
upper bound; we expect realistic 30-50% of that.

## Why mixed-model and not Opus end-to-end

  - **Reranker** stays DeepSeek: 12b showed Sonnet only nudges top-1
    by +2-3pp at 16-17× cost. Top-10 is identical at 98%. The
    downstream patch generator picks from top-K with K probably > 1
    on Phase 3, so the rerank top-1 lift may not propagate.
  - **Repro generator** stays DeepSeek: §8.6 of the Phase 2 design
    caps repro generation at $0.30/inst. Opus would blow past this
    on a single attempt; Sonnet would too. DeepSeek is the only
    model that fits the budget cleanly. If Opus repro turns out to
    be a dramatic quality win, that's a Phase 7 follow-up.
  - **Patch generator (both pipeline and agent paths)** swaps to
    Opus: this is where capability matters most. Patch generation
    is the load-bearing step; the marginal $/instance is justified.
  - **Patch minimizer** stays DeepSeek: minimization is mechanical
    pruning of accepted patches. Opus quality lift is unlikely to
    matter; cost would dominate.
  - **Selection reviewer** swaps to Sonnet: choosing among K
    candidate patches benefits from a stronger judge, but doesn't
    need Opus-tier reasoning. Sonnet is the right cost-quality
    tradeoff here.
  - **Escalation reviewer** stays DeepSeek (changed from the
    original V10_DESIGN.md §3.6 default of Opus): with Sonnet as
    the primary reviewer, escalation already steps up; promoting
    escalation to Opus would only matter on a small tail of
    contested cases. Cost-benefit doesn't justify it given the
    primary reviewer is now Sonnet, not DeepSeek. Re-evaluate after
    Phase 5 dev_50 numbers.

## Per-phase sanity checks (DeepSeek throughout)

At each phase boundary, run dev_50 with DeepSeek and verify the
phase's own acceptance gate. These are **NOT model ablations** —
they're "does the phase produce sensible numbers" checks.

| Phase | Sanity check | Acceptance criterion |
|---|---|---|
| 2 (repro) | dev_50 repro coverage | ≥60% fails-at-base |
| 3 (patch-gen) | dev_50 single-shot pass rate | TBD (per V10_DESIGN.md §3.4) |
| 4 (validation) | dev_50 false-pass rate | TBD |
| 5 (selection) | dev_50 selection accuracy on hand-graded sample | TBD |

If any sanity check fails on DeepSeek, the issue is in the phase's
implementation, not the model. Don't escalate to Sonnet/Opus before
debugging on DeepSeek — the cheaper model surfaces design flaws
faster.

## Models.yaml dev-time lock (this commit)

Per the strategy revision, every YAML role is set to `deepseek-chat`:

```yaml
roles:
  reranker:                       deepseek-chat
  repro_generator:                deepseek-chat
  repro_verifier:                 deepseek-chat
  patch_generator_pipeline:       deepseek-chat
  patch_generator_agent:          deepseek-chat
  patch_minimizer:                deepseek-chat
  selection_reviewer:             deepseek-chat
  selection_escalation_reviewer:  deepseek-chat   ← was claude-opus-4-7
```

The change to `selection_escalation_reviewer` (from `claude-opus-4-7`
to `deepseek-chat`) is the only YAML diff in this commit. The
mixed-model leaderboard run reaches Opus / Sonnet via env-var
overrides, NOT via the YAML default.

## Rollback plan

If the leaderboard mixed-model run produces a number that's WORSE
than the DeepSeek baseline (which would be surprising given Phase 1
+2-3pp evidence), the rollback is:

  - Stop the leaderboard run.
  - Report the DeepSeek baseline number as the V10 result.
  - Open a follow-up audit on why the model swap regressed (likely
    candidates: patch-gen prompt is over-tuned to DeepSeek's
    quirks; Sonnet selection-reviewer disagrees more often than
    DeepSeek's).
  - DeepSeek-only stays the published headline until the regression
    is understood.

## Cost summary

  - **Dev-time saving** (this revision): ~$280 (skipped per-phase
    Sonnet/Opus ablations across Phase 2-5).
  - **Leaderboard run delta** vs DeepSeek baseline: TBD; bounded
    above by §10 budget (V10_DESIGN.md §10).
  - **Total V10 LLM spend so far** (Phase 1 + audits): ~$30 (12a
    $0.24 + 12b $7.97 + 13b $0.04 + Phase 1 dev_50/dev_100 retrieval
    runs ~$15 + test_500 DeepSeek run in progress at ~$1.20).

## Cross-references

  - `docs/audits/model_agnosticism_audit.md` (commit 16a) — the
    structural enforcement that makes this swap plan possible.
  - `harness/config/models.yaml` — the role-to-model mapping that
    this plan locks at deepseek-chat.
  - `docs/audits/test500_reranker_decision.md` (commit 13a) — the
    Phase 1 reranker decision; supersedes nothing here, just covers
    the rerank-step ablation that already ran.
  - `tests/test_model_agnostic.py` (commit 16a) — fails CI if
    anyone hardcodes a model name in business logic.
