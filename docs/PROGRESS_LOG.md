# V10 Phase 2 → Phase 5 PROGRESS_LOG

Append-only log of every commit landed during the long-horizon Phase 2-5
batch. Each entry records: commit SHA + subject, files touched, LLM
spend, dev_50 / dev_100 result if applicable, acceptance gate status,
and cumulative spend. Pushed after each entry so a remote observer can
monitor.

Spec: docs/V10_TARGET_60PCT_PRO_SPEC.md (capability spec, §3 per-phase
gates, §4 cross-cutting capabilities, §7 execution framing).
Phase 2 design: docs/V10_DESIGN_PHASE2.md.
Model swap plan: docs/MODEL_SWAP_PLAN.md.

## Hard-stop budget tracker

| Item | Cap | Used | Remaining |
|---|---|---|---|
| Cumulative LLM spend (this batch) | $150.00 | $0.16 | $149.84 |
| Largest single dev_50 ablation | $30.00 cap | $0.16 (17d-run) | — |
| Phase 2 dev_50 iteration count | ≤5 | 1 (PASS @ 96%) | 4 |
| Phase 3 dev_50 iteration count | ≤8 | 0 | 8 |
| End-to-end dev_100 iteration count | ≤5 | 0 | 5 |

## Branch lineage

- v10/phase-1 — frozen at tag `v10-phase-1-complete` (652b26f).
- v10/phase-2 — branched off v10/phase-1 HEAD (06e4427).
- v10/phase-3 — TBD (will branch off Phase 2 tip).
- v10/phase-4 — TBD.
- v10/phase-5 — TBD.

## Entries (newest first)

### P3a — `docs/V10_DESIGN_PHASE3.md` design doc (2026-05-03)

**STOP for human ack before any Phase 3 code lands.**

Files: `docs/V10_DESIGN_PHASE3.md` (new).

LLM spend: $0.

Doc covers: contamination model (5 hard rules + 4 firewall test
extensions); architecture (pipeline path single-shot + agent path ACI);
3 routes with V0 simplification (everyone takes pipeline first);
PatchCandidate schema; cost model (~$50 total for full Phase 3 dev);
acceptance criteria (40% pipeline-only floor, 50% full-routing,
±5pp generalization to dev_100); module layout under `harness/patch_gen/`.

Five §11 open questions surfaced for explicit ack before P3b lands.

### 17d-run — dev_50 repro coverage measurement (2026-05-03)

Commit SHA: TBD on push. Branch: v10/phase-2.

Files: `docs/audits/dev_50_repro_coverage.md` (new),
`runs/v10_dev_50_repro_coverage/*.json` (50 instance checkpoints +
JSONL traces).

LLM spend: **$0.16** (vs. $15 budgeted; two orders of magnitude under).

Cumulative batch spend: $0.16.

**Headline: 48/50 (96.0%) usable repros.** Acceptance gate (≥60%)
**PASS by +36pp**.

Key data points:

  - Every usable repro accepted at attempt 0; attempts 1 and 2
    contributed zero acceptances on dev_50. Tilts the §8.2 N=2-vs-N=3
    decision toward N=1, but defers final call to dev_100 per spec.
  - Two failures, both `passes_at_base` (model wrote a test that
    didn't exercise the bug). The safe failure mode — no false-positive
    bug evidence enters the pipeline.
  - The 96% rate counts must-fail-at-base only; per Phase 2 §8.5 we
    accept "broken-repro" noise (test fails at base AND fails after
    fix) — Phase 5 selection's downstream signals will wash this
    out. The §7e audit (post-hoc, eval-only path) measures the
    actual broken-repro fraction.
  - Per-repo coverage: 100% on 10/12 repos; astropy 3/4 (75%) and
    matplotlib 4/5 (80%) the only repos with misses.
  - Cost p50/p90/max: $0.0029 / $0.0052 / $0.0104 per instance.
    The §8.6 $0.30/instance cap was never close to firing.
  - Wall-clock: 6.6 min serial. p50 7.4s/instance.

§4 capability check: this commit moves us closer to **§4.2
(self-verification beyond the base test)** — repro signal is the
core pre-validation pass/fail signal Phase 5 will use.

Acceptance gate: **PASS** (≥60%, hit 96.0%).

### 17d — dev_50 repro coverage script + verifier path-resolution fix (2026-05-03)

Commit SHA: c5210c5. Branch: v10/phase-2.

Files: `scripts/repro_coverage_eval.py` (new, 410 lines),
`harness/repro.py` (modified — verifier `_resolve_test_path` +
`_normalize_target_test_id` helpers), `tests/test_repro_retry.py`
(modified — 7 new path-resolution tests).

LLM spend: $0 (smoke run was 1 instance, $0.0027, included in 17d-run total).

Notable bug found and fixed during 1-instance smoke on
`astropy__astropy-12907`:

  - Old `_verify_repro_at_base` prepended `test_dirs[0]` to
    `case.test_filename` unconditionally, producing
    `astropy/astropy/modeling/tests/...` when the model emitted a
    full repo-relative path. Result: every repro attempt reported
    ImportError → 0% coverage on every instance.
  - New helpers handle 3 model output shapes (bare basename, path
    starting with declared test_dir, full repo-relative path
    elsewhere). System prompt tightened to ask for repo-relative
    path explicitly + new-file-only.

100/100 repro tests pass. 37/37 V10 baseline + smoke-strict pass.

§4 capability check: necessary infrastructure for §4.2 self-verification.

### Prefatory — SWE-bench Pro infrastructure + 60% capability spec (2026-05-03)

Commit SHA: 56fc1a3. Branch: v10/phase-2.

Files: `docs/V10_TARGET_60PCT_PRO_SPEC.md` (new),
`docs/PROGRESS_LOG.md` (new), `harness/dataset.py` (Pro split
registry), `harness/views.py` (`dockerhub_tag` field), `harness/sandbox.py`
(Pro image routing), `scripts/verify_images.py` (Pro preflight),
`scripts/build_pro_split.py` (new), `splits/test_pro.json` (new, 731
instances), `tests/test_dataset_pro.py` (new, 8 tests),
`tests/test_dataset_and_conventions.py` (whitelist updated).
`meta_harness_plus/tasks/data/swebench_pro.jsonl` (new, 24MB Pro corpus).

LLM spend: $0.

Lands the in-progress Pro-support work that was sitting in the
working tree at the start of the batch + the spec doc + the
PROGRESS_LOG scaffold.

37/37 V10 unit tests + 8/8 Pro adapter tests + smoke-strict pass.

§4 capability check: §4.4 (domain adaptation) infrastructure for
running on Pro instances.

