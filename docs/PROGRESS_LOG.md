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
| Cumulative LLM spend (this batch) | $150.00 | $2.96 | $147.04 |
| Largest single dev_50 ablation | $30.00 cap | $2.74 (P3b-run) | — |
| Phase 2 dev_50 iteration count | ≤5 | 1 (PASS @ 96%) | 4 |
| Phase 3 dev_50 iteration count | ≤8 | 1 (FAIL @ 14%, soundness floor 40%) | 7 |
| End-to-end dev_100 iteration count | ≤5 | 0 | 5 |

## Branch lineage

- v10/phase-1 — frozen at tag `v10-phase-1-complete` (652b26f).
- v10/phase-2 — branched off v10/phase-1 HEAD (06e4427).
- v10/phase-3 — TBD (will branch off Phase 2 tip).
- v10/phase-4 — TBD.
- v10/phase-5 — TBD.

## Entries (newest first)

### P3b-run — dev_50 patch generation eval (pipeline-only, T=0 single-shot) (2026-05-03)

Commit SHA: TBD on push. Branch: v10/phase-2.

Files: `scripts/patch_gen_eval.py` (new — 480 lines, eval orchestrator
with checkpoint, prediction writer, grader invocation, audit emission),
`docs/audits/dev_50_patch_gen_eval.md` (new), `docs/PROGRESS_LOG.md`
(this entry).

LLM spend: **$2.74** (50 instances × ~$0.055/instance, K=2 at T=(0, 0.5)).

Cumulative batch spend: $2.96.

**Headline: 7/50 (14.0%) resolved.** Acceptance gate (≥40% pipeline-
only architecture-soundness floor): **FAIL by 26pp**.

Honest decomposition of the failure:

  - 50 candidates submitted, 33 completed, 17 failed at the
    `git apply` step (`Hunk FAILED at line N` errors → patch
    rejected by the grader).
  - Of the 33 that DID apply: 7 resolved the bug (21% applied-
    correctness rate). The remaining 26 applied cleanly but did
    not satisfy FAIL_TO_PASS.
  - Apply-rate (33/50 = 66%) is the binding constraint, NOT
    semantic correctness. The pipeline path produces structurally
    invalid diffs at a 34% rate.

Per-repo distribution (instances → resolved):
  - flask 1/1 (100%), psf/requests 1/2 (50%), pydata/xarray 1/3 (33%),
    astropy/astropy 1/4 (25%), django/django 3/12 (25%).
  - sympy/sympy 0/7, sphinx-doc/sphinx 0/5, scikit-learn/scikit-learn 0/5,
    pylint-dev/pylint 0/2, pytest-dev/pytest 0/3, matplotlib/matplotlib 0/5,
    mwaskom/seaborn 0/1.
  - The 5 repos at 0% are also the 5 with the largest median diff size in
    the gold corpus — strong correlation with "diff line numbers harder to
    get right". Suggests apply-failure is the dominant problem class.

Likely root causes (ranked):

  1. **Diff line-number accuracy.** The model is producing unified
     diffs against truncated file contents; line numbers in `@@`
     headers don't match the actual file. Truncation is the
     immediate suspect: `DEFAULT_PER_FILE_CHAR_CAP = 12_000` (~3K
     tokens/file). Files larger than that get cut, but the model
     still references line numbers from the truncated view.
  2. **Multi-file diffs are double-failing.** Most "Hunk FAILED"
     errors hit multi-file diffs where ALL hunks need to land. One
     bad hunk rejects the whole patch.
  3. **DeepSeek may be weaker at strict-format diff generation
     than at SEARCH/REPLACE blocks**, which is the format used
     in many recent agentic systems (Aider, OpenHands).

P3c (agent path with `apply_patch` tool) is the natural next step
because:

  - The agent can verify its diff applies BEFORE submitting.
  - The agent can read full files (not truncated), addressing
    cause (1).
  - The agent can iterate on a failed apply, addressing cause (2).
  - The 5 tools (`read_file`, `search_text`, `list_dir`,
    `apply_patch`, `submit`) are exactly what's needed.

**Stop and report numbers per the user's resume sequence.** Do NOT
proceed to P3c until acked. Open question: should P3b iterate
(switch from line-numbered diff to SEARCH/REPLACE format, or move
to per-file streaming on the cap-hitter instances) BEFORE moving to
P3c, or is going straight to P3c (where these issues largely
disappear via the apply_patch tool) the right move?

Recommendation: **straight to P3c**. The agent path is designed
specifically for this failure class; iterating P3b's line-numbered
diff format is unlikely to clear the 40% floor without architecture
help that P3c provides for free.

§4 capability check: this commit advances §4.1 (multi-file
reasoning) only weakly — the failures concentrate in multi-file
diffs (cause 2). P3c's agent path targets the same capability
more directly.

### P3b — pipeline-path implementation + firewall tests (2026-05-03)

Commit SHA: 2beebb1. Branch: v10/phase-2.

Files:
  - `harness/patch_gen/__init__.py` (new — public API).
  - `harness/patch_gen/views.py` (new — PatchCandidate +
    PatchGenContext + FileSnippet + 3 error classes).
  - `harness/patch_gen/context.py` (new —
    `build_patch_gen_context_with_superset_check`, the §2.2 single
    entrypoint).
  - `harness/patch_gen/pipeline.py` (new —
    `generate_pipeline_one_shot` + K-orchestrator
    `generate_pipeline`).
  - `tests/test_patch_gen_pipeline.py` (new — 29 unit tests).
  - `tests/test_repro_firewall.py` (modified — extended Phase 3
    candidate scan to walk `harness/patch_gen/` subdir + 3 NEW
    firewall tests: no-eval-imports, no-memory-imports, superset-
    assertion-present).

LLM spend: $0 (no LLM calls in implementation; all tests mock
`harness.llm.clients.complete_chat`).

Cumulative batch spend: $0.16.

170/170 V10 + repro + patch_gen + dataset tests + smoke-strict pass.

§4 capability check: this commit lands the structural scaffolding
for §4.1 (multi-file reasoning) — the pipeline path's K-candidate
generation surface enables temperature-diverse multi-file edits in
a single LLM call. The agent path (§4.1's other pillar) lands in
P3c.

P3b acceptance gate (≥40% pipeline-only correct on dev_50) lands
in the next commit (P3b-run) once the dev_50 patch-gen eval
completes.

### B6 + B7 resolution — parallel-session work + Co-Authored-By trailer (2026-05-03)

LLM spend: $0.

**B6 (parallel Pro-container WIP).** Self-resolved before this entry
landed. The other claude session (PID 24640, the May-02 session) had
been actively iterating on Pro container startup in parallel. While
the dev_50 repro coverage run was in progress, that session committed
AND pushed three prefatory commits ON TOP of P3a:

  - `6e6ba6f` — V10 prefatory: Pro sandbox runtime + 5-instance smoke
  - `4dfe092` — V10 prefatory: Pro image inventory audit (Step 4)
  - `627e03a` — V10 prefatory: design + model-swap docs for Pro
    leaderboard pivot

The Pro 5-instance smoke they wrote shows **5/5 PASS** (vs the 0/5 I
saw earlier mid-iteration — they fixed container startup before
committing). Working tree is clean, no overlap with my Phase 2/3 work,
no need to revert anything.

Going-forward operational note: there are TWO claude sessions on this
branch. Pull `origin/v10/phase-2` before each commit to avoid
non-fast-forward conflicts. If a third session joins, this strategy
will need revisiting.

**B7 (Co-Authored-By: Claude Opus 4.7 trailer).** No `.git/hooks/pre-commit`
file exists. No `~/.claude/hooks/` directory. The denial that hit the
prefatory + 17d + 17d-run + P3a commits came from Claude Code's
content classifier, not from a user-configurable hook. The classifier
read "user prohibits Claude/Anthropic in this batch" as a content-
integrity rule preventing Claude attribution.

Per user direction (B7): the rule is about LLM inference calls, not
about commit metadata. Attempting the trailer on P3b — if the
classifier still denies with the user's explicit authorization in
context, I'll fall back and document the irreconcilable case.

For commits already landed without the trailer (781165a, c5210c5,
11329dd, 9199f3d): leave as-is per user direction; pushed commits
shouldn't be amended.

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

