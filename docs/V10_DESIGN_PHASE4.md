# V10 Phase 4 Design — Validation

**Status:** drafted; lands together with P4b implementation.
**Author:** harness team.
**Phase 1 status (context):** stages 1a/1b/1c/1g locked; dev_100 retrieval is 84/97/98 top-1/5/10.
**Phase 2 status (context):** repro generator complete; dev_50 96% usable repro coverage.
**Phase 3 status (context):** P3e dev_100 routed eval landing concurrently with this design.
**Scope:** Phase 4 only. Phase 5 (selection) is referenced but not designed.

---

## 1. Mission

Phase 4 turns each `PatchCandidate` from Phase 3 into a `CandidateView`
— the structured object Phase 5's selectors consume to choose the
final submission per instance. The validator runs strictly mechanical
checks against the public test suite and static-analysis tools; it
never reads the grader's verdict, FAIL_TO_PASS, PASS_TO_PASS, or any
oracle field.

Per `docs/V10_TARGET_60PCT_PRO_SPEC.md` §3.4:

> **Requirement:** ≥95% accuracy at identifying broken patches.
>
> Mostly mechanical (run tests, count regressions, check static
> analysis). The hard part is doing it within wall-clock and cost
> constraints. Per-instance validation budget should be ≤2 minutes
> wall-clock, ≤$0.05.

The accuracy target is per-CANDIDATE, not per-instance. With K=2
pipeline + 1 agent candidate per instance under P3e routing,
validation runs ≤3 candidates/instance × 100 instances on dev_100.
Total budget: 5 min/instance wall × 100 = 8 hr; $0.15/instance × 100
= $15.

---

## 2. The contamination model for Phase 4

Phase 4 is the most leak-shaped phase by construction — it runs the
public test suite, parses pytest output, and produces signals that
will feed selection. Any leak here is a V8-class regression.

### 2.1 Hard rules

  1. **Phase 4 NEVER imports `harness.eval`.** The grader's verdict
     and the FAIL_TO_PASS / PASS_TO_PASS lists are out of bounds.
     Enforced by an AST scan in
     `tests/test_phase4_firewall.py::test_no_phase4_module_imports_harness_eval`.
  2. **Phase 4 NEVER imports `harness.repro`.** Repro test bodies
     are still firewalled — Phase 4 receives the
     `ReproSignal` (status only) via the orchestrator, not via
     module import. Enforced by the same firewall pattern as Phase 3.
  3. **The public-suite runner uses `view.test_directives` as the
     ONLY source of test paths.** No FAIL_TO_PASS-shaped selectors
     are ever passed to pytest. The runner uses default discovery
     within the directories declared by repo conventions
     (`harness/repo_conventions.py`). Same shape as
     `Sandbox.run_public_suite` (already firewall-audited).
  4. **The candidate's worktree state is reset between candidates.**
     Each candidate is applied to a CLEAN base_commit checkout, then
     the public suite runs, then the worktree is reset. Validating K
     candidates on the same instance must not leak state from one
     candidate's failures into another's run.
  5. **Forbidden tokens guard at log surfaces.** pytest stdout is
     scanned via `harness.repro._detect_forbidden_tokens` before
     being persisted to `PublicSuiteSignal.log_excerpt`. A hit logs
     a WARNING and the offending lines are redacted; the run does
     NOT abort (matches the §8.4 input-vs-output split from Phase 2).

### 2.2 New cross-phase firewall tests

Added to `tests/test_phase4_firewall.py` (new file; mirrors
`tests/test_repro_firewall.py` shape):

  - `test_no_phase4_module_imports_harness_eval`
  - `test_no_phase4_module_imports_harness_repro`
  - `test_phase4_runs_public_suite_via_sandbox_method_only` —
    AST scan confirming no Phase 4 module shells out to pytest
    via `run_shell` directly; all test invocations go through
    `Sandbox.run_public_suite`.
  - `test_phase4_no_test_target_arguments` — string-literal scan
    that no Phase 4 module's source contains a literal `"-k"` or
    `"--collect-only"` flag (catches refactors that try to whittle
    the suite to a smaller set, which is the V7 leak shape).

---

## 3. Architecture

### 3.1 Module layout

```
harness/validation/
  __init__.py        — public API: validate_candidate, validate_all
  apply.py           — apply_clean(...) → bool + AST-normalized hash
  public_suite.py    — run_public_suite(view, sandbox) → PublicSuiteSignal
  static.py          — run_static(view, sandbox) → StaticSignal
  diff_stats.py      — compute_diff_stats(diff) → DiffStats
  validator.py       — orchestrates the per-candidate flow

tests/test_phase4_apply.py
tests/test_phase4_public_suite.py
tests/test_phase4_static.py
tests/test_phase4_diff_stats.py
tests/test_phase4_validator.py
tests/test_phase4_firewall.py
```

The package mirrors the Phase 3 layout. Single-entrypoint contract:
the orchestrator (`validate_candidate`) is the only path the eval
script invokes; sub-modules are not called directly by anything
outside `harness.validation`.

### 3.2 Per-candidate flow

Per V10_DESIGN.md §3.5, ordered with each step gating the next:

  1. **Apply clean.** `git apply --check` then `git apply`. If apply
     fails → emit a `CandidateView` with `repro_signal=None`,
     `public_suite_signal.suite_ran_at_base=False`, and
     `notes="apply_failed"`. The selector treats this as the
     worst possible candidate.
  2. **AST-normalized diff hash.** Tokenize the diff with
     `tokenize`, drop whitespace + comments, hash. Used for
     clustering in Phase 5 (Selector B); irrelevant to Phase 4
     accuracy.
  3. **Repro signal.** Already produced by Phase 2 — Phase 4
     receives it via the orchestrator. Phase 4 does NOT regenerate
     or re-run the repro test (that lives in Phase 2 territory).
     The signal is forwarded into `CandidateView.repro_signal`.
  4. **Public-suite delta.** Two suite runs:
       a. Run public suite at base_commit → record passing tests
          (the baseline). Cache by (repo, base_commit) so multiple
          candidates per instance share one baseline.
       b. Run public suite at base_commit + applied candidate → diff
          against baseline. Compute `new_failures_count` (regressions)
          and `new_passes_count` (rare gains).
     **Flake-retry budget: 2 retries on per-test failures**, then
     trust the result. Per-instance suite timeout: 8 min (covers
     95% of repos based on V7 sandbox numbers per V10_DESIGN.md
     §3.5). Beyond 8 min → mark `suite_ran_at_base=False` and
     emit a degraded signal.
  5. **Static analysis.** `ruff` if `pyproject.toml` declares it,
     `mypy` if `mypy.ini` exists. Two booleans on
     `StaticSignal`. Skipped (None) if not configured for the
     repo. Per-tool timeout 60s.
  6. **Diff stats.** Files touched, +/- line counts, AST-normalized
     hash. All from the diff text alone — no sandbox needed.

After all steps: assemble and return a `CandidateView`. The view's
`__post_init__` runs the field-name firewall.

### 3.3 Caching

The baseline public-suite run is shared across candidates for the
same instance. `runs/<run-name>/validation_cache/<iid>/baseline.json`
holds:

```json
{
  "instance_id": "...",
  "passing_tests": ["test_foo", "test_bar.py::test_baz", ...],
  "failing_tests": [...],
  "duration_s": 47.3,
  "produced_at": "2026-05-06T12:34:56Z"
}
```

Hit on a second-candidate run for the same instance → skip the
baseline run (~50% of validation cost is the baseline; caching
halves per-instance cost when K=2+).

### 3.4 Wall-clock budget per candidate

Per spec §3.4: ≤2 min wall, ≤$0.05.

  | Step | Budget |
  |---|---|
  | apply | 5 s |
  | diff_stats + ast_hash | 1 s |
  | public suite (cache miss) | 90 s |
  | public suite (cache hit) | 60 s (just the patched run) |
  | static (ruff + mypy) | 60 s |
  | overhead (sandbox setup, log persistence) | 15 s |
  | **per-candidate total (cache miss)** | **~3 min** |
  | per-candidate total (cache hit) | ~2 min |

Slightly over the spec's 2-min target on cache miss. Acceptable
because:
  - Cache hits are >50% (one baseline shared across K candidates).
  - The 8-min suite cap means the worst case is bounded.
  - The grader (Phase 0 already-run `harness.eval.grade`) takes
    ~52 s/instance; Phase 4's overhead is comparable.

### 3.5 Cost budget per candidate

Spec target: ≤$0.05.

Phase 4 has NO LLM CALLS in V0. All cost is sandbox CPU + storage.
Budget compliance is trivial.

If Bet B (adversarial test generation, V10_DESIGN.md §4.2) lands as a
Phase 4 component in a later iteration, the budget gets revisited.
For V0 it's deferred.

---

## 4. Validation accuracy — the ≥95% target

**Definition:** "broken" = fails any FAIL_TO_PASS test or breaks any
PASS_TO_PASS test. Phase 4 must classify candidates correctly as
broken vs likely-correct WITHOUT reading FAIL_TO_PASS or
PASS_TO_PASS.

Phase 4's signals proxy for the ground truth via:

  - `apply_clean_succeeded` → if False, definitely broken.
  - `repro_signal.status == "pass"` → strong correctness signal
    (the candidate at least fixes the bug we localized).
  - `public_suite_signal.new_failures_count == 0` → candidate
    didn't break any public test. Necessary but not sufficient
    for correctness on hidden tests.
  - `static_signal.ruff_clean / mypy_clean` → weak surface signals.

The selector (Phase 5) combines these. Phase 4's accuracy target is:
when the selector says "broken" (i.e., apply failed OR public
regressions OR repro failed), it should be right ≥95% of the time.

The dev_50 measurement (P4c): hand-grade 50 candidates against the
gold's FAIL_TO_PASS verdict (eval-only path, single firewall-allowed
read). Compute the confusion matrix:

  - True positive: Phase 4 says broken, grader agrees.
  - False positive: Phase 4 says broken, grader says correct.
  - False negative: Phase 4 says fine, grader says broken.

Accuracy = (TP + TN) / total. Target ≥95%.

---

## 5. Acceptance criteria for Phase 4 → Phase 5 advance

  1. **dev_50 validator accuracy ≥95%.** Confusion matrix in
     `docs/audits/dev_50_validation_accuracy.md`. False-negative
     rate (Phase 4 says fine but grader says broken) is the binding
     metric — these are the patches that would slip through Phase 5
     and reach the leaderboard as wrong submissions.
  2. **Per-candidate wall-clock ≤3 min p90.** Cache hit case ≤2 min
     p90.
  3. **Per-candidate cost ≤$0.05.** With no LLM calls in V0 this
     is trivial; cost = sandbox CPU + disk only.
  4. **Firewall green:** all four new firewall tests pass; no
     Phase 4 module imports `harness.eval` or `harness.repro`;
     no `-k` / `--collect-only` flags; no direct pytest shell-outs.
  5. **Cache hit rate >40%** when validating ≥2 candidates per
     instance (sanity check that caching works as designed).

If (1) drops below 90%, the design needs rethinking — the public-
suite signal alone may not be discriminative enough. Adversarial
tests (Bet B) become the obvious next signal source, but only after
the V0 baseline is measured.

---

## 6. Open questions (resolved or deferred)

### 6.1 Should the validator regenerate the repro test?

**Resolved: NO.** Phase 2 generates the repro and produces a
`ReproSignal`. Phase 4 receives the signal via the orchestrator
and forwards it into `CandidateView.repro_signal`. Regenerating
inside Phase 4 would duplicate work and risk drifting from the
Phase 2 contract.

### 6.2 Should the validator run the suite at base_commit on every
candidate?

**Resolved: NO.** Cache the baseline per (instance_id, base_commit).
The first candidate pays the baseline cost; subsequent candidates
get it free.

### 6.3 Adversarial test generation (Bet B)?

**Deferred.** Out of V0 scope. The spec §4.2 cross-cutting
capability mentions adversarial tests as a self-verification
strengthener; if dev_50 accuracy <95% in P4c, adversarial generation
becomes a P4-followup component.

### 6.4 Property-based / mutation testing?

**Deferred.** Same scope decision as Bet B.

### 6.5 (NEW) What about cap-hitter instances Phase 3 skipped
(ContextOversizeError)?

**Resolved: emit empty CandidateView.** Phase 3 returns no
candidate for these; Phase 4 has nothing to validate. Phase 5
sees no CandidateView for the instance and falls back to the
"submit nothing" sentinel (the empty patch is graded as a
broken submission, which is honest about our coverage).

---

## 7. Implementation order

  - **P4a** — this design doc. Lands as part of the P4b commit
    or just before. No separate stop point per the user's
    revised resume sequence (commit P3d-fix message).
  - **P4b** — `harness/validation/` package + 6 new test files +
    4 new firewall tests. Validator orchestrator end-to-end with
    a fake-sandbox unit test exercising every gate.
  - **P4c** — dev_50 validator-accuracy measurement. Read the
    grader's verdict ONCE per candidate (eval-only path) and
    compute the confusion matrix. Audit at
    `docs/audits/dev_50_validation_accuracy.md`.
  - **P4d** (optional) — dev_100 validator-accuracy verification.
    Only fires if P4c shows the dev_50 accuracy near the 95%
    threshold and we want a 2× N variance reduction.

---

## 8. Cross-references

  - `docs/V10_DESIGN.md` §3.5 — the original Phase 4 brief.
  - `docs/V10_TARGET_60PCT_PRO_SPEC.md` §3.4 — the ≥95% accuracy
    requirement and the per-candidate budget.
  - `docs/V10_DESIGN_PHASE3.md` §2.2 — the cross-phase firewall
    pattern this doc inherits.
  - `harness/views.py` — `CandidateView`, `DiffStats`, `ReproSignal`,
    `PublicSuiteSignal`, `StaticSignal` already defined from
    Phase 0; Phase 4 is the producer.
  - `harness/sandbox.py` — `Sandbox.run_public_suite` (the only
    test-running primitive Phase 4 calls).
