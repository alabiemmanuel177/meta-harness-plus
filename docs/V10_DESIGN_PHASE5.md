# V10 Phase 5 Design — Candidate Selection

**Status:** drafted; lands together with P5b implementation.
**Author:** harness team.
**Phase 4 status (context):** validator package + 35 unit tests landed; produces `CandidateView` per candidate.
**Phase 3 status (context):** P3e dev_100 routed eval landing concurrently.
**Scope:** Phase 5 only.

---

## 1. Mission

Phase 5 turns one or more `CandidateView` per instance into a single
patch submission. It picks among candidates using only signals on
`CandidateView` — never the grader's verdict, never FAIL_TO_PASS,
never PASS_TO_PASS.

Per `docs/V10_TARGET_60PCT_PRO_SPEC.md` §3.5:

> **Requirement:** ≥90% correct-patch picked when one exists in K
> candidates.

The selector exists primarily for the routed pipeline where K=1 per
instance (one strategy chosen by the router). In V0 selection is
near-trivial — the single candidate IS the submission. The
infrastructure is built for future K>1 work (e.g., when the
leaderboard run lands K=4 pipeline + K=2 agent per instance).

---

## 2. The contamination model for Phase 5

Selection is the V8 leak class — V8 picked candidates by reading the
grader's `resolved` field. V10 forbids this structurally.

### 2.1 Hard rules

  1. **Selectors take ONLY `CandidateView`** (and the
     instance's `InstanceView` for repo conventions). No raw
     `PatchCandidate`, no grader output, no FAIL_TO_PASS list.
     Enforced by `tests/test_no_oracle_leak.py` (already
     pre-Phase 0).
  2. **No selector imports `harness.eval`.** Per the firewall
     pattern from Phase 3/4. Lands in
     `tests/test_phase5_firewall.py`.
  3. **No selector imports `harness.repro`.** The repro test
     content is firewalled; selectors see only
     `CandidateView.repro_signal.status` (a string status enum).
  4. **Selector decisions are deterministic given inputs.**
     Same `CandidateView[]` → same selection. No LLM
     non-determinism in the selector core; if Selector A
     (LLM reviewer) is enabled, its output is hashed-cached
     for reproducibility.

### 2.2 New cross-phase firewall tests

Added in P5b at `tests/test_phase5_firewall.py`:

  - `test_no_phase5_module_imports_harness_eval`
  - `test_no_phase5_module_imports_harness_repro`
  - `test_no_phase5_module_imports_harness_memory`
  - `test_phase5_selectors_take_only_candidate_view` —
    AST scan: every selector function in `harness.selection` has
    its first non-self argument typed as `Sequence[CandidateView]`
    or `list[CandidateView]`.

---

## 3. Architecture

### 3.1 Three selectors (per V10_DESIGN.md §3.6)

  - **Selector A — LLM reviewer.** Sonnet (or DeepSeek for dev).
    Rubric: minimal, plausible, passes own repro, no public
    regression. Outputs ranked list with rationale. Runs in V0
    only when K>1.
  - **Selector B — Cluster-then-vote.** Group candidates by
    `CandidateView.cluster_id` (which is the AST-normalized hash
    truncated to 12 chars by Phase 4). Pick the largest cluster's
    representative by smallest diff. AST-hash equality is the
    clustering signal; tie-breaker is `diff_stats.additions +
    deletions` ascending.
  - **Selector C — Heuristic.** Highest
    `(repro_signal.status == "pass") AND
    (public_suite_signal.new_failures_count == 0)` wins. Ties
    broken by smallest diff, then by `static_signal.ruff_clean`
    (True > None > False).

### 3.2 Combination

Per V10_DESIGN.md §3.6: weighted vote with weights tuned on
dev_50, locked before dev_100. V0 weights:

```python
WEIGHTS = {
    "selector_a": 0.5,
    "selector_b": 0.25,
    "selector_c": 0.25,
}
```

Each selector emits a per-candidate score in [0, 1]. The final
pick is `argmax(sum(w_i * score_i))`. Ties broken
deterministically by `candidate_id` ascending.

When K=1 (the V0 routed regime), all selectors return 1.0 for the
sole candidate and the combination is trivial. The infrastructure
is built and tested but invisible.

### 3.3 Escalation

Per spec §3.6: disagreement among selectors → escalate to Opus
reviewer with full `CandidateView` for all candidates. Escalation
budget: 15% of instances.

V0 leaves escalation as a stub that always returns the
heuristic-selector's choice. Real escalation lands when K>1 is
the production regime.

### 3.4 Module layout

```
harness/selection/
  __init__.py        — public API: select, SelectionResult
  selector_a_llm.py  — LLM reviewer (DeepSeek by default)
  selector_b_cluster.py — cluster-then-vote
  selector_c_heuristic.py — heuristic
  combiner.py        — weighted vote + escalation hook
  views.py           — SelectionResult dataclass

tests/test_phase5_selector_b.py
tests/test_phase5_selector_c.py
tests/test_phase5_combiner.py
tests/test_phase5_firewall.py
```

Selector A's tests (LLM reviewer) are integration-style — mocked
LLM. Lands in P5b.

---

## 4. Acceptance criteria

  1. **dev_50 selection accuracy** when K>1: ≥90% of instances
     where at least one CandidateView resolves the bug, the
     selector picks a resolving candidate. Measured against the
     grader verdict (eval-only). For V0 with K=1, this gate is
     trivially met.
  2. **Determinism:** selector returns the same choice on
     repeated invocations with the same `CandidateView[]`.
     Tested with a property test.
  3. **Firewall green:** all four new firewall tests pass; no
     harness.selection module imports harness.eval / harness.repro
     / harness.memory.
  4. **Per-instance cost ≤$0.05** when Selector A is active.
     V0 K=1 has no selector cost.

---

## 5. V0 simplification

Phase 5 V0 implements the three selectors and the combiner but
defers the LLM reviewer to a stub when K=1 (saves cost). The
heuristic + cluster selectors run unconditionally. When K=1 the
choice is the sole candidate; the selector infrastructure exists
to support K>1 in the leaderboard run without code changes.

P5c lands the dev_50 selection-accuracy measurement only after
the leaderboard run produces K>1 candidates per instance. For
the dev_100 V0 number, selection is a passthrough.

---

## 6. Implementation order

  - **P5a** — this design doc. Lands together with P5b.
  - **P5b** — `harness/selection/` package + 4 test files.
  - **P5c** — dev_50 K>1 selection-accuracy measurement.
    Optional in V0 (lands when the leaderboard run K>1 is
    in scope).

---

## 7. Cross-references

  - V10_DESIGN.md §3.6 — original Phase 5 brief.
  - V10_TARGET_60PCT_PRO_SPEC.md §3.5 — the ≥90% requirement.
  - V10_DESIGN_PHASE4.md §2.2 — firewall pattern this doc inherits.
  - harness/views.py — `CandidateView`, the only shape selectors see.
