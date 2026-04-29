# V10 Design — SWE-bench Verified, oracle-free, leaderboard-eligible

**Status:** draft, awaiting review.
**Author:** harness team.
**Target:** 70%+ Pass@1 on SWE-bench Verified (test-500), oracle-free, $25–35/instance.
**Stretch:** 75–80%, contingent on Phase 7 (fine-tuned localizer).
**Scope:** This doc supersedes V7/V8. V10 is a clean rebuild of the contamination model and pipeline; we keep the sandbox plumbing (`agent_docker.py`, `swebench_adapter.py` shell) but rewrite anything that touches prompt construction or candidate selection.

---

## 1. Honest framing

We are an independent team building a public harness against opponents who have either (a) frontier-lab privileges, (b) fine-tuned models on private corpora, or (c) both. The brief's stretch target (75–80%) is achievable in principle but requires Phase 7. **The headline number we should plan around is 70%+ on test-500 within budget, oracle-free, with full reviewer rollouts shipped.** Anything above that is upside.

Two lessons we are taking from V7/V8 into V10:

- **V7's leak** was *prompt-string*: FAIL_TO_PASS selectors were pasted into the actor and reviewer prompts (`swebench_v7.py:146`, `:25`, `:32`). Stripping them drops V7 from 63.6% to ~52%.
- **V8's leak** was *selection-side*: `resolved_lookup` (`run_swebench_500_v8.py:67`, `:247`) consults the evaluator's `resolved` verdict to pick which seed wins per instance. This is invisible to a "no forbidden string in prompts" grep. The firewall must catch this class, not just prompt strings.

Everything in V10 is designed around those two failure shapes.

---

## 2. Contamination model

We define oracle leakage as **any pathway by which information derived from FAIL_TO_PASS, PASS_TO_PASS, hints_text, gold patch, or the evaluator's verdict influences the patch we submit**. There are three pathways and the firewall has to address all three.

| Pathway | Where it leaks | V7/V8 example | V10 mitigation |
|---|---|---|---|
| Prompt | Any string sent to an LLM | V7 prompts include FAIL_TO_PASS selectors | `InstanceView` dataclass; only `InstanceView` reaches prompt builders |
| Selector input | What the chooser sees per candidate | V8 `resolved_lookup` keys on eval verdict | `CandidateView` dataclass; selectors take only `CandidateView` |
| Test execution | Which tests we run during validation | "Run FAIL_TO_PASS" instructions to the agent | We run the **public** test suite at base_commit; we never read the FAIL_TO_PASS list |

### 2.1 The two views

```python
# Frozen, hash-stable, hand-audited. NO eval fields.
@dataclass(frozen=True)
class InstanceView:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str         # the issue text
    repo_skeleton: RepoSkeleton    # files, classes, functions; no test bodies
    test_directives: TestDirectives  # which dirs/globs the public suite lives in
    # NOTHING ELSE. No FAIL_TO_PASS, PASS_TO_PASS, hints_text, patch, test_patch, version.

@dataclass(frozen=True)
class CandidateView:
    candidate_id: str
    diff: str                                # the patch
    diff_stats: DiffStats                    # files touched, lines +/-, AST-normalized hash
    repro_signal: ReproSignal | None         # our own repro test result; never the official one
    public_suite_signal: PublicSuiteSignal   # delta vs base_commit on the **public** suite
    static_signal: StaticSignal              # ruff/mypy
    # NOTHING ELSE. No `resolved`, no `report`, no eval verdict.
```

Both views are `frozen=True`, both have `__post_init__` that asserts no field name matches a forbidden token.

### 2.2 The firewall test

`tests/test_no_oracle_leak.py` enforces three things:

1. **Type-level:** `InstanceView` and `CandidateView` carry no field whose name or annotation references forbidden symbols (`FAIL_TO_PASS`, `PASS_TO_PASS`, `hints_text`, `patch` excluding `diff`, `resolved`, `report`).
2. **Dataflow (static):** AST scan of every prompt builder confirms it accepts *only* `InstanceView` (and, where applicable, retrieval/skeleton results). Same for selectors, mutated to `CandidateView`. Any function that takes `dict` and gets called from a prompt builder fails the test.
3. **Dataflow (runtime):** A pytest fixture that wraps every LLM client call and every selector call. Inspects the call's keyword args; if any object exposes a forbidden field, fail. Runs in CI and in `make eval` for production runs.

The runtime check is the one V8 would have failed. The dataclass + AST check is the one V7 would have failed.

### 2.3 What we explicitly allow

- Reading the public test suite at `base_commit` and counting regressions on a candidate. **The dataset itself does not mark which tests are FAIL_TO_PASS — we don't peek at the metadata.**
- Generating our own reproduction tests from the issue text alone (Phase 2). These are *our oracle*; they may be wrong, and that's fine.
- Running `git log` / `git blame` at `base_commit`. No reading of post-fix commits.

### 2.4 What we explicitly forbid

- Accessing `swebench.harness.constants.FAIL_TO_PASS` or any equivalent. The only place this string should appear in V10 source is in the firewall test that forbids it.
- Loading `report.json` or evaluation result JSON during selection. Selectors only take `CandidateView`.
- Cross-instance memory built from candidates labeled "correct" by the evaluator. We may build cross-instance memory labeled by *our own* repro/public-suite signals.

---

## 3. Architecture

### 3.1 Phase 0 — Foundation

Concrete deliverables, in order:

1. `harness/views.py` — `InstanceView`, `CandidateView`, `RepoSkeleton`, `DiffStats`, `ReproSignal`, `PublicSuiteSignal`, `StaticSignal`. Frozen dataclasses, validated.
2. `harness/sandbox.py` — wraps `agent_docker.py`. Per-instance container at `base_commit`, deterministic seeds where possible, public-suite runner (timeout + flake-retry budget).
3. `harness/trajectory.py` — every LLM call, tool call, selector decision → `trajectories/{instance_id}/{candidate_id}/turn_NNN.jsonl`. Write-through, gzip on close.
4. `harness/cost.py` — per-stage token + dollar tracker, exposes a per-instance ceiling and graceful degradation hook.
5. `harness/dataset.py` — loads SWE-bench Verified, projects each row to `InstanceView`. The projection is the *only* place forbidden fields are dropped, and it lives behind an asserted boundary.
6. `tests/test_no_oracle_leak.py` — the firewall (see §2.2).
7. `splits/` — three frozen splits:
   - `dev_50.json` — for component sweeps. Hand-picked: 15 easy, 20 medium, 15 hard, repo-diverse.
   - `dev_100.json` — held out from dev-50; for decision making after components are tuned.
   - `test_500.json` — full Verified; touched **once** for the headline number.

**Phase 0 is done when** the firewall test is green, the dataset adapter produces `InstanceView` only, the sandbox runs the public suite for one instance end-to-end, and `make eval` works on a clean machine.

### 3.2 Phase 1 — Hierarchical localization

The brief is mostly right. Notes:

- Stage 1a (skeleton): cache per `(repo, base_commit)`. Verified has ~20 instances/repo on average, so cache hit rate is high.
- Stage 1b (file candidates): keep BM25 + embedding, **drop "LLM picks top-10 from skeleton"** until ablation shows it adds over BM25+embedding+grep. LLM file picking is a known cost trap; it's only worth running once we have a strong reranker.
- Stage 1d (symbol grep) is the highest-signal cheap stage. Run first, weight heavily.
- Stage 1e (git archeology): cap to 50 candidate commits per file, summarize via cheap model, don't dump full diffs into reranker context.
- Stage 1g (LLM rerank): a single Sonnet call producing top-5 with rationale. Rationale is logged; rationale is **not** fed back into patch generation (avoids interpretation collapse).
- Stage 1h (function/line): only for files Selector A will see. Don't pre-compute for all 30 candidates.

**Localization output:** `LocalizationResult` with ranked `(file, function, line_range, score, signals_used)` entries and a `confidence` ∈ [0,1].

**Ablation entry:** dev-50 with each sub-component (1c, 1d, 1e, 1f, 1g) removed in isolation.

### 3.3 Phase 2 — Reproduction oracle

Three tests per instance, generated by a sub-agent that sees only the `problem_statement` and `repo_skeleton`:

1. **Repro test** — should fail at `base_commit`, pass after a correct fix. Up to 3 attempts. If we can't make it fail at base, we mark `repro_status=no_repro` and proceed with `repro_signal=None`.
2. **Property tests (×2)** — invariants the patched code must satisfy. **Demoted from primary to tie-break-only** based on pushback #6. They're cheap to run, but we don't gate selection on them.
3. **Negative test** — explicitly *passes* at base_commit and should still pass after the fix. Catches over-broad patches that delete functionality.

Critical: repro tests are part of `CandidateView.repro_signal`, never part of `InstanceView`. The patch generator does **not** see the repro test contents — only the patch generator's *own* reasoning about what tests should exist. Otherwise we over-fit to our oracle.

### 3.4 Phase 3 — Hybrid generation

**Routing decision** (per instance, made before any expensive work):

```
route = pipeline_only        if est_difficulty <= EASY_THRESHOLD
route = pipeline_then_agent  if EASY_THRESHOLD < est_difficulty < HARD_THRESHOLD
route = agent_first          if est_difficulty >= HARD_THRESHOLD
```

Three routes with two configurable flags:

- `easy_threshold` — initial value: bottom 30% of dev-50 difficulty distribution.
- `agent_first_threshold` — initial value: top 15% of dev-50 difficulty distribution.

Both calibrated on dev-50, locked before dev-100. `pipeline_only` skips the agent path entirely. `pipeline_then_agent` is the gated path (pipeline first, then agent only if pipeline candidates fail validation). `agent_first` skips the pipeline entirely — a $4 pipeline pass on the hardest 15% rarely produces a winning candidate, and we don't want to pay for it.

`est_difficulty` is from cheap signals computed once before expensive stages: issue length, traceback present, candidate-file count from Phase 1, multi-file-scope mentions in the issue, repo (django/sympy skew harder than flask). The heuristic is itself ablatable.

**Pipeline path.** K=4 candidates at temperatures (0.0, 0.3, 0.7, 1.0). Sonnet for ≤medium, Opus for hard. ~$1–3/instance.

**Agent path.** 2 candidates, two prompt framings (cautious / exploratory). **Turn budget gated by difficulty:**

- `T_max_easy = 15` — easy instances finish by turn 12 in published agentic systems
- `T_max_medium = 30` — covers most published agentic baselines
- `T_max_hard = 50` — hard instances still make real progress past turn 30 (SWE-Agent, OpenHands logs)

Opus only on hard. Cost ~$2–3 (easy) / ~$4–8 (medium) / ~$14–20 (hard).

**Per-turn cost guard.** If cumulative per-instance spend reaches 80% of the per-instance cap ($28 of $35), force-finalize on the best partial candidate so far. Prevents a runaway hard instance from blowing the cap while still allowing legitimate long-horizon work. Logged as `degraded=true` in the trajectory so we can audit how often it fires.

**Pipeline gate (for `pipeline_then_agent`).** If pipeline produces a candidate that (a) applies cleanly, (b) passes its repro, (c) has zero public-suite regressions, and (d) all three selectors agree it's the pick, **we skip the agent entirely.** Expected to fire on ~30–50% of medium instances; we'll calibrate on dev-50.

### 3.5 Phase 4 — Validation

Per candidate, in this order, each gating the next:

1. Apply cleanly. If not → drop.
2. AST-normalized diff hash + cluster ID.
3. Repro signal: pass / fail / no_repro / errored.
4. Public-suite delta. **Flake-retry budget: 2 retries on per-test failures, then trust the result.** Per-instance suite timeout: 8 minutes (covers 95% of repos based on our sandbox numbers from V7).
5. Static (ruff if config exists, mypy if config exists). Boolean.
6. Diff stats.

Output is a `CandidateView`. No `resolved` field exists anywhere in this struct.

### 3.6 Phase 5 — Selection

Three selectors (per pushback #10, simpler than the brief):

- **Selector A — LLM reviewer (Sonnet).** Rubric: minimal, plausible, passes own repro, no public regression. Outputs ranked list with rationale.
- **Selector B — Cluster-then-vote.** k-means in code-embedding space (using `text-embedding-3-large` on the diff body). Pick the largest cluster's representative by smallest diff. AST hash is a tiebreaker, not the clustering basis.
- **Selector C — Heuristic.** Highest `(repro_passed AND public_regressions == 0)` candidate; ties broken by smallest diff, then by static-signal score.

Final pick: weighted vote with weights tuned on dev-50, locked before dev-100. Disagreement → escalate to Opus reviewer with full `CandidateView` for all candidates. Escalation budget: 15% of instances.

### 3.7 Phase 6 — Refinement (de-scoped)

Per pushback #5, refinement is one pass, not three:

- **Minimization** — keep. Shrink to smallest hunk that still passes repro + has zero new public-suite failures. Real signal in the literature.
- **Devil's advocate** — drop from Phase 6 default, move to `--experimental` flag. Run only on the dev-50 ablation; if it adds ≥1pt, promote to default.
- **One-shot polish** — keep, but it's a 200-token prompt to strip debug prints. Not a "phase" so much as a pre-submit lint.

### 3.8 Phase 7 — Fine-tuned localizer (escape hatch, scaffolded in Phase 0)

We **start collecting training data in Phase 0**, not in month 3. Pipeline:

1. Crawl public, pre-cutoff bug-fix commits (license-filtered) from a curated list of ~5k Python repos.
2. For each commit, emit `(issue_text, pre_commit_skeleton, fixed_files, fixed_functions, fixed_lines)`.
3. Hold this corpus zipped, ready to fine-tune a localizer head when we hit the dev-100 wall.

We don't *commit* to Phase 7 in V10, but we make it cheap to start. If dev-100 says we're at 65–70% with the rest of the stack, Phase 7 is the next move.

---

## 4. Original bets — chosen 2

Per pushback #7–9, I'm choosing **#3 (Repository pre-study) and #5 (Adversarial test generation)**.

### 4.1 Bet A — REPO_NOTES.md (#3)

**What.** Per-repo, one-time pass: study architecture, conventions, common patch patterns, where utilities live, how tests are organized. Output: `repo_cache/{repo}/REPO_NOTES.md`. ~500–1500 tokens. Inject as system context for all instances of that repo.

**Why.** Verified has high per-repo concentration (~20 instances/repo on average). Spending $1.50 once on REPO_NOTES amortizes to <$0.10/instance. The content is genuinely useful: "tests live in `tests/`, conventions use `assert_array_equal` not `==`, error classes inherit from `SympifyError`."

**Expected lift.** 1.5–3 pts honestly. The brief said 2–4; I'm shading down because some of this signal already shows up in Phase 1's skeleton.

**Cost.** ~$30 total amortized across all 500 instances. Negligible.

**Risk.** Low. Worst case: no lift, we eat $30.

**Ablation.** dev-50 with REPO_NOTES on/off.

### 4.2 Bet B — Adversarial test generation (#5)

**What.** After patch generation, model B (different prompt, possibly different model) writes tests that try to break each candidate. The union of the original repro + adversarial tests is the validation oracle.

**Why.** Property tests via Hypothesis are noisy (pushback #6). Adversarial tests are targeted: model B sees the patch and reasons about what edge it might miss. Better signal at similar cost.

**Expected lift.** 1.5–3 pts. The brief said 2–4; I'm shading down because adversarial tests can also be wrong-flag (rejecting correct patches), so we need a calibration on dev-50 before promoting to default.

**Cost.** ~$1–2/instance. Adds modestly to the budget.

**Risk.** Medium. Calibration matters — over-adversarial tests reject good patches. Mitigation: only treat adversarial-fail as a *negative signal* in selectors, not as a hard reject.

**Ablation.** dev-50 with adversarial generation on/off, and on/off-as-soft-vs-hard-signal.

### 4.3 Bets we considered and dropped

- **#1 cross-repo retrieval** — pushback #7. Without fine-tuning the retrieval-into-ICL signal is too weak. Defer to Phase 7.
- **#2 multi-interpretation patching** — pushback #8. Subsumed by temperature diversity. Drop.
- **#4 symbolic verification** — narrow domain (~10–20 instances). Real lift but tiny absolute. Defer to a stretch experiment.
- **#6 cross-instance pattern learning** — interesting, but the labeling story is fragile (we'd label by our own signals, which already feed selectors — risk of double-counting). Defer until we can label more reliably.

---

## 5. Cost model

Per pushback #2, the brief's design at K_pipe=5, K_agent=2, full Phase 6 was ~$35–55/instance. After de-scoping:

**Per-stage, per-instance cost** (unchanged from v1 except agent line):

| Stage | Easy ($) | Hard ($) | Notes |
|---|---|---|---|
| Phase 1 localization | 0.40 | 1.50 | Cached skeleton; Phase 1g reranker dominates |
| Phase 2 repro gen | 0.30 | 0.50 | Sonnet, capped at 3 attempts |
| Phase 3 pipeline (K=4) | 1.20 | 4.00 | Sonnet on easy, Opus on hard |
| Phase 3 agent (K=2) | 0 (skipped) | 14–20 | T_max_hard=50, per-turn cost guard at 80% cap |
| Phase 4 validation | 0.50 | 1.00 | Public suite + static |
| Phase 5 selection | 0.40 | 1.00 | Three selectors |
| Phase 6 minimization | 0.30 | 0.60 | Single pass |
| Bet A (REPO_NOTES) | 0.05 | 0.05 | Amortized over ~20 instances/repo |
| Bet B (adversarial) | 1.00 | 2.00 | Sonnet |

**Per-instance cost by route:**

| Route | Stages run | Cost |
|---|---|---|
| `pipeline_only` (easy) | loc + repro + pipeline + val + sel + min + A + B | ~$4 |
| `pipeline_then_agent`, pipeline wins | (no agent) | ~$4 |
| `pipeline_then_agent`, agent fires | + agent K=2 | ~$22–27 |
| `agent_first` (hard) | loc + repro + agent + val + sel + min + A + B (no pipeline) | ~$20–24 |

**Mix-weighted total under three-way routing** (dev-50 calibration target):

- 30% `pipeline_only` @ $4 → $1.20
- 55% `pipeline_then_agent` (assume 40–60% pipeline-success rate) → $9–14 weighted
- 15% `agent_first` @ ~$22 → $3.30
- + 15% Opus reviewer escalation: +$1.0
- + 10% retry/failure overhead: +$1.5–2.0
- **Expected: ~$17–22/instance, $35 hard cap with graceful degradation.**

That fits the $25 soft target on the optimistic side and clears the $35 hard cap with margin. **Test-500 budget: ~$8.5–11k headline, +$2k ablations and dev iteration → ~$10–13k all-in.**

If Phase 7 happens, add ~$3–5k for training data prep + compute.

---

## 6. Honest score build-up

Per pushback #1, the brief's ladder was independent-additive. Real behavior is multiplicative on residual error. I'm modeling it as:

```
final = 1 - (1 - baseline) * Π (1 - lift_i)
```

where `lift_i` is the *fraction of remaining errors fixed* by component i. With baseline 52%, residual 48%:

| Component | Optimistic err-fix | Pessimistic err-fix | Optimistic score | Pessimistic score |
|---|---|---|---|---|
| Baseline (V7 - leak) | 0% | 0% | 52.0% | 52.0% |
| + hierarchical localization | 25% | 12% | 64.0% | 57.8% |
| + repro oracle | 8% | 4% | 66.9% | 59.5% |
| + hybrid pipeline+agent | 10% | 5% | 70.2% | 61.5% |
| + multi-selector ensemble | 5% | 2% | 71.7% | 62.3% |
| + minimization | 3% | 1% | 72.6% | 62.7% |
| + Bet A (REPO_NOTES) | 4% | 1.5% | 73.6% | 63.3% |
| + Bet B (adversarial) | 4% | 1.5% | 74.6% | 63.9% |
| + Phase 7 fine-tuned localizer | 12% | 5% | **77.7%** | **65.7%** |

**Honest forecast: 66–74% without Phase 7, 70–78% with Phase 7.** That's the band I'd defend in a paper. The brief's 77–81% sits at the optimistic end of that range and assumes everything goes right.

We will treat **70% on test-500** as the success bar for V10. **75%+** is upside that we explicitly attribute to Phase 7 if we get there.

---

## 7. Ablation matrix

`make ablate` runs on dev-50, each cell logged with cost + score:

| Ablation | What's removed | Tells us |
|---|---|---|
| `full` | nothing | Reference |
| `no-loc-1c` | traceback parser | Localization signal value |
| `no-loc-1d` | symbol grep | (likely the highest-leverage cheap stage) |
| `no-loc-1e` | git archeology | Worth the cost? |
| `no-loc-1f` | dep-graph expansion | |
| `no-loc-1g` | LLM rerank | |
| `route-everyone-pipeline-first` | three-way routing → single threshold (everyone gets pipeline first, agent if pipeline fails) | Routing baseline #1 |
| `route-everyone-agent-first` | three-way routing → single threshold (everyone gets agent first, pipeline as fallback) | Routing baseline #2 |
| `route-three-way-tuned` | (default; nothing removed) | Confirms three-way is worth the calibration |
| `T-flat-30` | turn budget by difficulty → flat T_max=30 across all difficulties | Difficulty-gated turn budget worth it? |
| `T-by-difficulty` | (default; nothing removed) | |
| `pipeline-only` | agent path entirely | Pipeline-floor sanity |
| `agent-only` | pipeline path entirely | Agent-floor sanity |
| `no-repro` | Phase 2 | How much repro is worth |
| `no-property` | property tests only | (expected: ~0) |
| `no-adversarial` | Bet B | |
| `no-repo-notes` | Bet A | |
| `selector-A-only` | drop B,C | Reviewer alone |
| `selector-B-only` | cluster only | |
| `selector-C-only` | heuristic only | Floor |
| `no-minimization` | | |
| `no-escalation` | drop Opus reviewer escalation | |

We commit ablation results to `paper/ablations.md` after each run.

---

## 8. Eligibility

Tracked in `NOTES_ELIGIBILITY.md`, updated weekly:

- **Open source** — repo public from Phase 0 merge. Apache-2.0. `make eval` on clean machine.
- **arXiv preprint** — `paper/` from Phase 0. Sections drafted as phases land. Submit when test-500 number is in.
- **Reviewer rollouts** — `trajectories/` per instance, gzipped JSONL. Storage estimate: ~150MB per instance × 500 = 75GB compressed. Plan: HF dataset upload as the canonical artifact, S3 mirror as backup.
- **SWE-bench Pro** — `harness/pro/` adapter scaffolded in parallel; same core, only dataset/eval differs.

---

## 9. Working agreements

- **No phase merges without an ablation entry.** "We added it and the score went up" is not evidence.
- **Firewall is sacred.** If a technique only works because we read FAIL_TO_PASS "to debug," we kill it.
- **Dev split discipline.** Iterate on dev-50. Decide on dev-100. Touch test-500 once.
- **Cost cap is real.** $35/instance hard ceiling per instance with graceful degradation; ship best-so-far at the cap.
- **Trajectories from day one.** Every LLM call dumped. Storage is cheaper than re-running.
- **Negative results get committed.** If Bet B doesn't help, the ablation table says so and we drop it.

---

## 10. Phase 0 plan (next, after this doc is acked)

Concrete commits, in this order:

1. `harness/views.py` — both views, frozen, validated.
2. `tests/test_no_oracle_leak.py` — type-level + AST-level + runtime hooks, all three failing initially.
3. `harness/dataset.py` — `InstanceView` projection, asserted boundary.
4. Ports of sandbox + trajectory + cost from existing modules into the new `harness/` package, audited for forbidden field leaks.
5. `splits/dev_50.json` — hand-picked, repo-diverse, difficulty-balanced.
6. End-to-end smoke: one instance through the empty pipeline. No generation; just `InstanceView` → sandbox → public suite at base_commit → trajectory dump.
7. Firewall green. PR opened. Show diff.

I'll wait for ack on this doc before starting Phase 0.

---

## 11. Model substitution

We are testing on DeepSeek v4 first, then swapping to Claude Opus 4.6. Documenting expectations up front so we don't anchor on fantasy lifts.

### 11.1 Realistic delta

- **Same harness, DeepSeek v4 → Opus 4.6 swap: expect +5 to +10 points on Verified, NOT +15 to +20.** Public model deltas on agentic SWE-bench harnesses are smaller than people assume because the harness eats most of the variance once localization is solved. Once Phase 1 finds the right file with high probability, the patch generator is rarely the bottleneck — the bottleneck is whether the right edit can be derived from the localized context, and that's a function of localization quality more than raw model strength.
- If DeepSeek v4 hits 65–70% on V10, Opus 4.6 lands us in the **70–78%** band (matching §6).
- **To clear 80%, we still need Phase 7 (fine-tuned localizer) regardless of which model we swap in.** No frontier model gets us past the localization-quality wall on this benchmark without training data.

### 11.2 Model-agnostic call sites — concrete commitment

The harness must be **model-agnostic at the call site**. One env-driven config decides which model handles each role:

```yaml
# harness/config/models.yaml
roles:
  localizer:        ${LOCALIZER_MODEL:-claude-sonnet-4-6}
  reranker:         ${RERANKER_MODEL:-claude-sonnet-4-6}
  repro_gen:        ${REPRO_MODEL:-claude-sonnet-4-6}
  patch_gen_pipeline: ${PIPELINE_MODEL:-claude-sonnet-4-6}
  patch_gen_agent:  ${AGENT_MODEL:-claude-opus-4-6}
  reviewer:         ${REVIEWER_MODEL:-claude-sonnet-4-6}
  reviewer_escalation: ${ESCALATION_MODEL:-claude-opus-4-6}
  minimizer:        ${MINIMIZER_MODEL:-claude-sonnet-4-6}
  adversarial_test_gen: ${ADVERSARIAL_MODEL:-claude-sonnet-4-6}
```

**Hard rules:**

- No model name appears in any prompt string. Prompts reference `{role}` placeholders that the call site expands.
- No `if model == "claude-..."` branches in business logic. Model-specific tokenizer quirks live in a single adapter layer (`harness/llm/adapters/`).
- A new model is added by writing a thin adapter, not by editing prompts or pipeline code.
- A swap (DeepSeek → Opus) is a config change, not a code change. CI must include a smoke run against at least two providers to keep this honest.

### 11.3 What this enables

- We can run dev-50 ablations with DeepSeek v4 (cheaper, faster iteration), confirm the harness is sound, then swap to Opus 4.6 for the final headline number with one config flag.
- Reviewer rollouts include the model config, so reviewers can re-run any instance with the same role assignments.
- If a future Claude version ships, swap takes minutes, not days.

---

## 12. Sandbox audit (pre-Phase-0)

Before writing Phase 0 code, the user requested an audit of every test-invoking and test-list-constructing path in the existing `meta_harness_plus/agent_docker.py` and `meta_harness_plus/swebench_adapter.py`, with explicit proof of whether each path reads forbidden fields. **Findings: existing leak surfaces present in three classes of code path. Phase 0 must redesign around them, not just import them.**

### 12.1 `agent_docker.py` paths

| Path | Lines | Reads forbidden field? | Verdict |
|---|---|---|---|
| `_ensure_pytest()` | 123–132 | No — runs `python -m pytest --version` only | **clean** |
| `run_tests(test_targets, ...)` | 554–566 | The function itself does not read forbidden fields, but its **`test_targets` parameter is caller-supplied and the docstring (line 559) explicitly tells callers to pass FAIL_TO_PASS** ("…rarely what you want — typically pass FAIL_TO_PASS"). This is a leak-shaped API. | **leak surface** |
| `run()` | 154–185 | Generic shell exec — no field reads | **clean** |
| `run_python(code)` | 361–370 | Executes caller-supplied Python; no field reads inside the executor | **clean** (executor itself; caller-side risk) |
| `run_reproduction(code)` | 330–359 | Same as above | **clean** |
| `read_file` / `write_file` / `read_file_range` / `search_text` / `replace_text` / `go_to_definition` / `find_references` / `list_files` / `get_diff` / `create_checkpoint` / `reset_to_clean` / `restore_checkpoint` / `changed_files` | various | All file/git ops; no test-list construction | **clean** |

**Note on `get_diff(exclude_tests=True)` (line 570).** The comment at line 575 ("SWE-bench applies the hidden test patch during evaluation") is awareness of the gold test_patch's existence in the eval pipeline. The function itself does not read it; it just excludes test-dir paths from the submitted diff. **Clean, but the comment is a tell that callers know about the test patch — confirming the surrounding code is leak-aware.**

### 12.2 `swebench_adapter.py` paths

| Path | Lines | Reads forbidden field? | Verdict |
|---|---|---|---|
| `SWEBenchInstance` dataclass | 37–48 | **Carries `hints_text`, `test_patch`, `fail_to_pass`, `pass_to_pass`, `gold_patch` as fields.** The typed view itself is the leak surface. | **leak (severe)** |
| `load_swebench_verified(...)` | 51–103 | Lines 79–101 explicitly read `FAIL_TO_PASS`, `PASS_TO_PASS`, `test_patch`, `hints_text`, and `patch` from the dataset row and store them in `SWEBenchInstance`. | **leak (severe)** |
| `build_user_prompt(inst, include_hints=True, ...)` | 134–152 | Reads `inst.hints_text` (line 147) when `include_hints=True`. With v1's default this leaks `hints_text` directly into the LLM prompt. | **leak** |
| `extract_patch(text)` | 162–176 | Pure parser; no field reads | **clean** |
| `write_predictions(...)` | 181–200 | Writes the standard JSONL prediction format. No oracle reads. | **clean** |
| `run_swebench_eval(...)` | 205–269 | This is the **post-submission evaluator**. It calls the official `swebench.harness.run_evaluation` which legitimately reads FAIL_TO_PASS to grade. Its return value includes `resolved` and `report`. | **clean as evaluator, but its outputs must NEVER feed selection** (V8's bug was here). |

### 12.3 Leak findings — summary

Three existing leak surfaces, in order of severity:

1. **`SWEBenchInstance` is a leak-by-construction typed view.** It physically carries `fail_to_pass`, `pass_to_pass`, `test_patch`, `hints_text`, `gold_patch`. Any code path that takes a `SWEBenchInstance` has full oracle access. This is the V7 leak: `swebench_v7.py:146` reads `inst.fail_to_pass` straight into the actor prompt.
2. **`DockerShellExecutor.run_tests(test_targets)` API encourages leakage.** The docstring tells callers to pass FAIL_TO_PASS as the selector list. Callers across `agent_swebench_loop.py` (lines 215, 663, 991, 1014, 1025, 1115) duly do so, often quoting "FAIL_TO_PASS" by name in agent-facing prompts.
3. **`build_user_prompt(include_hints=True)` reads `inst.hints_text`.** `hints_text` is a forbidden field; this is a direct prompt leak via a default argument.

### 12.4 V10 redesign — what we change

Phase 0 cannot merely "wrap" `agent_docker.py` and `swebench_adapter.py`. Required changes:

**(a) Replace `SWEBenchInstance` entirely.**

`harness/dataset.py` projects raw dataset rows directly to `InstanceView`, dropping forbidden fields at the boundary. `SWEBenchInstance` is left in the legacy `meta_harness_plus/` tree for backwards compatibility with V7/V8 runners; **no V10 code imports it.** The firewall test enforces this.

**Tightening 1 (per ack):** the firewall AST scan flags **both static imports from forbidden legacy modules AND dynamic attribute access** that could reach forbidden fields. The scan rejects:
- `from meta_harness_plus.swebench_adapter import SWEBenchInstance` (and aliases)
- `from meta_harness_plus.swebench_v7 import *`
- `from meta_harness_plus.agent_swebench_loop import *`
- `meta_harness_plus.agent_docker.DockerShellExecutor.run_tests` references
- `getattr(x, "fail_to_pass")` / `getattr(x, "FAIL_TO_PASS")` / etc. — any `getattr` whose 2nd arg is a string-literal matching a forbidden token (case-insensitive substring)
- `x["fail_to_pass"]` / `x["FAIL_TO_PASS"]` / etc. — subscript access with forbidden token
- String literals inside `harness/` containing forbidden tokens (allow-listed: `harness/eval.py` may reference `swebench.harness.run_evaluation`; the firewall test itself may name forbidden tokens). All other matches fail the test.

The scan operates on AST, not regex, so cases like `getattr(x, "fail" + "_to_pass")` are not caught at that pass — defense in depth comes from the runtime guard (§12.5), which trips on the resolved string.

```python
# harness/dataset.py
def load_verified_views(...) -> list[InstanceView]:
    raw_rows = _load_raw_dataset(...)
    return [_project_to_view(row) for row in raw_rows]

def _project_to_view(row: dict) -> InstanceView:
    # The ONLY place a raw dataset row is touched. Asserted boundary.
    forbidden = {"FAIL_TO_PASS", "PASS_TO_PASS", "test_patch", "hints_text", "patch"}
    return InstanceView(
        instance_id=row["instance_id"],
        repo=row["repo"],
        base_commit=row["base_commit"],
        problem_statement=row["problem_statement"],
        repo_skeleton=_skeleton_for(row["repo"], row["base_commit"]),
        test_directives=_test_dirs_for(row["repo"]),  # repo conventions only
    )
    # No other fields are passed through. Test enforces no forbidden key reaches downstream.
```

**(b) Wrap `DockerShellExecutor` for primitives only; do not expose `run_tests`.**

`harness/sandbox.py` reuses `DockerShellExecutor` for container lifecycle, file ops, and `run()` (generic shell). It **does not expose `run_tests(test_targets)`**. Instead it provides:

```python
class Sandbox:
    def __init__(self, view: InstanceView): ...
    def run_public_suite(self, *, timeout_s: int = 480) -> SuiteResult:
        """Run the repo's public test suite at the directories in
        view.test_directives. Default pytest discovery — no test-name
        allowlist or denylist sourced from the dataset row."""
        cmd = f"python -m pytest -p no:cacheprovider --tb=short {' '.join(self._dirs)}"
        return self._exec.run(cmd, timeout_s=timeout_s)
    def run_repro(self, code: str) -> ExecResult: ...  # for our own repro tests
    # NO run_tests(test_targets) — that API is intentionally not exposed.
```

`self._dirs` is computed from `view.test_directives`, which in turn was derived in `harness/dataset.py` from **repo conventions only** (filesystem inspection of where `test_*.py` lives at base_commit, plus a hardcoded per-repo override map for known repos). It is never derived from a dataset field.

**Discovery-first semantics (revised in commit 7).** The override map is a *preferred default + sanity check*; **filesystem discovery inside the container at base_commit is the source of truth.** This handles repo refactors that happened mid-Verified-corpus: e.g., `psf/requests` Verified instances at older base_commits ship a top-level `test_requests.py` rather than a `tests/` directory. Discovery finds the actual layout; if the override and discovery disagree, we log a warning to the trajectory and proceed with the discovered set.

**Tightening 2 (per ack):** every entry in the per-repo override map at `harness/repo_conventions.py` carries a source comment citing the public artifact at base_commit it was derived from. Examples:

```python
REPO_TEST_DIRS: dict[str, list[str]] = {
    "django/django": ["tests/"],
    # Source: django/CONTRIBUTING.rst at sha 8f6a7a0... ("Run tests with `python tests/runtests.py`")
    "sympy/sympy": ["sympy/"],
    # Source: sympy/setup.cfg [tool:pytest] testpaths at sha 4f8b1c2...
    "pytest-dev/pytest": ["testing/"],
    # Source: pytest-dev/pytest pyproject.toml [tool.pytest.ini_options] testpaths
    # ...
}
```

Anyone reviewing the firewall can verify each entry traces to a public artifact, not to a dataset field. **Override-vs-discovery consistency check:** the test suite for `repo_conventions.py` runs filesystem fallback discovery on each known repo at its first-Verified base_commit and asserts the override matches what discovery would find. Mismatch fails the test — either the override is stale or filesystem discovery would find the answer anyway, in which case the override is unnecessary. This keeps the override list small and auditable.

**(c) Replace `build_user_prompt` and the entire prompt-builder layer.**

V10 prompt builders take only `InstanceView`. `include_hints` is removed; `hints_text` is not on `InstanceView`. The agent loop in `agent_swebench_loop.py` is **not carried forward** — Phase 3's agent is a clean rewrite that never references "FAIL_TO_PASS" in any string.

**(d) `run_swebench_eval` reused as-is, with strict isolation.**

It is the post-submission grader. Its return values (`resolved`, `report`) are written to `runs/<run_id>/eval_results.json` and **never read by any selector or prompt builder.** The firewall has a static test that no V10 module imports `run_swebench_eval` outside of `harness/eval.py`, and `harness/eval.py` itself is not callable from selectors.

### 12.5 Runtime guard inside the sandbox

**Tightening 3 (per ack): case-insensitive substring matching, expanded token list. Better to overflag and tune down than to miss.**

```python
# harness/sandbox.py
_FORBIDDEN_TOKENS = (
    # Base tokens — match any case via .casefold(); substring match.
    "fail_to_pass",
    "pass_to_pass",
    "test_patch",
    "hints_text",
    "hints",          # broader than hints_text; flags raw "hints" strings
    "gold_patch",
    "resolved",       # the V8 leak token: report.json's "resolved" verdict
)
# Variants matched automatically by casefold() + substring:
#   - SCREAMING_SNAKE_CASE:   FAIL_TO_PASS, PASS_TO_PASS, TEST_PATCH, HINTS_TEXT, HINTS, GOLD_PATCH, RESOLVED
#   - CamelCase:              FailToPass, PassToPass, TestPatch, HintsText, Hints, GoldPatch, Resolved
#   - mixed/lowercase/underscored: any composition with the underlying string still trips
# Also catches concatenation-after-resolution: getattr(x, "fail" + "_to_pass") -> "fail_to_pass" -> trips.

class OracleLeakError(RuntimeError):
    """Raised when a test selector or call argument contains a token matching
    a dataset-derived field name. Defense-in-depth alongside the static
    firewall (tests/test_no_oracle_leak.py)."""

class Sandbox:
    def _assert_no_forbidden_token(self, value, *, label: str) -> None:
        # Walk strings, lists/tuples of strings, and stringy objects. Any
        # casefolded substring match against _FORBIDDEN_TOKENS raises.
        if value is None:
            return
        if isinstance(value, str):
            haystacks = [value]
        elif isinstance(value, (list, tuple)):
            haystacks = [str(v) for v in value]
        else:
            haystacks = [str(value)]
        for hay in haystacks:
            folded = hay.casefold()
            for tok in _FORBIDDEN_TOKENS:
                if tok in folded:
                    raise OracleLeakError(
                        f"sandbox.{label}: argument contains forbidden token "
                        f"{tok!r} (matched in {hay[:120]!r}); this is an oracle "
                        f"leak surface. See docs/V10_DESIGN.md §12.5."
                    )

    def run_repro(self, code: str) -> ExecResult:
        self._assert_no_forbidden_token(code, label="run_repro.code")
        ...

    def run_public_suite(self, *, timeout_s: int = 480) -> SuiteResult:
        # Even though _dirs comes from TestDirectives (which is supposed to
        # be safe), we double-check at runtime. Cheap, catches mistakes.
        self._assert_no_forbidden_token(self._dirs, label="run_public_suite.dirs")
        ...
```

Same shape as the firewall test but at execution time. Logged + raised as `OracleLeakError`. CI greps `harness/sandbox.py` for `_assert_no_forbidden_token` to confirm the guard fires from every test-running path. The guard intentionally over-matches (e.g., `hints` will trip on benign strings containing the substring) — when that happens we prefer to rename the benign string rather than weaken the guard.

### 12.6 Eval output isolation (one-way pipe)

**Tightening 4 (per ack):** the post-submission grader's output is physically isolated from V10 selection code, both at the directory level and at the import level.

**Directory layout:**

```
runs/{run_id}/
  predictions.jsonl         # what V10 SUBMITS to the grader
  trajectories/             # written by V10 selection during candidate gen
  candidates.jsonl          # CandidateView per candidate (no `resolved`)
eval_outputs/{run_id}/
  report.json               # written by harness/eval.py ONLY
  per_instance/{iid}.json   # the grader's per-instance verdict
```

**Hard rules:**

- `harness/eval.py` is the only module that writes to `eval_outputs/`. Every other `harness/` module is forbidden from reading or writing under that path.
- The grader is a one-way pipe: V10 selection writes candidates → grader reads candidates and submission → grader writes verdicts to `eval_outputs/`. Nothing inside `harness/` reads back from `eval_outputs/`.
- The CI check in `tests/test_no_oracle_leak.py` performs three additional scans:
  1. AST scan of every `harness/` module (excluding `harness/eval.py`) for any string literal containing `eval_outputs` — fail if found.
  2. AST scan for any `open()`, `Path(...).read_*`, `json.load(`, `pickle.load(` whose first arg references `eval_outputs/` — fail if found.
  3. Filesystem-level: at the end of any pytest run that exercised the firewall fixture, scan `harness/` modules' actually-opened-paths via the runtime fixture's logger; assert none touched `eval_outputs/`.

This closes the V8 leak class for good and makes the boundary visible in the directory layout, not just in code.

### 12.7 V10 cache hygiene at startup

**Also per tightening 4:** any V10 cache directories must be either empty or contain only V10-tagged entries at startup. Cheap, prevents subtle reseeding leaks where a V7 prompt cache or trajectory dump contaminates a V10 run.

```python
# harness/cache.py
V10_CACHE_DIRS = [
    Path("runs"),                # only entries like runs/v10_*
    Path("trajectories"),        # only entries like trajectories/v10_*
    Path("repo_cache"),          # only entries like repo_cache/v10_*
    Path(".harness_cache"),      # internal harness cache
]
V10_TAG_PREFIX = "v10_"

def assert_clean_cache_at_startup() -> None:
    """Called from harness.__init__ once per process. Asserts that every
    directory in V10_CACHE_DIRS is either nonexistent, empty, or contains
    only entries whose names start with V10_TAG_PREFIX. Raises CacheLeakError
    otherwise."""
    for d in V10_CACHE_DIRS:
        if not d.exists():
            continue
        for entry in d.iterdir():
            if not entry.name.startswith(V10_TAG_PREFIX):
                raise CacheLeakError(
                    f"{d}/{entry.name}: pre-V10 cache entry. V10 caches "
                    f"must be V10-tagged. Move or delete legacy entries "
                    f"before starting a V10 run."
                )
```

`assert_clean_cache_at_startup` runs from `harness.__init__` so it fires before any other V10 code executes. CI exercises this in a test that seeds a non-V10-tagged file and asserts the import fails.

### 12.8 Conclusion of audit

**Existing leak paths exist in `agent_docker.py:run_tests`, `swebench_adapter.SWEBenchInstance`, `swebench_adapter.load_swebench_verified`, and `swebench_adapter.build_user_prompt`.** Phase 0 cannot proceed by importing these modules. The V10 `harness/` package replaces all four with leak-free analogues per §12.4.

The four tightenings (firewall AST scan for getattr/subscript, repo-conventions source citations + override-vs-discovery consistency, case-insensitive substring runtime guard, eval output isolation + V10 cache hygiene) are folded into §12.4–§12.7 and become Phase 0 acceptance criteria.
