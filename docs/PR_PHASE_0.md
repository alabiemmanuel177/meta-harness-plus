# V10 Phase 0: contamination firewall + sandbox + dev-50 split

**Branch:** `v10/phase-0`, 21 commits.
**Base:** `main @ 134747a` (last pre-V10 commit)
**Diff:** 58 files, +17566 / −1
**Status:** Ready to merge. Negative smoke (the explicit pre-merge blocker) verified passing. All Phase 0 gates green.

---

## Reading order for leaderboard eligibility review

The diff is large because Phase 0 also tracked V7/V8 baseline source for documentation per V10_DESIGN.md §13.3. To make this reviewable in an afternoon rather than a week, read in this order:

1. **`README.md`** — project framing (V10 = clean leaderboard submission; V7/V8 = documented contaminated baselines).
2. **`docs/V10_DESIGN.md`** — the full design (12 sections + asset inventory). Especially §2 (contamination model), §12 (sandbox audit + four tightenings), §13 (asset inventory).
3. **`harness/`** — the V10 implementation (~3K lines). `views.py` (frozen dataclasses with field-name validators), `dataset.py` (single asserted projection boundary), `sandbox.py` (leak-free wrapper around DockerShellExecutor; runtime guard with case-insensitive substring match), `cache.py` (cache hygiene with `V10_EXCLUSIVE_DIRS` vs `V10_NAMESPACED_PARENTS` split), `eval.py` (post-submission grader, isolated to eval_outputs/).
4. **`tests/test_no_oracle_leak.py`** — the firewall (type + AST + runtime layers).
5. **`scripts/smoke_phase0.py`** and **`scripts/negative_smoke.py`** — proof the firewall fires (positive smoke runs end-to-end on psf__requests-2317; negative smoke confirms `OracleLeakError` raises on tampered InstanceView and forbidden state_label).
6. **`splits/dev_50.json`** + **`docs/audits/v7_missing_evals.md`** + **`docs/audits/empty_v7_context_budget.md`** — the calibration data and risk audits driving Phase 1 decisions.

The remaining ~9K lines are documented V7/V8 baseline data:
- `runs/swebench_500_v7/` — V7 run artifacts (predictions + 467 eval reports + per-instance trajectories).
- `meta_harness_plus/{swebench_v7,swebench_adapter,agent_swebench_loop,agent_docker}.py` + tests + example runners — the contaminated V7 source. The firewall test statically forbids any V10 module from importing the leak surfaces; the runtime guard would raise `OracleLeakError` if any of their fields ever reached an LLM call.

These are included for reproducibility per V10_DESIGN.md §13.3 ("V7's success/failure pattern is OUR observation about OUR runs — legitimate, oracle-free signal") but are NOT part of the V10 submission surface. **Ignore for review purposes.**

---

## What's in this PR

The clean rebuild of the SWE-bench Verified harness defined in `docs/V10_DESIGN.md`. Phase 0 establishes the contamination firewall, the leak-free sandbox, the dev-50 split, and the asset-readiness gates that Phase 1 inherits as preconditions.

### Commit list

| # | SHA | Subject |
|---|-----|---------|
| 1 | `53a722d` | design doc + frozen views (`InstanceView`, `CandidateView`, …) |
| 2 | `04c4a1d` | contamination firewall: type + AST + runtime |
| 3 | `846b298` | dataset projection + repo conventions |
| 4 | `7878a09` | sandbox + trajectory + cost + eval + cache primitives |
| 5 | `407f7bf` | dev-50 split (50 instances, 12 repos, 13/22/15 by difficulty) |
| 6 | `5169cf7` | end-to-end smoke + Makefile |
| 7 | `009800a` | discovery-first override semantics (handles psf/requests pre-migration) |
| 8 | `6fe263c` | real execution smoke (`--exec` flag) |
| 9 | `e94e099` | **negative smoke (the merge blocker)** |
| 10 | `246ee06` | cache TODO + design doc §10.5/§13 + Phase 7 scaffold |
| 11 | `0de0880` | `make verify-images` + image-required + dataset boundary |
| 12 | `75e03a9` | V7 missing-evals audit |
| 13 | `48de73a` | V7 difficulty priors |

---

## Spot-check evidence (the (a–e) the user requested)

### (a) `InstanceView` / `CandidateView` are `frozen=True` with working `__post_init__`

`harness/views.py`. Both views have `frozen=True`; both `__post_init__` calls `_assert_no_forbidden_field_names(type(self))` against the canonical `FORBIDDEN_TOKENS`. Verified by `tests/test_no_oracle_leak.py::test_attempting_to_construct_view_with_forbidden_field_raises`, which synthesizes a class with a `gold_patch` field and confirms `ForbiddenFieldError`.

### (b) Runtime fixture actually wraps LLM client calls

`tests/conftest.py:LLMCallInspector` is a callable wrapper. It intercepts `*args`/`**kwargs`, recursively walks dataclasses (declared fields AND `__dict__` extras for monkey-patched smuggling), dicts, lists, and strings via `_walk_value_for_tokens`, normalizes via `_normalize` (casefold + strip underscores so SCREAMING_SNAKE / CamelCase / canonical all converge), and raises `OracleLeakError` *before* invoking the wrapped function. Verified by 4 tests covering kwarg strings, dataclass field values, dataclass field names, and nested collections.

### (c) Dataset projection is the single asserted boundary

`harness/dataset.py:_project_to_view` is whitelist-only on `_PROJECTED_KEYS = {"instance_id", "repo", "base_commit", "problem_statement"}`. No subscript or `getattr` reads any oracle-derived key. `_assert_dataset_source_of_truth` enforces the local jsonl as the only legal source — no HuggingFace fallback. Verified by `test_project_to_view_only_uses_whitelisted_keys` which synthesizes a row with bogus oracle keys and confirms the projection ignores them.

### (d) Smoke test ran default pytest discovery on a real instance, trajectory landed

```
$ make smoke-exec
[verify-images] OK 50/50 present for dev_50.json (0.05s)
[smoke] instance_id=psf__requests-2317
[smoke] checking V10 cache hygiene (scoped: trajectories/)…
[smoke] loading InstanceView…
        repo='psf/requests'  test_dirs=('tests/',)  source='override:repo_conventions'
[smoke] starting sandbox at base_commit=091991be0da1…
[smoke] running public test suite (execute, timeout=480s)…
[sandbox:psf__requests-2317] override-vs-discovery mismatch:
  declared ['tests/'] but discovery found ['test_requests.py']. Discovery wins.
        effective_paths=('test_requests.py',)
        exit_code=0  duration=4.4s   142 tests collected
                     61 failed, 81 passed, 3 warnings in 4.12s
[smoke] trajectory landed: trajectories/v10_smoke/psf__requests-2317/phase0_smoke/turn_0000.jsonl
```

The 61 failures are network-dependent timeout tests (`--network none` in the sandbox); pytest itself ran cleanly.

### (e) Four review-acked tightenings in place

1. **Firewall AST scan** flags `getattr` / `setattr` / `hasattr` / `delattr` with forbidden string literals AND subscript access (`x["fail_to_pass"]`). Tests: `test_no_harness_module_uses_getattr_with_forbidden_string_literal`, `test_no_harness_module_uses_subscript_with_forbidden_string_literal`.
2. **Override map source citations** — every entry in `harness/repo_conventions.py:REPO_TEST_DIRS` has a `# Source:` comment naming the public artifact (CONTRIBUTING.rst / setup.cfg / pyproject.toml) plus a Verified base_commit example. Override-vs-discovery consistency exercised on synthetic checkouts and on the real psf/requests pre-migration case (commit 7).
3. **Case-insensitive substring runtime guard** — `harness/sandbox.py:_normalize` does casefold + strip underscores so all variants (SCREAMING_SNAKE, CamelCase, canonical) converge. Verified by `test_runtime_guard_substring_matches_screaming_snake`, `..._matches_camelcase`, `..._blocks_resolved_token`.
4. **Eval output isolation + V10 cache hygiene** — `harness/eval.py` is the only module that may read/write `eval_outputs/`; `test_no_harness_module_reads_eval_outputs` AST-scans every other module. `harness/cache.py:assert_clean_cache_at_startup` enforces no non-`v10_` entries in V10 cache dirs. Phase 0 smoke uses scoped check; Phase 1+ wires global check at import (TODO in `harness/__init__.py`).

---

## Pre-merge follow-ups (all done as commits, not as a checklist)

The review's three pre-merge items landed as commits 7, 8, 9:

- **Commit 7** — discovery-first override refactor + psf/requests pre-migration test
- **Commit 8** — real execution smoke run with output captured
- **Commit 9** — negative smoke (the merge blocker)

```
$ make smoke-negative
[neg] tampered InstanceView via object.__setattr__ … OK (raised OracleLeakError)
[neg] forbidden token in state_label … OK (raised OracleLeakError)
2/2 negative assertions passed
OK: firewall fires at both real call paths
```

---

## Asset inventory ack (commits 10–13)

Per the asset inventory message, items (a)-(f) before Phase 1 starts:

| # | Item | Commit |
|---|------|--------|
| (a) | `make verify-images` (498/500 present, dev-50 fully covered) | 11 |
| (b) | `scripts/audit_missing_v7_evals.py` → `docs/audits/v7_missing_evals.md` | 12 |
| (c) | `scripts/analyze_v7_for_routing.py` → `splits/v7_difficulty_priors.json` | 13 |
| (d) | Negative smoke | 9 |
| (e) | Full execution smoke | 8 |
| (f) | Cache hygiene global wire-in TODO | 10 |

V7 missing-evals findings:

- 33 missing reports out of 500 submitted
- 30/33 are V7 empty-patch submissions (V7's actor produced nothing; eval harness pre-filtered)
- 3/33 are non-empty patches whose eval crashed mid-way
- Sandbox does NOT need extra hardening for these repos; failures are upstream of the container

V7 difficulty priors:

- 182 instances classified `hard` (V7 resolved=False — even with FAIL_TO_PASS in prompt, V7 failed)
- 272 `medium` (resolved with effort indicators)
- 46 `easy` (resolved cleanly: n_turns ≤ 25 AND stop_reason='done')
- Calibrated against actual V7 turn-count distribution; Phase 1 will recalibrate via dev-50 sweeps

`splits/v7_difficulty_priors.json` is read by `scripts/build_dev_split.py` (later: by the dev-50 stratifier in Phase 1) and explicitly NOT consumed by production routing logic at inference. The boundary is documented in V10_DESIGN.md §13.3 and will be enforced by the firewall in Phase 1.

Phase 7 corpus crawl: scaffolded at `scripts/phase7_crawl.py` per V10_DESIGN.md §10.5. Dry-run validates environment (disk: 1.25 TB free, well above the 50 GB minimum). NOT auto-started — running the crawl is a deliberate operator decision (requires `GH_TOKEN`, ~50 GB minimum disk for initial 100-repo run, ~1 TB for full corpus, multi-week wall-clock).

---

## V10 vs legacy test counts

```
$ make test-v10
34 passed in 0.07s

$ pytest tests/ -q
495 passed, 11 warnings in 4.44s
```

- **34 V10 tests** under `tests/test_no_oracle_leak.py`, `tests/test_dataset_and_conventions.py`, `tests/test_sandbox_and_cache.py`. These are the only tests that import from `harness/`. They cover the full Phase 0 surface: views, firewall (3 layers), dataset, repo conventions, sandbox primitives, runtime guard, cache hygiene, cost tracker.
- **461 legacy tests** cover V7/V8/Meta-Harness++ work. None of them exercise V10 modules — `grep -l "from harness" meta_harness_plus/` returns empty. Their passing is reassuring (we didn't break anything legacy) but **not informative** about V10 correctness. The 34 V10 tests are the load-bearing evidence for this PR.

---

## Phase 1 preconditions (deferred to Phase 1 commit 1, not this PR)

Per the review, Phase 1 starts after this PR merges and after:

1. **Strict global cache hygiene wired into `harness/__init__.py`** — replace the smoke's scoped check with a global one fired at import. Acceptance: `make smoke-strict` (Phase 1 deliverable) exits 0 only if every V10 cache dir is V10-tagged. Tracked TODO already in `harness/__init__.py`.
2. **`splits/dev_100.json`** — held-out 100-instance set distinct from dev-50, repo-diverse, difficulty-balanced, deterministic. Phase 1 commit 1 builds it.
3. **Trajectory log captures localization signals as a structured record** — schema for the per-stage signals (skeleton retrieval scores, BM25/embedding ranks, traceback frames, grep hits, git archeology priors, dep-graph expansion, LLM rerank rationale). Required for the ablation table.

Plus the side-quest (parallel with Phase 1, not after): scaffolded as `scripts/phase7_crawl.py`. Operator decision required before any real crawl — surfaced separately to the user in this PR's discussion.

---

## Design doc

`docs/V10_DESIGN.md` (`v10/phase-0` HEAD: 770 lines). 13 sections + the §12 contamination audit subsection set + §13 asset inventory subsection set. Specifically:

- §1 — honest framing
- §2 — contamination model (the two views + firewall test design)
- §3 — phase architecture (0 → 7)
- §4 — the two original bets (REPO_NOTES + adversarial test gen) + dropped bets with reasons
- §5 — cost model (per-route table; mix-weighted ~$17–22/instance)
- §6 — honest score build-up (66–74% without Phase 7, 70–78% with)
- §7 — ablation matrix
- §8 — eligibility plan
- §9 — working agreements
- §10 — Phase 0 plan + §10.5 Phase 7 corpus crawl
- §11 — model substitution (DeepSeek → Opus expectations)
- §12 — sandbox audit + four tightenings (12.4 redesign, 12.5 runtime guard, 12.6 eval output isolation, 12.7 cache hygiene)
- §13 — asset inventory (Docker images, dataset jsonl, V7 trajectories as DATA, missing 33, API auth, disk)

---

## How to test

```
make test-v10        # 34/34 V10 unit tests
make firewall        # 11/11 firewall layer tests
make verify-images   # 498/500 (full Verified); use --split for subsets
make smoke           # collect-only smoke (≤1 s)
make smoke-exec      # real pytest execution smoke (≤30 s)
make smoke-negative  # confirm firewall raises on real call paths
make build-split     # rebuild splits/dev_50.json (deterministic)
```

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
