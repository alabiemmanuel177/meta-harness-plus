# Phase 1 stages 1d / 1e / 1f decision

**Author:** harness team.
**Status:** recommendation, awaiting your call.
**Per the long-horizon batch spec (commit 5e):** decision rule is

> top-10 ≥ 85% → skip 1d/1e/1f, declare Phase 1 done, move to Phase 2 (reproduction oracle).
> top-10 ∈ [75%, 85%) → implement 1d only (symbol grep — cheapest), re-run, re-decide.
> top-10 < 75% → implement all three. Something is structurally missing.

---

## Recommendation: **skip 1d / 1e / 1f. Phase 1 is done. Move to Phase 2.**

The full dev_50 result with stages 1b + 1c + 1g lands at **98.0% top-10** — well past the 85% threshold. The only failing instance (`sphinx-doc__sphinx-11510`) miss is a Stage 1g rerank ordering issue, not a missing-signal-source issue (see analysis below).

---

## Numbers (full dev_50, paired ablation)

|                    | No rerank | + Stage 1g rerank | Δ          |
|--------------------|-----------|-------------------|------------|
| Top-1              | 24.0%     | **74.0%**         | +50 pp     |
| Top-5              | 62.0%     | **94.0%**         | +32 pp     |
| **Top-10**         | 78.0%     | **98.0%**         | +20 pp     |

The 78% top-10 *without* rerank is already in the [75%, 85%) zone where the decision rule says "implement 1d only." The 98% *with* rerank is in the ≥85% zone where the rule says "skip all three." The rerank turns out to be the load-bearing intervention, not the additional signal sources.

---

## Why stages 1d / 1e / 1f are not the bottleneck

### 1d (symbol grep)
**What it would add:** `ripgrep` for rare identifiers / quoted code / error strings extracted from the issue, weighted by inverse-frequency rarity.

**What we already have:** `bm25_extracted_symbols` strategy already pulls identifiers (CamelCase, snake_case, dotted refs, backtick code, quoted strings) and runs BM25 on them. The 5c per-strategy ablation shows `bm25_extracted_symbols` alone hits 16% top-1 / 70% top-10 — comparable to the other BM25 strategies but already integrated into the rerank prompt.

**What 1d would actually add over 1b's `bm25_extracted_symbols`:** stricter rarity weighting (current BM25 implicitly handles term frequency but not corpus-wide rarity scoring). Worth maybe ~2pp lift on hard instances, but the rerank already handles the disambiguation that rarity scoring would help with.

**Verdict:** redundant given Stage 1g.

### 1e (git archeology)
**What it would add:** `git log -S<token>` and `git log --all -- <candidate_file>` to find past commits that touched candidate files. Strong prior when the issue references a function/class that's been modified before, especially for fix commits (`fix:`, `bug`, etc.).

**What's the marginal lift on dev_50:** the misses happen on `sphinx/directives/other.py` (1 instance). Git archeology might find that this file has been touched in past `fix:` commits — but the rerank already sees the file as a candidate; the issue is ranking, not retrieval.

**Verdict:** would help only when the gold file is *not in any upstream signal's results*. On dev_50, **0 misses match this profile** — every miss has the gold file at *some rank in some strategy*. Git archeology doesn't fix ordering problems.

### 1f (dep-graph expansion)
**What it would add:** AST/import graph; expand each candidate one hop in both directions (importers + importees) to catch cases where the bug is in a caller or callee.

**What's the marginal lift:** same as 1e — would help only if the gold file isn't already a candidate. On dev_50, every miss has the gold file as a candidate. Dep-graph would flood the candidate pool with neighbors, increasing reranker prompt size without adding new gold-file candidates.

**Verdict:** would help only when the gold file is buried 2+ hops from any direct match. Not the dev_50 failure mode.

---

## Why the dev_50 result holds (and where it might not)

**Strengths of the current stack:**

1. The 1c traceback parser runs on 3/50 instances; it doesn't gate the headline but adds redundant signal where present.
2. Stage 1g consumes per-strategy ranks as features (per `RankedFile.upstream_best_rank`), exactly the design from V10_DESIGN.md §9. The +50pp top-1 lift validates that working agreement.
3. The reranker's per-strategy view recovers files that aggregation buried — `bm25_first_paragraph` hits 32% top-1 alone, but aggregated drops to 24%; rerank lifts back to 74% by trusting the strategy-1 picks.

**Cases where this might not generalize:**

- **Test-500 may have repos not in dev_50.** dev_50 covers all 12 Verified repos but with weighted instance counts. Any failure mode unique to a low-coverage repo could surface.
- **The single sphinx miss** (`sphinx-doc__sphinx-11510`) suggests sphinx's directive/extension architecture confuses the rerank. Worth a per-instance dive before claiming production-ready.
- **The reranker is deepseek-chat** — relatively cheap. Stronger models (Sonnet 4.6, Opus 4.7) might give a small further lift, but the dev_50 ceiling is already 98% top-10. Diminishing returns.
- **Dev-100 hasn't been evaluated** — that's the held-out set per V10_DESIGN.md split discipline. Phase 1 acceptance should be confirmed there before locking in the design.

---

## Concrete next moves (in priority order)

1. **Phase 2 — reproduction oracle** (per V10_DESIGN.md §3.3). The localizer is good enough; the next bottleneck is generating a repro test from the issue text. Phase 1 stages 1d/1e/1f are deferred unless dev-100 says we need them.
2. **Run dev-100** with the current stack (`make eval-fast`-shape but on `splits/dev_100.json`). Confirm 98% top-10 holds on held-out data. If dev-100 top-10 drops below 90%, revisit the 1d/1e/1f decision.
3. **Investigate the single sphinx miss.** Cheap (<30 min), might surface a generic ranking-rule fix that helps the whole stack.
4. **Reranker model ablation** — defer. The +50pp from rerank-with-deepseek is so large that swapping models won't change the macro picture; revisit during paper writeup.

---

## What 1d/1e/1f WOULD be useful for (separate concern)

These three stages produce **per-instance attribution data** that's interesting for the paper even if they don't move dev_50 numbers. Specifically:

- 1d: distribution of "rare identifier vs common identifier" in Verified issues — paper-worthy.
- 1e: distribution of "files with prior fix commits" overlapping gold patches — paper-worthy.
- 1f: distribution of "gold file is N hops from any upstream candidate" — paper-worthy.

I'd implement them as instrumentation-only (don't feed into the reranker) for the paper. Not as a Phase 1 production gate.

---

## Cost & cadence note

The dev_50 rerun-with-rerank cost ~$3.75 cumulative this session (≤$50 budget). A dev-100 rerun should land at ~$7.50 — also well under. Acceptance to advance to Phase 2 is bounded by the dev-100 result, not by budget.
