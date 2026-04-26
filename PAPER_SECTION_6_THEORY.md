# Paper Section 6 — Theoretical Framing (Draft)

This is the prose draft of section 6. Length target: ~1 page.
Full sketch lives in THEORY.md (Appendix D).

---

## 6. Theoretical Framing

We sketch two theoretical claims and the proof outlines. Tight
proofs are deferred to Appendix D / future work.

### 6.1 Pareto search dominates scalar search (frontier inclusion)

**Claim.** Let `F_pareto(T)` be the Pareto frontier admitted after
`T` search iterations under the multi-objective MH++ search rule,
and `F_scalar(T)` be the size-1 set admitted under
ScalarAccuracyFrontier (C1 ablation). Then under any sequence of
proposals,

    F_scalar(T)  ⊆  F_pareto(T)

with strict inclusion when the proposed harnesses span any
nontrivial accuracy/cost tradeoff (i.e. there exist `h_a, h_b` with
`acc(h_a) < acc(h_b)` but `tokens(h_a) << tokens(h_b)`).

**Proof sketch.** Any candidate admitted by ScalarAccuracyFrontier
must be the highest-accuracy point seen — hence weakly dominates
all other points on the accuracy axis. Any such point is therefore
on the multi-objective Pareto frontier (it dominates on at least
one axis). Strict inclusion when accuracy/cost tradeoffs exist:
the cheaper-than-best-acc point with second-best accuracy is
non-dominated multi-objectively but invisible to scalar search.

**Empirical implication.** Our 6-cell ablation shows the C1-disabled
condition uses +47% tokens and +79% latency for the same accuracy
as full MH++. The scalar search visits accuracy peaks at any cost;
the Pareto search retains cost-cheap variants of similar accuracy.

### 6.2 Sample complexity for component attribution

**Setup.** Suppose component-kind `k` has true mean accuracy delta
`Δ_k` (the population mean of accuracy gain when `k` is present
vs absent in a harness shape). Drop-one ablation with `m` snapshots
× `r` repeats gives an empirical estimate `Δ̂_k` with variance
`≤ σ²_k / (m·r)`.

**Claim.** Under iid score noise with bounded support `[a, b]`, the
probability of mis-ranking two kinds whose true mean gap is `ε`:

    P(rank-flip) ≤ 2 exp(-2 m·r·ε² / (b-a)²)        (Hoeffding)

**Sample size for confident ranking.** To distinguish kinds with
gap `ε = 0.05` and confidence `1-δ = 0.95`:

    m·r ≥ (b-a)² · log(2/δ) / (2ε²)
        ≈ 0.5² · log(40) / (2·0.0025)
        ≈ 73

**Empirical implication.** Our default `attribution_screen_size=12`
× `attribution_repeats=2 × eval_repeats=2` gives 48 effective
samples per kind per seed — under-powered by ~30% to distinguish
ε=0.05 gaps. This explains why our single-seed attribution signals
are noisy and why MultiSeedRunner aggregates across 5+ seeds: with
5 seeds we multiply effective `m·r` to 240, comfortably above the
threshold.

### 6.3 Regret bound for the attribution-guided proposer

**Setup (informal).** Let `HV(F)` be the dominated hypervolume of
frontier `F` against a fixed reference. Let `F*_k` be the optimal
size-`k` frontier under the score distribution. Define cumulative
hypervolume regret:

    R(T) = Σ_{t=1}^T [HV(F*_k) - HV(F_t)]

**Claim (informal).** Under iid evaluation noise with bounded
support, the attribution-guided LLMProposer achieves

    R(T) ≤ O(√(T · |H|_eff · log T))

where `|H|_eff` is the effective harness space size — the number
of shapes producing statistically distinguishable scores under our
finite eval budget.

The proof would follow standard multi-objective UCB analysis (Auer
et al. 2003 for MAB, then Belakaria et al. 2019 NSGA-II hypervolume
extensions). The structured-arms factorization needs new analysis
since components contribute non-additively to score; a tight
constant in front of the `√(T |H|_eff log T)` bound is open work.

### 6.4 What this section is *not*

This is a sketch, not a theorem. Section 6.1's frontier-inclusion
claim is straightforward; Sections 6.2 and 6.3 are open research
problems. The framework's empirical contribution stands without
theoretical backing — we include this section to (a) place our
work in the multi-objective bandits literature, (b) show that the
attribution signal's noise is bound-predicted (motivates multi-seed
aggregation as the right fix), and (c) point follow-on theory work
in the right direction.

A main-track-with-theory paper would close §6.3 with explicit
constants and an empirical validation of the bound on synthetic
harness spaces. We defer that to future work.
