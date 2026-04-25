# Theoretical Framing: Where MH++ Sits in the Optimization Literature

**Status:** sketch / outline. Not a proof. Sets up the formal claims that
would round out a main-track paper, with enough rigor to make a reviewer
nod and enough honesty about what's still open to point follow-on theory
work in the right direction.

---

## 1. The setting

We want to find a *harness*: an ordered pipeline of components
(retriever, reranker, fewshot, formatter, predictor, voter) parameterized
by configuration choices. Let:

- `H` = the discrete-but-large set of valid harness shapes
- For each `h ∈ H`, evaluating it on `n` items costs `n` LLM calls and
  produces a multi-dimensional score `s(h) ∈ [0,1] × R+ × R+`
  (accuracy, tokens, latency)
- We want to find the Pareto frontier `F* ⊆ H` over the partial order
  induced by `s`

This is **multi-objective black-box optimization with structured arms**:
- "Black-box": `s(h)` is an LLM call, no gradient
- "Structured": `H` factorizes — components are independently selectable
  and their attribution can be measured

## 2. Existing techniques we use

- **Successive Halving / Hyperband** (Jamieson & Talwalkar 2016, Li et
  al. 2017) — bandit-style budget allocation
- **Multi-objective Pareto search** with non-dominated sorting (Deb et
  al. 2002, NSGA-II)
- **Drop-one ablation** for component attribution — equivalent to
  Shapley-value approximation for grouping in coalitional games when
  components are independent
- **EWMA running estimates** — standard online statistics
- **Multi-arm bandit framing** for proposer guidance via attribution
  weights (softmax-temperature exploration)

None of these are individually novel. The novelty is in their composition
applied to the harness-search problem and shown to converge on
empirically useful frontier estimates.

## 3. What a regret bound for MH++ would look like

The cleanest theoretical claim would be a **regret bound on the
attribution-guided proposer's hypervolume gap from the optimal
frontier**.

**Setup.** Let `HV(F)` be the dominated hypervolume of frontier `F`
relative to a fixed reference. Let `F*_k` be the best size-`k` frontier
under our score distribution. Define cumulative regret after `T`
iterations as:

    R(T) = sum_{t=1}^{T} [ HV(F*_k) - HV(F_t) ]

where `F_t` is the frontier the search has admitted by iteration `t`.

**Claim (informal):** Under iid `s` evaluation noise with bounded
support, attribution-guided mutation proposers achieve

    R(T) ≤ O( sqrt(T · |H|_eff · log T) )

where `|H|_eff` is the **effective harness space size** — the number of
shapes that produce statistically distinguishable scores under the
search's eval budget. The `log T` factor comes from the standard UCB-style
concentration argument.

**Comparison vs scalar-only search:** scalar accuracy optimization
achieves the same `R(T)` rate but on a *strictly smaller objective space*.
Concretely: any `h` Pareto-optimal in scalar-accuracy search is also on
the Pareto frontier of `(acc, -tokens, -lat)` (because it dominates on
the accuracy axis, regardless of cost). But the converse fails — Pareto
search finds shapes scalar search ignores. So:

    F_pareto ⊇ F_scalar

With strict inclusion when there exist `h_a, h_b ∈ H` with
`acc(h_a) < acc(h_b)` but `tokens(h_a) << tokens(h_b)`. *This always
happens at sufficient `H` density* — specifically, whenever the
accuracy-vs-cost tradeoff is non-trivial.

**Bottom line:** Pareto search dominates scalar search in the strong
sense (frontier is a superset, never a subset). The empirical finding
of 7-of-10-seed strict Pareto dominance over hand-tuned RAG is
*consistent* with this — RAG is a single-point hand pick on what
amounts to a scalar (accuracy-prior) selection; Pareto search visits
shapes the hand-pick wouldn't.

A *tight* bound proving the constant in front of the `sqrt(T |H|_eff log T)`
is open work — the rough proof sketch follows the standard
multi-objective UCB analysis (Auer et al. for MAB, then NSGA-II
extensions in Belakaria et al. 2019), but the structured-arms
factorization needs new analysis since components contribute
non-additively to the score.

## 4. Sample complexity for attribution

A second formal claim would bound how much eval data attribution needs
to identify the highest-value component.

**Setup.** Suppose component-kind `k` has true mean accuracy delta
`Δ_k` (from the population distribution of harnesses where `k` is
present vs absent). Under our drop-one ablation with `m` snapshots and
`r` repeats per snapshot, the empirical estimate `Δ̂_k` has variance
roughly `σ²_k / (m · r)` where `σ²_k` is the across-harness variance
of `k`'s contribution.

By Hoeffding's inequality, the probability of mis-ranking `Δ_k > Δ_j`
when `Δ_k > Δ_j + ε` is at most:

    P(rank-flip) ≤ 2 exp(-2 m r ε² / (b - a)²)

where `(b - a)` is the support of the per-harness delta. To distinguish
two kinds whose true gap is `ε` with confidence `1 - δ`:

    m · r ≥ (b - a)² · log(2/δ) / (2ε²)

For our typical `b - a ≈ 0.5` (deltas in [-0.25, 0.25]) and `ε = 0.05`
(distinguishable per-component value), this requires `m · r ≥ 50` —
meaning ~50 ablation events per kind. We currently ship `m ≈ 6-10`
ablations × `r ≈ 2` repeats = `12-20`. **Underpowered by ~3×.**

This explains why our attribution signals are noisy in single-seed
runs — the bound predicts it. Multi-seed aggregation (which we use
via `MultiSeedRunner`) is exactly the right fix; with 5 seeds we
multiply `m·r` to `60-100`, comfortably above the threshold.

## 5. Connection to multi-objective bandits

Belakaria, Deshwal, Doppa (2019) and Hernandez-Lobato et al. (2016) frame
multi-objective Bayesian optimization with hypervolume-improvement
acquisition functions. Our framework can be cast in that language:

- Each "arm" = a harness shape `h ∈ H`
- Reward = `s(h) ∈ R^d` (we use `d = 3`)
- Acquisition = the proposer's attribution-guided + frontier-aware
  selection rule
- Hypervolume improvement = the per-iteration `Δ HV` we now log

**Open formalization:** showing that our attribution-guided proposer is
asymptotically equivalent to a hypervolume-improvement acquisition
function on a Gaussian-process surrogate over `H`. This is plausible
because (a) softmax over EWMA attribution mimics Thompson sampling on
the per-kind mean reward, and (b) the structured-arms factorization
gives the GP a natural kernel.

## 6. What would close the gap to a real proof

1. **Define `|H|_eff` precisely** — a noise-aware version of the
   structured-arms space size that accounts for our finite `m·r` budget.
2. **State and prove the regret bound** with explicit constants tying
   `|H|_eff` to the per-kind variance term.
3. **Lower bound for scalar search** showing it can be `Ω(sqrt(d))`
   worse than Pareto on multi-objective tasks where the cost axis
   matters.
4. **Empirical validation of the bound** — measure `R(T)` on synthetic
   harness spaces with known optima, verify the predicted scaling.

Steps 1-3 are weeks of focused theory work for a competent ML theorist.
Step 4 is engineering on top of our existing framework.

## 7. The honest claim for a paper

The framework's *empirical* contribution stands without theoretical
backing — the strict-Pareto-dominance result on Gemini news_hard_50
+ symptom_hard is statistically defensible regardless. The theoretical
contribution would be incremental: framing existing techniques in a
multi-objective bandits language, proving informal bounds, validating
empirically.

A main-track paper without theory but with strong empirical results on
public benchmarks is possible (the original Meta-Harness paper from
Stanford IRIS works that way). A paper *with* the theory would be
stronger but demands more time than we have today.

For now this document serves as the "future work" theoretical section of
a paper draft — it tells reviewers we know where the gap is and roughly
how to close it.

## References

- Jamieson & Talwalkar 2016 — Successive halving for hyperparameter
  identification (AISTATS)
- Li et al. 2017 — Hyperband (JMLR 18:185)
- Deb, Pratap, Agarwal, Meyarivan 2002 — NSGA-II
- Belakaria, Deshwal, Doppa 2019 — Max-value entropy search for
  multi-objective Bayesian optimization (NeurIPS)
- Hernandez-Lobato, Hoffman, Ghahramani 2016 — Predictive entropy search
  for Bayesian optimization with unknown constraints
- Auer 2003 — Using confidence bounds for exploitation-exploration trade-offs
- Lee, Nair, Zhang, Lee, Khattab, Finn 2026 — Meta-Harness: End-to-End
  Optimization of Model Harnesses (arXiv:2603.28052) — the original
  paper this work extends
