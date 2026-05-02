# V10 Target Specification: 60% on SWE-bench Pro

**Status:** capability spec. Defines WHAT the system must do, not HOW to build it.
**Headline target:** 60% pass rate on SWE-bench Pro test set.
**Realistic target band:** 50-60%. Anything above 45% is publishable.
**Current frontier (May 2026):** Claude Mythos Preview at 45.9% on Pro (Anthropic's best unreleased model).

---

## 1. Why this target matters

SWE-bench Verified is contaminated. Frontier labs have memorized substantial portions
of it during training. Claude Mythos Preview scores 93.9% on Verified vs 45.9% on Pro
— a 48-point gap on the same model. OpenAI stopped reporting Verified scores entirely
in early 2026.

SWE-bench Pro is the contamination-resistant successor. 731 instances across
GPL-licensed repos that frontier labs are less likely to have trained on. Open to
independent submitters (no academic affiliation gate). Leaderboard is much less
crowded — top 5 is achievable for an independent team.

A 60% score on Pro would beat Claude Mythos by 14 percentage points. It would set a
new state of the art on a contamination-resistant benchmark. Not impossible, but
requires building something materially better than current frontier-lab work.

---

## 2. The math anchoring the requirements

Final pipeline score ≈ (file recall) × (patch generation success | correct files) × (selection accuracy)

For 60% final score, with Phase 1 currently at ~95% top-10 file recall:

```
0.95 × patch_gen × selection = 0.60
patch_gen × selection = 0.63
```

Two viable parameter combinations:

- High-selection regime: selection = 0.95, patch_gen = 0.66
- High-patch regime: selection = 0.85, patch_gen = 0.74

Realistic ceiling on selection is ~0.90 (literature shows selection between K
candidates rarely hits 95%+ on hard benchmarks). So patch_gen needs to land in
the **66-74% band** when given the correct files.

Currently published patch generation success rates on Pro-difficulty instances are
**40-55%**. Hitting 70% means materially exceeding what's published.

This is the central engineering problem. Everything else is in service of this number.

---

## 3. Per-phase capability requirements

Working backwards from the 60% target, here is what each phase must deliver.

### 3.1 Phase 1 — Localization

**Requirement:** ≥95% top-10 file recall, ≥85% top-1 on Pro instances.

Currently delivering 95%/75% on Verified. Pro is harder (multi-file patches,
cross-module bugs). Same architecture might drop to 90%/70% on Pro.

The localizer must:

- Handle multi-file gold patches. Pro has many instances where the gold patch touches
  3+ files. Top-10 must include ALL the touched files, not just one. Current evaluation
  semantics are "any-of-gold"; for Pro this needs to shift toward "all-of-gold" or
  weighted-recall.
- Use semantic + structural signals beyond BM25 + embedding. Traceback parsing helps
  on the small fraction of Pro instances that ship with stack traces. The remaining
  instances need stronger signals: AST-aware retrieval, dependency-graph expansion,
  git-archaeology on related commits.
- Possibly require the fine-tuned localizer (Phase 7 corpus crawl, currently scaffolded
  but not active). Activating Phase 7 is a 2-3 week side project that could add 3-5pp
  on Pro specifically.

### 3.2 Phase 2 — Reproduction

**Requirement:** ≥75% usable repro rate on Pro.

Phase 2 currently targets 60% on dev_50 (commit 17d acceptance gate). Pro is harder.
To hit 75% the generator must:

- Handle bugs that require complex setup: database state, mock objects, configuration
  flags, dependency injection. Many Pro bugs are in framework code where the
  reproduction requires non-trivial scaffolding.
- Write tests that reproduce subtle behavioral bugs, not just obvious failures.
  "Function returns None instead of raising" is harder to test than "function crashes."
- Self-verify against the public test suite (not just at base_commit). A test that
  fails at base AND breaks unrelated things isn't a usable repro.

### 3.3 Phase 3 — Patch generation

**Requirement:** 66-74% correct-patch rate when given correct files.

This is the hardest requirement and the central engineering challenge. To hit this on
Pro the patch generator must:

- Reason about multi-file changes, not just single-function fixes. Pro's median
  instance touches 1.8 files; many touch 3-5.
- Understand library APIs deeply enough to use them correctly without examples.
  Pro repos include frameworks where the right fix uses an internal API the model
  hasn't seen heavily during training.
- Generate patches that pass the hidden FAIL_TO_PASS suite, not just the public tests.
  Many "looks correct" patches break hidden tests for edge cases.
- Handle edge cases (off-by-one, error handling, type coercion, None handling,
  thread safety) without explicit prompting.
- Likely needs Claude Opus 4.7 or equivalent, possibly with extended thinking enabled.
- Likely needs an agent loop with 30-50 turns on hard instances, not single-shot
  generation.
- Likely needs read access to the full repo (not just localized files) for the
  hardest instances.

### 3.4 Phase 4 — Validation

**Requirement:** ≥95% accuracy at identifying broken patches.

Mostly mechanical (run tests, count regressions, check static analysis). The hard
part is doing it within wall-clock and cost constraints. Per-instance validation
budget should be ≤2 minutes wall-clock, ≤$0.05.

### 3.5 Phase 5 — Selection

**Requirement:** ≥90% correct-patch picked when one exists in K candidates.

To hit 90% on Pro the selector must:

- Distinguish surface-correct patches (compile, look right) from semantically
  correct patches.
- Use multiple signals in combination: repro test pass, public test regressions,
  patch minimality, static analysis output, devil's-advocate critique, AST
  similarity to historical fixes in the same repo.
- Possibly need a learned reranker rather than just an LLM judge. Frontier work in
  late 2025 / early 2026 increasingly uses lightweight learned selectors.
- Handle ties without falling back to "pick first." Tie-breaking is a meaningful
  fraction of error.

---

## 4. Cross-cutting capabilities the system needs

These are the capabilities that don't sit in any single phase but the system as a
whole must demonstrate. Each one is a real research problem; some are
solved-but-unpublished at frontier labs, some are open.

### 4.1 Multi-file reasoning at near-human level

Pro's multi-file instances are where Claude Mythos drops to 46%. Beating that
requires the patch generator to do something different from single-file
generation.

Possible approaches (the system must do at least one):

- Two-phase generation: a "plan" phase that identifies which files need to change
  and what the change should be conceptually, followed by an "implement" phase
  that produces the actual diff for each file.
- Explicit dependency analysis before patch generation, building a small graph
  of "if I change X, what else must change."
- A critic loop that catches incomplete multi-file patches by checking whether
  the patched code's call sites still work.

### 4.2 Self-verification beyond the base test

Most published harnesses verify "does the patch apply, does the public test
suite still pass." That misses semantic bugs. To hit 60% on Pro the system needs
a stronger verification signal.

Possible approaches:

- Property-based testing of the modified code (which Phase 2 design explicitly
  removed; would need to come back as a Phase 4 component).
- Differential testing: compare patched vs unpatched behavior on synthesized
  inputs.
- Formal-ish reasoning about invariants the code should preserve.
- Mutation testing on a small scale: introduce known mutations to the patched
  code, verify the test suite catches them.

### 4.3 Smart candidate selection without an oracle

Selection above 85% accuracy on Pro is hard because the harness can't peek at
hidden tests. The selector must use signals that correlate with correctness
without being the test itself.

Possible approaches:

- Ensemble agreement: generate K patches from M models, look at consensus.
- Cross-signal consistency: a patch that passes the repro test AND doesn't
  regress public tests AND has minimal diff is more trustworthy than one that
  passes only the repro.
- Code-style and minimality heuristics weighted by repo conventions.
- An LLM judge with carefully engineered prompts, including chain-of-thought
  about why each candidate might be wrong.

### 4.4 Domain adaptation across diverse Pro repos

Pro's repos are unfamiliar to frontier models. The system needs a way to give the
patch generator context about repos it doesn't know intimately.

Possible approaches:

- REPO_NOTES (auto-generated documentation per repo, the existing Bet A from
  V10_DESIGN.md). Run once per repo, cache, inject into every instance prompt
  for that repo.
- Cross-instance learning: build memory across the 731 Pro instances during
  the run. Carefully — no oracle leakage. Patterns of "what kinds of fixes work
  in this repo" can be learned from earlier instances and applied to later ones.
- Retrieval-augmented prompting from the repo's own historical commits. Past
  bug-fix commits in the same repo are strong priors.

### 4.5 Cost-aware routing

Opus on 731 Pro instances at multi-candidate generation could easily run
$500-1500 per full eval. The system must be smart about which instances deserve
frontier-model spend.

Possible approaches:

- Instance difficulty estimation from cheap signals (issue length, traceback
  presence, candidate-file count, multi-file scope mentions). Easy instances
  routed to DeepSeek; hard instances routed to Opus.
- Tiered generation: try DeepSeek first, escalate to Opus only when validation
  signals are weak.
- Adaptive K: generate fewer candidates on easy instances, more on hard ones.

---

## 5. What this system has that frontier labs don't

Five edges that are real and exploitable:

- **The contamination firewall.** Methodology contribution that frontier labs
  haven't published. Helps marginally on Pro (where contamination is already
  controlled) but matters enormously for the paper's reception.
- **Cost discipline.** Frontier labs optimize for raw score; this system
  optimizes for cost-per-correctness. The cost story is publishable on its own.
- **Model-agnostic architecture.** When a new frontier model releases, swapping
  it in is one config change. Frontier labs can't easily benchmark against
  competitors' models.
- **Pro-first focus.** Frontier labs report Pro reluctantly because their
  Verified scores look better. This system is built for Pro and reports it as
  the headline.
- **Open reproducibility.** Every artifact (config, model weights for the
  open-weight components, prompts, eval code) is published. Anyone can rerun
  the harness and get the same number ±API noise.

---

## 6. What this system does NOT have that frontier labs do

Be honest about the deficits:

- Frontier model access at near-zero marginal cost. Pay retail; they pay
  inference cost only.
- Months of dev time and PhD-level engineering teams. The 5-7 week timeline
  is aggressive.
- Internal eval infrastructure. Each test_500 run is a real wall-clock event.
- Whatever proprietary tricks haven't been published. Frontier-lab harnesses
  almost certainly do things the literature doesn't describe.

---

## 7. Execution framing

The work over the next 5-7 weeks should be governed by a single question, asked
after every commit:

> Does this move us closer to one of the five cross-cutting capabilities in §4?

If yes, keep going. If no, redesign or remove.

Each phase boundary should produce a measurement against the per-phase
requirement in §3:

- End of Phase 2: dev_100 repro coverage ≥75%? If yes, proceed. If no,
  iterate prompt or architecture before Phase 3 starts.
- End of Phase 3: dev_100 patch generation success (given correct files) ≥66%?
  If yes, proceed. If no, this is the hardest stage and where most variance
  lives. Iterate.
- End of Phase 4 + 5: dev_100 end-to-end pipeline ≥35%? Below this, the
  60% Pro target is out of reach. Above this, in range.

Measure honestly. Adjust the target if the data demands it. A 50% Pro submission
that's honest and methodologically sound beats a 60% submission that hand-waves.

---

## 8. The acceptance criteria for the project

This work succeeds if it produces ALL of the following:

1. A clean SWE-bench Pro submission with score ≥45%. Below this is not a
   submission-grade result.
2. An arXiv preprint documenting the contamination firewall, the per-strategy
   rerank finding, and the harness architecture. Independent of the score.
3. Open-sourced harness with reproducible `make eval` on a clean machine.
4. Per-phase audit documents and ablation tables. The methodology, not just
   the result.
5. A published comparison between SWE-bench Verified and Pro on the same harness,
   contributing to the field's understanding of contamination.

A 50% Pro score with all five deliverables is a stronger contribution than a
60% Pro score with weak documentation.

---

## 9. The honest reframe

The 60% target is real but ambitious. It requires building something materially
better than current frontier-lab work on a benchmark Anthropic helped design to
be hard.

The path goes through 50% first. Hit 50%, understand where the misses come
from, then iterate the 10pp. If the architecture is sound and the data shows
clear next steps, 60% is reachable. If the architecture has a structural
problem, 60% is not the right target and the data will say so.

Treat 50% as success, 55% as exceptional, 60% as a research milestone. All
three are wins. The only loss is not finishing or publishing dishonest numbers.

Build toward the five capabilities in §4. Measure against the per-phase
requirements in §3. Ship the deliverables in §8.

That is the plan.