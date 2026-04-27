# WOW Push Plan — Ollama Pro + Local Compute

**Created:** 2026-04-27
**Goal:** Close the 6 gaps in `HONEST_ASSESSMENT.md` to take MH++ from "credible
workshop paper" to "wow." Stay under ~$200 all-in by routing the dominant
LLM-under-test cost through Ollama Pro (~$20/mo flat) and reserving a small
closed-API budget for transfer validation.

---

## Budget envelope

| Line item | Cost |
|---|---|
| Ollama Pro (cloud-hosted open models) | $20/mo |
| Closed-API spot validation (Sonnet 4.6 + GPT-5-mini, top-1 harness, 8 samples) | **$13–35 one-time** |
| Hosting (when shipping demo) | $0–20/mo |
| Compute (your machine) | $0 |
| **Total to ship** | **~$50–80 + $20/mo recurring** |

**Hard rule:** if a phase blows past its budget gate, stop and reassess — do
not silently overspend.

---

## What is already in place (don't rebuild)

| Capability | Location |
|---|---|
| Ollama wiring (native + OpenAI-compat) | `meta_harness_plus/llm/client.py:84` (`HTTPClient`) |
| Math task + predictor | `meta_harness_plus/tasks/math_task.py`, `meta_harness_plus/llm/predictor.py:MathLLMPredictor` |
| Pareto frontier + dominance | `meta_harness_plus/pareto.py:dominates` |
| Successive halving | `meta_harness_plus/halving.py` |
| Synergy / drop-pair ablation | `meta_harness_plus/synergy.py:SynergyTracker` |
| Continual loop (persistence, drift, gates, rollback) | `meta_harness_plus/continual.py` |
| Existing baselines (DSPy, OPRO, TextGrad, ProTeGi, RAG, CoT-RAG, random) | `examples/{dspy,opro,textgrad,protegi}_baseline.py`, `examples/hand_tuned_baselines.py` |
| Demo CLI + notebook + dashboard | `examples/demo.py`, `examples/demo_notebook.ipynb` |
| Local-Ollama search script | `examples/run_ollama_search.py` |
| Multi-seed runner | `examples/run_public_10seed.sh` (already supports `API=ollama`) |

The cloud Ollama port is **a URL + bearer-token swap**, not new code.

---

## Phased plan (each phase has a gate)

### Phase 0 — Cloud Ollama wiring + smoke test  [Day 1, $0–1]

- Set `OLLAMA_API_URL` and `OLLAMA_API_KEY` env vars (Ollama Pro provides a
  cloud endpoint; HTTPClient already supports bearer-token auth).
- Add a one-line wrapper `examples/run_ollama_cloud_search.py` (clone of
  `run_ollama_search.py`, default URL flipped to cloud).
- Smoke test: 5 AG News rows × `gpt-oss:120b` from cloud. Assert latency
  reasonable, tokens accounted, no auth errors.
- **Gate:** end-to-end call succeeds; cost <$1 in Ollama allowance.

### Phase 1 — AIME-25 task wiring  [Day 2–3, $0–2]

- Source AIME-25 problems (HuggingFace `Maxwell-Jia/AIME_2024` style; AIME-25 = 30 problems).
- Add `meta_harness_plus/tasks/aime_task.py` (mirrors `math_task.py`; numeric
  answer parser; canonicalization to integer 0–999 per AIME format).
- Add `tests/test_aime_task.py` with 3–5 deterministic-fixture tests.
- Validate: deepseek-v3.1 zero-shot CoT on 5 problems, expect ~50–70% baseline.
- **Gate:** loader green + 5-problem smoke run lands within expected accuracy band.

### Phase 2 — Baseline grid on AIME-25  [Day 4–5, $2–10]

- Run on `gpt-oss:120b`, `deepseek-v3.1`, `qwen3-coder:480b`:
  - Hand-tuned: zero-shot CoT, MAJ@8 self-consistency, MAJ@16
  - DSPy `BootstrapFewShot`
  - OPRO (instruction search)
  - Random search (uniform component sampling, 20 candidates)
- 5 seeds each. Persist to `runs/aime25_baselines_<model>_seed<n>/`.
- **Gate:** baseline table populated for ≥2 models × ≥4 baselines × 5 seeds.
  Best-baseline number recorded.

### Phase 3 — MH++ Pareto search  [Day 6–10, $5–25]

- Configure search space (existing components: predictor, voting, reranker,
  bootstrap demos, CoT compression, plan→execute, verifier-loop).
- Halving schedule: 32 candidates → 16 → 8 → 4 → 2 → 1, eval sizes
  `[5, 10, 15, 25, 30, 30]` problems.
- 5 seeds. Track strict Pareto vs best baseline on (accuracy ↑, tokens ↓, latency ↓).
- **Gate:** ≥3/5 seeds achieve strict Pareto over best baseline AND headline
  accuracy delta ≥ +5pt. If not, document the negative honestly and stop.

### Phase 4 — Synergy law mining  [Day 11–12, $2–8]

- Run `SynergyTracker` drop-pair ablation over the top-8 frontier harnesses
  on AIME-25.
- Rank pairs by signed synergy delta. Surface top-3 super-additive pairs.
- For each, write a one-paragraph mechanism hypothesis tied to math reasoning
  (e.g. "verifier-loop × MAJ@8 super-additive because verifier rejects
  arithmetic slips that majority would otherwise lock in").
- **Gate:** at least one pair with |Δ| ≥ 0.05 accuracy and a defensible
  mechanism story.

### Phase 5 — Closed-API transfer validation  [Day 13, **hard cap $35**]

Budget is tight. Run the lean spec and stop the moment the cap is hit.

- **Top-1 harness only** (the Phase 3 winner; do not validate runner-up).
- **Two providers:** Claude Sonnet 4.6 (carries the headline) + GPT-5-mini
  (cross-vendor evidence at minimal cost). Skip full GPT-5 unless ≥$20 of
  the cap remains unspent after Sonnet completes.
- **30 AIME-25 problems × 8 samples × 1 harness × 2 models ≈ 480 calls.**
- **Pre-flight:** before kicking off either model, run 5 problems × 8 samples
  on Sonnet 4.6, multiply observed cost × 6 to project total — abort if
  projection exceeds $25 on Sonnet alone.
- **Checkpointed eval:** persist per-problem results to
  `runs/aime25_transfer_<model>/results.jsonl` so a mid-run failure costs at
  most one batch ($3–5) to retry.
- **Cost log:** every batch writes a line to `runs/wow_push/cost_log.jsonl`
  with `(provider, model, calls, in_tokens, out_tokens, $usd)`.
- Report transfer: does the discovered harness still beat hand-tuned
  CoT/MAJ@8 on closed frontier models?
- **Gate:** transfer holds (effect ≥ +3pt vs best closed-model baseline) OR
  document the gap honestly in `RESULTS_AIME25_TRANSFER.md`. Either outcome
  is publishable.

**Per-call cost ballpark (Sonnet 4.6 on AIME-style CoT):**
~3K input × $3/MTok + ~2K output × $15/MTok ≈ **$0.04/call** → **~$10 for
240 calls**. GPT-5-mini at ~10× cheaper input/output ≈ **~$1–2 for 240 calls**.
**Expected total: ~$13.** $35 is a 2.7× safety margin, not a target.

### Phase 6 — Demo + dogfood + writeup  [Day 14–21, $0–20/mo]

- Update `examples/demo.py` to use Ollama Cloud by default; verify <10 min wall-clock end-to-end on AIME-25 subset.
- Wire `meta_harness_plus/continual.py` to ingest your own Claude Code
  prompt-traffic logs as the dogfood signal (read-only; no external traffic).
- Hosted demo (optional, Phase 6b): minimal FastAPI + static HTML; deploy to
  Render/Fly free tier; rate-limit per IP.
- Update `THESIS.md` and `RESULTS_FINAL_GRID.md` with AIME-25 result + named
  synergy law + transfer numbers.
- **Gate:** new headline added; CHANGELOG line; tests still green
  (`pytest -q` ≥ 363 tests passing).

---

## Risks & honest caveats

1. **Ollama Pro rate limits.** Cloud tier has hourly/daily caps. Long Phase 3
   sweeps may need overnight pacing. Build in checkpointing every 5 candidates.
2. **Open-model AIME-25 ceiling.** If `deepseek-v3.1` zero-shot is already
   ~85%, harness gains compress. Pivot to AIME problems where models score
   50–70% (more headroom) — possibly AIME 2024 if 2025 saturates.
3. **Transfer to closed models may fail.** Harnesses tuned to open-model
   failure modes might not help Sonnet 4.6. That's still a publishable result
   ("MH++ wins on open models; transfer is partial — here's why"), but pivots
   the thesis.
4. **Synergy mining may produce noise.** With only 30 problems, drop-pair
   variance is high. Use bootstrap CIs, require |Δ| ≥ 0.05 AND CI not crossing
   zero.
5. **Auth/network.** Ollama Pro endpoints may not be OpenAI-compatible at the
   bearer-token layer everyone assumes. Verify in Phase 0.
6. **Scope creep.** Resist adding SWE-bench/GAIA/τ-bench until AIME-25 lands.
   One frontier benchmark done well > three half-done.

---

## Kickoff prompt for Claude Code

Paste this into a fresh Claude Code session (run from
`/home/eao/workplace/projects/Meta-Harness`):

```
You are continuing a research project at /home/eao/workplace/projects/Meta-Harness
called MH++ (Pareto harness search). Read these in order:

1. Tasks                       — original 8-item roadmap
2. HONEST_ASSESSMENT.md        — gap analysis (what's missing for "wow")
3. WOW_PUSH_PLAN.md            — the phased plan you are executing
4. CLAUDE.md (if present)      — repo conventions
5. README.md                   — repo orientation
6. meta_harness_plus/llm/client.py  — how HTTPClient routes to Ollama / OpenAI /
                                      Anthropic / Gemini

Then execute WOW_PUSH_PLAN.md, Phase 0 first, gate-by-gate. Rules:

- Stop at every gate. Print a status block (pass/fail, evidence paths,
  cost-so-far in $). Wait for "go" before the next phase.
- Cost discipline: never exceed the per-phase budget without asking. Track
  cumulative spend in runs/wow_push/cost_log.jsonl (one line per API batch).
- Reuse existing infra: pareto.py, halving.py, synergy.py, continual.py, the
  baselines in examples/, run_public_10seed.sh. Do NOT rewrite these.
- For Ollama Pro: use HTTPClient with the cloud URL + bearer key from env vars
  OLLAMA_CLOUD_URL and OLLAMA_API_KEY. If the user hasn't set them yet, stop
  and ask.
- Honest negatives count as deliverables. If Phase 3 fails the strict-Pareto
  gate, write the negative result to RESULTS_AIME25_NEGATIVE.md and stop —
  don't paper over it.
- Use TaskCreate to track phase progress; mark each phase completed only after
  its gate passes.
- Tests: every new module gets a deterministic-fixture test (no real API
  calls in tests). Run `pytest -q` before declaring any phase done.
- Commits: one commit per phase, messages like "Phase 1: AIME-25 loader +
  smoke test (5 problems, deepseek-v3.1, ~$0.02)". Co-author tag per repo
  convention.
- Do NOT touch closed-API budget (Phase 5) until Phases 0–4 all green.

Begin with Phase 0. First action: read the three files listed above, then
print a 5-line plan for Phase 0 (the work you'll do, the gate criteria, and
what you need from me — specifically the Ollama Pro env vars).
```

---

## What you (the human) need to do before pasting

1. **Sign up for Ollama Pro** if you haven't, and grab:
   - Cloud endpoint URL (e.g. `https://ollama.com/v1/chat/completions` — verify
     against current Ollama docs)
   - API key
2. **Export env vars** in the shell where Claude Code will run:
   ```sh
   export OLLAMA_CLOUD_URL="<url>"
   export OLLAMA_API_KEY="<key>"
   ```
3. **(Optional) Pre-stage AIME-25 data** if you have a preferred source. Otherwise Claude Code will pick one in Phase 1.
4. **(Phase 5 only)** Have a Claude / OpenAI API key ready with **~$35 cap**.
   Set the cap on the provider dashboard so a runaway loop physically cannot
   overspend.

That's it. Paste the prompt; Claude Code drives from there with you as the gate-keeper.
