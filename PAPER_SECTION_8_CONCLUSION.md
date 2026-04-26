# Paper Section 8 — Conclusion (Draft)

This is the prose draft of section 8. Length target: ~½ page.

---

## 8. Conclusion

We extended the Meta-Harness framework with three contributions:
multi-objective Pareto search over `(accuracy, tokens, latency)`,
budget-aware evaluation via successive halving, and component-level
drop-one attribution surfaced to the LLM proposer. The combination
produces harness shapes that strictly Pareto-dominate hand-tuned RAG
on 7 of 10 Gemini seeds across two adversarial English benchmarks
at 6×8 search budget — same-or-higher accuracy, strictly fewer
tokens, not-worse latency, on every axis simultaneously.

Across a 6-cell `(provider × task)` grid spanning OpenAI
gpt-4.1-nano, Google gemini-2.5-flash-lite, two hand-curated English
benchmarks, and the public Chinese LawBench 2-2 dataset, MH++ shows
CI-excluding-zero accuracy gains over hand-tuned RAG on every cell
with sufficient search budget. The largest single gain is +29.3pt
absolute accuracy (paired t=+17.8, p=6×10⁻⁵, d=+7.98). Against the
strongest hand-tuned baseline (CoT-RAG / voting-RAG / diverse-RAG),
MH++ wins substantially on 5 of 6 cells (+5.9 to +12.9pt) and
reproducibly on the 6th (+2.0pt with spread 0.020).

We are honest about three failure modes: small-budget search loses
to strong RAG baselines (closes substantially with bigger budgets);
the attribution-guided LLMProposer only marginally beats a random
proposer at small budgets on easy tasks; the public-dataset coverage
in our experiments is limited to one task family.

The framework is open-source with 267 unit tests, 1-line CLI
replication, and total experimental compute under $1.50 USD. Run
logs, prompt caches, and per-seed aggregates are committed alongside
the code.

The qualitative jump we set out to demonstrate — that automated
harness search produces shapes that beat strong hand-tuning on
every axis simultaneously, not just on accuracy at higher cost —
is now empirically supported on multiple cells. Future work
extends this to more public benchmarks, larger-model experiments,
and the theoretical regret bounds sketched in Section 6 and
Appendix D.
