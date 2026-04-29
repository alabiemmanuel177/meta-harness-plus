"""V10 SWE-bench Verified harness — oracle-free, leaderboard-eligible.

This package is the clean rebuild defined in `docs/V10_DESIGN.md`. It is
deliberately isolated from the legacy ``meta_harness_plus`` tree, which
contains V7/V8 runners with known oracle-leak surfaces (see §12 of the
design doc). No module here may import from:

  - ``meta_harness_plus.swebench_v7``
  - ``meta_harness_plus.swebench_adapter.SWEBenchInstance``
  - ``meta_harness_plus.swebench_adapter.build_user_prompt``
  - ``meta_harness_plus.swebench_adapter.load_swebench_verified``
  - ``meta_harness_plus.agent_swebench_loop`` (any name)
  - ``meta_harness_plus.agent_docker.DockerShellExecutor.run_tests``

The firewall test in ``tests/test_no_oracle_leak.py`` enforces these
boundaries statically (AST scan) and at runtime (LLM-call wrapping).

Cache hygiene wire-in (Phase 0 commit 18, satisfying the prior commit-10
TODO): every import of ``harness`` triggers
``assert_clean_cache_at_startup()`` BEFORE any other V10 code can run.
The default check inspects V10_EXCLUSIVE_DIRS (the grader output dir
and ``.harness_cache/``) and refuses to start if either contains
non-V10-tagged entries. Shared parents (``runs/``, ``trajectories/``,
``repo_cache/``) are tolerated for legacy V7/V8 baseline data; their
V10-tagged children are checked per-run.

Acceptance: ``make smoke-strict`` runs the smoke under the global
import-time hygiene; passing means no leftover legacy entries can seed
a V10 cache. See V10_DESIGN.md §12.7 and ``harness/cache.py``.
"""

# Fail fast at import: refuse to load V10 if eval_outputs/ or
# .harness_cache/ have been seeded with non-V10 data.
from harness.cache import assert_clean_cache_at_startup as _assert_v10_cache_clean

_assert_v10_cache_clean()
del _assert_v10_cache_clean


__all__: list[str] = []
