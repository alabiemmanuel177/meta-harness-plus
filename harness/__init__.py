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

# TODO(phase-1): wire strict global cache hygiene at import time.
#
# Currently `assert_clean_cache_at_startup()` is invoked only by the
# Phase 0 smoke runner with a SCOPED check (`scope=(trajectories/v10_smoke,)`).
# That keeps Phase 0 runnable on a developer machine where legacy V7/V8
# runs/ entries still exist.
#
# Phase 1 commit N (the localizer entry point) MUST replace the smoke's
# scoped check with a global one fired here at import:
#
#     from harness.cache import assert_clean_cache_at_startup
#     assert_clean_cache_at_startup()  # global default — every V10_CACHE_DIR
#
# Acceptance gate: `make smoke-strict` (added in Phase 1) runs the smoke
# with the global hygiene enabled and exits 0 only if no non-V10-tagged
# entries exist in any V10_CACHE_DIR. Until that target is green, the
# smoke import below stays scoped.
#
# Tracked under V10_DESIGN.md §12.7 ("V10 cache hygiene at startup").
"""

__all__: list[str] = []
