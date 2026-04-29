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
"""

__all__: list[str] = []
