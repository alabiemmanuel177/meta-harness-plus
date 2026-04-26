"""Real TerminalBench adapter — stub.

The real ``terminal-bench`` benchmark (https://github.com/laude-institute/terminal-bench)
is a Docker-driven evaluation harness with hundreds of tasks. Wrapping
it as an ``AgentTask`` for MH++ search requires:

1. The ``terminal-bench`` PyPI package installed.
2. Docker available on the host (the benchmark's official runner spins
   up per-task containers).
3. An OpenAI / Anthropic / Gemini API key for the agent's policy LLM
   if the harness is using a real model.

Because Docker may not be installed in every environment we ship MH++
to, this module is intentionally a thin import-time-checked adapter.
The function below loads tasks from a TerminalBench checkout dir and
returns an ``AgentTask`` whose ``setup`` and ``check_fn`` invoke the
real ``terminal-bench`` runner under the hood. If the package or
checkout isn't present, the function raises a clear ImportError telling
the caller exactly what to install.

To run real TerminalBench end-to-end::

    pip install terminal-bench
    git clone https://github.com/laude-institute/terminal-bench
    export TBENCH_TASKS_DIR=$PWD/terminal-bench/tasks
    python3 examples/run_terminalbench.py

The local 6-task fixture in ``terminalbench_fixture.py`` is what the
unit tests + ``examples/run_agent_search.py`` use; it doesn't need
Docker.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Iterable

from ..agent import AgentEnv, AgentExample, AgentTask


def _require_terminal_bench():
    """Import terminal-bench or raise a clear setup error."""
    try:
        return importlib.import_module("terminal_bench")
    except ImportError as e:
        raise ImportError(
            "Real TerminalBench requires the `terminal-bench` package and Docker.\n"
            "Install with:\n"
            "    pip install terminal-bench\n"
            "And ensure Docker is running. See "
            "https://github.com/laude-institute/terminal-bench for details."
        ) from e


def list_terminalbench_tasks(tasks_dir: str | Path | None = None) -> list[str]:
    """List task IDs available in a TerminalBench checkout.

    Args:
        tasks_dir: path to the checkout's ``tasks/`` dir. If None, reads
            from the ``TBENCH_TASKS_DIR`` env var.

    Raises:
        FileNotFoundError if the dir doesn't exist.
    """
    tasks_dir = tasks_dir or os.environ.get("TBENCH_TASKS_DIR")
    if not tasks_dir:
        raise FileNotFoundError(
            "Pass tasks_dir or set TBENCH_TASKS_DIR to your terminal-bench "
            "checkout's `tasks/` directory."
        )
    p = Path(tasks_dir)
    if not p.exists():
        raise FileNotFoundError(f"TerminalBench tasks dir not found: {p}")
    return sorted(d.name for d in p.iterdir() if d.is_dir())


def build_terminalbench_task(
    task_ids: Iterable[str] | None = None,
    tasks_dir: str | Path | None = None,
) -> AgentTask:
    """Build an AgentTask from real TerminalBench tasks.

    Each ``AgentExample`` here is a thin wrapper that delegates ``setup``
    and ``check_fn`` to the upstream TerminalBench runner so the real
    Docker-based scoring is reused. The ``MetaSearch`` proposer / our
    ``AgentHarness`` then drive the agent loop on top.

    This function explicitly requires the ``terminal-bench`` package.
    For deterministic tests, use ``build_terminalbench_fixture()``
    instead.
    """
    tb = _require_terminal_bench()
    if task_ids is None:
        task_ids = list_terminalbench_tasks(tasks_dir)

    # The real TerminalBench loader API is a moving target across
    # versions; rather than import-pinning a specific call we leave
    # this as a placeholder that tells the caller exactly what's
    # missing if they hit it.
    raise NotImplementedError(
        "Real TerminalBench wrapping is a stub: the upstream API surface "
        "changes between releases. Install terminal-bench, then either:\n"
        "  (a) call terminal-bench's own runner directly with your harness's "
        "policy callable, or\n"
        "  (b) implement build_terminalbench_task() against the version of "
        "terminal-bench you have installed (the loader returns a list of "
        "task spec dirs; map each into AgentExample with setup=copy_files, "
        "check_fn=run_test_command).\n\n"
        f"For deterministic local evaluation use build_terminalbench_fixture() "
        f"from meta_harness_plus.tasks.terminalbench_fixture instead.\n"
        f"Detected terminal_bench: {tb.__name__}, "
        f"version={getattr(tb, '__version__', 'unknown')}"
    )
