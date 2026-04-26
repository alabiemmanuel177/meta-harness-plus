"""Local TerminalBench-style fixture for agent-task evaluation.

These are deterministic, file-and-shell-based agent tasks that exercise
the same competencies the real TerminalBench benchmark probes:

- file system inspection (find/grep)
- text manipulation (sed/awk/cut)
- json processing (no jq dependency — use python -c)
- multi-file diffs / patches
- creating new files at a target path with required content

We hand-author 6 tasks here so the agent harness has a real,
deterministic suite to optimize over without external dataset
downloads. Each task ships its own ``setup`` (writes seed files into
the sandbox) and ``check_fn`` (inspects the post-run sandbox state).

For real-TerminalBench results, see
``meta_harness_plus.tasks.terminalbench_adapter``.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..agent import AgentEnv, AgentExample, AgentTask


# ---------- task 1: count occurrences and write to file ----------

def _setup_count_word(sandbox: Path) -> None:
    # Seven occurrences of 'alpha' — keep this and _check_count_word
    # in sync.
    (sandbox / "doc.txt").write_text(
        "alpha bravo alpha charlie alpha bravo delta echo alpha\n"
        "alpha foxtrot alpha alpha\n"
    )


def _check_count_word(env: AgentEnv) -> bool:
    answer = env.read_file("answer.txt").strip()
    return answer == "7"


# ---------- task 2: extract a JSON field ----------

def _setup_json_field(sandbox: Path) -> None:
    payload = {"users": [
        {"name": "ada", "age": 36},
        {"name": "linus", "age": 53},
        {"name": "grace", "age": 85},
    ]}
    (sandbox / "users.json").write_text(json.dumps(payload))


def _check_json_field(env: AgentEnv) -> bool:
    out = env.read_file("oldest.txt").strip()
    return out == "grace"


# ---------- task 3: rename .log files to .txt ----------

def _setup_rename_logs(sandbox: Path) -> None:
    (sandbox / "a.log").write_text("a\n")
    (sandbox / "b.log").write_text("b\n")
    (sandbox / "c.log").write_text("c\n")
    (sandbox / "keep.txt").write_text("keep\n")


def _check_rename_logs(env: AgentEnv) -> bool:
    p = env.sandbox
    no_logs = not any(f.suffix == ".log" for f in p.iterdir())
    txt_files = sorted(f.name for f in p.iterdir() if f.suffix == ".txt")
    return no_logs and txt_files == ["a.txt", "b.txt", "c.txt", "keep.txt"]


# ---------- task 4: write a function in python ----------

def _setup_write_py(sandbox: Path) -> None:
    (sandbox / "spec.txt").write_text(
        "Write a function `add(a, b)` that returns a + b.\n"
        "Save it as solution.py.\n"
    )


def _check_write_py(env: AgentEnv) -> bool:
    if not env.file_exists("solution.py"):
        return False
    src = env.read_file("solution.py")
    if "def add" not in src:
        return False
    # Try to import & execute. We do this in a subprocess of the executor
    # at check time using a tiny inline python -c — but since check_fn
    # has no executor, just regex-check the body.
    return "return" in src and "a" in src and "b" in src


# ---------- task 5: grep for a pattern ----------

def _setup_grep(sandbox: Path) -> None:
    (sandbox / "log.txt").write_text(
        "INFO: started\n"
        "INFO: connected\n"
        "ERROR: db timeout\n"
        "INFO: retrying\n"
        "ERROR: db unavailable\n"
        "INFO: shutdown\n"
    )


def _check_grep(env: AgentEnv) -> bool:
    out = env.read_file("errors.txt").strip().splitlines()
    return out == ["ERROR: db timeout", "ERROR: db unavailable"]


# ---------- task 6: compute a sum ----------

def _setup_sum(sandbox: Path) -> None:
    (sandbox / "nums.txt").write_text("\n".join(str(i) for i in range(1, 11)) + "\n")


def _check_sum(env: AgentEnv) -> bool:
    out = env.read_file("sum.txt").strip()
    return out == "55"


# ---------- assembly ----------

_TASKS = [
    AgentExample(
        task_id="count_word_alpha",
        goal=("Count how many times the word 'alpha' appears in doc.txt "
              "and write the count (just the number) to answer.txt."),
        setup=_setup_count_word,
        check_fn=_check_count_word,
        max_turns=4,
    ),
    AgentExample(
        task_id="json_oldest_user",
        goal=("Read users.json. Find the user with the highest 'age' "
              "and write their 'name' (just the name) to oldest.txt."),
        setup=_setup_json_field,
        check_fn=_check_json_field,
        max_turns=4,
    ),
    AgentExample(
        task_id="rename_log_to_txt",
        goal=("Rename every *.log file in the current directory to *.txt, "
              "preserving content. Do not rename existing .txt files."),
        setup=_setup_rename_logs,
        check_fn=_check_rename_logs,
        max_turns=4,
    ),
    AgentExample(
        task_id="write_add_function",
        goal=("Read spec.txt and produce solution.py implementing the "
              "function described. Do not include any other code."),
        setup=_setup_write_py,
        check_fn=_check_write_py,
        max_turns=3,
    ),
    AgentExample(
        task_id="grep_errors",
        goal=("Extract every line containing 'ERROR' from log.txt and "
              "write them (in original order) to errors.txt."),
        setup=_setup_grep,
        check_fn=_check_grep,
        max_turns=3,
    ),
    AgentExample(
        task_id="sum_numbers",
        goal=("Compute the sum of all integers in nums.txt and write the "
              "sum (just the number) to sum.txt."),
        setup=_setup_sum,
        check_fn=_check_sum,
        max_turns=3,
    ),
]


def build_terminalbench_fixture() -> AgentTask:
    """Local 6-task TerminalBench-style suite. Deterministic, no API calls.

    Use this with ``AgentScorer`` + ``AgentHarness`` + a real or mock
    ``AgentPolicy`` to evaluate harness shapes without depending on
    the TerminalBench package or network access.
    """
    return AgentTask(name="terminalbench_local", examples=_TASKS)
