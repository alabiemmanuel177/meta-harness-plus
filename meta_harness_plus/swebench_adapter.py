"""SWE-bench Verified adapter — one-shot patch generation + official eval.

Most SWE-bench submissions use one-shot patch generation rather than a
multi-turn shell-using agent: the LLM reads the problem statement
(plus optionally retrieved repo context), emits a unified-diff patch,
and the official ``swebench.harness.run_evaluation`` harness applies
the patch in a per-task Docker container and runs the FAIL_TO_PASS
tests.

This module wires that flow into MH++'s search machinery:

- ``SWEBenchInstance`` — typed view of one Verified instance.
- ``load_swebench_verified`` — loader from HuggingFace (cached locally).
- ``generate_patch`` — single-call patch generator (Sonnet 4.6 et al.).
- ``score_patches_with_swebench_eval`` — invokes the official harness.

Speed:
- One LLM call per (instance × candidate × sample) — Sonnet ~3-8s/call.
- Anthropic concurrency >> Ollama Pro — 16+ concurrent comfortably.
- Eval is per-instance Docker container; ``swebench.harness`` parallelizes.

Durability:
- Per-(instance, candidate, seed) JSON checkpoint; resume-safe.
- Cost log appends a USD-flavored line per call to ``cost_log.jsonl``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class SWEBenchInstance:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: str
    test_patch: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    version: str
    gold_patch: str  # the reference solution; we never give this to the model


def load_swebench_verified(
    n: int | None = None,
    instance_ids: Iterable[str] | None = None,
    cached_path: str | Path | None = None,
) -> list[SWEBenchInstance]:
    """Load SWE-bench Verified rows. Caches the JSONL locally so future
    calls are network-free."""
    p = Path(cached_path) if cached_path else (
        Path("meta_harness_plus/tasks/data/swebench_verified.jsonl"))
    if p.exists():
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    else:
        from datasets import load_dataset
        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        rows = [dict(r) for r in ds]
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    if instance_ids is not None:
        wanted = set(instance_ids)
        rows = [r for r in rows if r["instance_id"] in wanted]
    if n is not None:
        rows = rows[:n]

    out: list[SWEBenchInstance] = []
    for r in rows:
        ftp = r.get("FAIL_TO_PASS", [])
        if isinstance(ftp, str):
            try:
                ftp = json.loads(ftp)
            except json.JSONDecodeError:
                ftp = [ftp]
        ptp = r.get("PASS_TO_PASS", [])
        if isinstance(ptp, str):
            try:
                ptp = json.loads(ptp)
            except json.JSONDecodeError:
                ptp = [ptp]
        out.append(SWEBenchInstance(
            instance_id=r["instance_id"],
            repo=r["repo"],
            base_commit=r["base_commit"],
            problem_statement=r["problem_statement"],
            hints_text=r.get("hints_text", "") or "",
            test_patch=r.get("test_patch", "") or "",
            fail_to_pass=list(ftp),
            pass_to_pass=list(ptp),
            version=str(r.get("version", "")),
            gold_patch=r.get("patch", "") or "",
        ))
    return out


# ---------------- Patch generation prompt ----------------

DEFAULT_SYSTEM_PROMPT = """You are an expert Python developer fixing a real-world bug in an open-source repository.

You will be given:
- The repository name and version.
- A bug description (the "problem statement").
- Optionally, hints and relevant file contents.

Your job: produce a single unified-diff patch (`diff --git ...` format) that fixes the bug. The patch must be applicable from the repository root with `git apply`.

Output ONLY the diff. Do NOT include explanation, markdown fences, or commentary outside the diff. The diff must start with `diff --git` and end with the final hunk's last line.
"""


PLAN_THEN_PATCH_PROMPT = """You are an expert Python developer. Solve a bug in an open-source repository.

Step 1: In a single short paragraph, identify the file you need to change and the root cause.
Step 2: Then, after the literal string "===PATCH===" on its own line, output a unified-diff patch (`diff --git ...` format). The patch must be applicable from the repo root with `git apply`.

Do not include markdown fences. The patch must end with the final hunk's last line."""


CONCISE_PATCH_PROMPT = """You fix bugs in Python repos by emitting unified-diff patches.

Output ONLY the diff (no commentary, no fences). It must start with `diff --git` and apply cleanly with `git apply` from the repo root."""


def build_user_prompt(
    inst: SWEBenchInstance,
    *,
    include_hints: bool = True,
    extra_context: str = "",
) -> str:
    parts = [
        f"Repository: {inst.repo} (version {inst.version})",
        f"Base commit: {inst.base_commit}",
        "",
        "## Problem statement",
        inst.problem_statement,
    ]
    if include_hints and inst.hints_text:
        parts += ["", "## Hints", inst.hints_text]
    if extra_context:
        parts += ["", "## Relevant context", extra_context]
    parts += ["", "Output the unified-diff patch now."]
    return "\n".join(parts)


# ---------------- Patch parsing ----------------

_DIFF_GIT_RE = re.compile(r"diff --git ", re.MULTILINE)
_FENCE_RE = re.compile(r"^```(?:diff|patch)?\s*\n(.*?)\n```\s*$",
                       re.DOTALL | re.MULTILINE)


def extract_patch(text: str) -> str:
    """Strip code fences, find the first ``diff --git`` block, return
    everything from there to end of text."""
    if not text:
        return ""
    s = text.strip()
    fence = _FENCE_RE.match(s)
    if fence:
        s = fence.group(1).strip()
    if "===PATCH===" in s:
        s = s.split("===PATCH===", 1)[1].strip()
    m = _DIFF_GIT_RE.search(s)
    if not m:
        return ""
    return s[m.start():].strip() + "\n"


# ---------------- Predictions file format ----------------

def write_predictions(
    patches: dict[str, str],
    *,
    model_name: str,
    out_path: str | Path,
) -> None:
    """Write the JSONL predictions file the swebench harness expects.

    Schema (per the swebench docs):
        {"instance_id": "...", "model_name_or_path": "...", "model_patch": "..."}
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for iid, patch in patches.items():
            f.write(json.dumps({
                "instance_id": iid,
                "model_name_or_path": model_name,
                "model_patch": patch,
            }) + "\n")


# ---------------- Eval invocation ----------------

def run_swebench_eval(
    *,
    predictions_path: str | Path,
    run_id: str,
    instance_ids: Iterable[str] | None = None,
    max_workers: int = 4,
    namespace: str | None = "swebench",
    cache_level: str = "instance",
) -> dict[str, dict]:
    """Invoke ``swebench.harness.run_evaluation`` to grade patches.

    Returns a mapping ``{instance_id: result_dict}`` where each result
    has at least ``"resolved": bool``.

    Uses pre-built images from Docker Hub by default (``namespace="swebench"``).
    Image naming on Hub uses ``_1776_`` as the instance separator
    (the dataset's ``__`` is rewritten by the harness). Pulling
    ~2-3 min per instance vs ~15-25 min to build from scratch.

    Set ``namespace=None`` to fall back to local builds (slower).
    """
    from swebench.harness.run_evaluation import main as swebench_run

    swebench_run(
        dataset_name="princeton-nlp/SWE-bench_Verified",
        split="test",
        instance_ids=list(instance_ids) if instance_ids else None,
        predictions_path=str(predictions_path),
        max_workers=max_workers,
        force_rebuild=False,
        cache_level=cache_level,
        clean=False,
        open_file_limit=4096,
        run_id=run_id,
        timeout=1800,
        namespace=namespace,
        rewrite_reports=False,
        modal=False,
        instance_image_tag="latest",
        env_image_tag="latest",
        report_dir=".",
    )

    # The harness writes results to ``logs/run_evaluation/<run_id>/<model>/<instance>/``.
    # Aggregate by reading the per-instance ``report.json`` files.
    log_root = Path("logs/run_evaluation") / run_id
    results: dict[str, dict] = {}
    if not log_root.exists():
        return results
    for model_dir in log_root.iterdir():
        if not model_dir.is_dir():
            continue
        for inst_dir in model_dir.iterdir():
            if not inst_dir.is_dir():
                continue
            report_path = inst_dir / "report.json"
            if report_path.exists():
                rec = json.loads(report_path.read_text())
                # Per-instance report shape varies by swebench version.
                resolved = rec.get(inst_dir.name, {}).get("resolved", False)
                results[inst_dir.name] = {
                    "resolved": resolved,
                    "report": rec.get(inst_dir.name, rec),
                }
    return results
