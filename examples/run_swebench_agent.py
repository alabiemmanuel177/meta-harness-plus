"""SWE-bench Verified multi-turn agent runner — Sonnet 4.6 ReAct loop
inside per-task Docker containers.

Runs N agent shapes × M tasks. Per-task flow:

1. Pull (or reuse) the swebench instance image, start a container.
2. Multi-turn agent loop with read_file / write_file / run_tests /
   list_files / bash / done tools.
3. After loop ends (done emitted, max_turns hit, or budget tripped),
   capture ``git diff`` of the working tree as the patch.
4. Write predictions JSONL, run ``swebench.harness.run_evaluation``
   to grade the patch against FAIL_TO_PASS + PASS_TO_PASS.
5. Persist per-(shape × task × seed) trajectory + result to disk.

This module is the BASELINE-runnable form of the multi-turn agent.
Halving search across multiple agent shapes lives in
``examples/run_swebench_agent_search.py`` (built on top of this).

Resume-safe: skips per-(shape × task × seed) cells whose result JSON
already exists. Per-task containers are cleaned up on every exit path
(success, timeout, error).

Usage::

    python3 examples/run_swebench_agent.py --n 1 --shape baseline \\
        --instance sympy__sympy-22914   # smoke
    python3 examples/run_swebench_agent.py --n 50 --shape baseline \\
        --seeds 1                        # full 50-task baseline
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from examples.run_ollama_cloud_search import (  # type: ignore
    _percentile, _load_dotenv_if_present, COST_LOG,
)
from meta_harness_plus.agent_docker import DockerShellExecutor
from meta_harness_plus.agent_swebench_loop import (
    BudgetGuard,
    Trajectory,
    price_for_model,
    run_agent_loop,
)
from meta_harness_plus.swebench_adapter import (
    SWEBenchInstance,
    load_swebench_verified,
    run_swebench_eval,
    write_predictions,
)


# ---------------- Anthropic chat_fn ----------------

def _build_anthropic_chat_fn(api_key: str, model: str):
    """Return a callable ``chat_fn(system, messages, tools, max_tokens,
    temperature) -> response_dict`` that hits Anthropic's Messages
    API with native tool-use."""
    import urllib.request, urllib.error

    URL = "https://api.anthropic.com/v1/messages"

    def chat_fn(*, system, messages, tools, max_tokens=4096, temperature=0.0):
        body = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": messages,
            "tools": tools,
        }
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            URL, data=data,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        # Retries: Anthropic 429 / 5xx with exponential backoff.
        max_retries = 4
        delay = 2.0
        for attempt in range(max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                code = e.code
                if code in {429, 500, 502, 503, 504} and attempt < max_retries:
                    time.sleep(delay)
                    delay = min(delay * 2, 30.0)
                    continue
                detail = e.read().decode("utf-8", errors="replace")[:300]
                raise RuntimeError(f"Anthropic HTTP {code}: {detail}") from e
        raise RuntimeError("retries exhausted")

    return chat_fn


# ---------------- Ollama (OpenAI-compat) chat_fn ----------------

def _anthropic_to_openai(messages: list[dict], system: str, tools: list[dict]) -> tuple[list[dict], list[dict]]:
    """Translate the loop's Anthropic-shaped state to OpenAI chat-completions
    shape. Returns (oai_messages, oai_tools)."""
    oai_msgs: list[dict] = [{"role": "system", "content": system}]
    for m in messages:
        role = m["role"]
        content = m["content"]
        if isinstance(content, str):
            oai_msgs.append({"role": role, "content": content})
            continue
        # content is a list of content blocks
        if role == "assistant":
            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            tool_calls: list[dict] = []
            for block in content:
                t = block.get("type")
                if t == "text":
                    text_parts.append(block.get("text", ""))
                elif t == "reasoning_content":
                    # DeepSeek thinking-mode round-trip: we captured this on
                    # the prior response, must echo back on next request.
                    reasoning_parts.append(block.get("text", ""))
                elif t == "tool_use":
                    tool_calls.append({
                        "id": block.get("id"),
                        "type": "function",
                        "function": {
                            "name": block.get("name"),
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    })
            text = "\n".join(text_parts)
            # OpenAI-compat (incl. DeepSeek) rejects assistant messages where
            # both content is null AND no tool_calls. Always have at least one
            # of them set; empty-string content is accepted alongside tool_calls.
            msg: dict = {"role": "assistant"}
            if tool_calls:
                msg["tool_calls"] = tool_calls
                msg["content"] = text  # empty string is fine; null isn't
            else:
                msg["content"] = text or " "  # avoid bare null on text-only paths
            if reasoning_parts:
                # DeepSeek requires reasoning_content from prior turn echoed back.
                msg["reasoning_content"] = "\n".join(reasoning_parts)
            oai_msgs.append(msg)
        elif role == "user":
            # tool_result blocks → individual {role: "tool"} messages.
            text_parts: list[str] = []
            for block in content:
                if block.get("type") == "tool_result":
                    obs = block.get("content", "")
                    if isinstance(obs, list):
                        obs = "\n".join(b.get("text", "") for b in obs if b.get("type") == "text")
                    oai_msgs.append({
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id"),
                        "content": obs or "",
                    })
                elif block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            if text_parts:
                oai_msgs.append({"role": "user", "content": "\n".join(text_parts)})
    oai_tools = [
        {"type": "function",
         "function": {
             "name": t["name"],
             "description": t["description"],
             "parameters": t["input_schema"],
         }}
        for t in tools
    ]
    return oai_msgs, oai_tools


def _openai_to_anthropic_response(resp: dict) -> dict:
    """Convert OpenAI chat-completions response → loop's Anthropic shape.

    Preserves DeepSeek's `reasoning_content` field (emitted by deepseek-reasoner
    in thinking mode) as a synthetic 'reasoning_content' block so it can be
    echoed back on subsequent turns — DeepSeek's API rejects requests that
    drop it.
    """
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message", {}) or {}
    blocks: list[dict] = []
    # DeepSeek thinking-mode: capture reasoning_content first so it round-trips.
    rc = msg.get("reasoning_content")
    if rc:
        blocks.append({"type": "reasoning_content", "text": rc})
    if msg.get("content"):
        blocks.append({"type": "text", "text": msg["content"]})
    for tc in msg.get("tool_calls", []) or []:
        fn = tc.get("function", {}) or {}
        try:
            inp = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            inp = {"_raw_arguments": fn.get("arguments", "")}
        blocks.append({
            "type": "tool_use",
            "id": tc.get("id") or f"call_{len(blocks)}",
            "name": fn.get("name", ""),
            "input": inp,
        })
    finish = choice.get("finish_reason", "end_turn")
    stop = {"tool_calls": "tool_use", "length": "max_tokens",
            "stop": "end_turn", "end_turn": "end_turn"}.get(finish, finish)
    usage = resp.get("usage", {}) or {}
    return {
        "content": blocks,
        "stop_reason": stop,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def _build_ollama_chat_fn(api_key: str, model: str, base_url: str = "https://ollama.com"):
    """Ollama Cloud OpenAI-compat tool-use chat_fn. Same shape as the
    Anthropic builder so the agent loop is provider-agnostic."""
    import urllib.request, urllib.error
    url = base_url.rstrip("/") + "/v1/chat/completions"

    def chat_fn(*, system, messages, tools, max_tokens=4096, temperature=0.0):
        oai_msgs, oai_tools = _anthropic_to_openai(messages, system, tools)
        body = {
            "model": model,
            "messages": oai_msgs,
            "tools": oai_tools,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            url, data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "content-type": "application/json",
            },
        )
        max_retries = 4
        delay = 2.0
        for attempt in range(max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    raw = json.loads(r.read())
                return _openai_to_anthropic_response(raw)
            except urllib.error.HTTPError as e:
                code = e.code
                if code in {429, 500, 502, 503, 504} and attempt < max_retries:
                    time.sleep(delay)
                    delay = min(delay * 2, 30.0)
                    continue
                detail = e.read().decode("utf-8", errors="replace")[:500]
                raise RuntimeError(f"Ollama HTTP {code}: {detail}") from e
        raise RuntimeError("retries exhausted")

    return chat_fn


# ---------------- DeepSeek API direct chat_fn ----------------

def _build_deepseek_chat_fn(api_key: str, model: str,
                            base_url: str = "https://api.deepseek.com",
                            request_timeout: float = 600.0):
    """DeepSeek API-direct OpenAI-compat tool-use chat_fn. Same shape as the
    Anthropic builder. Implements robust 429/5xx backoff with jitter."""
    import urllib.request, urllib.error, random
    url = base_url.rstrip("/") + "/v1/chat/completions"

    def chat_fn(*, system, messages, tools, max_tokens=4096, temperature=0.0):
        oai_msgs, oai_tools = _anthropic_to_openai(messages, system, tools)
        body = {
            "model": model,
            "messages": oai_msgs,
            "tools": oai_tools,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            url, data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "content-type": "application/json",
            },
        )
        max_retries = 6
        delay = 2.0
        for attempt in range(max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=request_timeout) as r:
                    raw = json.loads(r.read())
                return _openai_to_anthropic_response(raw)
            except urllib.error.HTTPError as e:
                code = e.code
                if code in {429, 500, 502, 503, 504} and attempt < max_retries:
                    # Exponential backoff with jitter; 429s get longer waits.
                    sleep_s = delay + random.uniform(0, delay * 0.5)
                    if code == 429:
                        sleep_s *= 1.5
                    time.sleep(min(sleep_s, 60.0))
                    delay = min(delay * 2, 60.0)
                    continue
                detail = e.read().decode("utf-8", errors="replace")[:500]
                raise RuntimeError(f"DeepSeek HTTP {code}: {detail}") from e
            except (TimeoutError, urllib.error.URLError) as e:
                if attempt < max_retries:
                    time.sleep(min(delay, 30.0))
                    delay = min(delay * 2, 60.0)
                    continue
                raise RuntimeError(f"DeepSeek timeout/url err: {e}") from e
        raise RuntimeError("DeepSeek retries exhausted")

    return chat_fn


# ---------------- Shapes ----------------

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert Python developer fixing a real-world bug in an "
    "open-source repository. You have shell tools to read and write files, "
    "run tests, and inspect the codebase.\n\n"
    "Process:\n"
    "1. Read the problem statement carefully.\n"
    "2. Locate the relevant file(s) (use search_text, list_files, and "
    "read_file_range for large files; avoid repeatedly reading whole files).\n"
    "3. Make a targeted production-code fix (prefer replace_text for small edits; use "
    "write_file only for small or new files).\n"
    "4. Run the failing tests or a focused reproduction (use run_tests or run_python).\n"
    "5. If tests pass, call `done`. If not, iterate.\n\n"
    "SWE-bench note: the official hidden test patch is usually not applied "
    "inside this checkout. If a requested pytest selector cannot be collected, "
    "do not add or edit tests to create it. Fix production code from the issue "
    "description, then run a nearby existing test or a focused run_python "
    "reproduction.\n\n"
    "Test-file edits are blocked. The submitted patch is production-code focused.\n\n"
    "Be efficient — bias toward small, surgical fixes over large rewrites. "
    "If an action fails or produces the same result twice, change approach "
    "instead of repeating it."
)

PLAN_FIRST_SYSTEM_PROMPT = (
    "You are an expert Python developer. You will fix a real bug in a "
    "production repo using a tool-using shell.\n\n"
    "On your FIRST turn: do not edit code. Instead read 1-3 files to "
    "understand the surface area. Prefer search_text and read_file_range "
    "before whole-file reads. Then in subsequent turns, plan the "
    "fix as a 1-paragraph thought, then write the production-code fix with replace_text "
    "when possible, then verify with run_tests. If a requested hidden "
    "SWE-bench test selector is not present, do not create tests for it; "
    "run a nearby existing test or run_python reproduction instead. Test-file "
    "edits are blocked. Call `done` only when tests pass."
)


@dataclass(frozen=True)
class AgentShape:
    """One harness configuration searchable by MH++."""
    name: str
    system_prompt: str
    tool_set: str            # "minimal" | "full"
    temperature: float
    max_turns: int
    max_tokens_per_turn: int

    def label(self) -> str:
        return self.name


SHAPES: dict[str, AgentShape] = {
    "baseline": AgentShape(
        name="baseline",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        tool_set="minimal",
        temperature=0.0,
        max_turns=15,
        max_tokens_per_turn=8192,
    ),
    "plan_first": AgentShape(
        name="plan_first",
        system_prompt=PLAN_FIRST_SYSTEM_PROMPT,
        tool_set="minimal",
        temperature=0.0,
        max_turns=20,
        max_tokens_per_turn=8192,
    ),
    "full_tools": AgentShape(
        name="full_tools",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        tool_set="full",
        temperature=0.0,
        max_turns=20,
        max_tokens_per_turn=8192,
    ),
    "high_temp": AgentShape(
        name="high_temp",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        tool_set="minimal",
        temperature=0.7,
        max_turns=15,
        max_tokens_per_turn=8192,
    ),
}


# ---------------- Per-task runner ----------------

def _abbrev_input(inp: dict) -> dict:
    """Abbreviate noisy fields (large file content) for trajectory dumps."""
    out = {}
    for k, v in inp.items():
        if isinstance(v, str) and len(v) > 400:
            out[k] = v[:400] + f"…[{len(v) - 400} more chars]"
        else:
            out[k] = v
    return out




def _record_anthropic_call(
    *, phase: str, label: str, model: str,
    in_tokens: int, out_tokens: int, latency_ms: float, usd: float,
) -> None:
    """Append an Anthropic cost line to runs/wow_push/cost_log.jsonl."""
    COST_LOG.parent.mkdir(parents=True, exist_ok=True)
    with COST_LOG.open("a") as f:
        f.write(json.dumps({
            "ts": time.time(),
            "phase": phase, "provider": "anthropic", "label": label,
            "model": model, "calls": 1,
            "in_tokens": in_tokens, "out_tokens": out_tokens,
            "latency_ms": round(latency_ms, 1),
            "usd": round(usd, 6),
        }) + "\n")


def run_one_task(
    *,
    inst: SWEBenchInstance,
    shape: AgentShape,
    seed: int,
    chat_fn,
    model: str,
    cache_dir: Path,
    budget_guard: BudgetGuard,
    cleanup_image: bool = False,
    verbose_log: bool = False,
) -> dict:
    """Run the agent loop on one (instance, shape, seed) and return a
    record with the patch + trajectory summary. Resume-safe."""
    cache_path = cache_dir / "trajectories" / (
        f"{inst.instance_id}_{shape.label()}_seed{seed}.json"
    )
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    sh = DockerShellExecutor(
        inst.instance_id,
        no_network=False,      # need network briefly to install pytest
        ensure_pytest=True,    # SWE-bench testbed envs ship without pytest
    )
    image_name = sh.image  # capture for optional cleanup
    rec: dict = {
        "instance_id": inst.instance_id,
        "shape": shape.label(),
        "seed": seed,
        "config": asdict(shape),
    }
    traj = None
    t0 = time.perf_counter()
    try:
        sh.start()
        traj = run_agent_loop(
            instance_id=inst.instance_id,
            problem_statement=inst.problem_statement,
            fail_to_pass=inst.fail_to_pass,
            sh=sh,
            chat_fn=chat_fn,
            system_prompt=shape.system_prompt,
            tool_set=shape.tool_set,
            temperature=shape.temperature,
            max_tokens_per_turn=shape.max_tokens_per_turn,
            budget_guard=budget_guard,
            price_per_mtok=price_for_model(model),
        )
        rec.update({
            "n_turns": len(traj.turns),
            "done_emitted": traj.done_emitted,
            "stop_reason": traj.stop_reason,
            "in_tokens": traj.total_in_tokens,
            "out_tokens": traj.total_out_tokens,
            "wall_s": traj.total_wall_s,
            "usd": traj.total_usd,
            "patch": traj.final_patch,
            "patch_len": len(traj.final_patch),
            "turns_detail": [
                {
                    "i": t.turn_idx,
                    "text": t.text[:1000],
                    "tool_calls": [
                        {"name": c["name"],
                         "input": _abbrev_input(c["input"])}
                        for c in t.tool_calls
                    ],
                    "tool_results": [
                        {"obs": (r.get("content", "") or "")[:600]}
                        for r in t.tool_results
                    ],
                    "in_tokens": t.in_tokens,
                    "out_tokens": t.out_tokens,
                    "wall_ms": round(t.wall_ms, 1),
                    "stop_reason": t.stop_reason,
                }
                for t in traj.turns
            ],
        })
        if traj.total_in_tokens or traj.total_out_tokens:
            _record_anthropic_call(
                phase="swebench_agent",
                label=f"{shape.label()}_seed{seed}_{inst.instance_id}",
                model=model,
                in_tokens=traj.total_in_tokens,
                out_tokens=traj.total_out_tokens,
                latency_ms=traj.total_wall_s * 1000,
                usd=traj.total_usd,
            )
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["patch"] = ""
    finally:
        rec["task_wall_s"] = round(time.perf_counter() - t0, 1)
        sh.cleanup()
        if cleanup_image:
            # Remove instance image to keep disk lean.
            # Base/env layers stay cached and shared with sibling instances.
            try:
                import subprocess as _sp
                _sp.run(["docker", "rmi", image_name],
                        capture_output=True, timeout=30)
            except Exception:
                pass

    if verbose_log and traj is not None:
        # Write FULL-fidelity trajectory dump (no abbreviation) for audit.
        verbose_dir = cache_dir / "trajectories_full"
        verbose_dir.mkdir(parents=True, exist_ok=True)
        verbose_path = verbose_dir / (
            f"{inst.instance_id}_{shape.label()}_seed{seed}.json"
        )
        try:
            verbose_payload = {
                **{k: v for k, v in rec.items() if k != "turns_detail"},
                "system_prompt": shape.system_prompt,
                "problem_statement": inst.problem_statement,
                "fail_to_pass": inst.fail_to_pass,
                "turns_full": [
                    {
                        "i": t.turn_idx,
                        "text": t.text,
                        "tool_calls": [
                            {"name": c["name"], "input": c["input"]}
                            for c in t.tool_calls
                        ],
                        "tool_results": [
                            {"obs": r.get("content", "") or r.get("text", "")}
                            for r in t.tool_results
                        ],
                        "in_tokens": t.in_tokens,
                        "out_tokens": t.out_tokens,
                        "wall_ms": round(t.wall_ms, 1),
                        "stop_reason": t.stop_reason,
                    }
                    for t in traj.turns
                ],
            }
            verbose_path.write_text(json.dumps(verbose_payload, indent=2))
        except Exception:
            pass

    cache_path.write_text(json.dumps(rec, indent=2))
    return rec


def evaluate_patches(
    *, records: list[dict], run_id: str, predictions_dir: Path,
    model: str,
) -> dict[str, bool]:
    """Write a predictions JSONL and run swebench eval. Returns
    ``{instance_id: resolved}``."""
    predictions = {r["instance_id"]: r.get("patch", "") for r in records}
    preds_path = predictions_dir / f"{run_id}.jsonl"
    write_predictions(predictions, model_name=model, out_path=preds_path)
    try:
        results = run_swebench_eval(
            predictions_path=preds_path,
            run_id=run_id,
            instance_ids=list(predictions.keys()),
            max_workers=8,
        )
    except Exception as e:
        print(f"[agent] eval crashed: {type(e).__name__}: {e}")
        return {}
    return {iid: bool(r.get("resolved", False)) for iid, r in results.items()}


# ---------------- Driver ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50,
                    help="Number of SWE-bench-Verified instances.")
    ap.add_argument("--instance", default=None,
                    help="Single instance for smoke; overrides --n.")
    ap.add_argument("--shape", default="baseline", choices=list(SHAPES.keys()))
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--provider", default="anthropic",
                    choices=["anthropic", "ollama", "deepseek"],
                    help="LLM provider for the agent loop.")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--max-workers", type=int, default=4,
                    help="Parallel agent loops (each owns a Docker container).")
    ap.add_argument("--max-turns", type=int, default=None,
                    help="Override shape's max_turns.")
    ap.add_argument("--max-usd", type=float, default=1.50,
                    help="Per-task USD budget cap.")
    ap.add_argument("--max-tokens-in", type=int, default=None,
                    help="Override cumulative input-token cap per task.")
    ap.add_argument("--max-wall-s", type=float, default=600.0)
    ap.add_argument("--out-dir", default="runs/swebench_agent")
    ap.add_argument("--run-id", default="agent_v1")
    ap.add_argument("--skip-eval", action="store_true",
                    help="Run agent loops but don't grade the patches.")
    args = ap.parse_args()

    _load_dotenv_if_present()
    if args.provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit("ANTHROPIC_API_KEY missing")
    elif args.provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise SystemExit("DEEPSEEK_API_KEY missing")
    else:
        api_key = os.environ.get("OLLAMA_API_KEY")
        if not api_key:
            raise SystemExit("OLLAMA_API_KEY missing")

    # Load instances.
    if args.instance:
        instances = load_swebench_verified(instance_ids=[args.instance])
    else:
        instances = load_swebench_verified(n=args.n)
    print(f"[agent] running shape={args.shape} seeds={args.seeds} "
          f"on {len(instances)} instances; max_workers={args.max_workers}")

    shape = SHAPES[args.shape]
    if args.max_turns is not None:
        shape = AgentShape(
            name=shape.name, system_prompt=shape.system_prompt,
            tool_set=shape.tool_set, temperature=shape.temperature,
            max_turns=args.max_turns,
            max_tokens_per_turn=shape.max_tokens_per_turn,
        )

    budget = BudgetGuard(
        max_turns=shape.max_turns,
        max_tokens_in=(
            args.max_tokens_in
            if args.max_tokens_in is not None
            else BudgetGuard().max_tokens_in
        ),
        max_usd=args.max_usd,
        max_wall_s=args.max_wall_s,
    )
    if args.provider == "anthropic":
        chat_fn = _build_anthropic_chat_fn(api_key, args.model)
    elif args.provider == "deepseek":
        chat_fn = _build_deepseek_chat_fn(api_key, args.model)
    else:
        chat_fn = _build_ollama_chat_fn(api_key, args.model)

    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "cache"

    # Run agent loops in parallel.
    pairs = [(inst, seed) for inst in instances for seed in range(args.seeds)]
    print(f"[agent] {len(pairs)} (instance × seed) cells to run")

    t0 = time.perf_counter()
    records: list[dict] = []
    n_done = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futs = {
            pool.submit(
                run_one_task,
                inst=inst, shape=shape, seed=seed, chat_fn=chat_fn,
                model=args.model, cache_dir=cache_dir, budget_guard=budget,
            ): (inst.instance_id, seed)
            for inst, seed in pairs
        }
        for fut in as_completed(futs):
            iid, seed = futs[fut]
            rec = fut.result()
            records.append(rec)
            n_done += 1
            elapsed = time.perf_counter() - t0
            n_turns = rec.get("n_turns", 0)
            usd = rec.get("usd", 0)
            stop = rec.get("stop_reason", "")
            patch_len = rec.get("patch_len", 0)
            print(f"[agent] +{elapsed:6.0f}s  {iid:35s} seed{seed} "
                  f"turns={n_turns:2d} usd={usd:.3f} "
                  f"patch={patch_len:5d}b stop={stop:18s}  "
                  f"({n_done}/{len(pairs)})")

    sweep_wall = time.perf_counter() - t0
    print(f"\n[agent] sweep wall: {sweep_wall:.0f}s")
    total_usd = sum(r.get("usd", 0) for r in records)
    print(f"[agent] total agent cost: ${total_usd:.3f}")

    # Evaluation.
    if args.skip_eval:
        out_path = out_dir / f"{args.run_id}_{args.shape}.json"
        out_path.write_text(json.dumps({
            "shape": args.shape, "model": args.model, "seeds": args.seeds,
            "n_instances": len(instances),
            "records": records, "sweep_wall_s": sweep_wall,
            "total_usd": total_usd, "evaluated": False,
        }, indent=2))
        print(f"[agent] wrote {out_path} (eval skipped)")
        return

    # Eval per seed (separate predictions JSONL per seed).
    per_seed_resolved: dict[int, dict[str, bool]] = {}
    for seed in range(args.seeds):
        seed_records = [r for r in records if r["seed"] == seed]
        run_id_full = f"{args.run_id}_{args.shape}_seed{seed}"
        print(f"\n[agent] grading seed {seed} via swebench eval...")
        res = evaluate_patches(
            records=seed_records, run_id=run_id_full,
            predictions_dir=out_dir / "predictions", model=args.model,
        )
        per_seed_resolved[seed] = res
        n_resolved = sum(1 for v in res.values() if v)
        print(f"[agent] seed {seed}: {n_resolved}/{len(res)} resolved "
              f"({n_resolved/max(1,len(res)):.1%})")

    out_path = out_dir / f"{args.run_id}_{args.shape}.json"
    out_path.write_text(json.dumps({
        "shape": args.shape, "model": args.model, "seeds": args.seeds,
        "n_instances": len(instances),
        "records": records, "per_seed_resolved": {
            str(k): v for k, v in per_seed_resolved.items()
        },
        "sweep_wall_s": sweep_wall,
        "total_usd": total_usd, "evaluated": True,
    }, indent=2))
    print(f"[agent] wrote {out_path}")


if __name__ == "__main__":
    main()
