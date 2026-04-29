"""SWE-bench Verified 1-task smoke — generate one patch with Sonnet 4.6
and grade it with the official harness.

What this validates end-to-end:
1. Anthropic API key + HTTPClient routing to Sonnet 4.6.
2. Patch extraction from the LLM output (handles fences / chatter).
3. Predictions JSONL file format.
4. swebench.harness.run_evaluation invocation (Docker build, test run).
5. Per-instance result file parsing.

Cost estimate: 1 Sonnet 4.6 call (~$0.05–0.30) + ~5–15 min Docker work.
Resume-safe: skips if ``runs/swebench_smoke/<instance>_<run_id>.json``
already exists.

Usage:
    python3 examples/run_swebench_smoke.py
    python3 examples/run_swebench_smoke.py --instance sympy__sympy-22914
    python3 examples/run_swebench_smoke.py --include-test-patch  # cheats: gives the model the failing tests for inspection
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from examples.run_ollama_cloud_search import (  # type: ignore
    _percentile, _record_ollama_batch, _load_dotenv_if_present, COST_LOG,
)
from meta_harness_plus.llm.client import HTTPClient
from meta_harness_plus.swebench_adapter import (
    DEFAULT_SYSTEM_PROMPT,
    SWEBenchInstance,
    build_user_prompt,
    extract_patch,
    load_swebench_verified,
    run_swebench_eval,
    write_predictions,
)


# Pricing (USD per million tokens) for closed-API cost-log lines.
SONNET_4_6_PRICE = {"in": 3.0, "out": 15.0}


def _record_anthropic_call(
    *, phase: str, label: str, model: str,
    in_tokens: int, out_tokens: int, latency_ms: float,
) -> None:
    """Cost log entry for closed-API calls (with usd estimate)."""
    cost = (in_tokens * SONNET_4_6_PRICE["in"]
            + out_tokens * SONNET_4_6_PRICE["out"]) / 1_000_000.0
    COST_LOG.parent.mkdir(parents=True, exist_ok=True)
    with COST_LOG.open("a") as f:
        f.write(json.dumps({
            "ts": time.time(),
            "phase": phase, "provider": "anthropic", "label": label,
            "model": model, "calls": 1,
            "in_tokens": in_tokens, "out_tokens": out_tokens,
            "latency_ms": round(latency_ms, 1),
            "usd": round(cost, 6),
        }) + "\n")


def _generate_patch(
    inst: SWEBenchInstance,
    *,
    api_key: str,
    model: str = "claude-sonnet-4-6",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    include_hints: bool = True,
) -> tuple[str, dict]:
    """One LLM call → (patch_text, telemetry_dict)."""
    client = HTTPClient(
        api_url="https://api.anthropic.com/v1/messages",
        api_key=api_key,
        model=model,
        timeout_s=300.0,
        max_retries=4,
        retry_base_delay=2.0, retry_max_delay=30.0,
    )
    user = build_user_prompt(inst, include_hints=include_hints)
    t0 = time.perf_counter()
    resp = client.complete(
        system=system_prompt, user=user,
        max_tokens=max_tokens, temperature=temperature,
    )
    wall_ms = (time.perf_counter() - t0) * 1000
    patch = extract_patch(resp.text)
    return patch, {
        "in_tokens": resp.input_tokens,
        "out_tokens": resp.output_tokens,
        "latency_ms": resp.latency_ms,
        "wall_ms": wall_ms,
        "raw_text_first_400": resp.text[:400],
        "patch_first_500": patch[:500],
        "patch_extracted": bool(patch),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="sympy__sympy-22914",
                    help="Instance ID. Default is a small 1-FAIL_TO_PASS sympy task.")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--include-hints", action="store_true",
                    help="Pass `hints_text` from the dataset (debugging only).")
    ap.add_argument("--skip-eval", action="store_true",
                    help="Generate the patch but do not invoke the Docker eval.")
    ap.add_argument("--out-dir", default="runs/swebench_smoke")
    ap.add_argument("--run-id", default="smoke_v1")
    args = ap.parse_args()

    _load_dotenv_if_present()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit(
            "ANTHROPIC_API_KEY not set. Add to .env or export."
        )

    print(f"[smoke] loading SWE-bench Verified instance {args.instance!r}")
    insts = load_swebench_verified(instance_ids=[args.instance])
    if not insts:
        raise SystemExit(f"instance {args.instance!r} not in SWE-bench Verified")
    inst = insts[0]
    print(f"[smoke] repo={inst.repo} version={inst.version} "
          f"FAIL_TO_PASS={len(inst.fail_to_pass)} tests")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    record_path = out_dir / f"{inst.instance_id}_{args.run_id}.json"

    if record_path.exists():
        print(f"[smoke] skip — {record_path} already exists")
        return

    # Step 1 — generate patch.
    print(f"[smoke] generating patch with {args.model} "
          f"(T={args.temperature}, max_tokens={args.max_tokens})...")
    patch, tel = _generate_patch(
        inst, api_key=api_key, model=args.model,
        temperature=args.temperature, max_tokens=args.max_tokens,
        include_hints=args.include_hints,
    )
    print(f"[smoke] patch generation done; in={tel['in_tokens']} "
          f"out={tel['out_tokens']} wall={tel['wall_ms']:.0f}ms "
          f"patch_extracted={tel['patch_extracted']}")
    _record_anthropic_call(
        phase="swebench_smoke", label=f"gen_{inst.instance_id}",
        model=args.model,
        in_tokens=tel["in_tokens"], out_tokens=tel["out_tokens"],
        latency_ms=tel["wall_ms"],
    )

    if not patch:
        print(f"[smoke] FAIL — no patch extractable from model output")
        print(f"[smoke] first 400 chars of raw response:")
        print(tel["raw_text_first_400"])
        record_path.write_text(json.dumps({
            "instance_id": inst.instance_id,
            "patch": "", "patch_extracted": False,
            "telemetry": tel, "resolved": False,
            "stage": "extract_failed",
        }, indent=2))
        sys.exit(2)

    print(f"[smoke] patch (first 400 chars):")
    print(tel["patch_first_500"][:400])

    # Step 2 — write predictions JSONL.
    preds_path = out_dir / f"predictions_{args.run_id}.jsonl"
    write_predictions(
        {inst.instance_id: patch},
        model_name=args.model,
        out_path=preds_path,
    )
    print(f"[smoke] wrote {preds_path}")

    if args.skip_eval:
        record_path.write_text(json.dumps({
            "instance_id": inst.instance_id,
            "patch": patch, "patch_extracted": True,
            "telemetry": tel,
            "resolved": None, "stage": "skipped_eval",
        }, indent=2))
        print(f"[smoke] PASS (eval skipped). Wrote {record_path}")
        return

    # Step 3 — run swebench harness eval.
    print(f"[smoke] invoking swebench.harness.run_evaluation "
          f"(may take 5–15 min for first-time Docker build)...")
    t0 = time.perf_counter()
    try:
        results = run_swebench_eval(
            predictions_path=preds_path,
            run_id=args.run_id,
            instance_ids=[inst.instance_id],
            max_workers=1,
        )
    except Exception as e:
        print(f"[smoke] FAIL — eval crashed: {type(e).__name__}: {e}")
        record_path.write_text(json.dumps({
            "instance_id": inst.instance_id,
            "patch": patch, "patch_extracted": True,
            "telemetry": tel, "resolved": False,
            "stage": "eval_crashed",
            "error": f"{type(e).__name__}: {e}",
        }, indent=2))
        sys.exit(3)
    eval_wall = time.perf_counter() - t0
    res = results.get(inst.instance_id, {})
    resolved = bool(res.get("resolved", False))

    print(f"[smoke] eval done in {eval_wall:.0f}s; resolved={resolved}")

    record_path.write_text(json.dumps({
        "instance_id": inst.instance_id,
        "model": args.model,
        "patch": patch, "patch_extracted": True,
        "telemetry": tel,
        "eval_wall_seconds": eval_wall,
        "resolved": resolved,
        "harness_report": res.get("report", {}),
        "stage": "complete",
    }, indent=2))
    print(f"[smoke] {'PASS' if resolved else 'FAIL'} on {inst.instance_id}; wrote {record_path}")


if __name__ == "__main__":
    main()
