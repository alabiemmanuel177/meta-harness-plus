"""Ollama Pro concurrency-ceiling probe.

Three-step measurement to find the actual Pro concurrency cap, not the
assumed 16. Output is written to ``runs/wow_push/PRO_CONCURRENCY_PROBE.md``
and a JSON appendix.

Step 1 — single-model burst control + step:
  Burst 16 concurrent on gpt-oss:20b (control), then 64 concurrent
  (step). Compare p50/p95 latency, count 429s.
  Knee criterion: p95@64 < 1.5×p95@16 AND zero 429s -> headroom exists.

Step 2 — multi-model burst:
  Fire 16 concurrent on each of {gpt-oss:20b, deepseek-v3.1:671b,
  kimi-k2:1t} simultaneously (48 total). Per-model p95 vs solo.
  If per-model p95 stays within 1.5× of solo, concurrency is per-model.

Step 3 — push higher (only if Step 1 passed):
  Burst 128 concurrent on gpt-oss:20b. Find the actual ceiling.

Decision rule (per spec):
- If measured ceiling ≥ 32: re-launch Phase 3 with that worker count.
- If ≤ 16: current plan is honest; resume.

Run-time: ~10 min. Cost: ~$0 (Pro flat-rate).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from examples.run_ollama_cloud_search import (
    _AccountingClient, _percentile, _record_ollama_batch,
    _load_dotenv_if_present, COST_LOG,
)
from meta_harness_plus.llm.client import HTTPClient


PROBE_PROMPT = (
    "Reply with exactly one short sentence summarizing why the sky appears blue."
)
SHORT_TOKENS = 64


def burst(model: str, n_concurrent: int, n_calls: int,
          url: str, key: str, label: str) -> dict:
    """Fire n_calls on `model` with n_concurrent workers; collect stats."""
    raw = HTTPClient(api_url=url, api_key=key, model=model,
                     timeout_s=180.0, max_retries=0)
    client = _AccountingClient(raw)
    errors_by_kind: dict[str, int] = {}
    errors_log: list[str] = []

    def one(_i):
        try:
            client.complete(system="You answer concisely.",
                            user=PROBE_PROMPT,
                            max_tokens=SHORT_TOKENS, temperature=0.0)
            return ("ok", None)
        except RuntimeError as e:
            msg = str(e)
            if "429" in msg:
                return ("http_429", msg[:120])
            if "HTTP 5" in msg:
                return ("http_5xx", msg[:120])
            return ("exc_runtime", msg[:120])
        except urllib.error.HTTPError as e:
            return (f"http_{e.code}", str(e)[:120])
        except Exception as e:
            return (f"exc_{type(e).__name__}", str(e)[:120])

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_concurrent) as pool:
        for fut in as_completed(pool.submit(one, i) for i in range(n_calls)):
            kind, msg = fut.result()
            if kind != "ok":
                errors_by_kind[kind] = errors_by_kind.get(kind, 0) + 1
                if msg:
                    errors_log.append(msg)
    wall = time.perf_counter() - t0

    p50 = _percentile(client.latencies_ms, 50)
    p95 = _percentile(client.latencies_ms, 95)
    p99 = _percentile(client.latencies_ms, 99)
    qps = client.calls / max(0.001, wall)
    return {
        "label": label,
        "model": model,
        "n_concurrent": n_concurrent,
        "n_calls_attempted": n_calls,
        "n_calls_succeeded": client.calls,
        "errors_by_kind": errors_by_kind,
        "p50_ms": round(p50, 1),
        "p95_ms": round(p95, 1),
        "p99_ms": round(p99, 1),
        "qps_effective": round(qps, 2),
        "wall_s": round(wall, 1),
        "in_tokens": client.in_tokens,
        "out_tokens": client.out_tokens,
        "errors_log_first_3": errors_log[:3],
    }


def run_step1(url: str, key: str, results: list[dict]) -> bool:
    """Step 1 — single-model 16 (control) then 64 (step)."""
    print("\n[probe step 1] single-model burst on gpt-oss:20b")
    print("[probe]   16-concurrent control...")
    ctrl = burst("gpt-oss:20b", 16, 32, url, key, "step1_control_16w")
    results.append(ctrl)
    print(f"[probe]   control: p50={ctrl['p50_ms']}ms p95={ctrl['p95_ms']}ms "
          f"qps={ctrl['qps_effective']} errors={ctrl['errors_by_kind']}")

    print("[probe]   64-concurrent step...")
    step = burst("gpt-oss:20b", 64, 64, url, key, "step1_step_64w")
    results.append(step)
    print(f"[probe]   step:    p50={step['p50_ms']}ms p95={step['p95_ms']}ms "
          f"qps={step['qps_effective']} errors={step['errors_by_kind']}")

    has_429 = "http_429" in step["errors_by_kind"]
    p95_ratio = step["p95_ms"] / max(1, ctrl["p95_ms"])
    passed = (not has_429) and p95_ratio < 1.5
    print(f"[probe]   step1 passed? p95_ratio={p95_ratio:.2f}, "
          f"429s={'NO' if not has_429 else step['errors_by_kind']['http_429']} "
          f"-> {'PASS' if passed else 'FAIL'}")
    return passed


def run_step2(url: str, key: str, results: list[dict]):
    """Step 2 — 16 concurrent on each of 3 models simultaneously."""
    print("\n[probe step 2] multi-model burst (16w each on 3 models)")

    def fire(model):
        return burst(model, 16, 16, url, key, f"step2_multi_16w_{model}")

    models = ["gpt-oss:20b", "deepseek-v3.1:671b", "kimi-k2:1t"]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=3) as outer:
        futures = {outer.submit(fire, m): m for m in models}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            print(f"[probe]   {r['model']:25s}: p50={r['p50_ms']}ms "
                  f"p95={r['p95_ms']}ms qps={r['qps_effective']} "
                  f"errors={r['errors_by_kind']}")
    wall = time.perf_counter() - t0
    print(f"[probe]   step2 wall={wall:.1f}s")


def run_step3(url: str, key: str, results: list[dict]):
    """Step 3 — push to 128 concurrent on gpt-oss:20b."""
    print("\n[probe step 3] push to 128-concurrent on gpt-oss:20b")
    r = burst("gpt-oss:20b", 128, 128, url, key, "step3_push_128w")
    results.append(r)
    print(f"[probe]   p50={r['p50_ms']}ms p95={r['p95_ms']}ms "
          f"qps={r['qps_effective']} errors={r['errors_by_kind']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/wow_push/PRO_CONCURRENCY_PROBE.md")
    ap.add_argument("--out-json", default="runs/wow_push/PRO_CONCURRENCY_PROBE.json")
    args = ap.parse_args()

    _load_dotenv_if_present()
    url = os.environ["OLLAMA_CLOUD_URL"]
    key = os.environ["OLLAMA_API_KEY"]
    results: list[dict] = []
    t0 = time.perf_counter()

    step1_passed = run_step1(url, key, results)
    run_step2(url, key, results)
    if step1_passed:
        run_step3(url, key, results)
    else:
        print("\n[probe] skipping step 3 — step 1 didn't pass headroom criterion")

    total = time.perf_counter() - t0
    print(f"\n[probe] total wall {total:.1f}s")

    # Persist JSON.
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(results, indent=2))

    # Markdown report.
    md = ["# Ollama Pro concurrency-ceiling probe", "",
          f"Total wall: {total:.1f}s. {len(results)} bursts."]
    md += ["", "## Per-burst results",
           "| label | model | concurrent | calls | succ | 429s | p50_ms | p95_ms | qps |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        n_429 = r["errors_by_kind"].get("http_429", 0)
        md.append(
            f"| {r['label']} | {r['model']} | {r['n_concurrent']} | "
            f"{r['n_calls_attempted']} | {r['n_calls_succeeded']} | "
            f"{n_429} | {r['p50_ms']} | {r['p95_ms']} | {r['qps_effective']} |"
        )

    # Verdict.
    md += ["", "## Decision"]
    step1_step = next((r for r in results if r["label"] == "step1_step_64w"), None)
    step1_ctrl = next((r for r in results if r["label"] == "step1_control_16w"), None)
    step3 = next((r for r in results if r["label"] == "step3_push_128w"), None)
    if step1_step and step1_ctrl:
        ratio = step1_step["p95_ms"] / max(1, step1_ctrl["p95_ms"])
        s_429 = step1_step["errors_by_kind"].get("http_429", 0)
        md.append(f"- **Step 1 (16→64)**: p95 ratio = {ratio:.2f}, 429s = {s_429}")
        if step1_passed:
            md.append("  → PASS, headroom present beyond 16 workers.")
        else:
            md.append("  → FAIL, ceiling somewhere ≤ 64 workers.")
    if step3:
        s_429 = step3["errors_by_kind"].get("http_429", 0)
        ratio = step3["p95_ms"] / max(1, step1_ctrl["p95_ms"]) if step1_ctrl else 0
        md.append(f"- **Step 3 (128w)**: p95={step3['p95_ms']}ms (ratio {ratio:.2f}× control), 429s={s_429}")
    # Step 2 cross-model.
    step2 = [r for r in results if r["label"].startswith("step2")]
    if step2:
        md.append("- **Step 2 (multi-model 16+16+16)**:")
        for r in step2:
            n_429 = r["errors_by_kind"].get("http_429", 0)
            md.append(f"  - {r['model']}: p50={r['p50_ms']}ms p95={r['p95_ms']}ms 429s={n_429}")

    md.append("")
    md.append("## Recommended Phase 3 worker counts")
    # Heuristic recommendation:
    if step3 and step3["errors_by_kind"].get("http_429", 0) == 0:
        md.append("- 128 workers per slot is safe on gpt-oss:20b.")
    elif step1_passed:
        md.append("- 64 workers per slot is safe on gpt-oss:20b. Bumping to 128 saw rate-limits.")
    else:
        md.append("- Stay at 16 workers per slot; higher burst hit rate-limits.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(md))
    print(f"\n[probe] wrote {args.out} and {args.out_json}")
    _record_ollama_batch(
        phase="phase3_concurrency_probe", label="multi_step",
        model="multi", url=url,
        calls=sum(r["n_calls_succeeded"] for r in results),
        in_tokens=sum(r["in_tokens"] for r in results),
        out_tokens=sum(r["out_tokens"] for r in results),
        latencies_ms=[],  # already aggregated per-burst
    )


if __name__ == "__main__":
    main()
