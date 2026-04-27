"""Phase 0 smoke runner — cloud Ollama (Ollama Pro / Turbo).

Validates that the existing ``HTTPClient`` works against a hosted
Ollama endpoint. ``HTTPClient`` already auto-detects whichever path
the user supplies — OpenAI-compat (``/v1/chat/completions``) or
native (``/api/chat``); the dashboard tells you which one Pro exposes.

Reads:
- ``OLLAMA_CLOUD_URL`` — full URL the dashboard publishes.
- ``OLLAMA_API_KEY`` — bearer token from Ollama Pro dashboard.

Usage:
    export OLLAMA_CLOUD_URL=...
    export OLLAMA_API_KEY=...
    python3 examples/run_ollama_cloud_search.py \
        --model gpt-oss:120b --task agnews --eval-size 5

Cost log schema for Ollama batches (Pro is flat-rate, not per-token):
    {ts, phase, label, model, url, calls, in_tokens, out_tokens,
     latency_ms_p50, latency_ms_p95}
The ``usd`` field is reserved for Phase 5 closed-API batches only and
is *not* written here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _load_dotenv_if_present() -> None:
    """Best-effort .env loader — populates os.environ for keys not
    already set. Matches the convention the bash runners use
    (``set -a; source .env; set +a``). Lines must be ``KEY=value``;
    quotes are stripped if they wrap the value.
    """
    p = _REPO_ROOT / ".env"
    if not p.exists():
        return
    try:
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip("'").strip('"')
            os.environ.setdefault(k, v)
    except Exception:
        # .env is optional; never let a malformed file kill the run.
        pass


_load_dotenv_if_present()

from meta_harness_plus.attribution import AttributionTracker
from meta_harness_plus.components import (
    BagOfWordsRetriever,
    MajorityVoter,
    NullFewShot,
    NullRetriever,
    NullVoter,
    SimpleFormatter,
    TopKFewShot,
    baseline_for,
)
from meta_harness_plus.harness import Harness
from meta_harness_plus.llm.client import HTTPClient
from meta_harness_plus.llm.predictor import LLMPredictor
from meta_harness_plus.runner import SearchConfig, SearchRunner
from meta_harness_plus.scorer import Scorer
from meta_harness_plus.search.mock_proposer import AttributionGuidedMutationProposer
from meta_harness_plus.tasks import build_agnews_task, build_toy_task


COST_LOG = Path("runs/wow_push/cost_log.jsonl")


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def _record_ollama_batch(
    *,
    phase: str,
    label: str,
    model: str,
    url: str,
    calls: int,
    in_tokens: int,
    out_tokens: int,
    latencies_ms: list[float],
) -> None:
    """One batch line for an Ollama (flat-rate) provider.

    Schema deliberately omits ``usd``: Pro is flat-rate, not per-token.
    Phase 5 closed-API logging uses a separate path that does record
    ``usd``.
    """
    COST_LOG.parent.mkdir(parents=True, exist_ok=True)
    with COST_LOG.open("a") as f:
        f.write(json.dumps({
            "ts": time.time(),
            "phase": phase,
            "provider": "ollama",
            "label": label,
            "model": model,
            "url": url,
            "calls": calls,
            "in_tokens": in_tokens,
            "out_tokens": out_tokens,
            "latency_ms_p50": round(_percentile(latencies_ms, 50.0), 1),
            "latency_ms_p95": round(_percentile(latencies_ms, 95.0), 1),
        }) + "\n")


class _AccountingClient:
    """Thin pass-through over HTTPClient that tallies per-call stats.

    Records each ``chat()`` call's (input_tokens, output_tokens, latency_ms)
    so the smoke runner can emit p50/p95 and exact in/out token totals.
    """

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0
        self.in_tokens = 0
        self.out_tokens = 0
        self.latencies_ms: list[float] = []

    def complete(self, *args, **kwargs):
        resp = self.inner.complete(*args, **kwargs)
        self.calls += 1
        self.in_tokens += resp.input_tokens
        self.out_tokens += resp.output_tokens
        self.latencies_ms.append(resp.latency_ms)
        return resp

    def reset(self) -> None:
        self.calls = 0
        self.in_tokens = 0
        self.out_tokens = 0
        self.latencies_ms = []


def _build_task(name: str, eval_size: int):
    if name == "toy":
        return build_toy_task(seed=0)
    if name == "agnews":
        t = build_agnews_task()
        t.eval_set = t.eval_set[:eval_size]
        return t
    raise SystemExit(f"unknown --task {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-oss:120b",
                    help="Ollama Pro model id, e.g. gpt-oss:120b, deepseek-v3.1, qwen3-coder:480b.")
    ap.add_argument("--task", default="agnews", choices=["toy", "agnews"])
    ap.add_argument("--eval-size", type=int, default=5)
    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument("--proposals", type=int, default=3)
    ap.add_argument("--url", default=None,
                    help="Override OLLAMA_CLOUD_URL env var.")
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    url = args.url or os.environ.get("OLLAMA_CLOUD_URL")
    api_key = os.environ.get("OLLAMA_API_KEY", "")
    if not url:
        raise SystemExit(
            "Missing OLLAMA_CLOUD_URL env var. "
            "Export the chat-completions URL from Ollama Pro and retry."
        )
    if not api_key:
        raise SystemExit(
            "Missing OLLAMA_API_KEY env var. Export it and retry."
        )

    print(f"[phase0] cloud-ollama smoke: model={args.model} task={args.task} "
          f"eval_size={args.eval_size}")
    print(f"[phase0] url={url}")

    task = _build_task(args.task, eval_size=args.eval_size)
    raw_client = HTTPClient(api_url=url, api_key=api_key, model=args.model,
                            timeout_s=120.0, max_retries=3)
    client = _AccountingClient(raw_client)

    # Tiny one-shot ping first — fail fast if auth/endpoint is wrong.
    try:
        ping = client.complete(system="reply with the literal word PONG",
                               user="ping", max_tokens=8, temperature=0.0)
    except Exception as e:
        # Auth / endpoint failure: do NOT retry blindly. Surface the URL
        # we used so the caller can verify the dashboard publishes it.
        raise SystemExit(
            f"[phase0] auth/endpoint smoke failed against {url}: {e}\n"
            f"  Verify OLLAMA_CLOUD_URL matches the endpoint Ollama Pro "
            f"publishes for your account, and OLLAMA_API_KEY is valid."
        )
    print(f"[phase0] ping ok in {ping.latency_ms:.0f}ms; "
          f"in_tokens={ping.input_tokens} out_tokens={ping.output_tokens} "
          f"text={ping.text[:60]!r}")
    _record_ollama_batch(
        phase="phase0", label="ping", model=args.model, url=url,
        calls=client.calls, in_tokens=client.in_tokens,
        out_tokens=client.out_tokens, latencies_ms=client.latencies_ms,
    )
    client.reset()

    def pred(n_samples: int) -> LLMPredictor:
        return LLMPredictor(
            client=client, classes=task.classes,
            n_samples=n_samples,
            temperature=0.3 if n_samples > 1 else 0.0,
            max_tokens=args.max_tokens,
        )

    mutators = {
        "retriever": [
            lambda: NullRetriever(),
            lambda: BagOfWordsRetriever(corpus=task.train, k=2),
        ],
        "fewshot":   [lambda: NullFewShot(), lambda: TopKFewShot(k=1)],
        "voter":     [lambda: NullVoter(), lambda: MajorityVoter()],
        "predictor": [lambda: pred(1), lambda: pred(3)],
    }
    scorer = Scorer(task)
    attribution = AttributionTracker(scorer, baseline_for)
    proposer = AttributionGuidedMutationProposer(
        mutators=mutators, seed=0, epsilon=0.3, temperature=0.5,
    )
    seed = Harness(components=[
        NullRetriever(), NullFewShot(), SimpleFormatter(),
        pred(1), NullVoter(),
    ])
    config = SearchConfig(
        n_iterations=args.iterations,
        proposals_per_iter=args.proposals,
        screen_size=args.eval_size,
        full_eval_size=args.eval_size,
    )
    runner = SearchRunner(
        task=task, scorer=scorer, proposer=proposer,
        attribution=attribution, config=config, seed_harnesses=[seed],
    )

    t0 = time.perf_counter()
    state = runner.run()
    wall = time.perf_counter() - t0

    best = state.frontier.best_by_accuracy()
    if best is None:
        print("[phase0] no candidates scored — failure")
        sys.exit(1)
    sv = best.score
    print(f"[phase0] best harness: acc={sv.accuracy:.3f} "
          f"tokens={sv.tokens:.1f} latency_ms={sv.latency_ms:.1f}")
    print(f"[phase0] search wall_seconds={wall:.2f}")
    print(f"[phase0] frontier size = {len(state.frontier)}")
    print(f"[phase0] search calls={client.calls} "
          f"in_tokens={client.in_tokens} out_tokens={client.out_tokens} "
          f"latency p50={_percentile(client.latencies_ms, 50):.0f}ms "
          f"p95={_percentile(client.latencies_ms, 95):.0f}ms")

    _record_ollama_batch(
        phase="phase0", label="smoke_search",
        model=args.model, url=url,
        calls=client.calls, in_tokens=client.in_tokens,
        out_tokens=client.out_tokens, latencies_ms=client.latencies_ms,
    )

    print(f"[phase0] PASS — cost log: {COST_LOG}")


if __name__ == "__main__":
    main()
