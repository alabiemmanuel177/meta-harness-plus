"""Dev-50 retrieval eval — top-1/top-5/top-10 file accuracy for Stage 1b.

Per the Phase 1 stage 1b acceptance gate: report retrieval accuracy
against the gold patch's touched files for each of the 50 instances
in splits/dev_50.json. The eval is the only place V10 reads the
``patch`` field; reads happen via ``harness.eval._load_gold_touched_files``,
the firewall-allowed channel.

Per-instance flow:
  1. Load InstanceView via the projection boundary.
  2. Start a Sandbox (memory 4 GB, observation cap 32 MB for big repos).
  3. Run ``run_stage_1b_retrieval`` (BM25 across 3 query strategies +
     embedding if sentence-transformers is installed).
  4. Aggregate the per-strategy candidates and emit a single ranked
     list (the candidates are already ordered by upstream-signal
     count + best-rank in retrieval.run_stage_1b_retrieval).
  5. Call ``evaluate_retrieval_recall`` to score against the gold.

Output: docs/audits/dev50_retrieval_eval.md.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import defaultdict

from harness.dataset import load_verified_view
from harness.eval import evaluate_retrieval_recall
from harness.retrieval import run_stage_1b_retrieval
from harness.sandbox import Sandbox


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEV_50 = PROJECT_ROOT / "splits" / "dev_50.json"
OUT = PROJECT_ROOT / "docs" / "audits" / "dev50_retrieval_eval.md"


def _try_make_embedder():
    """Return a LocalEmbedder if sentence-transformers is installed,
    else None. The eval still produces meaningful numbers from BM25
    alone; embedding lift is reported separately when available."""
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return None
    from harness.embedding import LocalEmbedder
    return LocalEmbedder()


def _ranked_files_from_retrieval(result) -> list[str]:
    """Flatten the retrieval candidates into a single ranked file list."""
    return [c["file_path"] for c in result.candidates]


def _ranked_files_per_strategy(result) -> dict:
    """Return per-strategy + per-signal ranked file lists for ablation."""
    out: dict[str, list[str]] = {}
    for strategy, hits in result.bm25_hits_per_strategy.items():
        out[f"bm25_{strategy}"] = [h.file_path for h in hits]
    out["embedding"] = [h.file_path for h in result.embedding_hits]
    out["aggregated"] = _ranked_files_from_retrieval(result)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N dev-50 instances")
    ap.add_argument("--no-embedding", action="store_true",
                    help="skip embedding even if sentence-transformers is installed")
    args = ap.parse_args()

    dev = json.loads(DEV_50.read_text())
    instances = dev["instances"]
    if args.limit is not None:
        instances = instances[: args.limit]
    print(f"[retr-eval] {len(instances)} instances from dev_50")

    embedder = None if args.no_embedding else _try_make_embedder()
    if embedder is None:
        print("[retr-eval] embedding: SKIPPED (sentence-transformers not installed)")
    else:
        print(f"[retr-eval] embedding: {embedder.model_name}")

    retrieved_per_instance: dict[str, list[str]] = {}
    per_strategy_retrieved: dict[str, dict[str, list[str]]] = {}
    n_files_indexed_per_instance: dict[str, int] = {}
    failed: list[tuple[str, str]] = []

    t0 = time.perf_counter()
    for i, entry in enumerate(instances):
        iid = entry["instance_id"]
        print(f"[retr-eval] [{i + 1}/{len(instances)}] {iid} … ", end="", flush=True)
        try:
            view = load_verified_view(iid)
            with Sandbox(view, max_observation_chars=32_000_000) as sb:
                result = run_stage_1b_retrieval(view, sb, embedder=embedder)
        except Exception as exc:
            failed.append((iid, f"{type(exc).__name__}: {exc}"))
            print(f"FAIL: {type(exc).__name__}")
            continue
        retrieved_per_instance[iid] = _ranked_files_from_retrieval(result)
        per_strategy_retrieved[iid] = _ranked_files_per_strategy(result)
        n_files_indexed_per_instance[iid] = result.n_files_indexed
        print(f"OK n_files={result.n_files_indexed} candidates={len(result.candidates)}")
    dur = time.perf_counter() - t0
    print(f"[retr-eval] retrieval done in {dur:.1f}s — "
          f"{len(retrieved_per_instance)} OK, {len(failed)} failed")

    # Score against gold.
    print("[retr-eval] scoring against gold patch (eval-only path)…")
    report = evaluate_retrieval_recall(
        retrieved_files_per_instance=retrieved_per_instance,
    )

    # Per-strategy ablation: score each strategy alone.
    strategy_reports = {}
    if per_strategy_retrieved:
        strategy_keys = list(next(iter(per_strategy_retrieved.values())).keys())
        for sk in strategy_keys:
            per_iid = {
                iid: per_strategy_retrieved[iid].get(sk, [])
                for iid in retrieved_per_instance
            }
            strategy_reports[sk] = evaluate_retrieval_recall(
                retrieved_files_per_instance=per_iid,
            )

    # Render report.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    out.append("# Dev-50 retrieval recall — Stage 1b acceptance gate")
    out.append("")
    out.append(
        "Generated by `scripts/retrieval_eval_dev50.py`. Reads gold "
        "patches via `harness.eval._load_gold_touched_files` "
        "(eval-only; firewall-protected). 'Hit at K' = at least one of "
        "the top-K retrieved files appears in the gold patch's touched "
        "file set."
    )
    out.append("")
    out.append(f"- **Instances evaluated:** {report.n_instances}")
    out.append(f"- **Embedding model:** "
               f"{embedder.model_name if embedder else 'NONE (sentence-transformers not available)'}")
    out.append(f"- **Wall-clock:** {dur:.1f}s")
    out.append("")
    pct = lambda n, d: f"{100.0 * n / d:.1f}" if d else "n/a"
    out.append("## Aggregated retrieval (all signals)")
    out.append("")
    out.append("| Metric | Hits | Recall |")
    out.append("|---|---|---|")
    out.append(f"| Top-1 | {report.n_top1}/{report.n_instances} | {pct(report.n_top1, report.n_instances)}% |")
    out.append(f"| Top-5 | {report.n_top5}/{report.n_instances} | {pct(report.n_top5, report.n_instances)}% |")
    out.append(f"| Top-10 | {report.n_top10}/{report.n_instances} | {pct(report.n_top10, report.n_instances)}% |")
    out.append("")

    if strategy_reports:
        out.append("## Per-strategy ablation (independent of aggregation)")
        out.append("")
        out.append("| Strategy | Top-1 | Top-5 | Top-10 |")
        out.append("|---|---|---|---|")
        for sk in sorted(strategy_reports.keys()):
            rep = strategy_reports[sk]
            out.append(
                f"| `{sk}` | "
                f"{pct(rep.n_top1, rep.n_instances)}% | "
                f"{pct(rep.n_top5, rep.n_instances)}% | "
                f"{pct(rep.n_top10, rep.n_instances)}% |"
            )
        out.append("")

    out.append("## Per-instance breakdown by repo")
    out.append("")
    by_repo = defaultdict(lambda: {"n": 0, "top1": 0, "top5": 0, "top10": 0})
    for entry in report.per_instance:
        r = by_repo[entry.repo]
        r["n"] += 1
        r["top1"] += int(entry.top1_hit)
        r["top5"] += int(entry.top5_hit)
        r["top10"] += int(entry.top10_hit)
    out.append("| Repo | Instances | Top-1 | Top-5 | Top-10 |")
    out.append("|---|---|---|---|---|")
    for repo in sorted(by_repo.keys()):
        r = by_repo[repo]
        out.append(
            f"| {repo} | {r['n']} | "
            f"{pct(r['top1'], r['n'])}% | "
            f"{pct(r['top5'], r['n'])}% | "
            f"{pct(r['top10'], r['n'])}% |"
        )
    out.append("")

    out.append("## Per-instance detail (failures highlighted)")
    out.append("")
    out.append("| Instance | Repo | Gold files | Top-10 hit? | n_indexed |")
    out.append("|---|---|---|---|---|")
    for entry in sorted(report.per_instance, key=lambda e: (e.top10_hit, e.repo, e.instance_id)):
        gold_str = ", ".join(f"`{p}`" for p in entry.gold_files[:3])
        if len(entry.gold_files) > 3:
            gold_str += f" (+{len(entry.gold_files) - 3})"
        n_idx = n_files_indexed_per_instance.get(entry.instance_id, "?")
        hit_marker = "✓" if entry.top10_hit else "**MISS**"
        out.append(
            f"| `{entry.instance_id}` | {entry.repo} | {gold_str} | {hit_marker} | {n_idx} |"
        )
    out.append("")

    if failed:
        out.append("## Retrieval failures")
        out.append("")
        for iid, err in failed:
            out.append(f"- `{iid}`: {err}")
        out.append("")

    out.append("## Phase 1 acceptance interpretation")
    out.append("")
    out.append(
        f"- Top-10 recall = **{pct(report.n_top10, report.n_instances)}%**. "
        f"This is the headline number for Stage 1b — the localizer's "
        f"job is to put the right file in the top-10."
    )
    out.append(
        f"- Top-1 recall = **{pct(report.n_top1, report.n_instances)}%**. "
        f"Top-1 is the harder bar — Phase 1 stages 1c-1g (traceback + "
        f"grep + archaeology + dep-graph + LLM rerank) exist to push "
        f"this number up from where Stage 1b alone leaves it."
    )
    out.append("")

    OUT.write_text("\n".join(out))
    print(f"[retr-eval] wrote {OUT.relative_to(PROJECT_ROOT)}")
    print(f"[retr-eval] HEADLINE: top-1 {report.n_top1}/{report.n_instances} "
          f"({pct(report.n_top1, report.n_instances)}%)  "
          f"top-5 {report.n_top5}/{report.n_instances} "
          f"({pct(report.n_top5, report.n_instances)}%)  "
          f"top-10 {report.n_top10}/{report.n_instances} "
          f"({pct(report.n_top10, report.n_instances)}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
