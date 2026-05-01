"""Dev-50 retrieval eval — top-1/top-5/top-10 file accuracy for Stage 1b.

Per the Phase 1 stage 1b acceptance gate: report retrieval accuracy
against the gold patch's touched files for each of the 50 instances
in splits/dev_50.json. The eval is the only place V10 reads the
``patch`` field; reads happen via ``harness.eval.load_eval_metadata``,
the firewall-allowed channel.

Per-instance flow:
  1. Load InstanceView via the projection boundary.
  2. Start a Sandbox (memory 4 GB, observation cap 32 MB for big repos).
  3. Run ``run_stage_1b_retrieval`` (BM25 across 3 query strategies +
     embedding if sentence-transformers is installed; +Stage 1c
     traceback if --include-traceback).
  4. Aggregate the per-strategy candidates and emit a single ranked
     list.
  5. Call ``evaluate_retrieval_recall`` to score against the gold.

Optimizations (commit 4b):
  - --batch-size N: embedder batch size (default 256 for the 32 GB
    AMD Radeon; safely under VRAM headroom).
  - --workers N: parallel per-instance execution via
    ProcessPoolExecutor. Each worker spins up its own Docker container
    and embedder; PyTorch serializes the actual GPU calls cleanly so
    CPU/IO work overlaps. Default 1 (deterministic); --workers 4 is
    the make eval-fast target.
  - Gold/repo lookups are pre-loaded once via
    harness.eval.load_eval_metadata; per-strategy ablation calls
    reuse the cache via the new ``preloaded=`` kwarg on
    evaluate_retrieval_recall.

Output: docs/audits/dev50_retrieval_eval.md.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEV_50 = PROJECT_ROOT / "splits" / "dev_50.json"
OUT = PROJECT_ROOT / "docs" / "audits" / "dev50_retrieval_eval.md"
CHECKPOINT_DIR = PROJECT_ROOT / "runs" / "v10_dev50_retr_eval" / "checkpoints"


def _paths_for_split(split_path: pathlib.Path) -> tuple:
    """Return (split_path, audit_out, checkpoint_dir) for a given split.
    The script's defaults point at dev_50 for backward compat; --split
    overrides everything via this helper."""
    split_name = split_path.stem  # 'dev_50' / 'dev_100' / etc.
    audit_out = PROJECT_ROOT / "docs" / "audits" / f"{split_name}_retrieval_eval.md"
    ckpt = PROJECT_ROOT / "runs" / f"v10_{split_name}_retr_eval" / "checkpoints"
    return split_path, audit_out, ckpt


def _try_make_embedder(batch_size: int):
    """Return a LocalEmbedder if sentence-transformers is installed,
    else None. The eval still produces meaningful numbers from BM25
    alone; embedding lift is reported separately when available.

    Each worker process calls this independently — fresh model load
    per worker (they share the GPU; PyTorch serializes CUDA calls).
    """
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return None
    from harness.embedding import LocalEmbedder
    return LocalEmbedder(default_batch_size=batch_size)


def _ranked_files_from_retrieval(result) -> list[str]:
    """Flatten the retrieval candidates into a single ranked file list."""
    return [c["file_path"] for c in result.candidates]


def _instance_best_rank_under(per_strategy_retrieved, entry, threshold: int) -> bool:
    """Return True iff for this entry, the gold file appears at any
    rank in any strategy."""
    iid = entry.instance_id
    gold_set = set(entry.gold_files)
    strategies = per_strategy_retrieved.get(iid, {})
    for rlist in strategies.values():
        for fp in rlist[:threshold]:
            if fp in gold_set or any(fp.endswith(g) or g.endswith(fp) for g in gold_set):
                return True
    return False


def _ranked_files_per_strategy(result) -> dict:
    """Return per-strategy + per-signal ranked file lists for ablation."""
    out: dict[str, list[str]] = {}
    for strategy, hits in result.bm25_hits_per_strategy.items():
        out[f"bm25_{strategy}"] = [h.file_path for h in hits]
    out["embedding"] = [h.file_path for h in result.embedding_hits]
    # Traceback hits are already in frame-priority order; preserve.
    if result.traceback_hits:
        seen: set[str] = set()
        traceback_paths: list[str] = []
        for sig in result.traceback_hits:
            if sig.file_path not in seen:
                seen.add(sig.file_path)
                traceback_paths.append(sig.file_path)
        out["traceback"] = traceback_paths
    out["aggregated"] = _ranked_files_from_retrieval(result)
    return out


@dataclass
class _WorkerArgs:
    """Picklable bundle for ProcessPoolExecutor work units."""
    instance_id: str
    use_traceback: bool
    no_shortlist: bool
    no_embedding: bool
    batch_size: int
    checkpoint_signature: str   # differentiates checkpoint dirs by flag combo
    use_rerank: bool = False
    retrieval_signature: str | None = None  # checkpoint key for retrieval-only data
    checkpoint_dir: str | None = None  # absolute path; overrides module global for multi-worker spawn


@dataclass
class _WorkerResult:
    """Picklable per-instance result returned by workers."""
    instance_id: str
    retrieved: list
    per_strategy: dict
    n_files_indexed: int
    error: str | None
    reranked_files: list | None = None  # [{file_path, final_score, rationale, ...}]
    rerank_cost_usd: float = 0.0
    rerank_duration_s: float = 0.0


def _checkpoint_path(iid: str, signature: str, base_dir: pathlib.Path | None = None) -> pathlib.Path:
    """Per-instance checkpoint path. The signature differentiates runs
    with different flags (so --no-shortlist vs shortlisted runs don't
    share checkpoints). ``base_dir`` overrides the module global —
    needed in spawn-mode worker processes where the parent's
    main()-time reassignment of CHECKPOINT_DIR isn't visible."""
    base = base_dir if base_dir is not None else CHECKPOINT_DIR
    return base / signature / f"{iid}.json"


def _load_checkpoint(iid: str, signature: str, base_dir: pathlib.Path | None = None) -> _WorkerResult | None:
    p = _checkpoint_path(iid, signature, base_dir=base_dir)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except Exception:
        return None
    return _WorkerResult(
        instance_id=d["instance_id"],
        retrieved=d["retrieved"],
        per_strategy=d["per_strategy"],
        n_files_indexed=d["n_files_indexed"],
        error=d.get("error"),
        reranked_files=d.get("reranked_files"),
        rerank_cost_usd=d.get("rerank_cost_usd", 0.0),
        rerank_duration_s=d.get("rerank_duration_s", 0.0),
    )


def _save_checkpoint(res: _WorkerResult, signature: str, base_dir: pathlib.Path | None = None) -> None:
    p = _checkpoint_path(res.instance_id, signature, base_dir=base_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "instance_id": res.instance_id,
        "retrieved": res.retrieved,
        "per_strategy": res.per_strategy,
        "n_files_indexed": res.n_files_indexed,
        "error": res.error,
        "reranked_files": res.reranked_files,
        "rerank_cost_usd": res.rerank_cost_usd,
        "rerank_duration_s": res.rerank_duration_s,
    }))


def _run_rerank_from_cached_retrieval(
    args: _WorkerArgs, retrieval: _WorkerResult,
) -> _WorkerResult:
    """Build CandidateForRerank from a cached retrieval result and run
    Stage 1g rerank. Skips the full retrieval re-run."""
    from harness.dataset import load_verified_view
    from harness.rerank import CandidateForRerank, RerankerError, RerankerInput, rerank
    from harness.sandbox import Sandbox
    from harness.skeleton import load_or_build_skeleton

    iid = args.instance_id
    try:
        view = load_verified_view(iid)
        # Need skeleton for one-line summaries in the rerank prompt.
        with Sandbox(view, max_observation_chars=32_000_000) as sb:
            skeleton = load_or_build_skeleton(view, sandbox=sb)

        # Build per-strategy ranks from the cached path lists.
        # Each strategy's list is in 1-indexed rank order.
        per_path_ranks: dict[str, dict] = {}
        for strategy_label, paths in retrieval.per_strategy.items():
            if strategy_label == "aggregated":
                # Skip — derived, not a primary strategy.
                continue
            for rank, path in enumerate(paths, start=1):
                per_path_ranks.setdefault(path, {})[strategy_label] = rank

        # Skeleton lookup for summaries.
        skel_summary: dict[str, str] = {}
        skel_lines: dict[str, int] = {}
        for fs in skeleton.files:
            cls_names = [c.name for c in fs.classes[:3]]
            fn_names = [f.name for f in fs.functions[:2]]
            bits = []
            if cls_names:
                bits.append("classes: " + ", ".join(cls_names))
            if fn_names:
                bits.append("functions: " + ", ".join(fn_names))
            skel_summary[fs.path] = "; ".join(bits) or "(empty)"
            max_line = 0
            for c in fs.classes:
                max_line = max(max_line, c.line_end)
                for m in c.methods:
                    max_line = max(max_line, m.line_end)
            for f in fs.functions:
                max_line = max(max_line, f.line_end)
            skel_lines[fs.path] = max_line

        # Build candidate list: union of every per-strategy list.
        cands = [
            CandidateForRerank(
                file_path=path,
                per_strategy_rank=ranks,
                per_strategy_score={},  # scores not preserved in checkpoint
                file_summary=skel_summary.get(path, ""),
                n_lines=skel_lines.get(path, 0),
            )
            for path, ranks in per_path_ranks.items()
        ]

        rerank_input = RerankerInput(view=view, candidates=tuple(cands), top_k=10)
        rerank_result = rerank(rerank_input)
    except (RerankerError, Exception) as exc:
        # Don't lose the retrieval data on rerank failure.
        return _WorkerResult(
            instance_id=iid,
            retrieved=retrieval.retrieved,
            per_strategy=retrieval.per_strategy,
            n_files_indexed=retrieval.n_files_indexed,
            error=f"rerank_failed: {type(exc).__name__}: {exc}",
        )

    reranked_dicts = [
        {
            "file_path": rf.file_path,
            "final_score": rf.final_score,
            "rationale": rf.rationale,
            "upstream_signals": list(rf.upstream_signals),
            "upstream_best_rank": rf.upstream_best_rank,
        }
        for rf in rerank_result.ranked_files
    ]
    # Add 'reranked' to per_strategy so downstream scoring + misses
    # analysis treats it as just another strategy.
    per_strategy_with_rerank = dict(retrieval.per_strategy)
    per_strategy_with_rerank["reranked"] = [d["file_path"] for d in reranked_dicts]

    return _WorkerResult(
        instance_id=iid,
        retrieved=[d["file_path"] for d in reranked_dicts],
        per_strategy=per_strategy_with_rerank,
        n_files_indexed=retrieval.n_files_indexed,
        error=None,
        reranked_files=reranked_dicts,
        rerank_cost_usd=rerank_result.cost_usd,
        rerank_duration_s=rerank_result.duration_s,
    )


def _run_one_instance(args: _WorkerArgs) -> _WorkerResult:
    """Worker entry point. Each worker process imports the harness
    modules fresh, instantiates its own embedder (loads model into VRAM),
    starts its own Docker container, runs retrieval, and returns the
    result. PyTorch serializes CUDA calls across processes when they
    share a GPU; CPU/IO/container work overlaps naturally.

    Checkpoints to disk after a successful run so a crash later in the
    pipeline (e.g., scoring/reporting) doesn't waste the retrieval cost.

    Two-tier cache when --rerank is set:
      1. Full checkpoint at the rerank signature (retrieval + rerank).
      2. Retrieval-only checkpoint at the no-rerank signature; if that
         exists, skip retrieval entirely and just run rerank.
    """
    # Resolve checkpoint base from args (multi-worker safe — workers
    # spawn fresh and don't inherit the parent's main()-time CHECKPOINT_DIR
    # reassignment).
    base_dir = pathlib.Path(args.checkpoint_dir) if args.checkpoint_dir else None

    # Tier 1: full checkpoint match.
    cached = _load_checkpoint(args.instance_id, args.checkpoint_signature, base_dir=base_dir)
    if cached is not None and cached.error is None:
        # If we wanted rerank but the cached has no rerank, fall through.
        if args.use_rerank and cached.reranked_files is None:
            pass  # will run rerank below
        else:
            return cached

    # Tier 2: retrieval-only checkpoint (rerun rerank only).
    if args.use_rerank and args.retrieval_signature is not None:
        retr_cached = _load_checkpoint(args.instance_id, args.retrieval_signature, base_dir=base_dir)
        if retr_cached is not None and retr_cached.error is None:
            # Skip retrieval; rerun rerank only.
            out = _run_rerank_from_cached_retrieval(args, retr_cached)
            if out.error is None:
                _save_checkpoint(out, args.checkpoint_signature, base_dir=base_dir)
            return out

    # Imports inside the worker so each process's site-packages cache
    # warms up independently and we don't accidentally fork a partly-
    # initialized parent.
    from harness.dataset import load_verified_view
    from harness.retrieval import run_stage_1b_retrieval
    from harness.sandbox import Sandbox
    from harness.skeleton import load_or_build_skeleton

    iid = args.instance_id
    embedder = None if args.no_embedding else _try_make_embedder(args.batch_size)
    try:
        view = load_verified_view(iid)
        with Sandbox(view, max_observation_chars=32_000_000) as sb:
            skeleton = None
            if args.use_traceback:
                skeleton = load_or_build_skeleton(view, sandbox=sb)
            result = run_stage_1b_retrieval(
                view, sb,
                embedder=embedder,
                embedding_use_shortlist=not args.no_shortlist,
                include_traceback=args.use_traceback,
                skeleton=skeleton,
            )
    except Exception as exc:
        out = _WorkerResult(
            instance_id=iid, retrieved=[], per_strategy={},
            n_files_indexed=0, error=f"{type(exc).__name__}: {exc}",
        )
        return out
    retrieval_only = _WorkerResult(
        instance_id=iid,
        retrieved=_ranked_files_from_retrieval(result),
        per_strategy=_ranked_files_per_strategy(result),
        n_files_indexed=result.n_files_indexed,
        error=None,
    )
    # Always persist the retrieval-only result (under retrieval_signature
    # if available, else under the checkpoint_signature). This makes the
    # retrieval reusable for future --rerank reruns with different LLM
    # configs.
    if args.retrieval_signature and args.retrieval_signature != args.checkpoint_signature:
        _save_checkpoint(retrieval_only, args.retrieval_signature, base_dir=base_dir)
    if not args.use_rerank:
        _save_checkpoint(retrieval_only, args.checkpoint_signature, base_dir=base_dir)
        return retrieval_only

    # Fresh-retrieval rerank path: use the just-built retrieval result
    # to feed the reranker. Same code path as the cached-retrieval
    # case below — keeps logic consistent.
    out = _run_rerank_from_cached_retrieval(args, retrieval_only)
    if out.error is None:
        _save_checkpoint(out, args.checkpoint_signature, base_dir=base_dir)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N dev-50 instances")
    ap.add_argument("--no-embedding", action="store_true",
                    help="skip embedding even if sentence-transformers is installed")
    ap.add_argument("--no-shortlist", action="store_true",
                    help="run embedding against the full repo file list "
                         "instead of the BM25 top-200 shortlist; "
                         "measures embedding's true upper bound (eval-only;"
                         "production keeps the shortlist for cost reasons)")
    ap.add_argument("--include-traceback", action="store_true",
                    help="run Stage 1c traceback parser; emits TracebackFrame "
                         "signals from issue text and includes them in candidate aggregation")
    ap.add_argument("--no-traceback", action="store_true",
                    help="explicit ablation flag — disable traceback even if it would otherwise run")
    ap.add_argument("--batch-size", type=int, default=256,
                    help="embedder batch size; default 256 (safe on 32 GB GPU). "
                         "Drop to 64 for CPU runs.")
    ap.add_argument("--workers", type=int, default=1,
                    help="number of parallel worker processes; default 1 "
                         "(deterministic). 4 saturates CPU+IO on this hardware.")
    ap.add_argument("--rerank", action="store_true",
                    help="run Stage 1g LLM rerank after retrieval; uses "
                         "harness/config/models.yaml role=reranker (default "
                         "deepseek-chat). Reuses retrieval-only checkpoints.")
    ap.add_argument("--split", type=str, default=str(DEV_50),
                    help="path to a split JSON (default: splits/dev_50.json). "
                         "Output audit path and checkpoint dir are derived "
                         "from the split's filename.")
    args = ap.parse_args()
    use_traceback = args.include_traceback and not args.no_traceback
    use_rerank = args.rerank

    # Resolve paths from --split.
    split_path = pathlib.Path(args.split)
    if not split_path.is_absolute():
        split_path = PROJECT_ROOT / split_path
    _split_path, audit_out, checkpoint_dir = _paths_for_split(split_path)
    # Re-bind module globals so checkpoint helpers + report writer use
    # the split-specific paths. Worker subprocesses spawn fresh and
    # re-import this module; their _paths_for_split is recomputed
    # via _WorkerArgs.checkpoint_signature too.
    global OUT, CHECKPOINT_DIR
    OUT = audit_out
    CHECKPOINT_DIR = checkpoint_dir
    print(f"[retr-eval] split: {split_path.relative_to(PROJECT_ROOT)}")
    print(f"[retr-eval] audit out: {OUT.relative_to(PROJECT_ROOT)}")
    print(f"[retr-eval] checkpoint dir: {CHECKPOINT_DIR.relative_to(PROJECT_ROOT)}")

    dev = json.loads(split_path.read_text())
    instances = dev["instances"]
    if args.limit is not None:
        instances = instances[: args.limit]
    print(f"[retr-eval] {len(instances)} instances from dev_50; "
          f"workers={args.workers} batch_size={args.batch_size}")
    if not args.no_embedding:
        # Quick check that sentence-transformers is available in this env;
        # the workers will repeat the check independently.
        try:
            import sentence_transformers  # noqa: F401
            print(f"[retr-eval] embedding: BAAI/bge-large-en-v1.5 (default)")
        except ImportError:
            print("[retr-eval] embedding: SKIPPED (sentence-transformers not installed)")
    else:
        print("[retr-eval] embedding: SKIPPED (--no-embedding)")

    # Checkpoint signature: identifies the flag combo so reruns with
    # different flags don't reuse stale checkpoints. The retrieval part
    # is the same regardless of --rerank, so we maintain BOTH a
    # retrieval-only signature (for cache reuse) and a rerank signature.
    sig_base = [
        "embed" if not args.no_embedding else "noembed",
        "noshortlist" if args.no_shortlist else "shortlist",
        "tb" if use_traceback else "notb",
        f"bs{args.batch_size}",
    ]
    retrieval_signature = "_".join(sig_base)
    signature = retrieval_signature + ("_rerank" if use_rerank else "")
    print(f"[retr-eval] checkpoint signature: {signature}")
    if use_rerank and retrieval_signature != signature:
        print(f"[retr-eval] retrieval-only signature (for cache reuse): "
              f"{retrieval_signature}")
    print(f"[retr-eval] checkpoint dir: {CHECKPOINT_DIR / signature}")

    # Build worker arg bundles.
    work = [
        _WorkerArgs(
            instance_id=e["instance_id"],
            use_traceback=use_traceback,
            no_shortlist=args.no_shortlist,
            no_embedding=args.no_embedding,
            batch_size=args.batch_size,
            checkpoint_signature=signature,
            use_rerank=use_rerank,
            retrieval_signature=retrieval_signature if use_rerank else None,
            checkpoint_dir=str(checkpoint_dir),
        )
        for e in instances
    ]

    retrieved_per_instance: dict[str, list[str]] = {}
    per_strategy_retrieved: dict[str, dict[str, list[str]]] = {}
    n_files_indexed_per_instance: dict[str, int] = {}
    failed: list[tuple[str, str]] = []
    t0 = time.perf_counter()

    if args.workers <= 1:
        # Serial path — preserves the original single-process flow.
        for i, w in enumerate(work):
            print(f"[retr-eval] [{i + 1}/{len(work)}] {w.instance_id} … ",
                  end="", flush=True)
            res = _run_one_instance(w)
            if res.error:
                failed.append((res.instance_id, res.error))
                print(f"FAIL: {res.error}")
                continue
            retrieved_per_instance[res.instance_id] = res.retrieved
            per_strategy_retrieved[res.instance_id] = res.per_strategy
            n_files_indexed_per_instance[res.instance_id] = res.n_files_indexed
            print(f"OK n_files={res.n_files_indexed} candidates={len(res.retrieved)}")
    else:
        # Parallel path — spawn N workers, each with its own embedder.
        # 'spawn' context to avoid fork issues with torch.
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
            futures = {pool.submit(_run_one_instance, w): w.instance_id for w in work}
            done = 0
            for fut in as_completed(futures):
                done += 1
                res = fut.result()
                if res.error:
                    failed.append((res.instance_id, res.error))
                    print(f"[retr-eval] [{done}/{len(work)}] {res.instance_id} FAIL: {res.error}")
                    continue
                retrieved_per_instance[res.instance_id] = res.retrieved
                per_strategy_retrieved[res.instance_id] = res.per_strategy
                n_files_indexed_per_instance[res.instance_id] = res.n_files_indexed
                print(f"[retr-eval] [{done}/{len(work)}] {res.instance_id} "
                      f"OK n_files={res.n_files_indexed} candidates={len(res.retrieved)}")

    dur = time.perf_counter() - t0
    print(f"[retr-eval] retrieval done in {dur:.1f}s — "
          f"{len(retrieved_per_instance)} OK, {len(failed)} failed")

    # Score against gold. Pre-load metadata ONCE so the per-strategy
    # ablation doesn't re-read the dataset for each strategy.
    print("[retr-eval] scoring against gold patch (eval-only path)…")
    from harness.eval import load_eval_metadata, evaluate_retrieval_recall
    preloaded_meta = load_eval_metadata(retrieved_per_instance.keys())

    report = evaluate_retrieval_recall(
        retrieved_files_per_instance=retrieved_per_instance,
        preloaded=preloaded_meta,
    )

    # Per-strategy ablation: score each strategy alone (reuses cache).
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
                preloaded=preloaded_meta,
            )

    # Render report.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    split_label = pathlib.Path(args.split).stem.replace("_", "-").capitalize()
    out.append(f"# {split_label} retrieval recall — Stage 1b acceptance gate")
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
    if args.no_embedding:
        embedding_label = "NONE (--no-embedding)"
    else:
        try:
            import sentence_transformers  # noqa: F401
            from harness.embedding import DEFAULT_LOCAL_MODEL
            embedding_label = f"{DEFAULT_LOCAL_MODEL} (batch={args.batch_size}, shortlist={'no' if args.no_shortlist else 'yes'})"
        except ImportError:
            embedding_label = "NONE (sentence-transformers not installed)"
    out.append(f"- **Embedding model:** {embedding_label}")
    out.append(f"- **Workers:** {args.workers}")
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

    # ---- Misses section: per-instance best-rank-seen-across-strategies ----
    misses = [e for e in report.per_instance if not e.top10_hit]
    if misses:
        out.append("## Misses — top-10 failures (Stage 1c targets)")
        out.append("")
        out.append(
            f"For every {split_label.lower()} instance where top-10 didn't hit, this "
            "table reports the BEST rank seen for any gold file across "
            "all retrieval strategies. If best_rank is small (e.g., 11-30), "
            "the gold IS reachable but ordering is wrong — Stage 1g "
            "rerank can fix it. If best_rank is large (>200) or `not_found`, "
            "the upstream signals never named the gold file — Stage 1c–1f "
            "(traceback / grep / archeology / dep-graph) are the path."
        )
        out.append("")
        out.append("| Instance | Repo | Gold files | best_rank | best_strategy | n_indexed |")
        out.append("|---|---|---|---|---|---|")
        for entry in sorted(misses, key=lambda e: (e.repo, e.instance_id)):
            iid = entry.instance_id
            gold_set = set(entry.gold_files)
            strategies = per_strategy_retrieved.get(iid, {})
            best_rank: int | None = None
            best_strategy = "n/a"
            for sk, rlist in strategies.items():
                for rank, fp in enumerate(rlist, start=1):
                    if fp in gold_set or any(fp.endswith(g) or g.endswith(fp) for g in gold_set):
                        if best_rank is None or rank < best_rank:
                            best_rank = rank
                            best_strategy = sk
                        break
            best_str = str(best_rank) if best_rank is not None else "**not_found**"
            gold_str = ", ".join(f"`{p}`" for p in entry.gold_files[:2])
            if len(entry.gold_files) > 2:
                gold_str += f" (+{len(entry.gold_files) - 2})"
            n_idx = n_files_indexed_per_instance.get(iid, "?")
            out.append(
                f"| `{iid}` | {entry.repo} | {gold_str} | {best_str} | "
                f"`{best_strategy}` | {n_idx} |"
            )
        out.append("")
        # Quick stats on the misses.
        not_found = sum(
            1 for e in misses
            if not _instance_best_rank_under(per_strategy_retrieved, e, threshold=10**6)
        )
        rescuable = len(misses) - not_found
        out.append(
            f"**Misses summary:** {len(misses)}/{report.n_instances} instances "
            f"missed top-10. Of those, {rescuable} have the gold file at SOME "
            f"rank in SOME strategy (rescuable by Stage 1g rerank); {not_found} "
            f"have the gold file in NO strategy's results (need new signal "
            f"sources from Stage 1c–1f)."
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
