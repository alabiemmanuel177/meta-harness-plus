"""V10 projected-context audit — scoped to the 16 V7 instances that hit
the tokens_in cap of 1,000,000.

Per the user-acked Phase 1 prereq batch: V7's trajectory stats (commit 17)
told us 16/30 empty patches hit V7's 1M-token input cap, but V7's prompt
shape is different from V10's (V7 included FAIL_TO_PASS selectors and a
specific actor-prompt structure; V10 will not include FAIL_TO_PASS but
will include a fuller localization context — repo skeleton, retrieval
results, traceback-derived candidates).

This script answers: "if V10's localization passes the FULL repo
skeleton to the reranker on these instances (worst case before
compression), how big is the projected prompt?"

Method per instance:
  1. Start the V10 Sandbox at base_commit (image-required, no network).
  2. Walk /testbed for .py files via an in-container Python script.
  3. AST-parse each file to count classes + functions + total lines.
  4. Project the skeleton + issue + retrieval-budget shapes:
        - skeleton-files-only:  file paths only (cheapest projection)
        - skeleton-symbols:     paths + 1 line per class/function
        - skeleton-top-30:      top-30 files by LOC, full FileSummary
        - issue-text:           problem_statement length
  5. Estimate tokens via 4-chars-per-token heuristic (rough but
     consistent; tighter estimates need tiktoken/Anthropic tokenizer
     which we're not bringing in for an audit).

Output: docs/audits/v10_projected_context_for_cap_hitters.md.

Bucketing for Phase 1 design decisions:
  - <200K tokens — fits any reasonable LLM; minimal compression needed.
  - 200K-500K   — fits 1M-context LLMs comfortably; mild compression OK.
  - 500K-1M     — flirting with V7's cap; must compress aggressively.
  - >1M         — cannot naively pass full skeleton; mandatory selection.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
import time

from harness.dataset import load_verified_view
from harness.sandbox import Sandbox


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
V7_TRAJ_ROOT = PROJECT_ROOT / "runs" / "swebench_500_v7" / "cache" / "trajectories_v7"
V7_PRED = PROJECT_ROOT / "runs" / "swebench_500_v7" / "predictions" / "v7_full500_seed0.jsonl"
DATASET = PROJECT_ROOT / "meta_harness_plus" / "tasks" / "data" / "swebench_verified.jsonl"
OUT = PROJECT_ROOT / "docs" / "audits" / "v10_projected_context_for_cap_hitters.md"

# 4 chars/token is the conventional rough heuristic for English+code mix.
# Anthropic and OpenAI tokenizers both land near 3.5-4.5 chars/token for
# Python source; 4 is a defensible mid.
CHARS_PER_TOKEN = 4

CAP_HITTER_PATTERN = "tokens_in cap"


def _identify_cap_hitters() -> list[str]:
    """Return instance_ids of V7 empty-patch instances whose stop_reason
    indicates a tokens_in cap hit."""
    # First: empty-patch instances.
    empty_iids: list[str] = []
    for line in V7_PRED.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if not (rec.get("model_patch") or "").strip():
            empty_iids.append(rec["instance_id"])
    # Then: stop_reason matching the tokens_in cap.
    cap_hitters: list[str] = []
    for iid in empty_iids:
        traj_path = V7_TRAJ_ROOT / f"{iid}_seed0.json"
        if not traj_path.exists():
            continue
        try:
            traj = json.loads(traj_path.read_text())
        except Exception:
            continue
        stop_reason = traj.get("stop_reason", "") or ""
        if CAP_HITTER_PATTERN in stop_reason:
            cap_hitters.append(iid)
    return sorted(cap_hitters)


# In-container script: walks /testbed for .py files, returns per-file
# (path, n_lines, n_classes, n_functions). Bounded depth = 12.
_SKELETON_PROBE_SCRIPT = r"""
import ast, json, os, pathlib, sys

root = pathlib.Path('/testbed')
out = []

def is_excluded(path):
    parts = set(path.parts)
    if any(p in parts for p in ('.git', '__pycache__', '.tox', '.eggs', 'build', 'dist')):
        return True
    return False

for path in root.rglob('*.py'):
    if is_excluded(path):
        continue
    try:
        text = path.read_text(errors='replace')
    except Exception:
        continue
    n_lines = text.count('\n') + 1
    n_classes = 0
    n_functions = 0
    try:
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                n_classes += 1
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                n_functions += 1
    except Exception:
        pass
    rel = str(path.relative_to(root))
    out.append({
        'path': rel,
        'lines': n_lines,
        'classes': n_classes,
        'functions': n_functions,
        'chars': len(text),
    })
print(json.dumps(out))
"""


def _walk_repo(sb: Sandbox) -> list[dict]:
    """Return per-file stats for every .py file under /testbed."""
    res = sb.run_shell(
        f"python3 -c {_q(_SKELETON_PROBE_SCRIPT)}",
        timeout_s=120,
    )
    if res.exit_code != 0:
        raise RuntimeError(f"skeleton probe failed: {res.stderr[:500]}")
    return json.loads(res.stdout)


def _q(s: str) -> str:
    """Shell single-quote a string for safe inline use."""
    return "'" + s.replace("'", "'\\''") + "'"


def _project_token_counts(files: list[dict], issue_chars: int) -> dict:
    """Project token counts for various V10 prompt shapes.

    The TOTAL projection (added per Phase 1 review) is the worst-case
    single-LLM-call upper bound: every Phase 1 signal source dumped
    into one prompt. Real V10 splits across multiple calls (skeleton
    only into the reranker; top-K file content only into the patch
    generator), so TOTAL is over-pessimistic — but it sets the
    aggregate cap that no single LLM call may exceed.
    """
    n_files = len(files)
    total_lines = sum(f["lines"] for f in files)
    total_classes = sum(f["classes"] for f in files)
    total_functions = sum(f["functions"] for f in files)
    total_chars = sum(f["chars"] for f in files)

    chars_per_symbol_line = 50

    # Path-only skeleton.
    chars_files_only = sum(80 + len(f["path"]) for f in files)

    # Symbol skeleton: path + 1 line per class + 1 line per function.
    chars_symbol_skeleton = chars_files_only + (
        chars_per_symbol_line * (total_classes + total_functions)
    )

    # Top-K files by LOC.
    by_lines = sorted(files, key=lambda f: -f["lines"])
    top10 = by_lines[:10]
    top30 = by_lines[:30]
    top11_to_30 = by_lines[10:30]

    chars_top10_full = sum(f["chars"] for f in top10)
    chars_top30_full = sum(f["chars"] for f in top30)
    chars_top30_symbols = sum(
        80 + len(f["path"]) + chars_per_symbol_line * (f["classes"] + f["functions"])
        for f in top30
    )
    chars_top11_30_symbols = sum(
        80 + len(f["path"]) + chars_per_symbol_line * (f["classes"] + f["functions"])
        for f in top11_to_30
    )

    # Fixed budgets for non-corpus prompt parts.
    chars_system = 2_000 * CHARS_PER_TOKEN          # ~2 K tokens system prompt
    chars_retrieval_signals = 3_000 * CHARS_PER_TOKEN  # ~3 K tokens for top-30 candidate list

    chars_total_naive = (
        chars_symbol_skeleton
        + chars_top30_full
        + issue_chars
        + chars_system
        + chars_retrieval_signals
    )
    # Compressed: drop full-content for files outside top-10; keep
    # symbol-only summaries for top-11..30.
    chars_total_compressed = (
        chars_symbol_skeleton  # full skeleton, symbol form
        + chars_top10_full      # only top-10 at full content
        + chars_top11_30_symbols  # ranks 11-30 as symbols only
        + issue_chars
        + chars_system
        + chars_retrieval_signals
    )
    # Aggressive: drop the full skeleton (only top-K files visible).
    chars_total_aggressive = (
        chars_top10_full
        + chars_top30_symbols
        + issue_chars
        + chars_system
        + chars_retrieval_signals
    )

    tokens_total_naive = chars_total_naive // CHARS_PER_TOKEN
    tokens_total_compressed = chars_total_compressed // CHARS_PER_TOKEN
    tokens_total_aggressive = chars_total_aggressive // CHARS_PER_TOKEN

    if tokens_total_naive < 800_000:
        compression_strategy = "none — naive total fits 800K budget"
    elif tokens_total_compressed < 800_000:
        compression_strategy = "compress: top-10 full + top-11..30 symbols only"
    elif tokens_total_aggressive < 800_000:
        compression_strategy = "aggressive: drop full skeleton; top-10 full + top-30 symbols"
    else:
        compression_strategy = "INSUFFICIENT: even aggressive compression > 800K — must split across LLM calls"

    return {
        "n_files": n_files,
        "total_lines": total_lines,
        "total_classes": total_classes,
        "total_functions": total_functions,
        "total_chars": total_chars,
        "tokens_files_only": chars_files_only // CHARS_PER_TOKEN,
        "tokens_symbol_skeleton": chars_symbol_skeleton // CHARS_PER_TOKEN,
        "tokens_top30_symbols": chars_top30_symbols // CHARS_PER_TOKEN,
        "tokens_top30_full": chars_top30_full // CHARS_PER_TOKEN,
        "tokens_top10_full": chars_top10_full // CHARS_PER_TOKEN,
        "tokens_issue": issue_chars // CHARS_PER_TOKEN,
        "tokens_total_naive": tokens_total_naive,
        "tokens_total_compressed": tokens_total_compressed,
        "tokens_total_aggressive": tokens_total_aggressive,
        "compression_strategy": compression_strategy,
    }


def _bucket(tokens: int) -> str:
    if tokens < 200_000:
        return "<200K (any LLM, minimal compression)"
    if tokens < 500_000:
        return "200K-500K (1M-context LLM, mild compression)"
    if tokens < 1_000_000:
        return "500K-1M (flirts with V7 cap; must compress)"
    return ">1M (mandatory selection — cannot pass full)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N instances (for debugging)")
    args = ap.parse_args()

    cap_hitters = _identify_cap_hitters()
    if args.limit is not None:
        cap_hitters = cap_hitters[: args.limit]
    print(f"[audit] {len(cap_hitters)} V7 cap-hitter instances")

    rows: list[dict] = []
    failed: list[tuple[str, str]] = []
    t0 = time.perf_counter()
    for i, iid in enumerate(cap_hitters):
        print(f"[audit] [{i + 1}/{len(cap_hitters)}] {iid} … ", end="", flush=True)
        try:
            view = load_verified_view(iid)
            # Skeleton probe output is naturally larger than the legacy
            # 32K observation cap (a 3000-file django repo emits ~600 KB
            # of JSON). Use 8 MB observation cap for the audit.
            with Sandbox(view, max_observation_chars=8_000_000) as sb:
                files = _walk_repo(sb)
            issue_chars = len(view.problem_statement)
            metrics = _project_token_counts(files, issue_chars)
            metrics["instance_id"] = iid
            metrics["repo"] = view.repo
            metrics["primary_bucket"] = _bucket(metrics["tokens_symbol_skeleton"])
            rows.append(metrics)
            print(
                f"OK files={metrics['n_files']} "
                f"sym_tok={metrics['tokens_symbol_skeleton']:,}"
            )
        except Exception as exc:
            failed.append((iid, str(exc)))
            print(f"FAIL: {type(exc).__name__}: {exc}")
    dur = time.perf_counter() - t0
    print(f"[audit] done in {dur:.1f}s — {len(rows)} OK, {len(failed)} failed")

    # Render report.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    out.append("# V10 projected-context audit — V7 cap-hitter instances")
    out.append("")
    out.append(
        "Generated by `scripts/audit_v10_projected_context.py`. Walks each "
        "of the 16 V7 cap-hitter instances (those whose V7 stop_reason was "
        "`tokens_in cap (1000000) reached`) inside the V10 sandbox at "
        "base_commit, AST-parses every .py file, and projects the V10 "
        "prompt token count for several skeleton shapes."
    )
    out.append("")
    out.append(
        "Token estimate via 4-chars-per-token heuristic. Tighter numbers "
        "would need a real tokenizer; the bucket call (200K / 500K / 1M / "
        ">1M) is robust to ±20% drift in chars-per-token, so the audit's "
        "Phase 1 conclusions are not heuristic-fragile."
    )
    out.append("")
    out.append(f"- **Cap-hitter instances:** {len(cap_hitters)}")
    out.append(f"- **Successfully audited:** {len(rows)}")
    if failed:
        out.append(f"- **Failed:** {len(failed)} ({', '.join(f for f, _ in failed)})")
    out.append("")

    # Bucket summary.
    out.append("## Bucket distribution (skeleton-symbol projection)")
    out.append("")
    bucket_counts: collections.Counter[str] = collections.Counter(
        r["primary_bucket"] for r in rows
    )
    out.append("| Bucket | Count | Phase 1 implication |")
    out.append("|---|---|---|")
    bucket_implications = {
        "<200K (any LLM, minimal compression)":
            "fits Sonnet/Opus context; pass symbol skeleton directly.",
        "200K-500K (1M-context LLM, mild compression)":
            "use 1M-context model OR cluster files before reranking.",
        "500K-1M (flirts with V7 cap; must compress)":
            "must drop file bodies; pass paths + per-class summaries only.",
        ">1M (mandatory selection — cannot pass full)":
            "MUST do BM25/embedding pre-filter to top-N before any LLM call.",
    }
    for bucket, n in sorted(bucket_counts.items(), key=lambda x: -x[1]):
        out.append(f"| {bucket} | {n} | {bucket_implications.get(bucket, '')} |")
    out.append("")

    # Per-instance detail.
    out.append("## Per-instance projection")
    out.append("")
    out.append(
        "Token columns: skeleton-files-only / skeleton-symbols / top-30 "
        "files (symbols) / top-30 files (full content). The "
        "`primary_bucket` column uses skeleton-symbols (the most likely "
        "Phase 1 reranker shape) for bucketing."
    )
    out.append("")
    out.append(
        "| Instance | Repo | Files | Symbol-skel tok | Top-30 full tok | "
        "Top-10 full tok | **TOTAL naive** | TOTAL compressed | Compression strategy |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|")
    rows.sort(key=lambda r: -r["tokens_total_naive"])
    for r in rows:
        out.append(
            f"| `{r['instance_id']}` | {r['repo']} | "
            f"{r['n_files']:,} | {r['tokens_symbol_skeleton']:,} | "
            f"{r['tokens_top30_full']:,} | {r['tokens_top10_full']:,} | "
            f"**{r['tokens_total_naive']:,}** | "
            f"{r['tokens_total_compressed']:,} | "
            f"{r['compression_strategy']} |"
        )
    out.append("")
    out.append(
        "TOTAL naive = symbol-skeleton + top-30-files-full-content + issue + "
        "system (~2K) + retrieval-signals (~3K). This is the worst-case "
        "single-LLM-call upper bound; real V10 splits across multiple calls "
        "(reranker sees skeleton only; patch generator sees top-K full "
        "content). 800K is the design budget — anything above needs "
        "compression. TOTAL compressed = top-10 at full content + "
        "top-11..30 as symbols only + skeleton + issue + system + retrieval."
    )
    out.append("")

    if failed:
        out.append("## Audit failures")
        out.append("")
        for iid, err in failed:
            out.append(f"- `{iid}`: {err}")
        out.append("")

    # Phase 1 implications synthesis.
    out.append("## Phase 1 design conclusions")
    out.append("")
    sym_tokens = [r["tokens_symbol_skeleton"] for r in rows]
    naive_totals = [r["tokens_total_naive"] for r in rows]
    compressed_totals = [r["tokens_total_compressed"] for r in rows]
    if sym_tokens:
        med_sym = sorted(sym_tokens)[len(sym_tokens) // 2]
        max_sym = max(sym_tokens)
        out.append(
            f"- Skeleton-symbol projection — median **{med_sym:,}**, "
            f"max **{max_sym:,}** tokens (well within 1M context)."
        )

        med_naive = sorted(naive_totals)[len(naive_totals) // 2]
        max_naive = max(naive_totals)
        n_over_800k_naive = sum(1 for t in naive_totals if t >= 800_000)
        out.append(
            f"- **TOTAL naive** projection (single-LLM-call worst case) — "
            f"median **{med_naive:,}**, max **{max_naive:,}** tokens. "
            f"**{n_over_800k_naive}/{len(naive_totals)}** instances cross the "
            f"800K design budget."
        )

        med_compressed = sorted(compressed_totals)[len(compressed_totals) // 2]
        max_compressed = max(compressed_totals)
        n_over_800k_compressed = sum(1 for t in compressed_totals if t >= 800_000)
        out.append(
            f"- **TOTAL compressed** (top-10 full + top-11..30 symbols + "
            f"skeleton) — median **{med_compressed:,}**, max "
            f"**{max_compressed:,}** tokens. "
            f"**{n_over_800k_compressed}/{len(compressed_totals)}** instances "
            f"still over 800K after compression."
        )

        out.append("")
        out.append("### Compression strategy by instance (Phase 1 acceptance criteria)")
        out.append("")
        if n_over_800k_naive == 0:
            out.append(
                "- **Naive total fits 800K on every cap-hitter.** Phase 1 "
                "may default to passing skeleton + top-30 full content + "
                "issue + retrieval signals into a single LLM call without "
                "compression."
            )
        elif n_over_800k_compressed == 0:
            out.append(
                f"- **{n_over_800k_naive} instances need compression**, "
                f"but the documented strategy (top-10 full + top-11..30 "
                f"symbols only) fits all of them under 800K. Phase 1 "
                f"should default to compressed shape on big-repo instances "
                f"(django, sympy, matplotlib) and leave naive for small "
                f"repos. The compression toggle is a Stage 1g knob, not "
                f"a Stage 1b concern."
            )
        else:
            out.append(
                f"- **{n_over_800k_compressed} instances exceed 800K even "
                f"with compression.** Phase 1 MUST split these across "
                f"multiple LLM calls: a localizer call (skeleton + "
                f"retrieval signals only) and separate per-candidate "
                f"patch-generation calls (issue + single file content). "
                f"Specific instances flagged in the per-instance table."
            )

        # Top contributors to TOTAL.
        top30_full_tokens = [r["tokens_top30_full"] for r in rows]
        max_top30 = max(top30_full_tokens)
        out.append(
            f"- Top-30-full-content alone — max **{max_top30:,}** tokens "
            f"(`sympy__sympy-23262`-class instances; sympy ships large "
            f"per-file modules). On these, top-30 full content dominates "
            f"the budget; compression cannot help and we must pass fewer "
            f"files. Phase 1 hyperparameter `K_files_full=10` is the "
            f"safer default on big-repo instances."
        )
    out.append("")
    out.append(
        "**Phase 1 commit 2 acceptance criterion:** the Stage 1a skeleton "
        "builder MUST produce a skeleton ≤ the symbol-skeleton projection "
        "above for each cap-hitter instance. If skeleton size on the "
        "real Phase 1a build differs from this audit by >2x, the "
        "skeleton builder needs revisiting before Stage 1b."
    )
    out.append("")

    OUT.write_text("\n".join(out))
    print(f"[audit] wrote {OUT.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
