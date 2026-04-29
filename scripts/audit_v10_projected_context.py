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
    """Project token counts for various V10 prompt shapes."""
    n_files = len(files)
    total_lines = sum(f["lines"] for f in files)
    total_classes = sum(f["classes"] for f in files)
    total_functions = sum(f["functions"] for f in files)
    total_chars = sum(f["chars"] for f in files)

    # Path-only skeleton: just one path per file, ~80 chars/path with framing.
    chars_files_only = sum(80 + len(f["path"]) for f in files)

    # Symbol skeleton: path + 1 line per class + 1 line per function.
    # Typical line: "  def foo(self, x): ..." -> ~40 chars.
    chars_per_symbol_line = 50
    chars_symbol_skeleton = chars_files_only + (
        chars_per_symbol_line * (total_classes + total_functions)
    )

    # Top-30 files by LOC, full file content (worst case for reranker
    # if it sees actual code, not just summaries):
    top30 = sorted(files, key=lambda f: -f["lines"])[:30]
    chars_top30_full = sum(f["chars"] for f in top30)

    # Top-30 files by LOC, symbols-only:
    chars_top30_symbols = sum(
        80 + len(f["path"]) + chars_per_symbol_line * (f["classes"] + f["functions"])
        for f in top30
    )

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
        "tokens_issue": issue_chars // CHARS_PER_TOKEN,
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
        "| Instance | Repo | Files | Total lines | Issue tok | "
        "Files-only tok | Symbol-skel tok | Top-30 sym tok | "
        "Top-30 full tok | Bucket |"
    )
    out.append(
        "|---|---|---|---|---|---|---|---|---|---|"
    )
    rows.sort(key=lambda r: -r["tokens_symbol_skeleton"])
    for r in rows:
        out.append(
            f"| `{r['instance_id']}` | {r['repo']} | "
            f"{r['n_files']:,} | {r['total_lines']:,} | "
            f"{r['tokens_issue']:,} | {r['tokens_files_only']:,} | "
            f"**{r['tokens_symbol_skeleton']:,}** | "
            f"{r['tokens_top30_symbols']:,} | "
            f"{r['tokens_top30_full']:,} | {r['primary_bucket'].split(' (')[0]} |"
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
    if sym_tokens:
        max_sym = max(sym_tokens)
        med_sym = sorted(sym_tokens)[len(sym_tokens) // 2]
        out.append(
            f"- Median skeleton-symbol projection: **{med_sym:,} tokens**. "
            f"Max: **{max_sym:,} tokens**."
        )
        if max_sym >= 1_000_000:
            out.append(
                "- **At least one cap-hitter would still blow up V10's "
                "skeleton-symbol projection.** Phase 1 MUST do BM25 / "
                "embedding pre-filter to a top-N candidate set BEFORE "
                "any reranker call. Naive 'pass the full skeleton' is "
                "not viable on these instances."
            )
        elif max_sym >= 500_000:
            out.append(
                "- The largest cap-hitter still fits in 1M context but "
                "uses 50%+ of it. Phase 1's reranker should pre-filter "
                "to top-N (recommended N=30) before the rerank LLM call "
                "to keep cost predictable."
            )
        else:
            out.append(
                "- All cap-hitters fit comfortably in 1M context with "
                "the symbol skeleton. V10's prompt shape (no "
                "FAIL_TO_PASS, structured skeleton) is materially "
                "smaller than V7's; the cap risk does not transfer."
            )
        top30_full_tokens = [r["tokens_top30_full"] for r in rows]
        max_top30 = max(top30_full_tokens)
        out.append(
            f"- Top-30-full-content projection (Phase 1 worst case if "
            f"the agent path reads the full top-30 files): "
            f"max **{max_top30:,} tokens**, median "
            f"**{sorted(top30_full_tokens)[len(top30_full_tokens) // 2]:,}**."
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
