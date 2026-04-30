"""Audit the gold-patch matcher's strictness.

Per the long-horizon batch spec (commit 6a): before trusting 98% top-10
as the headline, verify the matcher's "hit" semantics aren't leaking
lenient matches that inflate the score.

For each dev_50 instance:
  1. Load gold-patch touched files via the eval-only path
     (harness.eval.load_eval_metadata).
  2. Load the cached rerank result (top-10) from
     runs/v10_dev50_retr_eval/checkpoints/embed_noshortlist_tb_bs256_rerank/.
  3. For each "hit" claimed at top-K (K ∈ {1, 5, 10}), classify the
     match by type:
       - exact:      gold_file == retrieved_file (literal equality)
       - normalized: equal after harness.eval._normalize_path
       - suffix:     retrieved.endswith("/" + gold)
       - prefix:     gold.startswith(retrieved + "/")
       - basename:   PurePosixPath(gold).name == PurePosixPath(ret).name
                     but NEITHER endswith holds
       - none:       not actually a hit (shouldn't happen)

  Output: docs/audits/gold_match_strictness.md.

  Hard-stop threshold (per spec): if >20% of hits at top-10 are
  non-exact (anything except `exact` or `normalized`), the eval is
  using a lenient matcher and the headline must be redone with strict
  matching.
"""

from __future__ import annotations

import json
import pathlib
from collections import Counter
from pathlib import PurePosixPath


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEV_50 = PROJECT_ROOT / "splits" / "dev_50.json"
RERANK_CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "v10_dev50_retr_eval"
    / "checkpoints"
    / "embed_noshortlist_tb_bs256_rerank"
)
OUT = PROJECT_ROOT / "docs" / "audits" / "gold_match_strictness.md"


def _classify_match(gold: str, retrieved: str) -> str:
    """Return the most-strict match type that holds for (gold, retrieved).

    Strictness order:
      exact > normalized > suffix > prefix > basename > none
    """
    if gold == retrieved:
        return "exact"
    # normalized: same as eval._normalize_path
    g_norm = gold.strip().lstrip("./").lstrip("/")
    r_norm = retrieved.strip().lstrip("./").lstrip("/")
    if g_norm == r_norm:
        return "normalized"
    # suffix: retrieved ends with "/<gold>"  OR  gold ends with "/<retrieved>"
    if r_norm.endswith("/" + g_norm) or g_norm.endswith("/" + r_norm):
        return "suffix"
    # prefix: gold path is a directory, retrieved is something inside it
    if g_norm.startswith(r_norm.rstrip("/") + "/") or r_norm.startswith(g_norm.rstrip("/") + "/"):
        return "prefix"
    # basename only
    if PurePosixPath(g_norm).name == PurePosixPath(r_norm).name:
        return "basename"
    return "none"


def main() -> int:
    # The eval's actual hit semantics: set membership of normalized
    # paths. So a "hit" is when retrieved == gold exactly OR
    # _normalize_path strips the same leading ./ — which is type
    # "exact" or "normalized" only. Anything else is NOT a hit by the
    # eval's logic.
    from harness.eval import load_eval_metadata

    dev = json.loads(DEV_50.read_text())
    instances = dev["instances"]
    instance_ids = [e["instance_id"] for e in instances]

    print(f"[audit] loading gold metadata for {len(instance_ids)} instances")
    meta = load_eval_metadata(instance_ids)
    gold_per_iid = meta.gold_touched_files

    # Aggregate match counts at K = 1, 5, 10.
    K_VALUES = (1, 5, 10)
    k_match_counts: dict = {k: Counter() for k in K_VALUES}
    k_hit_counts: dict = {k: 0 for k in K_VALUES}
    k_miss_counts: dict = {k: 0 for k in K_VALUES}

    # Per-instance breakdown for the audit table.
    rows: list[dict] = []
    for iid in instance_ids:
        ckpt_path = RERANK_CHECKPOINT_DIR / f"{iid}.json"
        if not ckpt_path.exists():
            print(f"[audit] WARN: no rerank checkpoint for {iid}; skipping")
            continue
        ckpt = json.loads(ckpt_path.read_text())
        # The reranked top-K lives in "retrieved" (rerank pipeline
        # rewrote it to the reranked file list).
        retrieved = ckpt.get("retrieved") or []
        gold = gold_per_iid.get(iid, set())
        if not gold:
            print(f"[audit] WARN: no gold for {iid}; skipping")
            continue

        # For each K, find whether top-K hits any gold and what the
        # match type is.
        per_k: dict = {}
        for k in K_VALUES:
            top_k = retrieved[:k]
            best_type = "none"
            best_pair: tuple | None = None
            for r in top_k:
                for g in gold:
                    t = _classify_match(g, r)
                    if t == "none":
                        continue
                    # Pick the best (most strict) hit at this K.
                    if best_type == "none" or _strictness(t) > _strictness(best_type):
                        best_type = t
                        best_pair = (g, r)
            per_k[k] = (best_type, best_pair)
            if best_type != "none":
                k_hit_counts[k] += 1
                k_match_counts[k][best_type] += 1
            else:
                k_miss_counts[k] += 1

        rows.append({
            "iid": iid,
            "gold": sorted(gold),
            "top1": per_k[1],
            "top5": per_k[5],
            "top10": per_k[10],
            "n_retrieved": len(retrieved),
        })

    # Build the markdown report.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    out.append("# Gold-patch matcher strictness audit (commit 6a)")
    out.append("")
    out.append(
        "Generated by `scripts/audit_gold_match_strictness.py`. For each "
        "dev_50 instance, classifies every claimed hit at top-K by the "
        "tightest match type that holds:"
    )
    out.append("")
    out.append(
        "  - **exact** — `gold == retrieved` literally\n"
        "  - **normalized** — equal after stripping leading `./` and `/`\n"
        "  - **suffix** — `retrieved.endswith('/' + gold)` (lenient)\n"
        "  - **prefix** — gold path is a directory containing retrieved (or vice versa)\n"
        "  - **basename** — same filename, different parent directories\n"
        "  - **none** — not actually a hit"
    )
    out.append("")
    out.append(
        "Per V10_DESIGN.md, the production matcher in "
        "`harness.eval.evaluate_retrieval_recall` does set-membership "
        "of normalized paths — so it accepts only `exact` and "
        "`normalized` matches. Anything classified as `suffix`, "
        "`prefix`, or `basename` here would be a MISS by the production "
        "matcher; the audit is checking whether such cases also got "
        "counted as hits (i.e., whether there's a leak)."
    )
    out.append("")
    out.append(f"## Coverage")
    out.append("")
    out.append(f"- Instances audited: {len(rows)}")
    out.append(f"- Total dev_50 instances: {len(instance_ids)}")
    out.append("")

    # Headline distribution
    out.append("## Match-type distribution per K")
    out.append("")
    out.append("| K | Hits | exact | normalized | suffix | prefix | basename | misses |")
    out.append("|---|---|---|---|---|---|---|---|")
    for k in K_VALUES:
        c = k_match_counts[k]
        total_hits = k_hit_counts[k]
        misses = k_miss_counts[k]
        out.append(
            f"| {k} | {total_hits}/{len(rows)} | "
            f"{c.get('exact', 0)} | {c.get('normalized', 0)} | "
            f"{c.get('suffix', 0)} | {c.get('prefix', 0)} | "
            f"{c.get('basename', 0)} | {misses} |"
        )
    out.append("")

    # Verdict
    out.append("## Strictness verdict")
    out.append("")
    for k in K_VALUES:
        c = k_match_counts[k]
        strict_count = c.get("exact", 0) + c.get("normalized", 0)
        non_strict = (
            c.get("suffix", 0) + c.get("prefix", 0) + c.get("basename", 0)
        )
        total = strict_count + non_strict
        if total == 0:
            pct_strict = 100.0
            pct_non_strict = 0.0
        else:
            pct_strict = 100.0 * strict_count / total
            pct_non_strict = 100.0 * non_strict / total
        marker = "✓" if pct_non_strict <= 5.0 else ("WARN" if pct_non_strict <= 20.0 else "FAIL")
        out.append(
            f"- **Top-{k}:** {pct_strict:.1f}% strict (exact + normalized), "
            f"{pct_non_strict:.1f}% non-strict ({marker})"
        )
    out.append("")
    out.append(
        "Hard-stop threshold from the spec: >20% non-exact at top-10 "
        "fails the audit and the headline must be redone with strict "
        "matching."
    )
    out.append("")

    # Per-instance detail
    out.append("## Per-instance match types")
    out.append("")
    out.append("Format: `top-K: match_type (gold ↔ retrieved)`")
    out.append("")
    out.append("| Instance | top-1 | top-5 | top-10 |")
    out.append("|---|---|---|---|")
    for r in sorted(rows, key=lambda r: r["iid"]):
        cells = []
        for k in K_VALUES:
            t, pair = r[f"top{k}"]
            if t == "none":
                cells.append(f"miss")
            else:
                if pair:
                    g, ret = pair
                    if g == ret:
                        cells.append(f"`{t}`")
                    else:
                        cells.append(f"`{t}` (`{g}` ↔ `{ret}`)")
                else:
                    cells.append(f"`{t}`")
        out.append(f"| `{r['iid']}` | {cells[0]} | {cells[1]} | {cells[2]} |")
    out.append("")

    OUT.write_text("\n".join(out))
    print(f"[audit] wrote {OUT.relative_to(PROJECT_ROOT)}")
    for k in K_VALUES:
        c = k_match_counts[k]
        strict = c.get("exact", 0) + c.get("normalized", 0)
        non_strict = (
            c.get("suffix", 0) + c.get("prefix", 0) + c.get("basename", 0)
        )
        total = strict + non_strict
        print(
            f"[audit] top-{k}: {total} hits — "
            f"{strict} strict, {non_strict} non-strict "
            f"({100.0 * non_strict / total if total else 0:.1f}% non-strict)"
        )
    return 0


def _strictness(t: str) -> int:
    order = {
        "exact": 5, "normalized": 4, "suffix": 3,
        "prefix": 2, "basename": 1, "none": 0,
    }
    return order.get(t, 0)


if __name__ == "__main__":
    raise SystemExit(main())
