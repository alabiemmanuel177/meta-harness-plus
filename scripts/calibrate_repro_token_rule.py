"""Calibration spot-check for the §8.4 forbidden-token substring rule.

Per docs/V10_DESIGN_PHASE2.md §8.4: before shipping the input-layer
firewall, verify the substring rule doesn't trip on legitimate test
names that exist in the wild. Hard-stop #4 from the commit-17a spec:
if >2 legitimate test names trip, the rule needs narrowing.

This script reads cached repo skeletons under
``repo_cache/v10_skeletons/`` (built by Phase 1's localizer) and
collects every function name that looks like a test (starts with
"test_") across the 12 SWE-bench Verified repos. It then runs each
name through the same normalize-then-substring check the input
firewall applies and reports any hits.

Output: docs/audits/repro_token_rule_calibration.md.
"""

from __future__ import annotations

import json
import pathlib
from collections import defaultdict


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE = PROJECT_ROOT / "repo_cache" / "v10_skeletons"
OUT = PROJECT_ROOT / "docs" / "audits" / "repro_token_rule_calibration.md"


def main() -> int:
    from harness.repro import _detect_forbidden_tokens
    from harness.views import FORBIDDEN_TOKENS

    if not CACHE.is_dir():
        print(f"[calibrate] no cache at {CACHE}; nothing to scan")
        return 1

    # repo_dir -> commit -> [test names found in test files]
    per_repo_hits: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    repo_count = 0
    test_name_count = 0

    for repo_dir in sorted(CACHE.iterdir()):
        if not repo_dir.is_dir():
            continue
        # Pick the most recent commit's skeleton for each repo (any
        # commit's test corpus is representative for calibration).
        commits = sorted(repo_dir.iterdir())
        if not commits:
            continue
        skel_path = commits[-1] / "skeleton.json"
        if not skel_path.exists():
            continue
        repo_count += 1
        try:
            skel = json.loads(skel_path.read_text())
        except Exception:
            continue
        for f in skel.get("files", []):
            path = f.get("path", "")
            if "test" not in path.lower():
                continue
            for fn in f.get("functions", []):
                name = fn.get("name", "")
                if not name.startswith("test_"):
                    continue
                test_name_count += 1
                for tok in _detect_forbidden_tokens(name):
                    per_repo_hits[repo_dir.name].append(
                        (path, name, tok)
                    )
            for cls in f.get("classes", []):
                for method in cls.get("methods", []):
                    name = method.get("name", "")
                    if not name.startswith("test_"):
                        continue
                    test_name_count += 1
                    for tok in _detect_forbidden_tokens(name):
                        per_repo_hits[repo_dir.name].append(
                            (path, f"{cls['name']}.{name}", tok)
                        )

    total_hits = sum(len(v) for v in per_repo_hits.values())
    HARD_STOP_THRESHOLD = 2

    out: list[str] = []
    out.append("# Forbidden-token substring rule calibration (commit 17a)")
    out.append("")
    out.append(
        "Per V10_DESIGN_PHASE2.md §8.4 calibration step: before "
        "shipping the input-layer firewall, scan real test names in "
        "the SWE-bench Verified repos for any that would trip the "
        "substring rule on a legitimate name. The output-layer guard "
        "is informational, but a poorly-tuned rule would still spam "
        "the operator with WARNING noise."
    )
    out.append("")
    out.append("## Scope")
    out.append("")
    out.append(f"- Repos scanned: {repo_count}")
    out.append(f"- Test functions/methods inspected: {test_name_count}")
    out.append(f"- FORBIDDEN_TOKENS: {list(FORBIDDEN_TOKENS)}")
    out.append("")
    out.append("## Verdict")
    out.append("")
    if total_hits == 0:
        out.append(
            "**Zero legitimate test names trip the substring rule.** "
            "The current FORBIDDEN_TOKENS set is well-calibrated against "
            "real test corpora — no narrowing needed before shipping. "
            "Hard-stop #4 (commit-17a spec) NOT tripped."
        )
    elif total_hits <= HARD_STOP_THRESHOLD:
        out.append(
            f"**{total_hits} legitimate test name(s) trip the rule.** "
            f"Within the hard-stop threshold of ≤{HARD_STOP_THRESHOLD}. "
            f"Acceptable as informational-only noise (output-layer guard "
            f"per §8.4)."
        )
    else:
        out.append(
            f"**HARD STOP: {total_hits} legitimate test names trip the "
            f"rule (threshold: ≤{HARD_STOP_THRESHOLD}).** Narrow the "
            f"substring rule before shipping. Candidates: tighten "
            f"matching to require word-boundary, or drop ambiguous "
            f"tokens (e.g., 'hints' is the most generic)."
        )
    out.append("")

    if total_hits > 0:
        out.append("## Hits — per repo")
        out.append("")
        out.append("| Repo | Test file | Test name | Forbidden token |")
        out.append("|---|---|---|---|")
        for repo, hits in sorted(per_repo_hits.items()):
            for path, name, tok in hits:
                out.append(f"| {repo} | `{path}` | `{name}` | `{tok}` |")
        out.append("")

    out.append("## Methodology")
    out.append("")
    out.append(
        "  - Source: cached `repo_cache/v10_skeletons/<repo>/<commit>/skeleton.json`. "
        "  - One commit per repo (most recent in cache).\n"
        "  - All files with 'test' in their path; all functions and "
        "class methods that start with 'test_'.\n"
        "  - Normalization: lowercase + strip underscores/whitespace "
        "(matches `harness.repro._normalize_for_match`).\n"
        "  - Substring matching: each normalized FORBIDDEN_TOKENS "
        "entry is checked as a substring of the normalized test name."
    )
    out.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out) + "\n")
    print(f"[calibrate] scanned {test_name_count} test names across {repo_count} repos")
    print(f"[calibrate] hits: {total_hits}")
    print(f"[calibrate] wrote {OUT.relative_to(PROJECT_ROOT)}")
    return 0 if total_hits <= HARD_STOP_THRESHOLD else 2


if __name__ == "__main__":
    raise SystemExit(main())
