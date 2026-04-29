"""Build data/phase7_seed_repos.json — the curated seed list for the
Phase 7 corpus crawl.

Per V10_DESIGN.md §10.5, the full Phase 7 corpus targets ~5,000 Python
repos license-filtered for pre-cutoff bug-fix commits. This script
produces the SEED that the crawler reads first, and validates the
seed-list logic without hitting GitHub.

Two modes:

  --hardcoded (default, no network)
      Emits a v0 seed of ~50 well-known Python repos with known licenses.
      Sufficient to validate scripts/phase7_crawl.py end-to-end on a
      10-repo dry run. License values are best-effort from public
      repo metadata at the time of writing.

  --refresh (requires network + optionally GH_TOKEN)
      Hits PyPI's top-packages endpoint and the GitHub repo metadata
      API to expand the seed to ~5,000 repos. NOT IMPLEMENTED in this
      Phase 0 commit — gated as a separate operator decision because
      it needs API quotas and a curation review pass.

Output: data/phase7_seed_repos.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "data" / "phase7_seed_repos.json"

ACCEPTABLE_LICENSES = {"mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "isc"}


# ---------------------------------------------------------------------------
# Hardcoded seed — well-known Python repos with permissive licenses.
# Ordered roughly by category so a 10-repo dry run hits diverse codebases.
# All entries: known-licensed, active maintenance, real-world bug-fix
# commit history, sized appropriately for the crawl pipeline.
# ---------------------------------------------------------------------------

HARDCODED_SEED: list[dict] = [
    # Web frameworks
    {"repo": "pallets/flask",        "license": "bsd-3-clause", "category": "web"},
    {"repo": "pallets/click",        "license": "bsd-3-clause", "category": "web"},
    {"repo": "psf/requests",         "license": "apache-2.0",   "category": "web"},
    {"repo": "encode/httpx",         "license": "bsd-3-clause", "category": "web"},
    {"repo": "tiangolo/fastapi",     "license": "mit",          "category": "web"},
    {"repo": "tornadoweb/tornado",   "license": "apache-2.0",   "category": "web"},
    {"repo": "pyca/cryptography",    "license": "apache-2.0",   "category": "security"},

    # Data / scientific
    {"repo": "scikit-learn/scikit-learn", "license": "bsd-3-clause", "category": "ml"},
    {"repo": "pydata/xarray",        "license": "apache-2.0",   "category": "data"},
    {"repo": "scipy/scipy",          "license": "bsd-3-clause", "category": "scientific"},
    {"repo": "matplotlib/matplotlib","license": "bsd-3-clause", "category": "scientific"},
    {"repo": "astropy/astropy",      "license": "bsd-3-clause", "category": "scientific"},
    {"repo": "sympy/sympy",          "license": "bsd-3-clause", "category": "scientific"},
    {"repo": "pandas-dev/pandas",    "license": "bsd-3-clause", "category": "data"},

    # CLI / dev tools
    {"repo": "pytest-dev/pytest",    "license": "mit",          "category": "tools"},
    {"repo": "python/mypy",          "license": "mit",          "category": "tools"},
    {"repo": "pylint-dev/pylint",    "license": "gpl-2.0",      "category": "tools",
     "note": "GPL — exclude from any model fine-tune; included in seed so the license filter is exercised"},
    {"repo": "psf/black",            "license": "mit",          "category": "tools"},
    {"repo": "sphinx-doc/sphinx",    "license": "bsd-2-clause", "category": "tools"},
    {"repo": "PyCQA/flake8",         "license": "mit",          "category": "tools"},
    {"repo": "PyCQA/isort",          "license": "mit",          "category": "tools"},

    # Async / network
    {"repo": "aio-libs/aiohttp",     "license": "apache-2.0",   "category": "async"},
    {"repo": "python-trio/trio",     "license": "apache-2.0",   "category": "async"},
    {"repo": "MagicStack/asyncpg",   "license": "apache-2.0",   "category": "async"},

    # Database / ORM
    {"repo": "sqlalchemy/sqlalchemy","license": "mit",          "category": "db"},
    {"repo": "django/django",        "license": "bsd-3-clause", "category": "web"},
    {"repo": "MongoEngine/mongoengine","license": "mit",        "category": "db"},

    # ML / AI utilities
    {"repo": "huggingface/transformers", "license": "apache-2.0", "category": "ml"},
    {"repo": "huggingface/datasets", "license": "apache-2.0",   "category": "ml"},
    {"repo": "pytorch/vision",       "license": "bsd-3-clause", "category": "ml"},
    {"repo": "scikit-image/scikit-image", "license": "bsd-3-clause", "category": "ml"},

    # Visualization / docs
    {"repo": "mwaskom/seaborn",      "license": "bsd-3-clause", "category": "viz"},
    {"repo": "bokeh/bokeh",          "license": "bsd-3-clause", "category": "viz"},
    {"repo": "plotly/plotly.py",     "license": "mit",          "category": "viz"},

    # Utility libraries
    {"repo": "python-pillow/Pillow", "license": "mit",          "category": "media"},
    {"repo": "boto/boto3",           "license": "apache-2.0",   "category": "cloud"},
    {"repo": "ansible/ansible",      "license": "gpl-3.0",      "category": "ops",
     "note": "GPL — exclude from any model fine-tune; included so license filter is exercised"},
    {"repo": "Textualize/rich",      "license": "mit",          "category": "tools"},
    {"repo": "pydantic/pydantic",    "license": "mit",          "category": "tools"},
    {"repo": "tqdm/tqdm",            "license": "mpl-2.0",      "category": "tools"},
    {"repo": "psf/requests-toolbelt","license": "apache-2.0",   "category": "web"},
    {"repo": "more-itertools/more-itertools", "license": "mit", "category": "tools"},
    {"repo": "python-attrs/attrs",   "license": "mit",          "category": "tools"},
    {"repo": "spec-first/connexion", "license": "apache-2.0",   "category": "web"},
    {"repo": "pyyaml/pyyaml",        "license": "mit",          "category": "tools"},
    {"repo": "pypa/pip",             "license": "mit",          "category": "tools"},
    {"repo": "pypa/setuptools",      "license": "mit",          "category": "tools"},
    {"repo": "pyca/cryptography",    "license": "apache-2.0",   "category": "security"},
    {"repo": "werkzeug-team/werkzeug","license": "bsd-3-clause", "category": "web"},
    {"repo": "miguelgrinberg/Flask-SocketIO", "license": "mit", "category": "web"},
]


def _filter_by_license(seed: list[dict]) -> list[dict]:
    """Apply the license filter from V10_DESIGN.md §10.5: only MIT,
    Apache-2.0, BSD, and ISC are acceptable for fine-tune corpus."""
    accepted: list[dict] = []
    rejected: list[dict] = []
    for entry in seed:
        if entry["license"].lower() in ACCEPTABLE_LICENSES:
            accepted.append(entry)
        else:
            rejected.append(entry)
    return accepted, rejected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("hardcoded", "refresh"), default="hardcoded")
    args = ap.parse_args()

    if args.mode == "refresh":
        print("[seed] --refresh mode is not implemented in this commit.")
        print("[seed] Hit PyPI top packages + GitHub metadata to expand;")
        print("[seed] requires network + (recommended) GH_TOKEN.")
        return 2

    seed = list(HARDCODED_SEED)
    accepted, rejected = _filter_by_license(seed)

    out = {
        "version": "v10-phase7-seed-r1-hardcoded",
        "mode": "hardcoded",
        "description": (
            "Phase 7 corpus seed — v0 hardcoded sample of well-known "
            "Python repos for validating the crawler pipeline. The full "
            "~5000-repo seed requires PyPI + GitHub API curation; "
            "deferred to a network-enabled session per V10_DESIGN.md §10.5."
        ),
        "acceptable_licenses": sorted(ACCEPTABLE_LICENSES),
        "n_total": len(seed),
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        "rejected": [
            {"repo": r["repo"], "license": r["license"], "reason": "license"}
            for r in rejected
        ],
        "repos": [r["repo"] for r in accepted],
        "metadata": {r["repo"]: r for r in accepted},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[seed] wrote {OUT.relative_to(PROJECT_ROOT)}")
    print(f"[seed] total={len(seed)}  accepted={len(accepted)}  rejected={len(rejected)}")
    if rejected:
        print(f"[seed] license-rejected repos: {[r['repo'] for r in rejected]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
