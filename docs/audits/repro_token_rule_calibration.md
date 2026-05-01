# Forbidden-token substring rule calibration (commit 17a)

Per V10_DESIGN_PHASE2.md §8.4 calibration step: before shipping the input-layer firewall, scan real test names in the SWE-bench Verified repos for any that would trip the substring rule on a legitimate name. The output-layer guard is informational, but a poorly-tuned rule would still spam the operator with WARNING noise.

## Scope

- Repos scanned: 12
- Test functions/methods inspected: 43831
- FORBIDDEN_TOKENS: ['fail_to_pass', 'pass_to_pass', 'test_patch', 'hints_text', 'hints', 'gold_patch', 'resolved']

## Verdict

**2 legitimate test name(s) trip the rule.** Within the hard-stop threshold of ≤2. Acceptable as informational-only noise (output-layer guard per §8.4).

## Hits — per repo

| Repo | Test file | Test name | Forbidden token |
|---|---|---|---|
| django_django | `tests/test_client_regress/tests.py` | `RequestMethodTests.test_patch` | `test_patch` |
| django_django | `tests/test_client_regress/tests.py` | `RequestMethodStringDataTests.test_patch` | `test_patch` |

## Methodology

  - Source: cached `repo_cache/v10_skeletons/<repo>/<commit>/skeleton.json`.   - One commit per repo (most recent in cache).
  - All files with 'test' in their path; all functions and class methods that start with 'test_'.
  - Normalization: lowercase + strip underscores/whitespace (matches `harness.repro._normalize_for_match`).
  - Substring matching: each normalized FORBIDDEN_TOKENS entry is checked as a substring of the normalized test name.

