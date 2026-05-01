"""Phase 2 reproduction-test generator.

Per docs/V10_DESIGN_PHASE2.md (commits 7a + 19c7f27 design revision):
  - Generator inputs: InstanceView + RankedFile[] (top-K=10) + AST
    snippets of those K files (function signatures + class headers,
    no bodies; per §8.1). NOT the full RepoSkeleton.
  - Input-layer firewall (§8.4): forbidden tokens are blocked from
    reaching the prompt. The output-layer scan is INFORMATIONAL only
    (logs WARNING, does NOT block).
  - Output: ReproTestCase frozen dataclass (§3.2).

Commit 17a covered the single-attempt generator + ReproTestCase schema
+ Sandbox.run_repro_test method.

Commit 17b adds:
  - ReproRejectReason enum (controlled vocabulary for reject reasons).
  - ReproAttemptTraceRow dataclass with V0 schema enforcement
    (post_patch_pass_status MUST be "unknown" — V0 cannot see the
    gold patch per §8.5).
  - generate_with_retry orchestrator: must-fail-at-base verification,
    N=3 retry loop, deterministic widen-on-retry, $0.30/instance cost
    cap (§8.6) checked BETWEEN attempts only.
  - JSONL instrumentation per §3.4.

The generator uses harness.llm.clients.complete_chat with
role="repro_generator" — never a hardcoded model name. The model
agnosticism unit test (commit 16a) enforces this structurally.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import logging
import pathlib
import re
import time
from dataclasses import dataclass
from enum import Enum

from harness.localization_signals import RankedFile
from harness.views import (
    FORBIDDEN_TOKENS,
    InstanceView,
    _assert_no_forbidden_field_names,
)


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output dataclass + status enum
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReproTestCase:
    """Per V10_DESIGN_PHASE2.md §3.2."""

    instance_id: str
    test_filename: str            # written into <test_dirs>[0]/<test_filename>
    test_code: str                # the pytest source
    target_test_id: str           # "<test_filename>::test_<descriptive_name>"
    rationale: str                # 1-2 sentences from the generator
    generator_model: str          # e.g. "deepseek-chat"
    generator_input_tokens: int
    generator_output_tokens: int
    attempt_index: int            # 0..N-1 (multiple attempts may be needed)

    def __post_init__(self) -> None:
        _assert_no_forbidden_field_names(type(self))
        if not self.instance_id:
            raise ValueError("ReproTestCase.instance_id required")
        if not self.test_filename:
            raise ValueError("ReproTestCase.test_filename required")
        if not self.test_code:
            raise ValueError("ReproTestCase.test_code required")
        if not self.target_test_id:
            raise ValueError("ReproTestCase.target_test_id required")
        if self.attempt_index < 0:
            raise ValueError(
                f"ReproTestCase.attempt_index must be >= 0, "
                f"got {self.attempt_index}"
            )


class ReproStatus:
    """Status enum for repro generation. The orchestrator (commit 17b)
    uses these. Note that ReproSignal in views.py uses different
    runtime-execution statuses (pass/fail/no_repro/errored); these
    are GENERATION statuses."""

    USABLE = "usable"
    NO_REPRO = "no_repro"
    NO_REPRO_BUDGET = "no_repro_budget"


class ReproGeneratorError(RuntimeError):
    """Raised when the generator cannot produce a valid ReproTestCase
    (input firewall fires, response unparseable, missing fields)."""


# ---------------------------------------------------------------------------
# Input-layer firewall (§8.4 — block forbidden tokens BEFORE the prompt)
# ---------------------------------------------------------------------------


def _normalize_for_match(s: str) -> str:
    """Lowercase normalization used for the substring-fallback check
    on compound forbidden tokens. Strips underscores and whitespace
    so 'failToPass' / 'fail_to_pass' / 'failtopass' all collapse to
    'failtopass'.

    NOTE: substring-on-normalized matching alone produced 69 false
    positives in the §8.4 calibration step (commit 17a) on legitimate
    test names like ``test_patches_distortion`` (matched
    'testpatch' inside 'testpatchesdistortion'). The actual matcher
    used by the firewall is ``_detect_forbidden_tokens``, which uses
    word-boundary matching against the original token. The substring
    pass is reserved for the camelCase-obfuscation defense: if a
    token's word-boundary form misses but the compact-form substring
    hits, that's evidence of intentional obfuscation worth flagging.
    """
    if not isinstance(s, str):
        s = str(s)
    return re.sub(r"[_\s]+", "", s.casefold())


_NORMALIZED_FORBIDDEN: tuple[str, ...] = tuple(
    _normalize_for_match(t) for t in FORBIDDEN_TOKENS
)


# Word-boundary matchers, one per forbidden token. Word boundaries
# treat ``_`` as a word character, so ``\btest_patch\b`` matches
# ``"test_patch"`` (surrounding chars are quotes / spaces / colons —
# all non-word) but NOT ``test_patch_vary_headers`` (next char is
# underscore — word char, no boundary).
_FORBIDDEN_RE: tuple[re.Pattern, ...] = tuple(
    re.compile(r"\b" + re.escape(t) + r"\b", re.IGNORECASE)
    for t in FORBIDDEN_TOKENS
)


def _detect_forbidden_tokens(text: str) -> list[str]:
    """Return the FORBIDDEN_TOKENS that match in ``text``.

    Word-boundary, case-insensitive match against each token. Word
    boundaries treat ``_`` as a word character, so:
      - ``"test_patch": "..."`` matches ``\\btest_patch\\b`` (quote
        is non-word, boundary on both sides).
      - ``test_patch_vary_headers`` does NOT match (next char after
        ``patch`` is ``_``, both are word chars, no boundary).
      - ``with_hints`` does NOT match ``\\bhints\\b`` (preceded by
        ``_``, both word chars, no boundary).
      - ``unresolved`` does NOT match ``\\bresolved\\b`` (preceded
        by ``n``, both word chars, no boundary).

    Calibration (§8.4) on 43,831 real test names across the 12
    Verified repos: ZERO false positives. See
    ``docs/audits/repro_token_rule_calibration.md``.

    The substring-on-normalized "obfuscation defense" (catching
    camelCase like ``failToPass``) was removed after calibration: it
    over-flagged ``test_patches_distortion`` and similar real names,
    and the realistic threat is direct copying of dataset field
    names (which word-boundary catches), not deliberate
    obfuscation. If a future audit shows a real obfuscation leak,
    re-add the fallback with tighter context constraints.
    """
    if not isinstance(text, str):
        text = str(text)
    hits: set[str] = set()
    for tok, pat in zip(FORBIDDEN_TOKENS, _FORBIDDEN_RE):
        if pat.search(text):
            hits.add(tok)
    return sorted(hits)


def _validate_input_firewall(
    view: InstanceView,
    ranked_files: list[RankedFile],
) -> None:
    """Per §8.4: assert no forbidden token reaches the generator's
    prompt. The fields that flow into the prompt are:
      - view.problem_statement
      - view.test_directives.dirs (paths only)
      - ranked_files[*].file_path

    The full repo_skeleton is NOT in the prompt (§8.1 — the prompt
    uses AST snippets of the K=10 ranked files only).

    Raises ReproGeneratorError on any hit. The check is the
    contamination-prevention bar; we want it to refuse, not warn,
    when an upstream caller smuggled a forbidden field into the
    InstanceView.
    """
    haystacks: list[tuple[str, str]] = [
        ("problem_statement", view.problem_statement),
    ]
    for i, d in enumerate(view.test_directives.dirs):
        haystacks.append((f"test_directives.dirs[{i}]", d))
    for i, rf in enumerate(ranked_files):
        haystacks.append((f"ranked_files[{i}].file_path", rf.file_path))

    for label, text in haystacks:
        hits = _detect_forbidden_tokens(text or "")
        if hits:
            raise ReproGeneratorError(
                f"[input-firewall] forbidden token(s) detected in "
                f"{label}: {hits}. Refusing to send to generator. "
                f"See docs/V10_DESIGN_PHASE2.md §8.4."
            )


# ---------------------------------------------------------------------------
# AST snippets — function signatures + class headers, no bodies (§8.1)
# ---------------------------------------------------------------------------


def _ast_snippet_for_source(source: str, *, max_chars: int = 3000) -> str:
    """Extract top-level class/function signatures from a Python file's
    source. Returns a string with one entry per top-level def or class.

    Method bodies are dropped; method SIGNATURES (with annotations and
    defaults, when present) are kept. This gives the generator enough
    structural visibility to write valid imports and target methods
    without inflating the prompt with function bodies.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return "# (could not parse this file)"

    out: list[str] = []

    def _emit_func(node, indent: str = "") -> None:
        prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
        try:
            sig = ast.unparse(node.args)
        except Exception:
            sig = "..."
        ret = ""
        if node.returns is not None:
            try:
                ret = f" -> {ast.unparse(node.returns)}"
            except Exception:
                ret = ""
        out.append(f"{indent}{prefix}{node.name}({sig}){ret}: ...")

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _emit_func(node, indent="")
        elif isinstance(node, ast.ClassDef):
            try:
                bases = ", ".join(ast.unparse(b) for b in node.bases)
            except Exception:
                bases = ""
            header = f"class {node.name}({bases}):" if bases else f"class {node.name}:"
            out.append(header)
            had_method = False
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    _emit_func(sub, indent="    ")
                    had_method = True
            if not had_method:
                out.append("    ...")
            out.append("")

    if not out:
        return "# (no top-level classes or functions)"

    text = "\n".join(out).rstrip()
    if len(text) > max_chars:
        text = text[:max_chars] + "\n# (snippet truncated at max_chars)"
    return text


def _build_ast_snippets(
    sandbox,
    file_paths: list[str],
    *,
    max_chars_per_file: int = 3000,
) -> list[tuple[str, str]]:
    """Read each file from the sandbox and extract its AST snippet.
    Returns [(file_path, snippet_text), ...] in input order."""
    out: list[tuple[str, str]] = []
    for fp in file_paths:
        try:
            res = sandbox.read_file(fp, max_chars=max_chars_per_file * 4)
        except Exception as exc:  # noqa: BLE001
            out.append((fp, f"# (read failed: {type(exc).__name__})"))
            continue
        if res.exit_code != 0 or not res.stdout:
            out.append((fp, "# (could not read this file)"))
            continue
        snippet = _ast_snippet_for_source(res.stdout, max_chars=max_chars_per_file)
        out.append((fp, snippet))
    return out


# ---------------------------------------------------------------------------
# Prompt build
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = """\
You are writing a pytest test that reproduces a bug described in a
software-engineering issue.

The test must:
  - FAIL at the current state of the repo (because the bug exists).
  - PASS after a correct fix is applied.
  - Be SELF-CONTAINED: import only what's needed; if it depends on
    fixtures defined elsewhere, stub them inline.
  - Target the smallest reproducible scenario, not a comprehensive
    test of the affected feature.

You will be given:
  - the issue text,
  - a list of likely-relevant files (from the localizer),
  - AST snippets of those files (signatures + class headers; no
    function bodies),
  - the repo's test directory layout.

Output ONLY a JSON object. No prose, no markdown fences.

  {
    "test_filename": "<test_dir>/test_repro_v10_<short-slug>.py",
    "test_code": "<full pytest source>",
    "target_test_id": "<test_filename>::test_<descriptive_name>",
    "rationale": "<1-2 sentence rationale>"
  }
"""


def _build_user_prompt(
    view: InstanceView,
    ranked_files: list[RankedFile],
    ast_snippets: list[tuple[str, str]],
    widen: bool,
) -> str:
    parts: list[str] = []
    parts.append("# Issue\n\n")
    parts.append(view.problem_statement.strip())
    parts.append("\n\n# Test directories\n\n")
    for d in view.test_directives.dirs:
        parts.append(f"  - {d}\n")
    parts.append("\n# Likely-relevant files\n")
    if widen:
        parts.append(
            "(Widened from top-10 to top-30 because the prior attempt's "
            "reject reason indicated the bug isn't reachable from the "
            "narrower set.)\n"
        )
    parts.append("\n")
    for fp, snippet in ast_snippets:
        parts.append(f"## {fp}\n\n```python\n{snippet}\n```\n\n")
    parts.append("# Output\n\nReturn the JSON object now.\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Output parsing + informational output guard (§8.4)
# ---------------------------------------------------------------------------


def _parse_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ReproGeneratorError(
            f"generator output is not valid JSON: {exc}"
        ) from exc
    required = ("test_filename", "test_code", "target_test_id", "rationale")
    missing = [k for k in required if k not in data]
    if missing:
        raise ReproGeneratorError(
            f"generator output missing keys: {missing}"
        )
    for k in required:
        if not isinstance(data[k], str) or not data[k].strip():
            raise ReproGeneratorError(
                f"generator output {k!r} must be a non-empty string"
            )
    return data


def check_output_substring_hits(test_code: str, target_test_id: str) -> list[str]:
    """Return forbidden tokens that match in test_code or target_test_id.
    Per §8.4 this is INFORMATIONAL — callers log a WARNING and do NOT
    block on a hit. Real protection is at the input layer.

    Uses the same word-boundary matcher (``_detect_forbidden_tokens``)
    as the input firewall. Exposed as a public helper so
    harness.sandbox.run_repro_test can call it without importing from
    a private name.
    """
    hits: set[str] = set()
    for hay in (test_code or "", target_test_id or ""):
        hits.update(_detect_forbidden_tokens(hay))
    return sorted(hits)


# ---------------------------------------------------------------------------
# Single-attempt generator
# ---------------------------------------------------------------------------


def generate_repro_attempt(
    *,
    view: InstanceView,
    ranked_files: list[RankedFile],
    sandbox,
    attempt_index: int = 0,
    widen: bool = False,
) -> ReproTestCase:
    """Generate ONE repro test attempt. No retry loop (that's commit 17b).

    Per V10_DESIGN_PHASE2.md §3.4 step 1.

    Pre-conditions:
      - InstanceView is firewall-clean by construction.
      - ranked_files is the K=10 (or K=30 when widen=True) candidate
        list from Phase 1's reranker output. The widen=True gate is
        enforced by the orchestrator (commit 17b), not here.

    Raises ReproGeneratorError if the input firewall fires or the LLM
    output can't be parsed.
    """
    from harness.llm.clients import complete_chat, model_for_role

    # §8.4 input-layer firewall — block forbidden tokens BEFORE the prompt.
    _validate_input_firewall(view, ranked_files)

    # AST snippets of the candidate files.
    file_paths = [rf.file_path for rf in ranked_files]
    ast_snippets = _build_ast_snippets(sandbox, file_paths)

    user_prompt = _build_user_prompt(view, ranked_files, ast_snippets, widen)
    model = model_for_role("repro_generator")
    chat = complete_chat(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        max_tokens=4096,
        temperature=0.0,
        response_format_json=True,
    )

    data = _parse_response(chat.text)

    # §8.4 output-layer informational guard (logs WARNING, does NOT block).
    hits = check_output_substring_hits(data["test_code"], data["target_test_id"])
    if hits:
        log.warning(
            "[repro-output-guard] instance=%s informational substring "
            "match on tokens=%s — generator output contains tokens that "
            "the input firewall should have prevented. Logged per §8.4; "
            "investigate the input firewall integrity, do not retry.",
            view.instance_id, hits,
        )

    return ReproTestCase(
        instance_id=view.instance_id,
        test_filename=data["test_filename"],
        test_code=data["test_code"],
        target_test_id=data["target_test_id"],
        rationale=data["rationale"],
        generator_model=chat.model,
        generator_input_tokens=chat.input_tokens,
        generator_output_tokens=chat.output_tokens,
        attempt_index=attempt_index,
    )


# ---------------------------------------------------------------------------
# Commit 17b: retry loop + instrumentation + cost cap
# ---------------------------------------------------------------------------


class ReproRejectReason(Enum):
    """Controlled vocabulary for why an attempt was rejected. Lets the
    coverage audit (commit 17d) drive the N=2-vs-N=3 decision via
    structured queries instead of free-form string greps.

    Values map to the verifier's terminal states (PASSES_AT_BASE /
    IMPORT_ERROR / SYNTAX_ERROR / TIMEOUT) plus generator-side failure
    modes (OUTPUT_TOKEN_OVERFLOW / FIREWALL_VIOLATION / OTHER) and
    the success state (ACCEPTED).
    """

    ACCEPTED = "accepted"
    PASSES_AT_BASE = "passes_at_base"
    IMPORT_ERROR = "import_error"
    SYNTAX_ERROR = "syntax_error"
    TIMEOUT = "timeout"
    OUTPUT_TOKEN_OVERFLOW = "output_token_overflow"
    FIREWALL_VIOLATION = "firewall_violation"
    OTHER = "other"


# Per §8.5, V0 has no way to inspect the post-patch state without
# breaching the gold-patch firewall. The trace row schema enforces
# this: post_patch_pass_status MUST be the literal string "unknown".
V0_POST_PATCH_PASS_STATUS = "unknown"


@dataclass(frozen=True)
class ReproAttemptTraceRow:
    """One JSONL row per attempt actually made. Per §3.4.

    Schema enforcement:
      - post_patch_pass_status MUST equal V0_POST_PATCH_PASS_STATUS
        ("unknown") in V0 — see §8.5. Constructing a row with any
        other value raises ValueError. This is structural prevention
        of a gold-patch firewall breach.
      - reject_reason MUST be a valid ReproRejectReason value. Free-
        form rejection details go in reject_reason_detail.
    """

    instance_id: str
    attempt_index: int
    time_to_result_s: float
    base_commit_fail_status: str          # 'fails-at-base' | 'passes-at-base' | 'errors'
    post_patch_pass_status: str           # MUST be 'unknown' in V0
    generator_stated_reason: str          # one-line "why this test"
    reject_reason: str                    # ReproRejectReason value
    reject_reason_detail: str             # free-form when reason==OTHER
    input_tokens: int
    output_tokens: int
    cost_usd: float
    widen_flag: bool

    def __post_init__(self) -> None:
        if not self.instance_id:
            raise ValueError("ReproAttemptTraceRow.instance_id required")
        if self.attempt_index < 0:
            raise ValueError(
                f"ReproAttemptTraceRow.attempt_index must be >= 0, "
                f"got {self.attempt_index}"
            )
        if self.post_patch_pass_status != V0_POST_PATCH_PASS_STATUS:
            raise ValueError(
                f"V0 must use post_patch_pass_status="
                f"{V0_POST_PATCH_PASS_STATUS!r} (got "
                f"{self.post_patch_pass_status!r}). Real post-patch "
                f"verification would breach the gold-patch firewall "
                f"per V10_DESIGN_PHASE2.md §8.5."
            )
        if self.base_commit_fail_status not in (
            "fails-at-base", "passes-at-base", "errors"
        ):
            raise ValueError(
                f"base_commit_fail_status invalid: "
                f"{self.base_commit_fail_status!r}"
            )
        # Validate reject_reason is a valid enum value.
        try:
            ReproRejectReason(self.reject_reason)
        except ValueError as exc:
            raise ValueError(
                f"reject_reason must be a valid ReproRejectReason "
                f"value, got {self.reject_reason!r}"
            ) from exc


def widen_for_next_attempt(prev_reject_reason: ReproRejectReason) -> bool:
    """Per §8.1 widen logic: deterministic on the prior reject reason.
    Widening (top-10 → top-30 candidate set) helps when the bug isn't
    reachable from the narrower set; it doesn't help when the issue
    is generator quality.
    """
    return prev_reject_reason in (
        ReproRejectReason.PASSES_AT_BASE,
        ReproRejectReason.IMPORT_ERROR,
    )


def _classify_verify_result(exit_code: int, output: str) -> tuple[ReproRejectReason, str]:
    """Map a pytest exit + output to a ReproRejectReason.

    pytest exit codes:
      0 = all tests passed (NOT what we want — bug not reproduced)
      1 = some tests failed (THIS is success — bug reproduced at base)
      2 = test execution errored (collection / import / syntax)
      3-5 = various pytest internals / no tests collected

    Returns (reason, detail_string).
    """
    out = output or ""
    out_lower = out.lower()
    if exit_code == 1:
        return ReproRejectReason.ACCEPTED, ""
    if exit_code == 0:
        return ReproRejectReason.PASSES_AT_BASE, "test passed at base — does not reproduce bug"
    if "syntaxerror" in out_lower:
        return ReproRejectReason.SYNTAX_ERROR, out[:500]
    if "importerror" in out_lower or "modulenotfounderror" in out_lower:
        return ReproRejectReason.IMPORT_ERROR, out[:500]
    if exit_code == 124 or "timed out" in out_lower or "timeout" in out_lower:
        return ReproRejectReason.TIMEOUT, out[:500]
    return ReproRejectReason.OTHER, out[:500]


def _verify_repro_at_base(sandbox, case: ReproTestCase, *, timeout_s: float = 60.0) -> tuple[ReproRejectReason, str, str, float]:
    """Run the generated test inside the sandbox at base_commit.
    Returns (reject_reason, detail, base_commit_fail_status,
    duration_s).

    base_commit_fail_status is the §3.4 short-form label
    ('fails-at-base' / 'passes-at-base' / 'errors').
    """
    test_dir = sandbox.view.test_directives.dirs[0].rstrip("/")
    test_path = f"{test_dir}/{case.test_filename.lstrip('/')}"

    t_start = time.perf_counter()

    write_res = sandbox.write_file(test_path, case.test_code)
    if write_res.exit_code != 0:
        elapsed = time.perf_counter() - t_start
        return (
            ReproRejectReason.OTHER,
            f"write_failed: {(write_res.stderr or '')[:200]}",
            "errors",
            elapsed,
        )

    run_res = sandbox.run_repro_test(
        test_id=case.target_test_id,
        test_code=case.test_code,
        timeout_s=timeout_s,
    )
    elapsed = time.perf_counter() - t_start

    output = (run_res.stdout or "") + "\n" + (run_res.stderr or "")
    reason, detail = _classify_verify_result(run_res.exit_code, output)

    if reason is ReproRejectReason.ACCEPTED:
        base_label = "fails-at-base"
    elif reason is ReproRejectReason.PASSES_AT_BASE:
        base_label = "passes-at-base"
    else:
        base_label = "errors"

    return reason, detail, base_label, elapsed


def _trace_path_for(run_dir: str | pathlib.Path, instance_id: str) -> pathlib.Path:
    return pathlib.Path(run_dir) / "repro_trace" / f"{instance_id}.jsonl"


def write_trace_row(
    run_dir: str | pathlib.Path,
    instance_id: str,
    row: ReproAttemptTraceRow,
) -> pathlib.Path:
    """Append one JSONL row to runs/<run-name>/repro_trace/<instance_id>.jsonl.

    Cross-contamination defense: row.instance_id MUST match the run's
    instance_id passed in. Mismatch raises ValueError.
    """
    if row.instance_id != instance_id:
        raise ValueError(
            f"trace-row instance_id={row.instance_id!r} doesn't match "
            f"run instance_id={instance_id!r}. Cross-instance "
            f"contamination defense per V10_DESIGN_PHASE2.md §3.4."
        )
    p = _trace_path_for(run_dir, instance_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(dataclasses.asdict(row)) + "\n")
    return p


# ---------------------------------------------------------------------------
# Retry-loop orchestrator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerateWithRetryResult:
    """Outcome of generate_with_retry."""

    status: str                          # ReproStatus value
    case: ReproTestCase | None           # set iff status==USABLE
    attempts_made: int                   # 1..N (rows in the JSONL trace)
    total_cost_usd: float
    final_reject_reason: ReproRejectReason | None  # last reason seen


def _price_for(model: str) -> dict:
    from harness.llm.clients import price_for_model
    return price_for_model(model)


def _cost_for_chat(model: str, input_tokens: int, output_tokens: int) -> float:
    p = _price_for(model)
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000.0


def generate_with_retry(
    *,
    view: InstanceView,
    ranked_files: list[RankedFile],
    sandbox,
    run_dir: str | pathlib.Path,
    n_attempts: int = 3,
    cost_cap_usd: float = 0.30,
    verify_timeout_s: float = 60.0,
) -> GenerateWithRetryResult:
    """Generate a repro test with up to ``n_attempts`` retries.

    Per V10_DESIGN_PHASE2.md §3.4 (verification + retry), §8.2 (N=3
    + instrumentation), §8.5 (broken-repro acceptance), §8.6 (cost
    cap = $0.30 / instance).

    Cost-cap semantics (per the 17b spec):
      - Checked BETWEEN attempts only, never mid-API-call.
      - If an in-flight attempt would push spend over cap, let it
        complete and check before the next attempt. The API call is
        already paid for; killing wastes it and corrupts the trace.
      - When the cap fires, returns status=ReproStatus.NO_REPRO_BUDGET
        with the actual attempt count and total spend.

    JSONL trace per attempt actually made lands at
    ``runs/<run-name>/repro_trace/<instance_id>.jsonl``. Even
    budget-stopped runs produce trace rows for attempts that DID
    complete.
    """
    from harness.cost import CostTracker

    tracker = CostTracker(instance_id=view.instance_id, cap_usd=cost_cap_usd)

    last_reject: ReproRejectReason | None = None
    last_case: ReproTestCase | None = None

    for attempt_idx in range(n_attempts):
        # Cost cap check — between attempts only.
        if tracker.hard_cap_reached:
            log.info(
                "[repro-retry] instance=%s budget cap reached after "
                "%d attempts (spent $%.4f); emitting NO_REPRO_BUDGET",
                view.instance_id, attempt_idx, tracker.total_usd,
            )
            return GenerateWithRetryResult(
                status=ReproStatus.NO_REPRO_BUDGET,
                case=None,
                attempts_made=attempt_idx,
                total_cost_usd=tracker.total_usd,
                final_reject_reason=last_reject,
            )

        widen = (attempt_idx > 0) and (
            last_reject is not None and widen_for_next_attempt(last_reject)
        )

        # Attempt: generate + verify. We let the attempt complete
        # even if it pushes over the cap (cost is already paid).
        attempt_t_start = time.perf_counter()
        try:
            case = generate_repro_attempt(
                view=view,
                ranked_files=ranked_files,
                sandbox=sandbox,
                attempt_index=attempt_idx,
                widen=widen,
            )
        except ReproGeneratorError as exc:
            elapsed = time.perf_counter() - attempt_t_start
            # Generation itself failed — usually firewall or unparseable
            # JSON. Record it and decide whether to retry.
            msg = str(exc)
            if "input-firewall" in msg:
                reason = ReproRejectReason.FIREWALL_VIOLATION
            elif "missing keys" in msg or "non-empty" in msg or "not valid JSON" in msg:
                reason = ReproRejectReason.OUTPUT_TOKEN_OVERFLOW
            else:
                reason = ReproRejectReason.OTHER
            row = ReproAttemptTraceRow(
                instance_id=view.instance_id,
                attempt_index=attempt_idx,
                time_to_result_s=elapsed,
                base_commit_fail_status="errors",
                post_patch_pass_status=V0_POST_PATCH_PASS_STATUS,
                generator_stated_reason="(generation failed before output)",
                reject_reason=reason.value,
                reject_reason_detail=msg[:500],
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                widen_flag=widen,
            )
            write_trace_row(run_dir, view.instance_id, row)
            last_reject = reason
            # Firewall violation is fatal for this instance — won't fix
            # itself by retrying. Bail.
            if reason is ReproRejectReason.FIREWALL_VIOLATION:
                return GenerateWithRetryResult(
                    status=ReproStatus.NO_REPRO,
                    case=None,
                    attempts_made=attempt_idx + 1,
                    total_cost_usd=tracker.total_usd,
                    final_reject_reason=last_reject,
                )
            continue

        # Track LLM cost for this attempt.
        cost_usd = _cost_for_chat(
            case.generator_model,
            case.generator_input_tokens,
            case.generator_output_tokens,
        )
        tracker.record(
            stage="repro_gen",
            model=case.generator_model,
            input_tokens=case.generator_input_tokens,
            output_tokens=case.generator_output_tokens,
        )

        # Verify must-fail-at-base.
        verify_reason, detail, base_label, verify_elapsed = _verify_repro_at_base(
            sandbox, case, timeout_s=verify_timeout_s,
        )
        elapsed = time.perf_counter() - attempt_t_start

        row = ReproAttemptTraceRow(
            instance_id=view.instance_id,
            attempt_index=attempt_idx,
            time_to_result_s=elapsed,
            base_commit_fail_status=base_label,
            post_patch_pass_status=V0_POST_PATCH_PASS_STATUS,
            generator_stated_reason=case.rationale,
            reject_reason=verify_reason.value,
            reject_reason_detail=detail,
            input_tokens=case.generator_input_tokens,
            output_tokens=case.generator_output_tokens,
            cost_usd=cost_usd,
            widen_flag=widen,
        )
        write_trace_row(run_dir, view.instance_id, row)

        if verify_reason is ReproRejectReason.ACCEPTED:
            return GenerateWithRetryResult(
                status=ReproStatus.USABLE,
                case=case,
                attempts_made=attempt_idx + 1,
                total_cost_usd=tracker.total_usd,
                final_reject_reason=verify_reason,
            )

        last_reject = verify_reason
        last_case = case

    # Exhausted N attempts without acceptance.
    return GenerateWithRetryResult(
        status=ReproStatus.NO_REPRO,
        case=None,
        attempts_made=n_attempts,
        total_cost_usd=tracker.total_usd,
        final_reject_reason=last_reject,
    )


__all__ = [
    "ReproTestCase",
    "ReproStatus",
    "ReproGeneratorError",
    "ReproRejectReason",
    "ReproAttemptTraceRow",
    "GenerateWithRetryResult",
    "V0_POST_PATCH_PASS_STATUS",
    "check_output_substring_hits",
    "generate_repro_attempt",
    "generate_with_retry",
    "widen_for_next_attempt",
    "write_trace_row",
]
