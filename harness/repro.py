"""Phase 2 reproduction-test generator.

Per docs/V10_DESIGN_PHASE2.md (commits 7a + 19c7f27 design revision):
  - Generator inputs: InstanceView + RankedFile[] (top-K=10) + AST
    snippets of those K files (function signatures + class headers,
    no bodies; per §8.1). NOT the full RepoSkeleton.
  - Input-layer firewall (§8.4): forbidden tokens are blocked from
    reaching the prompt. The output-layer scan is INFORMATIONAL only
    (logs WARNING, does NOT block).
  - Output: ReproTestCase frozen dataclass (§3.2).

Commit 17a covers the single-attempt generator + ReproTestCase schema
+ Sandbox.run_repro_test method. The retry loop, instrumentation, and
$0.30/instance cost cap land in commit 17b.

The generator uses harness.llm.clients.complete_chat with
role="repro_generator" — never a hardcoded model name. The model
agnosticism unit test (commit 16a) enforces this structurally.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass

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


__all__ = [
    "ReproTestCase",
    "ReproStatus",
    "ReproGeneratorError",
    "check_output_substring_hits",
    "generate_repro_attempt",
]
