"""Defensive tests for the model-agnostic routing layer.

Per V10_DESIGN.md §11.2: every LLM call goes through
``harness.llm.clients.complete_chat`` with a ROLE label. The role's
concrete model is looked up from ``harness/config/models.yaml`` (with
``V10_<ROLE>_MODEL`` env-var overrides). NO business-logic file
should:

  - Import the anthropic SDK or openai SDK directly (only
    ``harness/llm/clients.py`` is allowed; the opt-in
    ``RemoteOpenAIEmbedder`` in ``harness/embedding.py`` is also
    allowed because it's a non-default ablation path).
  - Pass a model-name string literal to ``complete_chat(model=...)``.
    Pass ``role=`` instead.
  - Hardcode model names like ``"deepseek-chat"`` or
    ``"claude-sonnet-4-5"`` inside business logic. CLI plumbing
    that THREADS a model name through (e.g.,
    ``--reranker-model claude-sonnet-4-5``) is allowed because the
    string flows through user input, not through code.

These rules let us swap any model with ONE config-line change.
Commit 16a (this test) is the structural enforcement.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest
import yaml


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
HARNESS = PROJECT_ROOT / "harness"
MODELS_YAML = HARNESS / "config" / "models.yaml"

# Files inside harness/ that are EXEMPT from the "no SDK imports" rule.
# These are documented routing layers, not business logic.
SDK_IMPORT_ALLOWLIST = {
    HARNESS / "llm" / "clients.py",      # the routing gateway itself
    HARNESS / "embedding.py",            # opt-in RemoteOpenAIEmbedder
}

# Files that legitimately reference model name strings (in routing /
# config / cost-tracker reference data, not as a target of an LLM call).
MODEL_STRING_ALLOWLIST = {
    HARNESS / "llm" / "clients.py",      # _route_model uses prefixes
    HARNESS / "cost.py",                 # _DEFAULT_PRICES table
    HARNESS / "config" / "models.yaml",  # the config itself
}

FORBIDDEN_MODEL_PREFIXES = (
    "deepseek-",     # deepseek-chat, deepseek-reasoner, etc.
    "claude-",       # claude-sonnet-4-5, claude-opus-4-7, etc.
    "gpt-",          # gpt-4o, gpt-4.1-nano, etc.
    "o1-",
    "o3-",
)
SDK_NAMES = ("anthropic", "openai")


def _walk_python_files(root: pathlib.Path):
    for p in root.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        yield p


# ---------------------------------------------------------------------------
# 1. The config file is parseable YAML.
# ---------------------------------------------------------------------------


def test_models_yaml_is_parseable():
    text = MODELS_YAML.read_text()
    cfg = yaml.safe_load(text)
    assert isinstance(cfg, dict)
    assert "roles" in cfg
    assert "prices" in cfg
    assert isinstance(cfg["roles"], dict)
    assert isinstance(cfg["prices"], dict)


# ---------------------------------------------------------------------------
# 2. Every required role label is present in the YAML.
# ---------------------------------------------------------------------------


REQUIRED_ROLES = {
    # Phase 1
    "reranker",
    # Phase 2 (pre-commitment for commits 7b/7c)
    "repro_generator",
    "repro_verifier",
    # Phase 3 (pre-commitment)
    "patch_generator_pipeline",
    "patch_generator_agent",
    "patch_minimizer",
    # Phase 5 (pre-commitment)
    "selection_reviewer",
    "selection_escalation_reviewer",
}


def test_required_roles_are_mapped():
    cfg = yaml.safe_load(MODELS_YAML.read_text())
    roles = cfg["roles"]
    missing = REQUIRED_ROLES - set(roles)
    assert not missing, f"required roles missing from models.yaml: {sorted(missing)}"
    # Every mapped role's model must have a price entry too (otherwise
    # cost reporting silently falls back to a generic estimate).
    prices = cfg["prices"]
    for role, model in roles.items():
        assert model in prices, (
            f"role {role!r} maps to model {model!r} which has NO price "
            f"entry under prices: in models.yaml"
        )


# ---------------------------------------------------------------------------
# 3. No business-logic file imports anthropic or openai SDKs.
# ---------------------------------------------------------------------------


def _imports_sdk(py_path: pathlib.Path) -> bool:
    try:
        tree = ast.parse(py_path.read_text())
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in SDK_NAMES:
                    return True
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None and node.module.split(".")[0] in SDK_NAMES:
                return True
    return False


def test_no_business_logic_imports_sdk():
    bad: list[str] = []
    for py in _walk_python_files(HARNESS):
        if py.resolve() in {p.resolve() for p in SDK_IMPORT_ALLOWLIST}:
            continue
        if _imports_sdk(py):
            bad.append(str(py.relative_to(PROJECT_ROOT)))
    assert not bad, (
        "These business-logic files import anthropic / openai SDKs "
        "directly — they should route through harness.llm.clients "
        "instead:\n  " + "\n  ".join(bad)
    )


# ---------------------------------------------------------------------------
# 4. No file inside harness/ hardcodes a model-name string literal
#    outside the routing/config/reference-data allowlist.
# ---------------------------------------------------------------------------


def _hardcoded_model_strings(py_path: pathlib.Path) -> list[tuple[int, str]]:
    """Return [(lineno, value)] for every string literal in py_path
    that starts with a forbidden model-name prefix."""
    try:
        tree = ast.parse(py_path.read_text())
    except SyntaxError:
        return []
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            v = node.value.strip().lower()
            if v.startswith(FORBIDDEN_MODEL_PREFIXES):
                hits.append((node.lineno, node.value))
    return hits


def test_no_hardcoded_model_names_in_harness():
    bad: list[str] = []
    for py in _walk_python_files(HARNESS):
        if py.resolve() in {p.resolve() for p in MODEL_STRING_ALLOWLIST}:
            continue
        hits = _hardcoded_model_strings(py)
        if hits:
            for ln, val in hits:
                bad.append(f"{py.relative_to(PROJECT_ROOT)}:{ln}  {val!r}")
    assert not bad, (
        "These files contain hardcoded model-name string literals. "
        "Replace with a role lookup via harness.llm.clients."
        "model_for_role() or pass through complete_chat(role=...):\n  "
        + "\n  ".join(bad)
    )


# ---------------------------------------------------------------------------
# 5. The env-var override path works (V10_<ROLE>_MODEL takes precedence
#    over the YAML default).
# ---------------------------------------------------------------------------


def test_env_var_override_path(monkeypatch):
    from harness.llm.clients import model_for_role

    # YAML default is deepseek-chat; override should win.
    monkeypatch.setenv("V10_RERANKER_MODEL", "claude-sonnet-4-5")
    assert model_for_role("reranker") == "claude-sonnet-4-5"

    # Without the env var, falls back to YAML.
    monkeypatch.delenv("V10_RERANKER_MODEL", raising=False)
    # Bust the module-level cache so the next call re-reads.
    import harness.llm.clients as clients
    clients._CACHED_CONFIG = None
    assert model_for_role("reranker") == "deepseek-chat"


def test_v10_use_opus_patch_gen_meta_flag(monkeypatch):
    """P3f Opus K=1 ablation hook. When V10_USE_OPUS_PATCH_GEN=1 is set,
    the two patch-gen roles resolve to claude-opus-4-7 — but ONLY those
    two; every other role stays on deepseek-chat."""
    import harness.llm.clients as clients
    from harness.llm.clients import model_for_role

    # Clear caches + per-role overrides.
    clients._CACHED_CONFIG = None
    for role in (
        "reranker", "repro_generator", "repro_verifier",
        "patch_generator_pipeline", "patch_generator_agent",
        "patch_minimizer", "selection_reviewer",
        "selection_escalation_reviewer",
    ):
        monkeypatch.delenv(f"V10_{role.upper()}_MODEL", raising=False)
    monkeypatch.delenv("V10_USE_OPUS_PATCH_GEN", raising=False)

    # OFF: defaults to deepseek-chat.
    assert model_for_role("patch_generator_pipeline") == "deepseek-chat"
    assert model_for_role("patch_generator_agent") == "deepseek-chat"
    assert model_for_role("reranker") == "deepseek-chat"

    # ON: patch-gen flips, others stay.
    monkeypatch.setenv("V10_USE_OPUS_PATCH_GEN", "1")
    assert model_for_role("patch_generator_pipeline") == "claude-opus-4-7"
    assert model_for_role("patch_generator_agent") == "claude-opus-4-7"
    assert model_for_role("reranker") == "deepseek-chat"
    assert model_for_role("repro_generator") == "deepseek-chat"
    assert model_for_role("selection_reviewer") == "deepseek-chat"

    # Per-role override beats the meta-flag (per-role is explicit).
    monkeypatch.setenv("V10_PATCH_GENERATOR_PIPELINE_MODEL", "deepseek-chat")
    assert model_for_role("patch_generator_pipeline") == "deepseek-chat"
    assert model_for_role("patch_generator_agent") == "claude-opus-4-7"


# ---------------------------------------------------------------------------
# 6. complete_chat call sites use role= (not a hardcoded model=) in
#    business logic. Detected via AST.
# ---------------------------------------------------------------------------


def _complete_chat_call_sites(py_path: pathlib.Path) -> list[tuple[int, set[str]]]:
    """Return [(lineno, set_of_kw_names)] for every complete_chat(...)
    call. The kw set lets us check whether 'role' or 'model' was passed."""
    try:
        tree = ast.parse(py_path.read_text())
    except SyntaxError:
        return []
    out: list[tuple[int, set[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            callee = node.func
            name = None
            if isinstance(callee, ast.Name):
                name = callee.id
            elif isinstance(callee, ast.Attribute):
                name = callee.attr
            if name == "complete_chat":
                kws = {kw.arg for kw in node.keywords if kw.arg is not None}
                out.append((node.lineno, kws))
    return out


def test_complete_chat_callers_use_role_label():
    """Every complete_chat call site that passes a STRING-LITERAL
    model= is a violation. Threading a model variable
    (e.g. model=chosen_model) is allowed because chosen_model itself
    came from model_for_role(role) or a CLI override."""
    violations: list[str] = []
    for py in _walk_python_files(HARNESS):
        if py.resolve() == (HARNESS / "llm" / "clients.py").resolve():
            continue  # the function definition itself
        try:
            tree = ast.parse(py.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            callee = node.func
            name = (
                callee.id if isinstance(callee, ast.Name)
                else callee.attr if isinstance(callee, ast.Attribute)
                else None
            )
            if name != "complete_chat":
                continue
            for kw in node.keywords:
                if kw.arg == "model":
                    # If the value is a string literal, that's a violation.
                    if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                        violations.append(
                            f"{py.relative_to(PROJECT_ROOT)}:{node.lineno}  "
                            f"model={kw.value.value!r} (use role= instead)"
                        )
    assert not violations, (
        "complete_chat callers passing a string-literal model= argument:\n  "
        + "\n  ".join(violations)
    )
