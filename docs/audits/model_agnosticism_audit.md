# Model-agnosticism audit (commit 16a)

Pre-Phase-2 verification: confirm V10's "model-agnostic at the call
site" design (V10_DESIGN.md §11.2) actually holds in code, and that
no Phase 1 module hardcodes a model name in a way that would block a
future swap. This audit covers the eight items from the batch spec.

## 1. Every LLM-related grep hit, categorized

Grep target: `complete_chat`, `anthropic.Client`, `openai.Client`,
`Anthropic(`, `OpenAI(`, `claude-`, `deepseek-`, `gpt-`, `sonnet`,
`opus`, `haiku`, `deepseek` — restricted to V10 in-scope dirs
(`harness/`, `scripts/`, `tests/`). The legacy `examples/` and
`meta_harness_plus/` trees are out of V10 scope and not audited here.

### `harness/` — V10 production code

| File:line | Match | Routing |
|---|---|---|
| `harness/llm/clients.py:11-12` | `"deepseek"` prefix in routing-rule docstring | **Routing layer** (not a call) |
| `harness/llm/clients.py:105-106` | `name.startswith("deepseek")` | **Routing layer** (`_route_model`) |
| `harness/llm/clients.py:138-148` | `provider == "deepseek"` / `"anthropic"` dispatch | **Routing layer** |
| `harness/llm/clients.py:151` | `_call_deepseek` function definition | **Routing layer** (provider impl) |
| `harness/llm/clients.py:163` | `OpenAI(api_key=..., base_url="https://api.deepseek.com/v1")` | **SOLE OpenAI SDK call site for chat** (gateway) |
| `harness/llm/clients.py:196` | `anthropic.Anthropic(api_key=...)` | **SOLE anthropic SDK call site** (gateway) |
| `harness/cost.py:21-29` | `_DEFAULT_PRICES` dict with model names | **Reference data** (not a call) |
| `harness/embedding.py:247` | `OpenAI(api_key=api_key)` inside `RemoteOpenAIEmbedder._ensure_client` | **Opt-in ablation path** (lazy-loaded; not the default `LocalEmbedder`) |
| `harness/rerank.py:341` | `from harness.llm.clients import complete_chat, model_for_role, price_for_model` | **Routes through gateway** ✓ |
| `harness/rerank.py:356` | `chosen_model = model_override or model_for_role(role)` | **Role lookup** ✓ |
| `harness/rerank.py:363` | `chat = complete_chat(messages=..., model=chosen_model, ...)` | **Routes through gateway** ✓ |
| `harness/config/models.yaml` | role-to-model map + price table | **Config** (the only place model names live as data) |

**Verdict:** the only places that import `anthropic` / `openai` are
`harness/llm/clients.py` (the gateway) and `harness/embedding.py`
(opt-in `RemoteOpenAIEmbedder` for ablations; not used by the default
local-embedding path). No business-logic file makes a direct API call.
Hard-stop #1 (any business-logic file directly imports anthropic or
openai SDK → routing bypass) is **NOT tripped**.

### `scripts/` — orchestration

| File:line | Match | Notes |
|---|---|---|
| `scripts/retrieval_eval_dev50.py:446-450` | `--reranker-model` CLI help text mentions example values | CLI threading; not a hardcoded model in code path |
| `scripts/preflight_test500.py:64-71, 227-297, 373-379` | `RERANKER_COSTS` / `RERANKER_RERANK_S` dicts keyed on model name | **Reference data** for the projection script (not a call site) |
| `scripts/test500_autoresume.sh:30` | log file name `test500_deepseek.log` | string in filename; not a model selector |

The scripts thread model names as CLI args / projection-script
constants, not as hardcoded targets of LLM calls. Hard-stop #2 (any
call site uses a string literal for the model name outside of
harness/config/ or harness/llm/) is **NOT tripped**.

### `tests/` — test code

| File:line | Match | Notes |
|---|---|---|
| `tests/conftest.py:122` | docstring example `"claude-..."` | comment text; harmless |
| `tests/test_no_oracle_leak.py:335,345` | `model="claude-sonnet-4-6"` in firewall-test fixtures | test data; the test mocks `complete_chat` |
| `tests/test_localization_signals.py:147,272` | `model="..."` in dataclass fixtures | test data; structural assertions only |
| `tests/test_swebench_adapter.py:98,106` | `model_name="claude-sonnet-4-6"` in adapter test | test data |
| `tests/test_sandbox_and_cache.py:99-111` | `model="claude-sonnet-4-6"` in CostTracker test | test data |
| `tests/test_rerank.py:257-316` | mocks of `complete_chat` returning `model="deepseek-chat"` | mock data; the test patches `complete_chat` itself |
| `tests/test_llm_client.py:30-86` | `api_key="sk-x", model="..."` in client routing tests | tests of the routing layer itself; expected to hardcode names |

All test-side model strings are fixture / mock data, not production
call paths.

## 2. Roles in use by `complete_chat` callers

Programmatically, the only file in `harness/` that calls
`complete_chat` is `harness/rerank.py`, which invokes it with a
threaded `model` argument that came from `model_for_role(role)`
(line 356) where `role` is the `RerankerInput` field defaulting to
`"reranker"`.

Roles currently exercised by V10 production code:

| Role | Caller | Default model |
|---|---|---|
| `reranker` | `harness/rerank.py` (Stage 1g) | `deepseek-chat` |

That's it. Phase 2/3/5 callers don't exist yet — but their roles are
pre-committed in `models.yaml` as of this commit (see §6/§7 below).

## 3. `models.yaml` contents and unmapped-role check

After this commit, `harness/config/models.yaml` defines:

| Role | Model | Phase |
|---|---|---|
| `reranker` | `deepseek-chat` | 1 |
| `repro_generator` | `deepseek-chat` | 2 (stub) |
| `repro_verifier` | `deepseek-chat` | 2 (stub) |
| `patch_generator_pipeline` | `deepseek-chat` | 3 (stub) |
| `patch_generator_agent` | `deepseek-chat` | 3 (stub; design intent: swap to opus when Phase 3 lands) |
| `patch_minimizer` | `deepseek-chat` | 3 (stub) |
| `selection_reviewer` | `deepseek-chat` | 5 (stub) |
| `selection_escalation_reviewer` | `deepseek-chat` | 5 (stub; was `claude-opus-4-7`, locked to DeepSeek in commit 16b per `docs/MODEL_SWAP_PLAN.md`) |

Every role used by `complete_chat` callers in V10 is mapped. Every
mapped model has a price entry under `prices:`. **No unmapped
roles.** The unit test (§8 below) enforces this structurally.

## 4. Env-var override path

From `harness/llm/clients.py:65-83`:

```python
def model_for_role(role: str) -> str:
    """Map a role label to a model name.

    Lookup order:
      1. Env var ``V10_<ROLE_UPPER>_MODEL`` (e.g. ``V10_RERANKER_MODEL``)
      2. ``roles.<role>`` in models.yaml
      3. KeyError if neither is set.
    """
    env_key = f"V10_{role.upper()}_MODEL"
    env_val = os.environ.get(env_key)
    if env_val:
        return env_val
    cfg = get_config()
    if role in cfg.get("roles", {}):
        return cfg["roles"][role]
    raise KeyError(...)
```

Lookup order is **env-var first**, YAML fallback, KeyError on miss.
Verified by `tests/test_model_agnostic.py::test_env_var_override_path`
(passes 6/6).

To swap reranker to Sonnet for one session without code changes:

```bash
V10_RERANKER_MODEL=claude-sonnet-4-5 ./scripts/retrieval_eval_dev50.py ...
```

Per-role env vars:
`V10_RERANKER_MODEL`, `V10_REPRO_GENERATOR_MODEL`,
`V10_REPRO_VERIFIER_MODEL`, `V10_PATCH_GENERATOR_PIPELINE_MODEL`,
`V10_PATCH_GENERATOR_AGENT_MODEL`, `V10_PATCH_MINIMIZER_MODEL`,
`V10_SELECTION_REVIEWER_MODEL`,
`V10_SELECTION_ESCALATION_REVIEWER_MODEL`.

## 5. Phase 2 stub roles — added

```yaml
repro_generator: deepseek-chat
repro_verifier:  deepseek-chat
```

Pre-committed so when Phase 2 code lands (commits 7b/7c/7d/7e per
the revised design at `docs/V10_DESIGN_PHASE2.md`), the implementation
must use the role label, not invent a new lookup.

## 6. Phase 3 stub roles — added

```yaml
patch_generator_pipeline: deepseek-chat
patch_generator_agent:    deepseek-chat
patch_minimizer:          deepseek-chat
```

All three default to `deepseek-chat` per the spec ("all defaulting to
deepseek-chat"). The design intent for `patch_generator_agent` is
`claude-opus-4-7` (V10_DESIGN.md §3.4), but starting at deepseek
keeps the routing layer honest until Phase 3 ships and we measure the
actual lift.

## 7. Phase 5 stub roles — added

```yaml
selection_reviewer:            deepseek-chat
selection_escalation_reviewer: claude-opus-4-7
```

`selection_reviewer` defaults to deepseek-chat; the escalation reviewer
goes to Opus per the design intent ("Opus-tier when the primary
reviewer is uncertain", V10_DESIGN.md §3.6).

## 8. Unit test — `tests/test_model_agnostic.py`

Six tests, all green:

| Test | What it checks |
|---|---|
| `test_models_yaml_is_parseable` | Config is parseable YAML with `roles:` and `prices:` keys. |
| `test_required_roles_are_mapped` | Every role in `REQUIRED_ROLES` (8 entries: 1 active + 7 stubs) is present in the YAML; every mapped model has a price entry. |
| `test_no_business_logic_imports_sdk` | AST scan: no `harness/` file imports `anthropic` or `openai` outside the allowlist (`llm/clients.py` + `embedding.py`). |
| `test_no_hardcoded_model_names_in_harness` | AST scan: no string literal under `harness/` starts with `deepseek-`, `claude-`, `gpt-`, `o1-`, `o3-` outside the routing/config/cost-table allowlist. |
| `test_env_var_override_path` | `V10_RERANKER_MODEL=claude-sonnet-4-5` makes `model_for_role("reranker")` return Sonnet; unset → falls back to YAML default. |
| `test_complete_chat_callers_use_role_label` | AST scan: no `complete_chat(model="literal-string", ...)` call sites in `harness/`. Threading a variable (e.g., `model=chosen_model`) is allowed. |

**Status: 6/6 pass.** Hard-stop #3 (unit test fails after stubs land)
**NOT tripped**.

The test catches future violations structurally — adding a hardcoded
model name to a new module, importing the SDK directly, or forgetting
to map a role in YAML all fail the test, not a code review.

## Hard-stop verdict

| Hard-stop | Status |
|---|---|
| #1: business-logic imports anthropic/openai SDK | ✓ NOT tripped (only the gateway + opt-in remote embedder) |
| #2: call site uses string literal model name outside config/llm | ✓ NOT tripped (test data, CLI args, and projection-script constants only) |
| #3: unit test fails after stub entries land | ✓ NOT tripped (6/6 pass) |

All three clear. The routing layer is solid; Phase 2/3/5 code can
land without re-litigating model-agnosticism.

## Cumulative spend

LLM cost on this audit batch: **$0.00**. Pure infrastructure.
