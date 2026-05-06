"""Phase 3 — patch generation.

Public API:

    from harness.patch_gen import generate_pipeline, PatchCandidate

The package layout is:

  pipeline.py — single-shot K-candidate generator (commit P3b)
  agent.py    — multi-turn ACI loop (commit P3c)
  context.py  — context assembly + superset assertion (commit P3b)
  routing.py  — difficulty estimation + route selection (commit P3d)
  views.py    — PatchCandidate frozen dataclass + helpers (commit P3b)

Per docs/V10_DESIGN_PHASE3.md §2.1, NO module under harness.patch_gen
imports harness.repro or harness.eval. The cross-phase firewall test
in tests/test_repro_firewall.py + tests/test_phase3_firewall.py
enforces this at AST level.
"""

from harness.patch_gen.views import (
    ContextOversizeError,
    FileSnippet,
    PatchCandidate,
    PatchGenContext,
    PatchGenError,
)
from harness.patch_gen.context import (
    DEFAULT_PER_FILE_CHAR_CAP,
    DEFAULT_PROJECTED_TOKEN_LIMIT,
    DEFAULT_TOP_K,
    build_patch_gen_context_with_superset_check,
)
from harness.patch_gen.pipeline import (
    DEFAULT_PIPELINE_TEMPERATURES,
    PipelineGenerationResult,
    generate_pipeline,
    generate_pipeline_one_shot,
)
from harness.patch_gen.agent import (
    AgentGenerationResult,
    DEFAULT_AGENT_COST_CAP_USD,
    DEFAULT_AGENT_T_MAX,
    generate_agent,
)
from harness.patch_gen.router import (
    PatchGenStrategy,
    RouterFeatures,
    extract_features,
    route,
)


__all__ = [
    "AgentGenerationResult",
    "ContextOversizeError",
    "DEFAULT_AGENT_COST_CAP_USD",
    "DEFAULT_AGENT_T_MAX",
    "DEFAULT_PER_FILE_CHAR_CAP",
    "DEFAULT_PIPELINE_TEMPERATURES",
    "DEFAULT_PROJECTED_TOKEN_LIMIT",
    "DEFAULT_TOP_K",
    "FileSnippet",
    "PatchCandidate",
    "PatchGenContext",
    "PatchGenError",
    "PatchGenStrategy",
    "PipelineGenerationResult",
    "RouterFeatures",
    "build_patch_gen_context_with_superset_check",
    "extract_features",
    "generate_agent",
    "generate_pipeline",
    "generate_pipeline_one_shot",
    "route",
]
