"""Real-LLM integration layer.

Introduces:
- ``LLMClient`` protocol — pluggable provider.
- ``ScriptedClient`` — deterministic fake for tests.
- ``HTTPClient`` — talks to Anthropic or OpenAI chat endpoints via stdlib urllib
  (no SDK dependency).
- ``LLMPredictor`` — Predictor component backed by an LLMClient.
- ``LLMProposer`` — reads the filesystem run log and produces structured harness
  proposals via an LLMClient.

Design principle: the LLM produces *structured JSON* specs of harnesses, which
are then instantiated via a component registry. This trades off expressiveness
vs safety/analyzability. See README for the case for the code-generation
alternative.
"""

from .client import LLMClient, LLMResponse, ScriptedClient, HTTPClient
from .registry import ComponentRegistry, default_registry, llm_search_registry
from .predictor import LLMPredictor
from .proposer import LLMProposer, DiagnosticContext

__all__ = [
    "LLMClient",
    "LLMResponse",
    "ScriptedClient",
    "HTTPClient",
    "ComponentRegistry",
    "default_registry",
    "llm_search_registry",
    "LLMPredictor",
    "LLMProposer",
    "DiagnosticContext",
]
