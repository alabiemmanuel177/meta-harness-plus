"""Built-in tasks. ``toy_classification`` is a deterministic offline task used
for unit tests — no external LLM required."""

from .toy_classification import build_toy_task, mock_llm

__all__ = ["build_toy_task", "mock_llm"]
