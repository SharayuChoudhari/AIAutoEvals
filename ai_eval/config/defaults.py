"""Built-in defaults — lowest precedence layer of the config resolution order."""

from __future__ import annotations

from typing import Any

SCHEMA_VERSION: int = 1

# Default judge models. Phase 1 decision: Ollama local first, cloud regression judge.
DEFAULT_JUDGE: str = "ollama/qwen2.5-coder:7b"
DEFAULT_REGRESSION_JUDGE: str = "openai/gpt-4o-mini"

# Directories the tool creates / reads, relative to the project cwd.
DEFAULT_EVAL_DIR: str = "eval"
DEFAULT_STATE_DIR: str = ".ai-eval"

# Runner defaults.
DEFAULT_PARALLEL: int = 4
DEFAULT_TOLERANCE: float = 0.02
DEFAULT_CACHE: bool = True

# Frameworks the Phase 1 detector set knows about.
KNOWN_FRAMEWORKS: tuple[str, ...] = (
    "openai",
    "langchain",
    "llamaindex",
    "chromadb",
    "pinecone",
    "pydantic_ai",
)

# Frontier models that should use the complex CoT judge prompt tier.
# Plain substrings; matched case-insensitively in `judge.tiering`.
COMPLEX_MODEL_HINTS: tuple[str, ...] = (
    "gpt-4",
    "claude-3",
    "claude-opus",
    "claude-sonnet",
    "gemini-1.5-pro",
    "o1",
)


def builtin_defaults() -> dict[str, Any]:
    """Return the built-in defaults as a dict ready to be merged."""
    return {
        "schema_version": SCHEMA_VERSION,
        "project_type": "custom",
        "judge": {
            "default": DEFAULT_JUDGE,
            "regression_check": DEFAULT_REGRESSION_JUDGE,
            "fallback": [],
        },
        "defaults": {
            "parallel": DEFAULT_PARALLEL,
            "cache": DEFAULT_CACHE,
            "tolerance": DEFAULT_TOLERANCE,
        },
        "tasks": {},
    }
