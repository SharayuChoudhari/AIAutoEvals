"""Shared type-name constants for inference + scaffold layers.

These constants encode the set of constructor/type names the runner, harness,
and synthesis-time demotion all agree are runnable without a live backend
(LLM clients + compiled framework graphs) or are str-bindable by the auto-seed
scalar path. Centralized here so the inference layer (``signatures.py``) and
the scaffold layer (``harness_writer.py``) reference the same allow-lists
without a cross-layer import.

Keep this module free of any non-stdlib imports — it is imported very early by
``inference/signatures.py`` and must not pull in scaffold / config layers.
"""

from __future__ import annotations

#: Known LLM-client type names. Assignments to these in ``__init__`` are NOT
#: stubbed — the client runs real (needs an API key, not canned data). The
#: synthesis-time demotion also treats these as str-bindable-safe: a class
#: whose ``__init__`` takes only an LLM-client-typed param is still runnable
#: because the client is constructed fresh on each call.
LLM_CLIENT_TYPES: frozenset[str] = frozenset(
    {
        "OpenAI",
        "AzureOpenAI",
        "AsyncOpenAI",
        "ChatOpenAI",
        "Anthropic",
        "ChatAnthropic",
        "Ollama",
        "HuggingFaceHub",
        "HuggingFacePipeline",
        "ChatGoogleGenerativeAI",
    }
)

#: Known compiled framework-graph object names. Assignments to these run real
#: (they orchestrate the LLM; stubbing them would defeat the eval). The
#: synthesis-time demotion treats these as runnable: a class whose ``__init__``
#: takes only a graph-typed param is still runnable.
GRAPH_TYPES: frozenset[str] = frozenset(
    {
        "StateGraph",
        "RunnableSequence",
        "CompiledGraph",
        "CompiledStateGraph",
    }
)

#: Annotation names that ``_build_call_args`` (engine.py) can satisfy from the
#: auto-seed scalar input, plus types the harness/runner already know how to
#: construct (LLM clients, compiled graphs). A class ``__init__`` whose
#: required params (after ``self``) are all in this set is NOT demoted — it's
#: runnable as a top-level entry. Required params outside this set (``session``,
#: ``config``, ``db``, …) demote the class method to ``top_level=False``.
RUNNABLE_INIT_TYPES: frozenset[str] = (
    LLM_CLIENT_TYPES
    | GRAPH_TYPES
    | frozenset(
        {
            "str",
            "Optional[str]",
            "str | None",
            "Any",
        }
    )
)

__all__ = [
    "GRAPH_TYPES",
    "LLM_CLIENT_TYPES",
    "RUNNABLE_INIT_TYPES",
]
