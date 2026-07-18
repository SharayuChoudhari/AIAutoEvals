"""litellm + instructor wrapper used by the SLM rubric builder.

The contract is intentionally tiny: one function, :func:`complete`, that takes a
Pydantic response model, a chat message list, and a LiteLLM model id, and
returns a validated instance of the response model. Retries on validation
failure are handled by instructor; transport/network failures are surfaced to
the caller as :class:`RubricEngineError` (defined in :mod:`builder`) so the CLI
can hard-fail with a remediation hint.

Tests inject a fake ``complete`` via the ``complete_fn`` parameter of
:func:`build_rubrics_slm` rather than mocking litellm at the HTTP layer.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, cast

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

#: A callable matching :func:`complete`'s signature. Tests build fakes of this
#: type to avoid network/ollama dependencies.
CompleteFn = Callable[..., Any]

#: Instructor mode used for SLM calls. ``JSON_SCHEMA`` sends a strict JSON
#: schema via ``response_format`` instead of a function-call tool, so there is
#: no tool name for the model to echo — weak local models (e.g.
#: ``ollama/qwen2.5-coder:7b``) frequently hallucinate a semantically-named
#: tool under the default ``Mode.TOOLS`` and the provider rejects it with
#: "Tool name does not match". JSON_SCHEMA keeps the single SLM call and the
#: existing Pydantic validation contract intact.
_INSTRUCTOR_MODE = "json_schema_mode"


def complete(
    model: str,
    messages: list[dict[str, str]],
    response_model: type[T],
    *,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    retries: int = 2,
) -> T:
    """Call ``litellm.completion`` wrapped by ``instructor`` with validation.

    Raises ``RubricEngineError`` (imported lazily to avoid a circular import at
    module import time) when the provider is unreachable after retries.
    """
    try:
        import instructor
        import litellm
    except ImportError as exc:  # pragma: no cover - deps are now core
        raise _engine_error(
            "the SLM rubric engine requires `litellm` and `instructor`",
            "dependencies are not installed",
            "pip install ai-evals or run with --rubric-engine rules",
        ) from exc

    client = instructor.from_litellm(litellm.completion, mode=instructor.Mode(_INSTRUCTOR_MODE))
    try:
        return client.chat.completions.create(
            model=model,
            messages=cast(Any, messages),
            response_model=response_model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=retries,
        )
    except Exception as exc:
        raise _engine_error(
            f"SLM call to {model!r} failed after {retries} retries",
            str(exc),
            (
                "ensure the model is reachable (e.g. `ollama serve` and "
                "`ollama pull qwen2.5-coder:7b` for the default), set a valid "
                "API key for cloud models, or fall back with --rubric-engine rules"
            ),
        ) from exc


def _engine_error(message: str, why: str, fix: str) -> Exception:
    from ai_eval.inference.slm.builder import RubricEngineError

    return RubricEngineError(message, why=why, fix=fix)


__all__ = ["CompleteFn", "complete"]
