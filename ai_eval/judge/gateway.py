"""Async judge gateway.

:func:`score` is the single public entry point. It:
1. Builds the metric's prompt (delegated to the metric registry's
   ``prompt_builder``), selecting the tier via :func:`~ai_eval.judge.tiering.select_tier`.
2. Consults :class:`~ai_eval.judge.cache.JudgeCache` (content-addressed).
3. Calls the async transport — by default the cached instructor-over-litellm
   client, but tests pass a fake async ``complete_fn``.
4. Retries-with-fallback across ``judge.default`` → ``judge.fallback[*]`` and
   collects failures into :class:`~ai_eval.judge.schemas.JudgeError` records.

The transport contract for ``complete_fn`` (async):

    async def complete_fn(
        *, model: str, messages: list[dict], response_model: type[T],
        temperature: float
    ) -> T

mirrors ``instructor``'s ``client.chat.completions.create`` kwargs, so the
default client and the test fake share a signature.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from pydantic import BaseModel

from ai_eval.judge.cache import JudgeCache, cache_key
from ai_eval.judge.instructor_glue import JudgeUnavailable, get_async_client
from ai_eval.judge.schemas import JudgeError, JudgeRequest, JudgeResponse
from ai_eval.judge.tiering import select_tier

T = TypeVar("T", bound=BaseModel)

#: Async callable signature shared by the default client and test fakes.
ACompleteFn = Callable[..., Awaitable[Any]]

_SCHEMA_ID = "judge_response_v1"


async def score(
    request: JudgeRequest,
    *,
    model: str,
    prompt_messages: list[dict[str, str]],
    scored_dimension: str,
    complete_fn: ACompleteFn | None = None,
    fallback_models: tuple[str, ...] = (),
    cache: JudgeCache | None = None,
    temperature: float = 0.0,
    tier_hint: str | None = None,
    complex_models: tuple[str, ...] | None = None,
) -> tuple[JudgeResponse | None, list[JudgeError]]:
    """Score one request, retrying across the fallback chain.

    Returns ``(response_or_None, errors)``. ``response`` is ``None`` only when
    every model in the chain failed; the caller (runner) records the errors and
    marks the example as ``error``.
    """
    prompt_text = json.dumps(prompt_messages, sort_keys=True, ensure_ascii=False)
    key = cache_key(model, prompt_text, schema_id=_SCHEMA_ID)
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            try:
                return JudgeResponse.model_validate(cached), []
            except Exception:
                pass  # corrupt cache → fall through to live call

    chain = [model, *fallback_models]
    errors: list[JudgeError] = []
    tier = tier_hint or select_tier(model, complex_models=complex_models)
    # Tier is advisory; surfaced via raw for debugging. The prompt was already
    # built by the metric registry using the tier — we just record it.
    do_complete = complete_fn if complete_fn is not None else _default_complete

    for mdl in chain:
        try:
            result = await do_complete(
                model=mdl,
                messages=prompt_messages,
                response_model=JudgeResponse,
                temperature=temperature,
            )
            response = (
                result
                if isinstance(result, JudgeResponse)
                else JudgeResponse.model_validate(result)
            )
            # Normalize: if a sub_score matches scored_dimension, lift it.
            response = _lift_dimension(response, scored_dimension)
            if cache is not None:
                cache.put(key, response.model_dump(mode="json"))
            response.raw = {"tier": tier, "model": mdl}
            return response, errors
        except Exception as exc:
            retryable = not isinstance(exc, (KeyboardInterrupt, SystemExit))
            errors.append(
                JudgeError(
                    model=mdl,
                    message=f"{type(exc).__name__}: {exc}",
                    retryable=retryable,
                )
            )
            continue
    return None, errors


def _lift_dimension(response: JudgeResponse, scored_dimension: str) -> JudgeResponse:
    """If a sub_score named ``scored_dimension`` exists, prefer it as ``score``.

    Metric authors define which dimension is canonical; the judge may also emit
    an aggregate ``score``. The dimension (when present) wins so metric
    semantics stay consistent across runs.
    """
    if scored_dimension and scored_dimension in response.sub_scores:
        dim = response.sub_scores[scored_dimension]
        try:
            val = float(dim)
        except (TypeError, ValueError):
            return response
        if 0.0 <= val <= 1.0:
            response.score = val
    return response


async def _default_complete(
    *,
    model: str,
    messages: list[dict[str, str]],
    response_model: type[T],
    temperature: float = 0.0,
    max_retries: int = 0,
) -> T:
    """The real transport: cached async instructor client over litellm.

    ``max_retries`` defaults to 0 so pings and short-timeout probes fail fast
    rather than litellm's default 2 retries x backoff. The runner may pass a
    higher value for real scoring.
    """
    client = get_async_client()
    # litellm prints "Provider List" banners to stdout on provider lookup,
    # corrupting the JSON data channel. Suppress stdout (not stderr) during
    # the call; instructor returns the response object, not via stdout.
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        return await client.chat.completions.create(
            model=model,
            messages=messages,
            response_model=response_model,
            temperature=temperature,
            max_retries=max_retries,
        )


async def ping(
    model: str,
    *,
    complete_fn: ACompleteFn | None = None,
    timeout: float = 10.0,
) -> tuple[bool, str]:
    """One-shot reachability probe. Returns ``(ok, detail)``.

    Sends a trivial 1-token completion. Never raises — failures map to
    ``(False, message)`` so the CLI / doctor can render a check row.
    """

    class _Ok(BaseModel):
        ok: bool = True

    do_complete = complete_fn if complete_fn is not None else _default_complete
    try:
        await asyncio.wait_for(
            do_complete(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                response_model=_Ok,
                temperature=0.0,
            ),
            timeout=timeout,
        )
        return True, "reachable"
    except JudgeUnavailable as exc:
        return False, exc.message
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


__all__ = ["ACompleteFn", "JudgeUnavailable", "ping", "score"]
