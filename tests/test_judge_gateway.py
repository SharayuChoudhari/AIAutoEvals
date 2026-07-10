"""Tests for the judge gateway: tier selection, cache, fallback, errors.

The transport is fully faked — no network/ollama. A fake async ``complete_fn``
mirrors the real instructor client signature so the gateway can't tell them
apart.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from ai_eval.judge.cache import JudgeCache, cache_key
from ai_eval.judge.gateway import ping, score
from ai_eval.judge.schemas import JudgeError, JudgeRequest, JudgeResponse


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


def _fake_complete(responses_by_model):
    """Return an async complete_fn that returns canned JudgeResponse dicts.

    ``responses_by_model`` maps model id -> dict (or Exception instance to raise).
    A model absent from the map raises ``RuntimeError`` (simulating an
    unreachable provider) so the fallback chain is exercised.
    """
    async def complete_fn(*, model, messages, response_model, temperature=0.0):
        if model not in responses_by_model:
            raise RuntimeError(f"model {model!r} not configured")
        val = responses_by_model[model]
        if isinstance(val, Exception):
            raise val
        if isinstance(val, JudgeResponse):
            return val
        return response_model.model_validate(val)
    return complete_fn


def _request():
    return JudgeRequest(
        task_name="t", task_type="rag", metric="faithfulness",
        example={"input": {"query": "q"}}, output="answer",
    )


# ---------------------------------------------------------------------------
# score: happy path + dimension lift
# ---------------------------------------------------------------------------

def test_score_returns_response(tmp_path: Path) -> None:
    complete = _fake_complete({
        "m1": {"score": 0.9, "rationale": "ok",
               "sub_scores": {"faithfulness": 0.9}},
    })
    resp, errors = _run(score(
        _request(), model="m1", prompt_messages=[{"role": "user", "content": "x"}],
        scored_dimension="faithfulness", complete_fn=complete,
    ))
    assert resp is not None
    assert resp.score == 0.9
    assert errors == []


def test_score_lifts_scored_dimension(tmp_path: Path) -> None:
    # aggregate score differs from the canonical dimension -> dimension wins
    complete = _fake_complete({
        "m1": {"score": 0.5, "sub_scores": {"faithfulness": 0.8}},
    })
    resp, _ = _run(score(
        _request(), model="m1", prompt_messages=[{"role": "user", "content": "x"}],
        scored_dimension="faithfulness", complete_fn=complete,
    ))
    assert resp.score == 0.8


def test_score_dimension_out_of_range_keeps_aggregate(tmp_path: Path) -> None:
    complete = _fake_complete({
        "m1": {"score": 0.5, "sub_scores": {"faithfulness": 5.0}},
    })
    resp, _ = _run(score(
        _request(), model="m1", prompt_messages=[{"role": "user", "content": "x"}],
        scored_dimension="faithfulness", complete_fn=complete,
    ))
    assert resp.score == 0.5


# ---------------------------------------------------------------------------
# fallback chain + error collection
# ---------------------------------------------------------------------------

def test_score_falls_back_when_default_fails(tmp_path: Path) -> None:
    complete = _fake_complete({
        "default": RuntimeError("boom"),
        "fb1": {"score": 0.7, "sub_scores": {}},
    })
    resp, errors = _run(score(
        _request(), model="default",
        prompt_messages=[{"role": "user", "content": "x"}],
        scored_dimension="score", complete_fn=complete,
        fallback_models=("fb1",),
    ))
    assert resp is not None
    assert resp.score == 0.7
    assert len(errors) == 1
    assert errors[0].model == "default"
    assert "boom" in errors[0].message


def test_score_all_fail_returns_none_with_errors(tmp_path: Path) -> None:
    complete = _fake_complete({})  # no model configured -> all raise
    resp, errors = _run(score(
        _request(), model="a", prompt_messages=[{"role": "user", "content": "x"}],
        scored_dimension="score", complete_fn=complete,
        fallback_models=("b", "c"),
    ))
    assert resp is None
    assert [e.model for e in errors] == ["a", "b", "c"]
    assert all(e.retryable is not False for e in errors)  # default retryable=True


# ---------------------------------------------------------------------------
# cache hit/miss
# ---------------------------------------------------------------------------

def test_score_cache_hit_skips_call(tmp_path: Path) -> None:
    cache = JudgeCache(tmp_path, enabled=True)
    calls = []
    async def complete_fn(*, model, messages, response_model, temperature=0.0):
        calls.append(model)
        return response_model.model_validate({"score": 0.9})
    # first call: miss -> writes
    _run(score(
        _request(), model="m1", prompt_messages=[{"role": "user", "content": "x"}],
        scored_dimension="score", complete_fn=complete_fn, cache=cache,
    ))
    # second call with same key -> hit, no new transport call
    _run(score(
        _request(), model="m1", prompt_messages=[{"role": "user", "content": "x"}],
        scored_dimension="score", complete_fn=complete_fn, cache=cache,
    ))
    assert calls == ["m1"]  # only the first (miss) call hit the transport
    assert cache.stats.hits == 1
    assert cache.stats.misses == 1


def test_score_no_cache_disabled(tmp_path: Path) -> None:
    cache = JudgeCache(tmp_path, enabled=False)
    async def complete_fn(*, model, messages, response_model, temperature=0.0):
        return response_model.model_validate({"score": 0.9})
    _run(score(
        _request(), model="m1", prompt_messages=[{"role": "user", "content": "x"}],
        scored_dimension="score", complete_fn=complete_fn, cache=cache,
    ))
    assert cache.stats.hits == 0
    assert cache.stats.misses == 0


# ---------------------------------------------------------------------------
# cache key determinism
# ---------------------------------------------------------------------------

def test_cache_key_deterministic() -> None:
    a = cache_key("m1", "prompt", schema_id="v1")
    b = cache_key("m1", "prompt", schema_id="v1")
    c = cache_key("m1", "different", schema_id="v1")
    d = cache_key("m2", "prompt", schema_id="v1")
    assert a == b
    assert a != c
    assert a != d


# ---------------------------------------------------------------------------
# ping
# ---------------------------------------------------------------------------

def test_ping_ok() -> None:
    async def complete_fn(*, model, messages, response_model, temperature=0.0):
        return response_model.model_validate({"ok": True})
    ok, detail = _run(ping("m1", complete_fn=complete_fn))
    assert ok is True


def test_ping_failure_returns_false_not_raise() -> None:
    async def complete_fn(*, model, messages, response_model, temperature=0.0):
        raise RuntimeError("unreachable")
    ok, detail = _run(ping("m1", complete_fn=complete_fn))
    assert ok is False
    assert "unreachable" in detail


def test_judge_response_rejects_out_of_range_score() -> None:
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        JudgeResponse(score=1.5)


def test_judge_error_serializes() -> None:
    e = JudgeError(model="m", message="boom", retryable=False)
    d = e.model_dump()
    assert d == {"model": "m", "message": "boom", "retryable": False}
