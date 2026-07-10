"""Thread-local task stack + golden-example emission for the bootstrapper.

Each patched framework call records ``{kind, name, args, result, latency_ms}``
into the active task's trace via :func:`record_call`. The tracer keeps a
thread-local stack so nested calls fold into the enclosing top-level task.

A *top-level task* is one entered via :func:`enter_task` — typically the user's
``main``/entry function the wrapper decides to wrap. On :func:`exit_task` the
tracer emits one golden example to the capture sink (an append-only JSONL
writer): ``{id, captured_at, input, expected?, trace:{calls:[...]}, tags}``.

Dedup by ``sha256(json.dumps(input, sort_keys=True))`` per task name.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

#: Regex for secrets to strip from captured args/results. Matches the common
#: env-var and header key names; the matched *value* (following ``=`` or in a
#: dict slot) is masked.
_SECRET_RE = re.compile(
    r"(?i)(api[_-]?key|bearer|secret|token|password|passwd|authorization)"
)
_SECRET_VALUE_RE = re.compile(
    r"(?i)((?:api[_-]?key|bearer|secret|token|password|passwd|authorization)"
    r"\s*[:=]\s*)(\S+)"
)


def redact(obj: Any) -> Any:
    """Recursively mask secret-looking values in ``obj``.

    Strings matching the secret pattern have their ``value`` part replaced with
    ``***``. Dict keys matching the pattern have their value masked. Returns a
    new structure; the input is not mutated.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if _SECRET_RE.search(str(k)):
                out[k] = "***"
            else:
                out[k] = redact(v)
        return out
    if isinstance(obj, (list, tuple)):
        return type(obj)(redact(v) for v in obj)
    if isinstance(obj, str):
        masked = _SECRET_VALUE_RE.sub(lambda m: f"{m.group(1)}***", obj)
        # Also catch bare "OPENAI_API_KEY=sk-..." style
        masked = re.sub(
            r"(?i)(OPENAI_API_KEY|ANTHROPIC_API_KEY)=\S+",
            lambda m: f"{m.group(1)}=***",
            masked,
        )
        return masked
    return obj


@dataclass
class _Call:
    kind: str
    name: str
    args: Any
    result: Any
    latency_ms: float


@dataclass
class _ActiveTask:
    task_name: str
    input: Any
    calls: list[_Call] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)


class _TaskStack:
    """Thread-local stack of active tasks + a capture sink."""

    def __init__(self) -> None:
        self._local = threading.local()
        self.sink: Callable[[dict], None] | None = None
        self.redact_fields: list[str] | None = None
        #: ``(task_name -> set[input_hash])`` for dedup within a run.
        self._seen: dict[str, set[str]] = {}

    def _stack(self) -> list[_ActiveTask]:
        st = getattr(self._local, "stack", None)
        if st is None:
            st = []
            self._local.stack = st
        return st

    @property
    def current(self) -> _ActiveTask | None:
        st = self._stack()
        return st[-1] if st else None

    def enter(self, task_name: str, input: Any) -> None:
        self._stack().append(_ActiveTask(task_name=task_name, input=input))

    def exit(self, output: Any) -> dict | None:
        st = self._stack()
        if not st:
            return None
        active = st.pop()
        expected = None
        if isinstance(output, dict) and "expected" in output:
            expected = output["expected"]
        elif isinstance(output, dict) and "answer" in output:
            expected = output
        example = {
            "id": _example_id(active.task_name, active.input),
            "captured_at": active.started_at,
            "task": active.task_name,
            "input": active.input,
            "expected": expected,
            "trace": {
                "calls": [
                    {
                        "kind": c.kind,
                        "name": c.name,
                        "args": c.args,
                        "result": c.result,
                        "latency_ms": c.latency_ms,
                    }
                    for c in active.calls
                ]
            },
            "tags": [],
        }
        # dedup per task by input hash
        h = _input_hash(active.input)
        seen = self._seen.setdefault(active.task_name, set())
        if h in seen:
            return None
        seen.add(h)
        if self.sink is not None:
            self.sink(example)
        return example

    def record_call(self, kind: str, name: str, args: Any, result: Any, latency_ms: float) -> None:
        active = self.current
        if active is None:
            return
        if self.redact_fields:
            args = redact(args)
            result = redact(result)
        active.calls.append(
            _Call(kind=kind, name=name, args=args, result=result, latency_ms=latency_ms)
        )

    def reset(self) -> None:
        """Clear the seen-set (tests that re-run a capture call this)."""
        self._seen.clear()


def _input_hash(input: Any) -> str:
    return hashlib.sha256(
        json.dumps(input, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _example_id(task_name: str, input: Any) -> str:
    return f"{task_name}_{_input_hash(input)[:12]}"


#: Module-level singleton used by the wrappers.
_TRACER = _TaskStack()


def get_tracer() -> _TaskStack:
    return _TRACER


def configure_sink(sink: Callable[[dict], None], *, redact_fields: list[str] | None) -> None:
    _TRACER.sink = sink
    _TRACER.redact_fields = redact_fields


def enter_task(task_name: str, input: Any) -> None:
    _TRACER.enter(task_name, input)


def exit_task(output: Any = None) -> dict | None:
    return _TRACER.exit(output)


def record_call(
    kind: str, name: str, args: Any, result: Any, latency_ms: float
) -> None:
    _TRACER.record_call(kind, name, args, result, latency_ms)


@contextmanager
def task_scope(task_name: str, input: Any):
    enter_task(task_name, input)
    try:
        yield
    finally:
        exit_task()


__all__ = [
    "configure_sink",
    "enter_task",
    "exit_task",
    "get_tracer",
    "record_call",
    "redact",
    "task_scope",
]
