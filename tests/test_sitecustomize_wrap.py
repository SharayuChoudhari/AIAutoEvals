"""Tests for the bootstrap sitecustomize shim's dotted + async wrapping (D3).

The shim is a generated ``sitecustomize.py`` template. We render it, exec it in
a controlled namespace, and drive ``_wrap_entry`` directly to verify:
  - dotted ``Class.method`` entries rebind on the class (instances see the wrapper),
  - bare ``fn`` entries rebind on the module (unchanged behavior),
  - ``async def`` methods get an async wrapper that awaits,
  - the wrapper records enter/exit-task tracer calls.
"""

from __future__ import annotations

import asyncio
import sys
import textwrap

import pytest

from ai_eval.bootstrap.sitecustomize import _SHIM_TEMPLATE


@pytest.fixture
def shim_ns(monkeypatch, tmp_path):
    """Render + exec the shim template in a fresh namespace with a fake tracer.

    The shim calls ``from ai_eval.bootstrap.tracer import enter_task, exit_task``
    at wrap time; we inject a fake tracer module so no real hooks install.
    """
    events: list[tuple[str, str, dict]] = []

    fake_tracer = type(sys)("ai_eval.bootstrap.tracer")
    fake_tracer.enter_task = lambda name, ctx: events.append(("enter", name, ctx))
    fake_tracer.exit_task = lambda: events.append(("exit", "", {}))
    fake_tracer.configure_sink = lambda **kw: None
    fake_tracer.get_tracer = lambda: None
    fake_wrappers = type(sys)("ai_eval.bootstrap.wrappers")
    fake_wrappers.install_hooks = lambda **kw: None
    # ``ai_eval`` package stub so ``from ai_eval.bootstrap...`` resolves.
    fake_pkg = type(sys)("ai_eval")
    fake_bs = type(sys)("ai_eval.bootstrap")
    monkeypatch.setitem(sys.modules, "ai_eval", fake_pkg)
    monkeypatch.setitem(sys.modules, "ai_eval.bootstrap", fake_bs)
    monkeypatch.setitem(sys.modules, "ai_eval.bootstrap.tracer", fake_tracer)
    monkeypatch.setitem(sys.modules, "ai_eval.bootstrap.wrappers", fake_wrappers)
    monkeypatch.setenv("AI_EVAL_BOOTSTRAP_TASKS", "[]")
    monkeypatch.setenv("AI_EVAL_BOOTSTRAP_SINK", str(tmp_path / "sink.jsonl"))

    content = textwrap.dedent(_SHIM_TEMPLATE).format(shim_dir=str(tmp_path))
    ns: dict = {}
    exec(compile(content, "sitecustomize.py", "exec"), ns)
    ns["_events"] = events
    return ns


def test_wrap_dotted_sync_method_rebinds_on_class(shim_ns, monkeypatch, tmp_path):
    """A dotted ``Class.method`` entry wraps the method on the class, so an
    instance call goes through the wrapper and records enter/exit-task."""
    # Write a user module with a class + sync method.
    (tmp_path / "svc.py").write_text(
        "class Svc:\n"
        "    def process(self, q):\n"
        "        return q + 1\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    wrap_entry = shim_ns["_wrap_entry"]
    events: list = shim_ns["_events"]
    wrap_entry({"name": "task", "file_path": "svc.py", "entry": "Svc.process"})

    # Now import the user module (triggers the deferred __import__ wrap).
    sys.modules.pop("svc", None)
    import svc  # type: ignore[import-not-found]

    inst = svc.Svc()
    result = inst.process(41)
    assert result == 42
    # The wrapper recorded enter + exit for the task.
    assert any(e[0] == "enter" and e[1] == "task" for e in events)
    assert any(e[0] == "exit" for e in events)
    sys.modules.pop("svc", None)


def test_wrap_bare_function_rebinds_on_module(shim_ns, monkeypatch, tmp_path):
    """A bare ``fn`` entry rebinds on the module (unchanged behavior)."""
    (tmp_path / "mod.py").write_text(
        "def run(q):\n"
        "    return q * 2\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    wrap_entry = shim_ns["_wrap_entry"]
    events: list = shim_ns["_events"]
    wrap_entry({"name": "task", "file_path": "mod.py", "entry": "run"})

    sys.modules.pop("mod", None)
    import mod  # type: ignore[import-not-found]

    assert mod.run(21) == 42
    assert any(e[0] == "enter" and e[1] == "task" for e in events)
    sys.modules.pop("mod", None)


def test_wrap_dotted_async_method_awaits(shim_ns, monkeypatch, tmp_path):
    """A dotted ``Class.method`` entry on an ``async def`` method installs an
    async wrapper that awaits the coroutine, so the task records enter/exit
    and the awaited result is returned (not a coroutine object)."""
    (tmp_path / "async_svc.py").write_text(
        "class Svc:\n"
        "    async def process(self, q):\n"
        "        return q + 1\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    wrap_entry = shim_ns["_wrap_entry"]
    events: list = shim_ns["_events"]
    wrap_entry({"name": "task", "file_path": "async_svc.py", "entry": "Svc.process"})

    sys.modules.pop("async_svc", None)
    import async_svc  # type: ignore[import-not-found]

    inst = async_svc.Svc()
    result = asyncio.run(inst.process(41))
    assert result == 42  # awaited, not a coroutine
    assert any(e[0] == "enter" and e[1] == "task" for e in events)
    assert any(e[0] == "exit" for e in events)
    sys.modules.pop("async_svc", None)


def test_wrap_entry_no_file_path_is_noop(shim_ns):
    """A spec without ``file_path`` returns early (no import hook installed)."""
    wrap_entry = shim_ns["_wrap_entry"]
    # Should not raise; nothing wrapped.
    wrap_entry({"name": "task", "entry": "fn"})


def test_wrap_entry_missing_entry_falls_back_to_main(shim_ns, monkeypatch, tmp_path):
    """A spec with ``entry=None`` falls back to ``main`` (the documented default)."""
    (tmp_path / "main_mod.py").write_text(
        "def main(q):\n"
        "    return q\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    wrap_entry = shim_ns["_wrap_entry"]
    events: list = shim_ns["_events"]
    wrap_entry({"name": "task", "file_path": "main_mod.py"})

    sys.modules.pop("main_mod", None)
    import main_mod  # type: ignore[import-not-found]

    assert main_mod.main(7) == 7
    assert any(e[0] == "enter" for e in events)
    sys.modules.pop("main_mod", None)
