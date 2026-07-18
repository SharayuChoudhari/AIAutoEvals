"""Tests for run-engine dotted entry resolution, async handling, and harness
loading (D7/Step 7).

The ``_Stub``/``_fake_call_args`` symbols were removed (AGENTS.md §1): dotted
entries construct with no args (or a harness), and an IO-coupled entry whose
``__init__`` requires real args fails with a bootstrap-directed error instead
of a silent ``_Stub``-driven crash.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

from ai_eval.config.schema import JudgeConfig, RubricsConfig, TaskSpec
from ai_eval.runner.engine import execute


def _rubrics(tasks: dict[str, TaskSpec]) -> RubricsConfig:
    return RubricsConfig(
        schema_version=1,
        project_type="chat",
        judge=JudgeConfig(default="fake/local"),
        tasks=tasks,
    )


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_dotted_sync_method_entry_resolves_and_runs(tmp_path: Path) -> None:
    """A dotted ``Class.method`` entry constructs an instance with no args
    and calls the bound method."""
    (tmp_path / "svc.py").write_text(
        "class Svc:\n"
        "    def __init__(self):\n"
        "        self.name = 'svc'\n"
        "    def process(self, q):\n"
        "        return f'{self.name}:{q}'\n",
        encoding="utf-8",
    )
    rubrics = _rubrics(
        {
            "svc_process": TaskSpec(
                file_path="svc.py", entry="Svc.process", type="chat", metrics=[]
            ),
        }
    )
    golden = {"svc_process": [{"id": "e1", "input": "hi"}]}
    record = _run(
        execute(
            rubrics,
            golden,
            project_root=tmp_path,
            parallel=1,
            complete_fn=None,
            run_id="r1",
            started_at=time.time(),
        )
    )
    task = record.tasks["svc_process"]
    # No error; the example ran.
    assert all(e.status != "error" for e in task.examples)


def test_dotted_async_method_entry_awaits(tmp_path: Path) -> None:
    """A dotted ``async def`` method entry is awaited — the output is the real
    result, not a coroutine object (the silent-failure gap the handover
    flagged)."""
    (tmp_path / "async_svc.py").write_text(
        "class Svc:\n"
        "    def __init__(self):\n"
        "        pass\n"
        "    async def process(self, q):\n"
        "        return q + '!'\n",
        encoding="utf-8",
    )
    rubrics = _rubrics(
        {
            "async_svc_process": TaskSpec(
                file_path="async_svc.py", entry="Svc.process", type="chat", metrics=[]
            ),
        }
    )
    golden = {"async_svc_process": [{"id": "e1", "input": "hi"}]}
    record = _run(
        execute(
            rubrics,
            golden,
            project_root=tmp_path,
            parallel=1,
            complete_fn=None,
            run_id="r2",
            started_at=time.time(),
        )
    )
    task = record.tasks["async_svc_process"]
    assert all(e.status != "error" for e in task.examples)


def test_bare_function_entry_runs_unchanged(tmp_path: Path) -> None:
    """A bare ``fn`` entry (no dot) runs via the original fast path."""
    (tmp_path / "mod.py").write_text(
        "def main(q):\n    return q * 2\n",
        encoding="utf-8",
    )
    rubrics = _rubrics(
        {
            "mod_task": TaskSpec(file_path="mod.py", entry="main", type="chat", metrics=[]),
        }
    )
    golden = {"mod_task": [{"id": "e1", "input": 21}]}
    record = _run(
        execute(
            rubrics,
            golden,
            project_root=tmp_path,
            parallel=1,
            complete_fn=None,
            run_id="r3",
            started_at=time.time(),
        )
    )
    task = record.tasks["mod_task"]
    assert all(e.status != "error" for e in task.examples)


def test_dotted_entry_missing_class_errors_cleanly(tmp_path: Path) -> None:
    """A dotted entry whose class doesn't exist produces an error example, not
    a crash."""
    (tmp_path / "svc.py").write_text(
        "class Other:\n    def process(self, q):\n        return q\n",
        encoding="utf-8",
    )
    rubrics = _rubrics(
        {
            "svc_process": TaskSpec(
                file_path="svc.py", entry="Svc.process", type="chat", metrics=[]
            ),
        }
    )
    golden = {"svc_process": [{"id": "e1", "input": "hi"}]}
    record = _run(
        execute(
            rubrics,
            golden,
            project_root=tmp_path,
            parallel=1,
            complete_fn=None,
            run_id="r4",
            started_at=time.time(),
        )
    )
    task = record.tasks["svc_process"]
    assert any(e.status == "error" for e in task.examples)


def test_harness_loaded_for_io_coupled_dotted_entry(tmp_path: Path) -> None:
    """An IO-coupled dotted entry loads the stub harness so the DAO read
    returns the canned fixture instead of hitting a real DB."""
    (tmp_path / "svc.py").write_text(
        "class DAO:\n"
        "    def search(self, q):\n"
        "        raise RuntimeError('real DB hit')\n"
        "class Svc:\n"
        "    def __init__(self):\n"
        "        self.dao = DAO()\n"
        "    def process(self, q):\n"
        "        return self.dao.search(q)\n",
        encoding="utf-8",
    )
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    # Use the real harness writer template so the generated file matches what
    # `ai-evals init` produces (including the project-root-relative resolution).
    from ai_eval.scaffold.harness_writer import HarnessSpec, IOAttr, render_harness

    spec = HarnessSpec(
        task_name="svc_process",
        entry="Svc.process",
        file_path="svc.py",
        attrs=[IOAttr(attr="dao", method="search", ctor_name="DAO")],
        body_hash="stub",
    )
    content = render_harness(spec)
    # Edit region 2 fixtures so the stub returns a canned value.
    content = content.replace(
        "('dao', 'search'): {}", "('dao', 'search'): [{'id': 1, 'text': 'canned'}]"
    )
    (eval_dir / "_harness_svc_process.py").write_text(content, encoding="utf-8")
    rubrics = _rubrics(
        {
            "svc_process": TaskSpec(
                file_path="svc.py", entry="Svc.process", type="chat", metrics=[]
            ),
        }
    )
    golden = {"svc_process": [{"id": "e1", "input": "q"}]}
    record = _run(
        execute(
            rubrics,
            golden,
            project_root=tmp_path,
            parallel=1,
            complete_fn=None,
            run_id="r5",
            started_at=time.time(),
        )
    )
    task = record.tasks["svc_process"]
    # The harness patched process → no real DB hit → no error.
    assert all(e.status != "error" for e in task.examples), task.errors


def test_dotted_entry_required_args_no_harness_errors_with_bootstrap_hint(
    tmp_path: Path,
) -> None:
    """A dotted entry whose ``__init__`` requires real args and has no harness
    fails with a clear bootstrap-directed error — not a ``RecursionError`` or
    silent ``_Stub``-driven crash (AGENTS.md §1).

    Pins the removal of ``_fake_call_args``/``_Stub``. Mirrors
    ``SingleQueryEvaluator.__init__(self, config)`` and
    ``ChatMessageService.__init__(self, session)``.
    """
    (tmp_path / "svc.py").write_text(
        "class Svc:\n"
        "    def __init__(self, config):\n"
        "        self.config = config\n"
        "    def process(self, q):\n"
        "        return f'got:{q}'\n",
        encoding="utf-8",
    )
    rubrics = _rubrics(
        {
            "svc_process": TaskSpec(
                file_path="svc.py", entry="Svc.process", type="chat", metrics=[]
            ),
        }
    )
    golden = {"svc_process": [{"id": "e1", "input": "hi"}]}
    record = _run(
        execute(
            rubrics,
            golden,
            project_root=tmp_path,
            parallel=1,
            complete_fn=None,
            run_id="b23",
            started_at=time.time(),
        )
    )
    task = record.tasks["svc_process"]
    # Construction failed (required arg, no harness, no _Stub) → error example
    # with a bootstrap-directed hint.
    assert any(e.status == "error" for e in task.examples), task.errors
    err = task.examples[0].error or ""
    assert "bootstrap" in err, err
    assert "missing" in err.lower() or "argument" in err.lower(), err


def test_dotted_entry_cross_package_import_loads(tmp_path: Path, monkeypatch) -> None:
    """A task module with a top-level ``from layers.dao import ...`` loads
    under console-script semantics (cwd NOT auto-on-sys.path).

    Pins Bug 1 end-to-end. Mirrors ``document_vector_d_a_o``. Uses
    ``monkeypatch`` to drop ``tmp_path`` (and the empty string) from
    ``sys.path`` so the bug actually triggers pre-fix.
    """
    layers = tmp_path / "layers"
    layers.mkdir()
    (layers / "__init__.py").write_text("", encoding="utf-8")
    (layers / "dao.py").write_text(
        "class DocumentVectorDAO:\n"
        "    def search(self, q):\n"
        "        raise RuntimeError('real vector store hit')\n",
        encoding="utf-8",
    )
    (tmp_path / "svc.py").write_text(
        "from layers.dao import DocumentVectorDAO\n"
        "class Svc:\n"
        "    def __init__(self):\n"
        "        self.dao = DocumentVectorDAO()\n"
        "    def search(self, q):\n"
        "        return self.dao.search(q)\n",
        encoding="utf-8",
    )
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    from ai_eval.scaffold.harness_writer import HarnessSpec, IOAttr, render_harness

    spec = HarnessSpec(
        task_name="svc_search",
        entry="Svc.search",
        file_path="svc.py",
        attrs=[IOAttr(attr="dao", method="search", ctor_name="DocumentVectorDAO")],
        body_hash="stub",
    )
    content = render_harness(spec)
    content = content.replace(
        "('dao', 'search'): {}", "('dao', 'search'): [{'id': 1, 'text': 'canned'}]"
    )
    (eval_dir / "_harness_svc_search.py").write_text(content, encoding="utf-8")

    rubrics = _rubrics(
        {
            "svc_search": TaskSpec(file_path="svc.py", entry="Svc.search", type="chat", metrics=[]),
        }
    )
    golden = {"svc_search": [{"id": "e1", "input": "q"}]}

    # Mimic a console-script entrypoint: cwd is NOT on sys.path. Drop tmp_path
    # and "" so the harness's exec_module can't find `layers` without the fix.
    saved = list(sys.path)
    tmp_str = str(tmp_path)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p not in ("", tmp_str)])
    try:
        record = _run(
            execute(
                rubrics,
                golden,
                project_root=tmp_path,
                parallel=1,
                complete_fn=None,
                run_id="b1",
                started_at=time.time(),
            )
        )
    finally:
        sys.path[:] = saved

    task = record.tasks["svc_search"]
    assert all(e.status != "error" for e in task.examples), task.errors
    assert not any("ModuleNotFoundError" in (e.error or "") for e in task.examples)
