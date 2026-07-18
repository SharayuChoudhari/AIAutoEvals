"""Tests for the FastAPI route-handler detector.

Covers the plan's §"FastAPI detector" test matrix: app-decorated routes,
router-decorated routes, non-route functions ignored, and the import gate.
"""

from __future__ import annotations

import ast
from pathlib import Path

from ai_eval.inference.ast_scan import scan_repo
from ai_eval.inference.detectors.fastapi import FastAPIDetector
from ai_eval.inference.signatures import collect_imports, find_callable_defs


def _parse(source: str) -> tuple[ast.AST, list]:
    tree = ast.parse(source)
    return tree, collect_imports(tree)


def test_fastapi_detector_finds_post_route(tmp_path: Path) -> None:
    path = tmp_path / "main.py"
    source = (
        "from fastapi import FastAPI\n"
        "app = FastAPI()\n"
        "\n"
        "@app.post('/chat')\n"
        "async def chat(req):\n"
        "    return {'answer': 'hi'}\n"
    )
    path.write_text(source, encoding="utf-8")
    tree, imports = _parse(source)
    calls = []  # route detection walks decorators, not call sites
    defs = find_callable_defs(tree)

    det = FastAPIDetector()
    assert det.matches(tree, imports)
    tasks = det.extract(tree, imports, path, tmp_path, calls=calls, defs=defs)
    assert len(tasks) == 1
    task = tasks[0]
    assert task.entry == "chat"
    assert task.name == "chat"
    assert task.top_level is True
    assert task.framework == "fastapi"
    assert task.type == "chat"
    assert task.inputs == ["query"]
    assert task.outputs == ["response"]
    assert any("post" in e for e in task.evidence)
    assert any("main.py" in e for e in task.evidence)


def test_fastapi_detector_finds_router_route(tmp_path: Path) -> None:
    path = tmp_path / "routes.py"
    source = (
        "from fastapi import APIRouter\n"
        "router = APIRouter()\n"
        "\n"
        "@router.get('/items')\n"
        "async def list_items():\n"
        "    return []\n"
    )
    path.write_text(source, encoding="utf-8")
    tree, imports = _parse(source)
    defs = find_callable_defs(tree)

    det = FastAPIDetector()
    tasks = det.extract(tree, imports, path, tmp_path, calls=[], defs=defs)
    assert len(tasks) == 1
    assert tasks[0].entry == "list_items"
    assert tasks[0].top_level is True
    assert any("get" in e for e in tasks[0].evidence)


def test_fastapi_detector_finds_multiple_routes(tmp_path: Path) -> None:
    """Each route handler is one task; dedup by entry name within a file."""
    path = tmp_path / "api.py"
    source = (
        "from fastapi import FastAPI\n"
        "app = FastAPI()\n"
        "\n"
        "@app.get('/health')\n"
        "async def health():\n"
        "    return {'ok': True}\n"
        "\n"
        "@app.post('/chat')\n"
        "async def chat(req):\n"
        "    return req\n"
        "\n"
        "@app.put('/items/{id}')\n"
        "async def update_item(id, item):\n"
        "    return item\n"
        "\n"
        "@app.delete('/items/{id}')\n"
        "async def delete_item(id):\n"
        "    return {'deleted': id}\n"
    )
    path.write_text(source, encoding="utf-8")
    tree, imports = _parse(source)
    defs = find_callable_defs(tree)

    det = FastAPIDetector()
    tasks = det.extract(tree, imports, path, tmp_path, calls=[], defs=defs)
    entries = {t.entry for t in tasks}
    assert entries == {"health", "chat", "update_item", "delete_item"}
    assert all(t.top_level for t in tasks)


def test_fastapi_detector_ignores_non_route(tmp_path: Path) -> None:
    """A plain ``def helper():`` with no route decorator emits no task."""
    path = tmp_path / "util.py"
    source = "from fastapi import FastAPI\napp = FastAPI()\n\ndef helper():\n    return 1\n"
    path.write_text(source, encoding="utf-8")
    tree, imports = _parse(source)
    defs = find_callable_defs(tree)

    det = FastAPIDetector()
    tasks = det.extract(tree, imports, path, tmp_path, calls=[], defs=defs)
    assert tasks == []


def test_fastapi_detector_ignores_non_route_decorator(tmp_path: Path) -> None:
    """A function with a non-route decorator (``@staticmethod``) is ignored."""
    path = tmp_path / "util.py"
    source = (
        "from fastapi import FastAPI\n"
        "app = FastAPI()\n"
        "\n"
        "@staticmethod\n"
        "def helper():\n"
        "    return 1\n"
    )
    path.write_text(source, encoding="utf-8")
    tree, imports = _parse(source)
    defs = find_callable_defs(tree)

    det = FastAPIDetector()
    tasks = det.extract(tree, imports, path, tmp_path, calls=[], defs=defs)
    assert tasks == []


def test_fastapi_detector_ignores_non_fastapi_file(tmp_path: Path) -> None:
    """``matches`` returns False when the file has no ``fastapi`` import."""
    path = tmp_path / "plain.py"
    source = (
        "from flask import Flask\n"
        "app = Flask(__name__)\n"
        "\n"
        "@app.route('/chat')\n"
        "def chat():\n"
        "    return 'hi'\n"
    )
    path.write_text(source, encoding="utf-8")
    tree, imports = _parse(source)

    det = FastAPIDetector()
    assert det.matches(tree, imports) is False


def test_fastapi_detector_ignores_class_methods(tmp_path: Path) -> None:
    """Route handlers are module-level functions; a method named ``get``
    inside a class is NOT a route (no ``@app.get`` decorator)."""
    path = tmp_path / "svc.py"
    source = (
        "from fastapi import FastAPI\n"
        "app = FastAPI()\n"
        "\n"
        "class Repository:\n"
        "    def get(self, id):\n"
        "        return id\n"
    )
    path.write_text(source, encoding="utf-8")
    tree, imports = _parse(source)
    defs = find_callable_defs(tree)

    det = FastAPIDetector()
    tasks = det.extract(tree, imports, path, tmp_path, calls=[], defs=defs)
    assert tasks == []


def test_fastapi_detector_full_scan(tmp_path: Path) -> None:
    """End-to-end: ``scan_repo`` finds the FastAPI route handler and emits
    it as a top-level task. The internal DAO method (if any) is demoted by
    the signature-inspection path, not by the detector itself."""
    (tmp_path / "api.py").write_text(
        "from fastapi import FastAPI\n"
        "app = FastAPI()\n"
        "\n"
        "@app.post('/chat')\n"
        "async def chat(req):\n"
        "    return {'answer': 'hi'}\n",
        encoding="utf-8",
    )
    result = scan_repo(tmp_path)
    assert "fastapi" in result.frameworks_seen
    tasks = [t for t in result.tasks if t.framework == "fastapi"]
    assert len(tasks) == 1
    assert tasks[0].entry == "chat"
    assert tasks[0].top_level is True


def test_fastapi_route_demotes_io_coupled_service_it_calls(tmp_path: Path) -> None:
    """Integration: a FastAPI route handler that calls a service whose
    ``__init__`` requires a ``session`` arg → the service method is demoted
    to ``top_level=False`` by signature inspection, while the route handler
    stays top-level. This is the AGENTS.md §1 end-to-end flow: the route is
    the scored entry; the service is scored from the captured trace."""
    from ai_eval.inference.ast_scan import ScanResult
    from ai_eval.inference.detectors.base import DetectedTask
    from ai_eval.inference.synthesize import build_rubrics

    (tmp_path / "api.py").write_text(
        "from fastapi import FastAPI\n"
        "from services.chat import ChatMessageService\n"
        "app = FastAPI()\n"
        "\n"
        "@app.post('/chat')\n"
        "async def chat(req):\n"
        "    svc = ChatMessageService(session=None)\n"
        "    return svc.process_query(req)\n",
        encoding="utf-8",
    )
    (tmp_path / "services").mkdir()
    (tmp_path / "services" / "chat.py").write_text(
        "class ChatMessageService:\n"
        "    def __init__(self, session):\n"
        "        self.session = session\n"
        "    def process_query(self, query):\n"
        "        return self.session.run(query)\n",
        encoding="utf-8",
    )
    # Simulate what the detectors would emit: the route (FastAPI) + the
    # service method (OpenAI detector firing on an internal call, or any
    # detector that picked it up). build_rubrics then applies signature
    # demotion to the service method.
    scan = ScanResult(
        files_scanned=2,
        elapsed_seconds=0.0,
        tasks=[
            DetectedTask(
                name="chat",
                framework="fastapi",
                type="chat",
                file_path="api.py",
                entry="chat",
                inputs=["query"],
                outputs=["response"],
            ),
            DetectedTask(
                name="ChatMessageService.process_query",
                framework="openai",
                type="chat",
                file_path="services/chat.py",
                entry="ChatMessageService.process_query",
            ),
        ],
        frameworks_seen={"fastapi", "openai"},
    )
    rubrics = build_rubrics(scan, project_root=tmp_path)
    assert rubrics.tasks["chat"].top_level is True
    assert rubrics.tasks["chat_message_service_process_query"].top_level is False
