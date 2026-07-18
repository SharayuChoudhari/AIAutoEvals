"""Tests for the cross-file call-graph builder (R2) and root selection (D1).

Covers the plan's edge cases:
  - shared retrieval helper called by 2 endpoints → helper demoted, both
    endpoints promoted (roots).
  - framework-object entry (LlamaIndex ``.aquery()``) → root, nothing reaches
    it.
  - ``__init__`` attr tracking resolves ``self.<attr>.<m>()`` across files.
  - dynamic dispatch / lazy init → no edge → safe over-promotion (both roots).
"""

from __future__ import annotations

from pathlib import Path

from ai_eval.inference.ast_scan import ScanResult
from ai_eval.inference.callgraph import SiteKey, build_call_graph, compute_roots
from ai_eval.inference.detectors.base import DetectedTask


def _scan(tasks: list[DetectedTask]) -> ScanResult:
    return ScanResult(
        files_scanned=len({t.file_path for t in tasks}),
        elapsed_seconds=0.0,
        tasks=tasks,
        frameworks_seen={t.framework for t in tasks},
    )


def _task(name: str, entry: str | None, file_path: str, **kw) -> DetectedTask:
    return DetectedTask(
        name=name,
        framework=kw.get("framework", "openai"),
        type=kw.get("type", "chat"),
        file_path=file_path,
        entry=entry,
        inputs=kw.get("inputs", []),
        outputs=kw.get("outputs", []),
    )


# ---------------------------------------------------------------------------
# __init__ attr tracking: self.dao.search() reaches the DAO task
# ---------------------------------------------------------------------------


def test_init_attr_resolves_cross_file_edge(tmp_path: Path) -> None:
    """``self.dao.search(...)`` in a service method reaches the DAO's
    ``search`` method (assigned in ``__init__`` via ``DocumentVectorDAO()``)
    when the DAO class is imported from a repo module."""
    (tmp_path / "layers").mkdir()
    (tmp_path / "layers" / "dao.py").write_text(
        "class DocumentVectorDAO:\n"
        "    def search_similar_vectors(self, vec):\n"
        "        return self.session.execute('SELECT ...')\n",
        encoding="utf-8",
    )
    (tmp_path / "services").mkdir()
    (tmp_path / "services" / "chat.py").write_text(
        "from layers.dao import DocumentVectorDAO\n"
        "\n"
        "class ChatService:\n"
        "    def __init__(self):\n"
        "        self.dao = DocumentVectorDAO()\n"
        "    def process_query(self, q):\n"
        "        return self.dao.search_similar_vectors(q)\n",
        encoding="utf-8",
    )
    tasks = [
        _task("ChatService.process_query", "ChatService.process_query", "services/chat.py"),
        _task(
            "DocumentVectorDAO.search_similar_vectors",
            "DocumentVectorDAO.search_similar_vectors",
            "layers/dao.py",
        ),
    ]
    scan = _scan(tasks)
    edges, _ = build_call_graph(tmp_path, scan)
    assert SiteKey("services/chat.py", "ChatService.process_query") in edges
    src = SiteKey("services/chat.py", "ChatService.process_query")
    dst = SiteKey("layers/dao.py", "DocumentVectorDAO.search_similar_vectors")
    assert dst in edges[src]


def test_compute_roots_demotes_reached_helper(tmp_path: Path) -> None:
    """The shared helper (DAO) is reached by the endpoint → demoted; the
    endpoint is the sole root."""
    (tmp_path / "layers").mkdir()
    (tmp_path / "layers" / "dao.py").write_text(
        "class DocumentVectorDAO:\n"
        "    def search_similar_vectors(self, vec):\n"
        "        return self.session.execute('SELECT ...')\n",
        encoding="utf-8",
    )
    (tmp_path / "services").mkdir()
    (tmp_path / "services" / "chat.py").write_text(
        "from layers.dao import DocumentVectorDAO\n"
        "\n"
        "class ChatService:\n"
        "    def __init__(self):\n"
        "        self.dao = DocumentVectorDAO()\n"
        "    def process_query(self, q):\n"
        "        return self.dao.search_similar_vectors(q)\n",
        encoding="utf-8",
    )
    tasks = [
        _task("ChatService.process_query", "ChatService.process_query", "services/chat.py"),
        _task(
            "DocumentVectorDAO.search_similar_vectors",
            "DocumentVectorDAO.search_similar_vectors",
            "layers/dao.py",
        ),
    ]
    scan = _scan(tasks)
    edges, _ = build_call_graph(tmp_path, scan)
    roots, reached = compute_roots(edges, tasks)
    assert {t.entry for t in roots} == {"ChatService.process_query"}
    assert {t.entry for t in reached} == {"DocumentVectorDAO.search_similar_vectors"}


# ---------------------------------------------------------------------------
# Shared helper called by 2 endpoints → helper demoted, both endpoints roots
# ---------------------------------------------------------------------------


def test_shared_helper_called_by_two_endpoints_is_demoted(tmp_path: Path) -> None:
    (tmp_path / "shared").mkdir()
    (tmp_path / "shared" / "retrieval.py").write_text(
        "def retrieve(query):\n    return vs.similarity_search(query)\n",
        encoding="utf-8",
    )
    (tmp_path / "api").mkdir()
    (tmp_path / "api" / "a.py").write_text(
        "from shared.retrieval import retrieve\n"
        "def endpoint_a(query):\n"
        "    return retrieve(query)\n",
        encoding="utf-8",
    )
    (tmp_path / "api" / "b.py").write_text(
        "from shared.retrieval import retrieve\n"
        "def endpoint_b(query):\n"
        "    return retrieve(query)\n",
        encoding="utf-8",
    )
    tasks = [
        _task("endpoint_a", "endpoint_a", "api/a.py"),
        _task("endpoint_b", "endpoint_b", "api/b.py"),
        _task("retrieve", "retrieve", "shared/retrieval.py"),
    ]
    scan = _scan(tasks)
    edges, _ = build_call_graph(tmp_path, scan)
    roots, reached = compute_roots(edges, tasks)
    assert {t.entry for t in roots} == {"endpoint_a", "endpoint_b"}
    assert {t.entry for t in reached} == {"retrieve"}


# ---------------------------------------------------------------------------
# Framework-object entry (no self.attr reachability) → root
# ---------------------------------------------------------------------------


def test_framework_object_entry_is_root(tmp_path: Path) -> None:
    """A LlamaIndex ``index.aquery()`` call site is a root: nothing reaches it
    (no ``self.<attr>`` reads into a detected site), so it's never demoted."""
    (tmp_path / "app.py").write_text(
        "from llama_index import VectorStoreIndex\n"
        "def ask(index, query):\n"
        "    return index.aquery(query)\n",
        encoding="utf-8",
    )
    tasks = [_task("ask", "ask", "app.py", type="rag")]
    scan = _scan(tasks)
    edges, _ = build_call_graph(tmp_path, scan)
    roots, reached = compute_roots(edges, tasks)
    assert {t.entry for t in roots} == {"ask"}
    assert reached == []


# ---------------------------------------------------------------------------
# Safe degradation: lazy init / dynamic dispatch → no edge → over-promotion
# ---------------------------------------------------------------------------


def test_lazy_init_attr_not_in_init_draws_no_edge(tmp_path: Path) -> None:
    """When ``self.dao`` is assigned outside ``__init__`` (lazy init in the
    method body), no edge is drawn — both sites stay roots. This is the R2
    safe-degradation contract: never wrongly demote a real task."""
    (tmp_path / "layers").mkdir()
    (tmp_path / "layers" / "dao.py").write_text(
        "class DAO:\n    def search(self, q):\n        return q\n",
        encoding="utf-8",
    )
    (tmp_path / "services").mkdir()
    (tmp_path / "services" / "svc.py").write_text(
        "from layers.dao import DAO\n"
        "\n"
        "class Svc:\n"
        "    def process(self, q):\n"
        "        # lazy init — not in __init__\n"
        "        if not hasattr(self, 'dao'):\n"
        "            self.dao = DAO()\n"
        "        return self.dao.search(q)\n",
        encoding="utf-8",
    )
    tasks = [
        _task("Svc.process", "Svc.process", "services/svc.py"),
        _task("DAO.search", "DAO.search", "layers/dao.py"),
    ]
    scan = _scan(tasks)
    edges, _ = build_call_graph(tmp_path, scan)
    roots, reached = compute_roots(edges, tasks)
    # Both stay roots — the DAO is NOT demoted (lazy init unresolved).
    assert {t.entry for t in roots} == {"Svc.process", "DAO.search"}
    assert reached == []


# ---------------------------------------------------------------------------
# Transitive reachability: A → B → C demotes both B and C
# ---------------------------------------------------------------------------


def test_transitive_reachability_demotes_chain(tmp_path: Path) -> None:
    (tmp_path / "mid.py").write_text(
        "def helper(q):\n    return inner(q)\ndef inner(q):\n    return q\n",
        encoding="utf-8",
    )
    (tmp_path / "top.py").write_text(
        "from mid import helper\ndef top(q):\n    return helper(q)\n",
        encoding="utf-8",
    )
    tasks = [
        _task("top", "top", "top.py"),
        _task("helper", "helper", "mid.py"),
        _task("inner", "inner", "mid.py"),
    ]
    scan = _scan(tasks)
    edges, _ = build_call_graph(tmp_path, scan)
    roots, reached = compute_roots(edges, tasks)
    assert {t.entry for t in roots} == {"top"}
    assert {t.entry for t in reached} == {"helper", "inner"}


# ---------------------------------------------------------------------------
# No tasks → empty graph, clean exit
# ---------------------------------------------------------------------------


def test_empty_scan_produces_empty_graph(tmp_path: Path) -> None:
    scan = _scan([])
    edges, _ = build_call_graph(tmp_path, scan)
    assert edges == {}
    roots, reached = compute_roots(edges, [])
    assert roots == []
    assert reached == []
