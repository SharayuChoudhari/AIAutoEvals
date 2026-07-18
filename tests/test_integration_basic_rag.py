"""End-to-end integration test for the universal task-detection plan.

Mirrors the Basic_RAG verification case as a synthetic multi-file fixture and
asserts the full motivating flow:
  - the DAO (internal node) is demoted — not emitted as a task,
  - the RAG endpoint (``ChatMessageService.process_query``) is the sole task,
  - its dotted ``Class.method`` entry is resolvable,
  - a stub harness is generated for the IO-coupled task,
  - ``ai-evals run`` produces a non-empty ``seed: auto`` result.

Plus the plan's edge-case fixtures: shared helper called by 2 endpoints,
framework-object root, judge harness excluded, async method, orphan-no-root
repo.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ai_eval.cli.app import app
from ai_eval.inference.ast_scan import scan_repo
from ai_eval.inference.synthesize import build_rubrics


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _basic_rag_repo(tmp_path: Path) -> Path:
    """A multi-file fixture mirroring Basic_RAG: a LangGraph-compiled RAG
    service whose ``process_query`` reads a DAO via ``self.document_vector_dao``,
    plus an internal DAO and an evaluations judge harness.
    """
    (tmp_path / "layers" / "dao").mkdir(parents=True)
    (tmp_path / "layers" / "dao" / "document_vectors_dao.py").write_text(
        "from sqlalchemy import select\n"
        "from pgvector.sqlalchemy import Vector\n"
        "\n"
        "class DocumentVectorDAO:\n"
        "    def __init__(self, session=None):\n"
        "        self.session = session\n"
        "    def search_similar_vectors(self, query_embedding):\n"
        "        distance = DocumentVector.embedding.op('<=>')(query_embedding)\n"
        "        stmt = select(DocumentVector).order_by(distance).limit(5)\n"
        "        return self.session.execute(stmt).scalars().all()\n"
        "\n"
        "class DocumentVector:\n"
        "    embedding = None\n",
        encoding="utf-8",
    )
    (tmp_path / "services").mkdir()
    (tmp_path / "services" / "chat_messages.py").write_text(
        "from layers.dao.document_vectors_dao import DocumentVectorDAO\n"
        "from langgraph.graph import END, StateGraph\n"
        "from openai import OpenAI\n"
        "\n"
        "class ChatMessageService:\n"
        "    def __init__(self):\n"
        "        self.document_vector_dao = DocumentVectorDAO()\n"
        "        self.client = OpenAI()\n"
        "\n"
        "    def process_query(self, query):\n"
        "        # Direct DAO read (R2 __init__ attr tracking resolves this).\n"
        "        docs = self.document_vector_dao.search_similar_vectors(query)\n"
        "        graph = StateGraph(dict)\n"
        "        graph.add_node('gen', lambda s: s)\n"
        "        graph.set_entry_point('gen')\n"
        "        graph.add_edge('gen', END)\n"
        "        compiled = graph.compile()\n"
        "        state = {'q': query, 'documents': docs}\n"
        "        return compiled.invoke(state)\n",
        encoding="utf-8",
    )
    (tmp_path / "evaluations").mkdir()
    (tmp_path / "evaluations" / "single_eval.py").write_text(
        "class EvaluationResult:\n"
        "    def __init__(self, faithfulness):\n"
        "        self.faithfulness = faithfulness\n"
        "def evaluate(query, answer):\n"
        "    return EvaluationResult(0.9)\n",
        encoding="utf-8",
    )
    return tmp_path


def test_basic_rag_dao_demoted_endpoint_is_sole_task(tmp_path: Path) -> None:
    """The DAO (reached via __init__ attr tracking) is demoted; the endpoint
    is the sole surviving task. The judge harness is excluded (J2∧J3)."""
    root = _basic_rag_repo(tmp_path)
    scan = scan_repo(root)
    rubrics = build_rubrics(scan, project_root=root)
    entries = {spec.entry for spec in rubrics.tasks.values()}
    # The endpoint survives as a dotted entry.
    assert "ChatMessageService.process_query" in entries
    # The DAO is demoted (reached by the endpoint via self.document_vector_dao).
    assert not any("search_similar_vectors" in (e or "") for e in entries)
    # The judge harness is excluded (J2: evaluations/ AND J3: faithfulness return).
    assert not any("evaluate" in (e or "") for e in entries)


def test_basic_rag_generates_harness_for_io_coupled_task(
    tmp_path: Path, runner: CliRunner, clean_env: None
) -> None:
    """``ai-evals init`` generates ``eval/_harness_<task>.py`` for the
    IO-coupled endpoint task (it reads self.document_vector_dao)."""
    root = _basic_rag_repo(tmp_path)
    result = runner.invoke(
        app, ["-C", str(root), "--format", "json", "init", "--force", "--rubric-engine", "rules"]
    )
    assert result.exit_code == 0, result.stderr or result.output
    eval_dir = root / "eval"
    harnesses = list(eval_dir.glob("_harness_*.py"))
    assert harnesses, "expected a harness file for the IO-coupled task"
    # The harness should reference the DAO's search method.
    content = harnesses[0].read_text(encoding="utf-8")
    assert "search_similar_vectors" in content


def test_basic_rag_run_produces_non_empty_seed_auto_result(
    tmp_path: Path, runner: CliRunner, monkeypatch, clean_env: None
) -> None:
    """Full init→run on a self-contained IO-coupled task: the golden set is
    auto-seeded (D6) and the run produces a non-empty ``seed: auto`` result.
    Uses a task with no external framework imports so the run can execute
    without installing langgraph/openai — the DAO is stubbed by the harness."""
    # Self-contained IO-coupled task: a service reading a DAO, detected via
    # the openai detector (inline OpenAI call) so no package install is needed.
    (tmp_path / "svc.py").write_text(
        "from openai import OpenAI\n"
        "class DAO:\n"
        "    def search(self, q):\n"
        "        raise RuntimeError('real DB hit')\n"
        "class Svc:\n"
        "    def __init__(self):\n"
        "        self.dao = DAO()\n"
        "        self.client = OpenAI()\n"
        "    def process(self, q):\n"
        "        docs = self.dao.search(q)\n"
        "        return self.client.chat.completions.create(\n"
        "            messages=[{'role':'user','content':docs}])\n",
        encoding="utf-8",
    )
    # Stub the openai module so the import doesn't need the real package.
    import sys
    import types

    fake_oai = types.ModuleType("openai")

    class _FakeOpenAI:
        def __init__(self, **kw):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(
                    create=lambda **kw: {"choices": [{"message": {"content": "ok"}}]}
                )
            )

    fake_oai.OpenAI = _FakeOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_oai)

    # Fake the judge so no network is hit.
    async def complete_fn(*, model, messages, response_model, temperature=0.0):
        return response_model.model_validate({"score": 0.9, "rationale": "ok", "sub_scores": {}})

    import ai_eval.judge.gateway as gw

    monkeypatch.setattr(gw, "_default_complete", complete_fn)

    init_result = runner.invoke(
        app,
        ["-C", str(tmp_path), "--format", "json", "init", "--force", "--rubric-engine", "rules"],
    )
    assert init_result.exit_code == 0, init_result.stderr or init_result.output
    # A harness should be generated for the IO-coupled task.
    assert list((tmp_path / "eval").glob("_harness_*.py"))
    # Edit the harness fixture so the DAO stub returns a canned value.
    harness_path = next((tmp_path / "eval").glob("_harness_*.py"))
    content = harness_path.read_text(encoding="utf-8")
    content = content.replace(
        "('dao', 'search'): {}", "('dao', 'search'): [{'id': 1, 'text': 'canned'}]"
    )
    harness_path.write_text(content, encoding="utf-8")

    # The golden set should have auto-seeded examples for the top-level task.
    # Under the new model (AGENTS.md §1), all top-level tasks get the 5
    # pure-LLM shape variants — the IO-coupled single-green-pipeline branch
    # was removed (the ``_Stub`` path is gone; IO-coupled entries need a
    # harness or bootstrap).
    golden = json.loads((tmp_path / "eval" / "golden_set.json").read_text())
    import yaml

    rubrics_data = yaml.safe_load((tmp_path / "eval" / "rubrics.yaml").read_text())
    task_name = next(
        n for n, spec in rubrics_data["tasks"].items() if "process" in spec.get("entry", "")
    )
    examples = golden["tasks"].get(task_name, [])
    assert len(examples) == 5
    assert all(e.get("seed") == "auto" for e in examples)

    run_result = runner.invoke(app, ["-C", str(tmp_path), "--format", "json", "run"])
    assert run_result.exit_code == 0, f"run failed: {run_result.stderr or run_result.output}"
    assert run_result.stdout, "run produced empty stdout"
    payload = json.loads(run_result.stdout)
    assert payload["summary"]["examples"] >= 1, payload
    assert task_name in payload["tasks"]


# ---------------------------------------------------------------------------
# Edge-case fixtures (plan §8)
# ---------------------------------------------------------------------------


def test_shared_helper_called_by_two_endpoints_both_promoted(tmp_path: Path) -> None:
    """Two endpoints (each with its own detected framework call) also call a
    shared retrieval helper → helper demoted, both endpoints promoted as roots."""
    (tmp_path / "shared").mkdir()
    (tmp_path / "shared" / "retrieval.py").write_text(
        "from langchain_community.vectorstores import FAISS\n"
        "def retrieve(query):\n"
        "    return FAISS.load_local('i').similarity_search(query)\n",
        encoding="utf-8",
    )
    (tmp_path / "api").mkdir()
    (tmp_path / "api" / "a.py").write_text(
        "from openai import OpenAI\n"
        "from shared.retrieval import retrieve\n"
        "def endpoint_a(query):\n"
        "    docs = retrieve(query)\n"
        "    return OpenAI().chat.completions.create(messages=[{'role':'user','content':docs}])\n",
        encoding="utf-8",
    )
    (tmp_path / "api" / "b.py").write_text(
        "from openai import OpenAI\n"
        "from shared.retrieval import retrieve\n"
        "def endpoint_b(query):\n"
        "    docs = retrieve(query)\n"
        "    return OpenAI().chat.completions.create(messages=[{'role':'user','content':docs}])\n",
        encoding="utf-8",
    )
    scan = scan_repo(tmp_path)
    rubrics = build_rubrics(scan, project_root=tmp_path)
    entries = {spec.entry for spec in rubrics.tasks.values()}
    assert "endpoint_a" in entries
    assert "endpoint_b" in entries
    assert "retrieve" not in entries  # demoted (reached by both endpoints)


def test_framework_object_entry_is_root(tmp_path: Path) -> None:
    """A LangChain chain ``.invoke()`` call site with nothing reaching it is a
    root (promoted as a task) — no other detected site calls into it."""
    (tmp_path / "app.py").write_text(
        "from langchain_openai import ChatOpenAI\n"
        "def ask(query):\n"
        "    return ChatOpenAI().invoke(query)\n",
        encoding="utf-8",
    )
    scan = scan_repo(tmp_path)
    rubrics = build_rubrics(scan, project_root=tmp_path)
    entries = {spec.entry for spec in rubrics.tasks.values()}
    assert "ask" in entries


def test_judge_harness_excluded(tmp_path: Path) -> None:
    """A RAGAS-style judge harness under evaluations/ returning a
    faithfulness score is excluded (J2∧J3)."""
    (tmp_path / "evaluations").mkdir()
    (tmp_path / "evaluations" / "judge.py").write_text(
        "import ragas\n"
        "def evaluate(answer, contexts):\n"
        "    return ragas.evaluate(answer, contexts=contexts)\n",
        encoding="utf-8",
    )
    (tmp_path / "app.py").write_text(
        "from openai import OpenAI\n"
        "def ask(q):\n"
        "    return OpenAI().chat.completions.create(messages=[q])\n",
        encoding="utf-8",
    )
    scan = scan_repo(tmp_path)
    rubrics = build_rubrics(scan, project_root=tmp_path)
    entries = {spec.entry for spec in rubrics.tasks.values()}
    assert "ask" in entries
    assert "evaluate" not in entries


def test_async_method_entry_resolvable(tmp_path: Path) -> None:
    """An ``async def`` method entry is detected with a dotted entry and
    survives selection (it's a root)."""
    (tmp_path / "svc.py").write_text(
        "from openai import AsyncOpenAI\n"
        "class Svc:\n"
        "    def __init__(self):\n"
        "        self.client = AsyncOpenAI()\n"
        "    async def astream(self, q):\n"
        "        return await self.client.chat.completions.create(messages=[q])\n",
        encoding="utf-8",
    )
    scan = scan_repo(tmp_path)
    rubrics = build_rubrics(scan, project_root=tmp_path)
    entries = {spec.entry for spec in rubrics.tasks.values()}
    assert "Svc.astream" in entries


def test_orphan_no_root_repo_clean_exit(tmp_path: Path) -> None:
    """A pure-library repo with no detected framework calls produces an empty
    rubrics config and classifies to ``custom`` — clean exit, no crash."""
    (tmp_path / "lib.py").write_text(
        "def add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )
    scan = scan_repo(tmp_path)
    rubrics = build_rubrics(scan, project_root=tmp_path)
    assert rubrics.tasks == {}
    assert rubrics.project_type == "custom"
