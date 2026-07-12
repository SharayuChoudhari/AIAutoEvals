"""Tests for task selection (plan D1): call-graph demotion + judge-exclusion.

These build repos on disk and run ``build_rubrics`` with ``project_root`` so
the full selection pipeline (call graph → demotion → J1/J2∧J3/J5 exclusion →
hints overrides) is exercised end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ai_eval.inference.ast_scan import ScanResult
from ai_eval.inference.detectors.base import DetectedTask
from ai_eval.inference.synthesize import build_rubrics


def _scan(tasks: list[DetectedTask]) -> ScanResult:
    return ScanResult(
        files_scanned=len({t.file_path for t in tasks}),
        elapsed_seconds=0.0,
        tasks=tasks,
        frameworks_seen={t.framework for t in tasks},
    )


def _task(entry: str | None, file_path: str, **kw) -> DetectedTask:
    return DetectedTask(
        name=kw.get("name", entry or "task"),
        framework=kw.get("framework", "openai"),
        type=kw.get("type", "chat"),
        file_path=file_path,
        entry=entry,
        inputs=kw.get("inputs", []),
        outputs=kw.get("outputs", []),
    )


def test_j1_judge_framework_import_excludes_site(tmp_path: Path) -> None:
    """J1: a site whose module imports ``ragas`` is excluded as judge code."""
    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / "judge.py").write_text(
        "import ragas\n"
        "def score(answer, contexts):\n"
        "    return ragas.evaluate(answer, contexts=contexts)\n",
        encoding="utf-8",
    )
    (tmp_path / "app.py").write_text(
        "from openai import OpenAI\n"
        "def ask(q):\n"
        "    return OpenAI().chat.completions.create(messages=[q])\n",
        encoding="utf-8",
    )
    tasks = [
        _task("ask", "app.py"),
        _task("score", "evals/judge.py"),
    ]
    rubrics = build_rubrics(_scan(tasks), project_root=tmp_path)
    entries = {spec.entry for spec in rubrics.tasks.values()}
    assert "ask" in entries
    assert "score" not in entries  # excluded via J1 (ragas import)


def test_j2_j3_eval_dir_with_score_return_excludes(tmp_path: Path) -> None:
    """J2 AND J3: a file under ``evaluations/`` whose callable returns a
    ``.faithfulness`` field is excluded. Bare ``Result`` suffix is NOT enough."""
    (tmp_path / "evaluations").mkdir()
    (tmp_path / "evaluations" / "single_eval.py").write_text(
        "class EvaluationResult:\n"
        "    def __init__(self, faithfulness):\n"
        "        self.faithfulness = faithfulness\n"
        "def evaluate(query, answer):\n"
        "    return EvaluationResult(0.9)\n",
        encoding="utf-8",
    )
    (tmp_path / "main.py").write_text(
        "from openai import OpenAI\n"
        "def run(q):\n"
        "    return OpenAI().chat.completions.create(messages=[q])\n",
        encoding="utf-8",
    )
    tasks = [
        _task("run", "main.py"),
        _task("evaluate", "evaluations/single_eval.py"),
    ]
    rubrics = build_rubrics(_scan(tasks), project_root=tmp_path)
    entries = {spec.entry for spec in rubrics.tasks.values()}
    assert "run" in entries
    assert "evaluate" not in entries  # J2 (evaluations/) AND J3 (faithfulness)


def test_j2_without_j3_keeps_site(tmp_path: Path) -> None:
    """J2 alone is NOT sufficient — a file under ``eval/`` whose callable does
    NOT return a score-shaped object survives (avoid false negatives)."""
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "runner.py").write_text(
        "from openai import OpenAI\n"
        "def run(q):\n"
        "    return OpenAI().chat.completions.create(messages=[q])\n",
        encoding="utf-8",
    )
    tasks = [_task("run", "eval/runner.py")]
    rubrics = build_rubrics(_scan(tasks), project_root=tmp_path)
    entries = {spec.entry for spec in rubrics.tasks.values()}
    assert "run" in entries  # J2 true but J3 false → kept


def test_bare_result_suffix_not_score_shaped(tmp_path: Path) -> None:
    """A ``SearchResult`` return under ``evaluations/`` is NOT excluded — bare
    ``Result`` suffix is insufficient (would false-positive on SearchResult)."""
    (tmp_path / "evaluations").mkdir()
    (tmp_path / "evaluations" / "search.py").write_text(
        "class SearchResult:\n"
        "    def __init__(self, hits):\n"
        "        self.hits = hits\n"
        "def search(q):\n"
        "    return SearchResult([])\n",
        encoding="utf-8",
    )
    tasks = [_task("search", "evaluations/search.py")]
    rubrics = build_rubrics(_scan(tasks), project_root=tmp_path)
    assert "search" in {spec.entry for spec in rubrics.tasks.values()}


def test_j5_generated_tests_excluded(tmp_path: Path) -> None:
    """J5: a site in ai-evals' generated ``eval/tests.py`` is excluded."""
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "tests.py").write_text(
        "from openai import OpenAI\n"
        "def test_thing():\n"
        "    return OpenAI().chat.completions.create(messages=[])\n",
        encoding="utf-8",
    )
    (tmp_path / "app.py").write_text(
        "from openai import OpenAI\n"
        "def ask(q):\n"
        "    return OpenAI().chat.completions.create(messages=[q])\n",
        encoding="utf-8",
    )
    tasks = [
        _task("ask", "app.py"),
        _task("test_thing", "eval/tests.py"),
    ]
    rubrics = build_rubrics(_scan(tasks), project_root=tmp_path)
    entries = {spec.entry for spec in rubrics.tasks.values()}
    assert "ask" in entries
    assert "test_thing" not in entries


def test_judge_code_hint_glob_excludes(tmp_path: Path) -> None:
    """``judge_code`` hints globs force-exclude matching files even without
    the J1/J2/J3 signals firing."""
    (tmp_path / "custom_judge").mkdir()
    (tmp_path / "custom_judge" / "scorer.py").write_text(
        "def score(q, a):\n"
        "    return 0.5\n",
        encoding="utf-8",
    )
    (tmp_path / "app.py").write_text(
        "from openai import OpenAI\n"
        "def ask(q):\n"
        "    return OpenAI().chat.completions.create(messages=[q])\n",
        encoding="utf-8",
    )
    tasks = [
        _task("ask", "app.py"),
        _task("score", "custom_judge/scorer.py"),
    ]
    rubrics = build_rubrics(
        _scan(tasks),
        project_root=tmp_path,
        judge_code_globs=["custom_judge/*.py"],
    )
    entries = {spec.entry for spec in rubrics.tasks.values()}
    assert "ask" in entries
    assert "score" not in entries


def test_force_task_hint_immunizes_against_exclusion(tmp_path: Path) -> None:
    """A ``force_task`` key matching a J1-excluded site keeps it as a task."""
    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / "judge.py").write_text(
        "import ragas\n"
        "def score(answer, contexts):\n"
        "    return ragas.evaluate(answer, contexts=contexts)\n",
        encoding="utf-8",
    )
    (tmp_path / "app.py").write_text(
        "from openai import OpenAI\n"
        "def ask(q):\n"
        "    return OpenAI().chat.completions.create(messages=[q])\n",
        encoding="utf-8",
    )
    tasks = [
        _task("ask", "app.py"),
        _task("score", "evals/judge.py"),
    ]
    rubrics = build_rubrics(
        _scan(tasks),
        project_root=tmp_path,
        force_task_keys={("evals/judge.py", "score")},
    )
    entries = {spec.entry for spec in rubrics.tasks.values()}
    assert "ask" in entries
    assert "score" in entries  # forced in despite J1


def test_call_graph_demotion_drops_reached_internal_node(tmp_path: Path) -> None:
    """P2 demotion: the DAO (reached by the endpoint via __init__ attr) is
    dropped; the endpoint is the sole task."""
    (tmp_path / "layers").mkdir()
    (tmp_path / "layers" / "dao.py").write_text(
        "class DAO:\n"
        "    def search(self, q):\n"
        "        return q\n",
        encoding="utf-8",
    )
    (tmp_path / "services").mkdir()
    (tmp_path / "services" / "svc.py").write_text(
        "from layers.dao import DAO\n"
        "class Svc:\n"
        "    def __init__(self):\n"
        "        self.dao = DAO()\n"
        "    def process(self, q):\n"
        "        return self.dao.search(q)\n",
        encoding="utf-8",
    )
    tasks = [
        _task("Svc.process", "services/svc.py"),
        _task("DAO.search", "layers/dao.py"),
    ]
    rubrics = build_rubrics(_scan(tasks), project_root=tmp_path)
    entries = {spec.entry for spec in rubrics.tasks.values()}
    assert "Svc.process" in entries
    assert "DAO.search" not in entries  # demoted (reached)


def test_hints_yaml_round_trip_judge_code_and_force_task(tmp_path: Path) -> None:
    """A full hints YAML with both ``judge_code`` and a ``force_task`` task
    parses and threads through to selection. The hint task with force_task
    survives even though it lives under a judge_code glob."""
    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / "judge.py").write_text(
        "def score(a):\n"
        "    return a\n",
        encoding="utf-8",
    )
    (tmp_path / "app.py").write_text(
        "from openai import OpenAI\n"
        "def ask(q):\n"
        "    return OpenAI().chat.completions.create(messages=[q])\n",
        encoding="utf-8",
    )
    hints = tmp_path / "ai-evals.hints.yaml"
    hints.write_text(
        yaml.safe_dump(
            {
                "judge_code": ["evals/*.py"],
                "tasks": [
                    {
                        "name": "forced_score",
                        "file_path": "evals/judge.py",
                        "entry": "score",
                        "type": "chat",
                        "force_task": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    from ai_eval.cli.rubric_engine import _hints_overrides
    from ai_eval.inference.hints import merge_hints

    judge_globs, force_keys = _hints_overrides(hints)
    assert "evals/*.py" in judge_globs
    assert ("evals/judge.py", "score") in force_keys

    tasks = [_task("ask", "app.py")]
    scan = _scan(tasks)
    scan = merge_hints(scan, hints)
    rubrics = build_rubrics(
        scan,
        project_root=tmp_path,
        judge_code_globs=judge_globs,
        force_task_keys=force_keys,
    )
    entries = {spec.entry for spec in rubrics.tasks.values()}
    assert "ask" in entries
    assert "score" in entries  # force_task beat the judge_code glob


def test_empty_repo_no_roots_clean_exit(tmp_path: Path) -> None:
    """A scan with zero surviving tasks produces an empty rubrics config — no
    crash, project_type falls to ``custom``."""
    rubrics = build_rubrics(_scan([]), project_root=tmp_path)
    assert rubrics.tasks == {}
    assert rubrics.project_type == "custom"
