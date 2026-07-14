"""Tests for the project-local metric registry (``eval/metrics.yaml``).

Covers the registry split (AGENTS.md §2): niche metrics removed from built-ins
are re-declarable in a project's ``eval/metrics.yaml``, loaded and merged at
runtime by :mod:`ai_eval.metrics.local`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_eval.metrics import registry
from ai_eval.metrics.local import load_local_judge_metrics, load_local_metrics


@pytest.fixture(autouse=True)
def _reset():
    registry.reset_cache()
    yield
    registry.reset_cache()


def _write_metrics_yaml(project_root: Path, body: str) -> None:
    eval_dir = project_root / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "metrics.yaml").write_text(body, encoding="utf-8")


def test_local_metrics_yaml_loads(tmp_path: Path) -> None:
    _write_metrics_yaml(
        tmp_path,
        """
metrics:
  - name: slot_filling_accuracy
    description: Accuracy of slots filled for booking.
    applicable_task_types: [booking, workflow]
    scored_dimension: slot_filling_accuracy
    threshold: 0.9
    weight: 1.0
    prompt_template: |
      Evaluate slot filling for {{input}} vs expected {{expected}}.
      Output: {{output}}.
""",
    )
    metrics = load_local_metrics(tmp_path)
    assert any(m.name == "slot_filling_accuracy" for m in metrics)
    m = metrics[0]
    assert m.default_threshold == 0.9
    assert m.applicable_task_types == ("booking", "workflow")


def test_local_judge_metric_resolves_via_registry(tmp_path: Path) -> None:
    _write_metrics_yaml(
        tmp_path,
        """
metrics:
  - name: slot_filling_accuracy
    description: Accuracy of slots filled for booking.
    scored_dimension: slot_filling_accuracy
    prompt_template: |
      Evaluate slot filling for {{input}}.
""",
    )
    jm = registry.get_judge_metric("slot_filling_accuracy", project_root=tmp_path)
    assert jm is not None
    assert jm.scored_dimension == "slot_filling_accuracy"
    assert jm.prompt_builder is not None


def test_local_metric_in_all_names(tmp_path: Path) -> None:
    _write_metrics_yaml(
        tmp_path,
        """
metrics:
  - name: my_custom_metric
    description: A custom metric.
    prompt_template: |
      Score {{output}}.
""",
    )
    names = registry.all_names(tmp_path)
    assert "my_custom_metric" in names


def test_builtin_niche_metrics_removed(tmp_path: Path) -> None:
    """Without a local file, niche metrics are NOT registered."""
    for niche in (
        "slot_filling_accuracy",
        "translation_bleu",
        "classification_f1",
        "extraction_field_accuracy",
        "scoring_accuracy",
        "summary_faithfulness",
    ):
        assert not registry.is_registered(niche, project_root=tmp_path)


def test_local_metric_cannot_override_builtin(tmp_path: Path) -> None:
    _write_metrics_yaml(
        tmp_path,
        """
metrics:
  - name: faithfulness
    description: impostor
    prompt_template: |
      impostor {{output}}.
""",
    )
    m = registry.get("faithfulness", project_root=tmp_path)
    assert m is not None
    assert m.description != "impostor"


def test_local_prompt_builder_substitutes_placeholders(tmp_path: Path) -> None:
    _write_metrics_yaml(
        tmp_path,
        """
metrics:
  - name: my_metric
    description: test metric
    scored_dimension: my_metric
    prompt_template: |
      input: {{input}}
      output: {{output}}
      expected: {{expected}}
""",
    )
    jm = registry.get_judge_metric("my_metric", project_root=tmp_path)
    assert jm is not None and jm.prompt_builder is not None

    class _T:
        type = "chat"
        purpose = "test purpose"

    example = {"input": {"q": 1}, "expected": "ref"}
    msgs = jm.prompt_builder(_T(), example, "out")
    assert isinstance(msgs, list) and len(msgs) == 2
    assert msgs[0]["role"] == "system"
    content = msgs[1]["content"]
    assert json.dumps({"q": 1}, default=str) in content
    assert '"out"' in content
    assert "ref" in content


def test_local_metric_runs_in_engine(tmp_path: Path) -> None:
    """A rubrics task using a local-only metric produces a non-error result
    via a fake complete_fn."""
    import asyncio
    import time

    from ai_eval.config.schema import (
        JudgeConfig,
        MetricSpec,
        RubricsConfig,
        TaskSpec,
    )
    from ai_eval.runner.engine import execute

    _write_metrics_yaml(
        tmp_path,
        """
metrics:
  - name: my_local_metric
    description: A local metric.
    scored_dimension: my_local_metric
    threshold: 0.5
    prompt_template: |
      Score the output: {{output}}
""",
    )

    rubrics = RubricsConfig(
        schema_version=1,
        project_type="chat",
        judge=JudgeConfig(default="fake/local"),
        tasks={
            "t1": TaskSpec(
                file_path="t.py",
                entry="t1",
                type="chat",
                metrics=[MetricSpec(name="my_local_metric", threshold=0.5)],
            ),
        },
    )
    golden = {"t1": [{"id": "e1", "input": "", "seed": "auto"}]}

    (tmp_path / "t.py").write_text("def t1(q=''): return 'hello'\n", encoding="utf-8")

    async def fake_complete(*, model, messages, response_model, temperature=0.0):
        return response_model.model_validate(
            {"score": 0.8, "rationale": "ok", "sub_scores": {"my_local_metric": 0.8}}
        )

    record = asyncio.run(
        execute(
            rubrics,
            golden,
            project_root=tmp_path,
            parallel=1,
            complete_fn=fake_complete,
            run_id="r",
            started_at=time.time(),
        )
    )
    task = record.tasks["t1"]
    assert all(e.status != "error" for e in task.examples), task.errors
    assert "my_local_metric" in task.examples[0].metric_scores


def test_malformed_local_entry_is_skipped(tmp_path: Path) -> None:
    _write_metrics_yaml(
        tmp_path,
        """
metrics:
  - description: missing name
    prompt_template: |
      x
  - name: good_metric
    description: ok
    prompt_template: |
      Score {{output}}.
""",
    )
    metrics = load_local_metrics(tmp_path)
    names = {m.name for m in metrics}
    assert "good_metric" in names
    assert len(metrics) == 1


def test_missing_local_file_returns_empty(tmp_path: Path) -> None:
    assert load_local_metrics(tmp_path) == ()
    assert load_local_judge_metrics(tmp_path) == ()
