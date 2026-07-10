"""Tests for the metric registry and the MetricSpec schema validator.

Covers plan task H.18:
- built-in registry contents (the 4 legacy + new metrics);
- entry-point plugin loading (mirrors `load_entrypoint_detectors`);
- `MetricSpec` validation rejects unknown metric names under strict mode and
  warns (one-release deprecation) otherwise.
"""

from __future__ import annotations

import warnings

import pytest

from ai_eval.config.schema import MetricSpec
from ai_eval.metrics import registry


def _patch_entry_points(monkeypatch: pytest.MonkeyPatch, eps):
    """Patch ``importlib.metadata.entry_points`` to return ``eps``.

    ``registry.load_metrics`` imports ``entry_points`` lazily inside its body
    via ``from importlib.metadata import entry_points``, so we patch the
    attribute on the importlib.metadata module itself. ``eps`` is a list of
    fake entry-point objects with ``.name`` and ``.load()``.

    The registry memoizes its result; the autouse ``_reset_metrics_cache``
    fixture clears it before/after each test so patched entry points are seen.
    """
    import importlib.metadata as md

    def fake(group=None):
        return eps

    monkeypatch.setattr(md, "entry_points", fake, raising=True)
    registry.reset_cache()


@pytest.fixture(autouse=True)
def _reset_metrics_cache():
    """Clear the memoized metric set around every test in this module."""
    registry.reset_cache()
    yield
    registry.reset_cache()


# ---------------------------------------------------------------------------
# Built-in registry contents
# ---------------------------------------------------------------------------

def test_builtin_metrics_include_legacy_four() -> None:
    names = {m.name for m in registry.BUILTIN_METRICS}
    # The 4 metrics the rule engine already emits must stay registered so both
    # engines validate identically.
    assert {
        "hallucination_rate",
        "context_precision",
        "faithfulness",
        "argument_accuracy",
    } <= names


def test_builtin_metrics_include_new_task_types() -> None:
    names = {m.name for m in registry.BUILTIN_METRICS}
    # New metrics for the open task_type vocabulary.
    assert {
        "scoring_accuracy",
        "extraction_field_accuracy",
        "classification_f1",
        "summary_faithfulness",
        "translation_bleu",
        "task_completion",
        "slot_filling_accuracy",
    } <= names


def test_all_names_includes_builtins_and_judge_metrics() -> None:
    # No plugins installed in the test env → all_names == legacy builtins +
    # the judge metrics + latency metrics.
    legacy = {m.name for m in registry.BUILTIN_METRICS}
    judge = {m.name for m in registry.load_judge_metrics()}
    assert set(registry.all_names()) == legacy | judge
    # sanity: the judge metrics we added are present
    assert {"latency_p50", "latency_p95"} <= set(registry.all_names())


def test_get_returns_metric_or_none() -> None:
    m = registry.get("faithfulness")
    assert m is not None
    assert m.name == "faithfulness"
    assert m.default_threshold == 0.9
    assert registry.get("does_not_exist") is None


def test_is_registered() -> None:
    assert registry.is_registered("context_precision")
    assert not registry.is_registered("totally_made_up_metric")


def test_builtin_metrics_have_unique_names() -> None:
    names = [m.name for m in registry.BUILTIN_METRICS]
    assert len(names) == len(set(names)), "duplicate builtin metric names"


# ---------------------------------------------------------------------------
# Entry-point plugin loading
# ---------------------------------------------------------------------------

class _FakeEP:
    """Minimal stand-in for importlib.metadata.EntryPoint."""

    def __init__(self, name: str, obj) -> None:
        self.name = name
        self._obj = obj

    def load(self):
        return self._obj


def test_plugin_metric_object_is_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A plugin exposing a Metric dataclass is appended."""
    fake = registry.Metric(
        name="my_plugin_metric",
        description="from a plugin",
        default_threshold=0.5,
    )
    _patch_entry_points(monkeypatch, [_FakeEP("my_plugin_metric", fake)])

    metrics = registry.load_metrics()
    assert any(m.name == "my_plugin_metric" for m in metrics)
    plugin = registry.get("my_plugin_metric")
    assert plugin is not None and plugin.default_threshold == 0.5


def test_plugin_dict_is_coerced(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = {
        "name": "dict_metric",
        "description": "dict plugin",
        "default_threshold": 0.3,
        "default_weight": 2.0,
        "applicable_task_types": ["chat"],
    }
    _patch_entry_points(monkeypatch, [_FakeEP("dict_metric", fake)])
    m = registry.get("dict_metric")
    assert m is not None
    assert m.default_threshold == 0.3
    assert m.default_weight == 2.0
    assert m.applicable_task_types == ("chat",)


def test_plugin_string_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare string plugin value registers a name with placeholder metadata."""
    _patch_entry_points(monkeypatch, [_FakeEP("string_metric", "string_metric")])
    assert registry.is_registered("string_metric")


def test_plugin_failure_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A plugin that raises on load is silently skipped (never crashes)."""

    class _Boom:
        name = "bad"

        def load(self):
            raise RuntimeError("boom")

    _patch_entry_points(monkeypatch, [_Boom()])
    metrics = registry.load_metrics()
    assert all(m.name != "bad" for m in metrics)


def test_plugin_does_not_shadow_builtin(monkeypatch: pytest.MonkeyPatch) -> None:
    """A plugin re-declaring a builtin name is ignored — builtins win."""
    impostor = registry.Metric(
        name="faithfulness", description="impostor", default_threshold=0.0
    )
    _patch_entry_points(monkeypatch, [_FakeEP("faithfulness", impostor)])
    m = registry.get("faithfulness")
    assert m is not None
    assert m.description != "impostor"


# ---------------------------------------------------------------------------
# MetricSpec schema validation
# ---------------------------------------------------------------------------

def test_metricspec_accepts_registered_name() -> None:
    # Should not raise.
    spec = MetricSpec(name="faithfulness", threshold=0.9)
    assert spec.name == "faithfulness"


def test_metricspec_warns_on_unknown_name_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AI_EVAL_STRICT_METRICS", raising=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spec = MetricSpec(name="unregistered_metric_xyz")
    assert spec.name == "unregistered_metric_xyz"
    assert any(
        issubclass(w.category, DeprecationWarning)
        and "unregistered_metric_xyz" in str(w.message)
        for w in caught
    ), "expected a DeprecationWarning for the unknown metric"


def test_metricspec_rejects_unknown_name_in_strict_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_EVAL_STRICT_METRICS", "1")
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as excinfo:
        MetricSpec(name="unregistered_metric_xyz")
    assert "not registered" in str(excinfo.value)

