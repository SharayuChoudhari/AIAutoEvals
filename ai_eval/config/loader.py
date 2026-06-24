"""Merged-config loader.

Resolution order (highest wins):
  1. CLI flag (passed in as `cli_overrides` dict)
  2. Env var `AI_EVAL_<UPPER_SNAKE>`
  3. eval/rubrics.yaml (in the project)
  4. ~/.config/ai-eval/config.yaml (user-global)
  5. Built-in defaults from `ai_eval.config.defaults`

`load_resolved()` returns the merged view annotated with source per key, which
`ai-eval config --print` renders.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ai_eval.config.defaults import builtin_defaults
from ai_eval.config.schema import RubricsConfig


@dataclass
class ResolvedConfig:
    """The merged config plus per-key source provenance for `config --print`."""

    data: dict[str, Any]
    sources: dict[str, str] = field(default_factory=dict)
    rubrics_path: Path | None = None
    user_config_path: Path | None = None

    def as_rubrics(self) -> RubricsConfig:
        """Validate the merged dict against the Pydantic schema."""
        return RubricsConfig.model_validate(self.data)


def user_config_path() -> Path:
    """Location of the optional user-global config."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "ai-eval" / "config.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        raise ConfigLoadError(
            f"failed to parse YAML at {path}: {exc}",
            path=path,
        ) from exc
    if not isinstance(data, dict):
        raise ConfigLoadError(
            f"top-level YAML at {path} must be a mapping, got {type(data).__name__}",
            path=path,
        )
    return data


def _env_overrides() -> dict[str, Any]:
    """Lift `AI_EVAL_FOO_BAR=value` into nested dict `{foo: {bar: value}}`.

    Phase 1 supports a small allowlist of dotted keys; arbitrary nesting is rejected
    to avoid surprises.
    """
    allowlist = {
        "AI_EVAL_JUDGE_DEFAULT": ("judge", "default"),
        "AI_EVAL_JUDGE_REGRESSION_CHECK": ("judge", "regression_check"),
        "AI_EVAL_PARALLEL": ("defaults", "parallel"),
        "AI_EVAL_TOLERANCE": ("defaults", "tolerance"),
        "AI_EVAL_CACHE": ("defaults", "cache"),
    }
    out: dict[str, Any] = {}
    for env_key, path in allowlist.items():
        if env_key not in os.environ:
            continue
        raw = os.environ[env_key]
        value: Any = raw
        if path[-1] in {"parallel"}:
            try:
                value = int(raw)
            except ValueError:
                continue
        elif path[-1] in {"tolerance"}:
            try:
                value = float(raw)
            except ValueError:
                continue
        elif path[-1] in {"cache"}:
            value = raw.strip().lower() in {"1", "true", "yes", "on"}
        cursor = out
        for key in path[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[path[-1]] = value
    return out


def _deep_merge(
    base: dict[str, Any],
    overlay: dict[str, Any],
    sources: dict[str, str],
    source_label: str,
    prefix: str = "",
) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value, sources, source_label, dotted)
        else:
            out[key] = value
            sources[dotted] = source_label
    return out


class ConfigLoadError(Exception):
    """Raised when on-disk config cannot be loaded or validated."""

    def __init__(self, message: str, *, path: Path | None = None) -> None:
        super().__init__(message)
        self.path = path


def load_resolved(
    *,
    project_root: Path,
    config_path: Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> ResolvedConfig:
    """Load the merged config from all layers.

    Does not raise if rubrics.yaml is missing (that's the case before `init`).
    """
    sources: dict[str, str] = {}
    merged = _deep_merge({}, builtin_defaults(), sources, "builtin")

    user_path = user_config_path()
    if user_path.is_file():
        merged = _deep_merge(merged, _read_yaml(user_path), sources, str(user_path))

    rubrics_path = config_path or (project_root / "eval" / "rubrics.yaml")
    if rubrics_path.is_file():
        merged = _deep_merge(merged, _read_yaml(rubrics_path), sources, str(rubrics_path))

    env_layer = _env_overrides()
    if env_layer:
        merged = _deep_merge(merged, env_layer, sources, "env")

    if cli_overrides:
        merged = _deep_merge(merged, cli_overrides, sources, "cli")

    return ResolvedConfig(
        data=merged,
        sources=sources,
        rubrics_path=rubrics_path if rubrics_path.is_file() else None,
        user_config_path=user_path if user_path.is_file() else None,
    )


__all__ = ["ConfigLoadError", "ResolvedConfig", "load_resolved", "user_config_path"]
