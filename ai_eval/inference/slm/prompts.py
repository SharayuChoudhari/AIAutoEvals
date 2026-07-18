"""Loader for the bundled SLM prompt templates.

Templates live under ``ai_eval/inference/prompts/`` and are versioned with the
code (not user-editable). ``render`` substitutes the named placeholders; unknown
placeholders are left intact so prompt authors can spot typos by inspection.
"""

from __future__ import annotations

import re
from functools import cache
from importlib import resources

import yaml

_PROMPTS_PACKAGE = "ai_eval.inference.prompts"

_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


@cache
def _load(name: str) -> str:
    """Return the raw template text for ``name`` (e.g. ``classify_task``)."""
    return resources.files(_PROMPTS_PACKAGE).joinpath(f"{name}.txt").read_text(encoding="utf-8")


@cache
def _few_shot_examples() -> list[dict]:
    """Return the parsed few-shot examples from ``few_shot.yaml``."""
    raw = resources.files(_PROMPTS_PACKAGE).joinpath("few_shot.yaml").read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    return data if isinstance(data, list) else []


def render_few_shot() -> str:
    """Render the few-shot examples as a compact text block for prompts."""
    lines: list[str] = []
    for ex in _few_shot_examples():
        lines.append(f"## {ex.get('example', '?')}")
        lines.append(ex.get("context", "").rstrip())
        exp = ex.get("expected", {})
        lines.append("expected:")
        lines.append(f"  purpose: {exp.get('purpose', '')}")
        lines.append(f"  type: {exp.get('type', '')}")
        lines.append(f"  inputs: {exp.get('inputs', [])}")
        lines.append(f"  outputs: {exp.get('outputs', [])}")
        lines.append(f"  metrics: {exp.get('metrics', [])}")
        lines.append("")
    return "\n".join(lines)


def render(name: str, **fields: str) -> str:
    """Render template ``name`` substituting only the given ``fields``.

    Placeholders for which no value is supplied are left as ``{name}`` so the
    template author can spot missing substitutions.
    """
    template = _load(name)
    return _PLACEHOLDER_RE.sub(lambda m: str(fields.get(m.group(1), m.group(0))), template)


__all__ = ["render", "render_few_shot"]
