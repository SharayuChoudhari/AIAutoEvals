"""Write the pytest entry point that calls `ai-evals run`."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def write(path: Path) -> None:
    """Write `eval/tests.py` from the bundled template."""
    template = files("ai_eval.scaffold.templates").joinpath("tests_py.tmpl").read_text(
        encoding="utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template, encoding="utf-8")


__all__ = ["write"]
