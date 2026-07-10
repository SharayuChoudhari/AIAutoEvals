"""Shared rubric-engine selection logic for `init` and `analyze`.

Both commands expose the same flag surface (`--rubric-engine`, `--rubric-model`,
and the cap flags) and the same branch logic. Centralizing it here keeps the two
entry points identical and makes the `rules`/`slm`/`hybrid` tri-state easy to
reason about.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import typer

from ai_eval.config.schema import RubricsConfig
from ai_eval.inference.ast_scan import ScanResult
from ai_eval.inference.slm.builder import BuildStats, EvidenceCaps, build_rubrics_slm
from ai_eval.inference.slm.cache import ResponseCache
from ai_eval.inference.synthesize import build_rubrics

VALID_ENGINES = ("rules", "slm", "hybrid")


@dataclass
class EngineResult:
    rubrics: RubricsConfig
    stats: BuildStats | None = None  # None when engine == rules


def _err(message: str, *, what: str, why: str, fix: str) -> None:
    typer.echo(f"error: {message}", err=True)
    typer.echo(f"  what: {what}", err=True)
    typer.echo(f"  why:  {why}", err=True)
    typer.echo(f"  fix:  {fix}", err=True)


def make_caps(
    max_snippet_chars: int,
    max_tasks: int,
    budget_tokens: int,
) -> EvidenceCaps:
    return EvidenceCaps(
        max_snippet_chars=max_snippet_chars,
        max_tasks=max_tasks,
        budget_tokens=budget_tokens,
    )


def build_with_engine(
    *,
    engine: str,
    scan: ScanResult,
    project_root: Path,
    model: str,
    judge_default: str,
    judge_regression: str,
    caps: EvidenceCaps,
    cache_enabled: bool = True,
    complete_fn=None,
) -> EngineResult:
    """Run the selected rubric engine and return a ``RubricsConfig``.

    - ``rules``: existing :func:`build_rubrics` (no SLM, no ollama).
    - ``slm``: skip rule classification; still uses the AST scan's
      ``DetectedTask`` list as evidence (the SLM owns type/metrics). Falls back
      to the empty-evidence path when the scan found nothing.
    - ``hybrid``: run rules (populating ``scan.tasks`` and a per-task type
      hint), then hand that evidence + the rule type hint to the SLM for
      classification. The SLM still owns the final type but sees the rule's
      opinion as prior evidence.
    """
    if engine == "rules":
        rubrics = build_rubrics(
            scan, judge_default=judge_default, judge_regression=judge_regression
        )
        rubrics.rubric_engine = "rules"
        return EngineResult(rubrics=rubrics)

    # hybrid runs the rule engine first and feeds its per-task classified type
    # to the SLM as prior evidence (the SLM still owns the final type). slm
    # skips rule classification entirely — the SLM classifies from AST evidence
    # alone. Both feed scan.tasks to the SLM; the difference is the rule hints.
    rule_hints: dict[str, str] | None = None
    if engine == "hybrid":
        rule_rubrics = build_rubrics(
            scan, judge_default=judge_default, judge_regression=judge_regression
        )
        rule_hints = {name: spec.type for name, spec in rule_rubrics.tasks.items()}

    cache = ResponseCache(project_root, enabled=cache_enabled)
    rubrics, stats, _report = build_rubrics_slm(
        scan,
        project_root=project_root,
        model=model,
        judge_default=judge_default,
        judge_regression=judge_regression,
        caps=caps,
        complete_fn=complete_fn,
        cache=cache,
        rule_hints=rule_hints,
    )
    rubrics.rubric_engine = engine
    return EngineResult(rubrics=rubrics, stats=stats)


def fail_engine_error(exc: Exception) -> None:
    """Render a :class:`RubricEngineError` with what/why/fix, then exit 1."""
    from ai_eval.inference.slm.builder import RubricEngineError

    if isinstance(exc, RubricEngineError):
        _err(
            exc.message,
            what="SLM rubric engine unavailable",
            why=exc.why or "the configured model could not be reached",
            fix=exc.fix or "use --rubric-engine rules to skip the SLM",
        )
    else:
        _err(
            "rubric engine failed",
            what=str(exc),
            why="unexpected error during SLM rubric generation",
            fix="re-run with --rubric-engine rules, or inspect the error above",
        )
    raise typer.Exit(code=1)


__all__ = [
    "VALID_ENGINES",
    "EngineResult",
    "build_with_engine",
    "fail_engine_error",
    "make_caps",
]
