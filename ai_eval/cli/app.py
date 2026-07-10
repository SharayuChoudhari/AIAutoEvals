"""Typer root app + global flags + entry point.

Design contract:
- `ai-evals <command> [flags]`, max 2 levels of nesting.
- Global flags resolved here and stashed in a `GlobalOptions` object on the Typer
  context so subcommands can read them.
- Exit codes:
    0  success
    1  general / runtime error
    2  usage error (bad flags, `--no-input` hit a prompt, command not implemented)
    3  evaluation failed thresholds (Phase 4 `run --fail-on-regression`)
- TTY detection drives `--format` auto: human on TTY, tsv on pipe.
- CI auto-mode: when env `CI=true`, the app behaves as if invoked with
  `--no-input --format json --no-color` unless the user passes overrides.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import typer

from ai_eval import __version__
from ai_eval.telemetry.logger import configure as configure_logging


class OutputFormat(str, Enum):
    AUTO = "auto"
    HUMAN = "human"
    JSON = "json"
    TSV = "tsv"


# Exit code constants, also exported as a module attribute for tests.
EXIT_OK = 0
EXIT_GENERAL = 1
EXIT_USAGE = 2
EXIT_REGRESSION = 3


@dataclass
class GlobalOptions:
    """Resolved global state, stashed on Typer context.obj."""

    cwd: Path
    config_path: Path | None
    format: OutputFormat
    no_color: bool
    quiet: bool
    verbose: int
    assume_yes: bool
    no_input: bool
    is_tty_stdout: bool
    is_tty_stderr: bool
    is_ci: bool

    @property
    def effective_format(self) -> OutputFormat:
        """Resolve `auto` based on TTY / CI state."""
        if self.format != OutputFormat.AUTO:
            return self.format
        if self.is_ci:
            return OutputFormat.JSON
        if not self.is_tty_stdout:
            return OutputFormat.TSV
        return OutputFormat.HUMAN


def _detect_ci() -> bool:
    """Return True when running in a CI environment. Conservative: only `CI=true`."""
    return os.environ.get("CI", "").lower() == "true"


app = typer.Typer(
    name="ai-evals",
    help="ai-evals — zero-config AI evaluation for your repo.",
    add_completion=False,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    rich_markup_mode="rich",
)


def _version_callback(value: bool) -> None:
    if value:
        py = f"python {sys.version_info.major}.{sys.version_info.minor}"
        typer.echo(f"ai-evals {__version__} ({py})")
        raise typer.Exit(code=EXIT_OK)


@app.callback()
def _root(
    ctx: typer.Context,
    cwd: Path = typer.Option(
        Path.cwd(),
        "-C",
        "--cwd",
        help="Run as if invoked from <dir>.",
        show_default=False,
    ),
    config_path: Path | None = typer.Option(
        None,
        "--config",
        help="Path to rubrics.yaml (default: eval/rubrics.yaml under --cwd).",
        show_default=False,
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.AUTO,
        "--format",
        help="Output format. auto: human on TTY, tsv on pipe, json in CI.",
        case_sensitive=False,
    ),
    no_color: bool = typer.Option(
        False,
        "--no-color",
        help="Disable ANSI colors.",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress progress; errors still print.",
    ),
    verbose: int = typer.Option(
        0,
        "-v",
        "--verbose",
        count=True,
        help="Verbose logs (repeatable: -vv for debug).",
    ),
    assume_yes: bool = typer.Option(
        False,
        "-y",
        "--yes",
        help="Assume 'yes' on all confirmations.",
    ),
    no_input: bool = typer.Option(
        False,
        "--no-input",
        help="Fail instead of prompting.",
    ),
    _version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    is_ci = _detect_ci()

    # CI auto-mode (user decision #6): when CI=true, default to non-interactive +
    # JSON + no-color unless the user explicitly overrode.
    if is_ci:
        if not no_input:
            no_input = True
        if not no_color:
            no_color = True
        # `format` already picks JSON in `effective_format` when is_ci=True.

    is_tty_stdout = sys.stdout.isatty()
    is_tty_stderr = sys.stderr.isatty()

    # If output is piped or JSON-requested, color is meaningless.
    if not is_tty_stdout and output_format != OutputFormat.HUMAN:
        no_color = True
    if output_format == OutputFormat.JSON:
        no_color = True

    configure_logging(verbosity=verbose, quiet=quiet)

    ctx.obj = GlobalOptions(
        cwd=cwd.resolve(),
        config_path=config_path.resolve() if config_path is not None else None,
        format=output_format,
        no_color=no_color,
        quiet=quiet,
        verbose=verbose,
        assume_yes=assume_yes,
        no_input=no_input,
        is_tty_stdout=is_tty_stdout,
        is_tty_stderr=is_tty_stderr,
        is_ci=is_ci,
    )


# Per-command registration. Imports happen here to keep startup fast and avoid
# circular imports between cli modules.
from ai_eval.cli import (
    analyze as _analyze,
)
from ai_eval.cli import bootstrap as _bootstrap
from ai_eval.cli import (
    config_cmd as _config_cmd,
)
from ai_eval.cli import (
    diff as _diff,
)
from ai_eval.cli import (
    doctor as _doctor,
)
from ai_eval.cli import (
    history as _history,
)
from ai_eval.cli import (
    init as _init,
)
from ai_eval.cli import (
    judge as _judge,
)
from ai_eval.cli import (
    report as _report,
)
from ai_eval.cli import (
    run as _run,
)
from ai_eval.cli import (
    stubs as _stubs,
)

app.command("init", help="Scan the repo and scaffold eval/ (rubrics + tests).")(
    _init.init_command
)
app.command("analyze", help="Re-run repo inference and refresh rubrics.yaml.")(
    _analyze.analyze_command
)
app.command("bootstrap", help="Capture real runtime traces into a golden set.")(
    _bootstrap.bootstrap_command
)
app.command("run", help="Execute evaluations against the current code.")(
    _run.run_command
)
app.command("diff", help="Show regression diff vs. a baseline run.")(
    _diff.diff_command
)
app.command("report", help="Render a past run as a human or JSON report.")(
    _report.report_command
)
app.command("judge", help="Inspect / test the configured judge models.")(
    _judge.judge_command
)
app.command("doctor", help="Verify environment, providers, and config.")(
    _doctor.doctor_command
)
app.command("config", help="Print or edit resolved configuration.")(
    _config_cmd.config_command
)
app.command("history", help="List, prune, or export past runs.")(
    _history.history_command
)
app.command("version", help="Print the version and exit.")(_stubs.version_command)


def main() -> None:
    """Console-script entry point.

    Relies on Click's standalone mode for exit-code propagation: `typer.Exit`
    raised by command bodies is converted to `sys.exit(exit_code)` automatically.
    """
    app()


__all__ = [
    "EXIT_GENERAL",
    "EXIT_OK",
    "EXIT_REGRESSION",
    "EXIT_USAGE",
    "GlobalOptions",
    "OutputFormat",
    "app",
    "main",
]
