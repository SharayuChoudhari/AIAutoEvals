"""Every command must respond to --help with exit 0 and a non-empty body."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from ai_eval.cli.app import app

ALL_COMMANDS = [
    "init",
    "analyze",
    "bootstrap",
    "run",
    "diff",
    "report",
    "judge",
    "doctor",
    "config",
    "history",
    "version",
]


def test_root_help(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    for cmd in ALL_COMMANDS:
        assert cmd in result.output, f"command `{cmd}` missing from root --help"


@pytest.mark.parametrize("command", ALL_COMMANDS)
def test_subcommand_help(runner: CliRunner, command: str) -> None:
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0, f"--help for `{command}` failed: {result.output}"
    assert "Usage:" in result.output
