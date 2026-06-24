"""Color + icon palette per the design doc §1.5.

Never communicate state with color alone — icon + label are always present.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class State:
    icon: str
    color: str       # rich style name
    label: str


PASS = State(icon="\u2713", color="green", label="pass")           # ✓
FAIL = State(icon="\u2717", color="red", label="fail")             # ✗
WARN = State(icon="!", color="yellow", label="warn")
INFO = State(icon="-", color="bright_black", label="info")
IDENT = State(icon=" ", color="cyan", label="id")
LINK = State(icon=" ", color="blue underline", label="link")


def styled(text: str, *, color: str, no_color: bool) -> str:
    """Return a rich-markup string honoring the global `--no-color` switch."""
    if no_color:
        return text
    return f"[{color}]{text}[/{color}]"


def state_glyph(state: State, *, no_color: bool) -> str:
    """`✓ pass` / `✗ fail` etc — icon + label, optionally colored."""
    text = f"{state.icon} {state.label}"
    return styled(text, color=state.color, no_color=no_color)


__all__ = [
    "FAIL",
    "IDENT",
    "INFO",
    "LINK",
    "PASS",
    "State",
    "WARN",
    "state_glyph",
    "styled",
]
