"""Tests for judge tier selection."""

from __future__ import annotations

from ai_eval.judge.tiering import select_tier


def test_gpt4_is_complex() -> None:
    assert select_tier("openai/gpt-4o-mini") == "complex"


def test_claude_sonnet_is_complex() -> None:
    assert select_tier("anthropic/claude-3-sonnet") == "complex"


def test_local_ollama_is_basic() -> None:
    assert select_tier("ollama/qwen2.5-coder:7b") == "basic"


def test_o1_is_complex() -> None:
    assert select_tier("openai/o1-mini") == "complex"


def test_override_complex_models_replaces_builtins() -> None:
    # Built-in hints ignored when an override is provided
    assert select_tier("openai/gpt-4o", complex_models=()) == "basic"
    assert select_tier("my/local-model", complex_models=("my/local",)) == "complex"


def test_case_insensitive() -> None:
    assert select_tier("OpenAI/GPT-4O") == "complex"


def test_empty_hint_string_ignored() -> None:
    assert select_tier("anything", complex_models=("", "  ")) == "basic"
