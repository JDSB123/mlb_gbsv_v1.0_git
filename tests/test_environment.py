"""Tests for runtime environment bootstrap behavior."""

from __future__ import annotations

import os
from pathlib import Path

from mlbv1.environment import bootstrap_environment, should_load_dotenv


def test_bootstrap_skips_dotenv_in_test_context(
    monkeypatch, tmp_path: Path
) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("ODDS_API_KEY=from-dotenv\n", encoding="utf-8")

    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_environment.py::test")

    state = bootstrap_environment(dotenv_path=dotenv_path)

    assert state.environment == "test"
    assert state.loaded_dotenv is False
    assert os.getenv("ODDS_API_KEY", "") == ""


def test_bootstrap_loads_dotenv_for_local_environment(
    monkeypatch, tmp_path: Path
) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("ODDS_API_KEY=from-dotenv\n", encoding="utf-8")

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("ODDS_API_KEY", raising=False)

    state = bootstrap_environment(environment="local", dotenv_path=dotenv_path)

    assert state.loaded_dotenv is True
    assert state.environment == "local"
    assert state.dotenv_path == dotenv_path
    assert os.getenv("ODDS_API_KEY", "") == "from-dotenv"


def test_should_load_dotenv_respects_explicit_override(monkeypatch) -> None:
    monkeypatch.setenv("MLBV1_LOAD_DOTENV", "false")
    assert should_load_dotenv("local") is False
