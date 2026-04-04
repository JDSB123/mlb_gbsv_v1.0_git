"""Runtime environment bootstrap and helpers.

This module keeps local development convenience (.env support) while avoiding
implicit cross-environment contamination in tests, CI, and Azure deployments.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

_ENVIRONMENT_VAR = "MLBV1_ENVIRONMENT"
_LOAD_DOTENV_VAR = "MLBV1_LOAD_DOTENV"
_ACA_MARKERS = ("CONTAINER_APP_NAME", "CONTAINER_APP_REVISION_NAME")


@dataclass(frozen=True)
class EnvironmentBootstrap:
    """Metadata describing environment bootstrap behavior."""

    environment: str
    repo_root: Path
    dotenv_path: Path
    loaded_dotenv: bool


_bootstrap_state: EnvironmentBootstrap | None = None


def repository_root() -> Path:
    """Return the repository root based on the source layout."""
    return Path(__file__).resolve().parents[2]


def _parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def detect_environment() -> str:
    """Infer the current runtime environment."""
    explicit = os.getenv(_ENVIRONMENT_VAR, "").strip().lower()
    if explicit:
        return explicit
    if any(os.getenv(marker, "").strip() for marker in _ACA_MARKERS):
        return "aca"
    if os.getenv("GITHUB_ACTIONS", "").strip().lower() == "true":
        return "ci"
    if os.getenv("CI", "").strip():
        return "ci"
    if os.getenv("PYTEST_CURRENT_TEST") or "pytest" in sys.modules:
        return "test"
    return "local"


def should_load_dotenv(environment: str | None = None) -> bool:
    """Decide whether repo-local .env should be loaded."""
    override = _parse_bool(os.getenv(_LOAD_DOTENV_VAR))
    if override is not None:
        return override
    return (environment or detect_environment()) == "local"


def load_dotenv(dotenv_path: str | Path | None = None) -> Path | None:
    """Load a .env file into process environment without requiring python-dotenv."""
    path = Path(dotenv_path) if dotenv_path else repository_root() / ".env"
    if not path.exists():
        return None

    with open(path, encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            os.environ.setdefault(key, value)

    return path


def bootstrap_environment(
    *,
    environment: str | None = None,
    dotenv_path: str | Path | None = None,
) -> EnvironmentBootstrap:
    """Initialize environment state for the current process."""
    global _bootstrap_state

    repo_root = repository_root()
    env_name = environment or detect_environment()
    env_file = Path(dotenv_path) if dotenv_path else repo_root / ".env"
    loaded_dotenv = False

    if should_load_dotenv(env_name):
        loaded_dotenv = load_dotenv(env_file) is not None

    _bootstrap_state = EnvironmentBootstrap(
        environment=env_name,
        repo_root=repo_root,
        dotenv_path=env_file,
        loaded_dotenv=loaded_dotenv,
    )
    return _bootstrap_state


def get_bootstrap_state() -> EnvironmentBootstrap:
    """Return cached environment bootstrap state, bootstrapping if needed."""
    return _bootstrap_state or bootstrap_environment()


def env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable."""
    parsed = _parse_bool(os.getenv(name))
    if parsed is None:
        return default
    return parsed


def env_int(name: str, default: int) -> int:
    """Read an integer environment variable with fallback."""
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    """Read a float environment variable with fallback."""
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default
