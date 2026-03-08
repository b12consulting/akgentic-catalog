"""Shared fixtures for CLI tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from akgentic.catalog.cli.main import app


@pytest.fixture()
def runner() -> CliRunner:
    """Create a Typer CLI test runner."""
    return CliRunner()


@pytest.fixture()
def catalog_dir(tmp_path: Path) -> Path:
    """Create a temporary catalog directory with subdirectories."""
    for name in ("templates", "tools", "agents", "teams"):
        (tmp_path / name).mkdir()
    return tmp_path


@pytest.fixture()
def cli_app() -> object:
    """Return the Typer app under test."""
    return app


def write_yaml(path: Path, data: object) -> None:
    """Write a Python object as YAML to *path*."""
    path.write_text(yaml.dump(data, default_flow_style=False))
