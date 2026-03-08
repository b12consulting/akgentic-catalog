"""Tests for the main CLI app and help output."""

from __future__ import annotations

from typer.testing import CliRunner

from akgentic.catalog.cli.main import app

runner = CliRunner()


class TestCliHelp:
    """Verify --help output lists expected subcommands and global options (AC-2)."""

    def test_help_lists_subcommands(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for cmd in ("template", "tool", "agent", "team", "import", "validate"):
            assert cmd in result.output, f"Expected subcommand '{cmd}' in help output"

    def test_help_lists_global_options(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "--catalog-dir" in result.output
        assert "--format" in result.output


class TestImportValidateHelp:
    """Verify import and validate commands show help and accept arguments."""

    def test_import_help(self) -> None:
        result = runner.invoke(app, ["import", "--help"])
        assert result.exit_code == 0
        assert "PYTHON_FILE" in result.output
        assert "--dry-run" in result.output

    def test_validate_help(self) -> None:
        result = runner.invoke(app, ["validate", "--help"])
        assert result.exit_code == 0
        assert "--catalog" in result.output
