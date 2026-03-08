"""Tests for the main CLI app, help output, and stub commands."""

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


class TestStubCommands:
    """Verify stub commands print 'Not yet implemented' and exit cleanly."""

    def test_import_stub(self) -> None:
        result = runner.invoke(app, ["import"])
        assert result.exit_code == 0
        assert "Not yet implemented" in result.output

    def test_validate_stub(self) -> None:
        result = runner.invoke(app, ["validate"])
        assert result.exit_code == 0
        assert "Not yet implemented" in result.output
