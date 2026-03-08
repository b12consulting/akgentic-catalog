"""Typer CLI application for the Akgentic catalog.

Provides the ``ak-catalog`` entry point with global options and subcommand
groups for managing catalog entries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import typer

from akgentic.catalog.cli._output import OutputFormat

__all__ = ["app"]

logger = logging.getLogger(__name__)


@dataclass
class GlobalState:
    """Shared state passed through Typer context."""

    catalog_dir: Path = field(default_factory=lambda: Path("catalog"))
    format: OutputFormat = OutputFormat.table


app = typer.Typer(name="ak-catalog", help="Manage Akgentic catalog entries.")


@app.callback()
def main(
    ctx: typer.Context,
    catalog_dir: Path = typer.Option(
        Path("catalog"),
        "--catalog-dir",
        help="Root directory for catalog YAML files.",
    ),
    fmt: OutputFormat = typer.Option(
        OutputFormat.table,
        "--format",
        help="Output format: table, json, or yaml.",
    ),
) -> None:
    """Akgentic catalog CLI -- manage templates, tools, agents, and teams."""
    ctx.ensure_object(dict)
    ctx.obj = GlobalState(catalog_dir=catalog_dir, format=fmt)


@app.command("import")
def import_cmd() -> None:
    """Import catalog entries from external sources."""
    typer.echo("Not yet implemented")


@app.command("validate")
def validate_cmd() -> None:
    """Validate catalog entries for consistency."""
    typer.echo("Not yet implemented")


def _get_state(ctx: typer.Context) -> GlobalState:
    """Retrieve the global state from the Typer context.

    Args:
        ctx: The current Typer command context.

    Returns:
        The ``GlobalState`` instance stored in ``ctx.obj``.
    """
    state: GlobalState = ctx.obj
    return state


def _register_subcommands() -> None:
    """Register all subcommand groups on the app.

    Called at module load time. Deferred to a function to keep imports
    after all module-level definitions, avoiding E402.
    """
    from akgentic.catalog.cli.template_cmd import template_app
    from akgentic.catalog.cli.tool_cmd import tool_app

    app.add_typer(template_app, name="template")
    app.add_typer(tool_app, name="tool")

    # Stub subcommand groups (placeholders for stories 6-2 and 6-3)
    agent_app = typer.Typer(name="agent", help="Manage agent catalog entries.")
    team_app = typer.Typer(name="team", help="Manage team catalog entries.")
    app.add_typer(agent_app, name="agent")
    app.add_typer(team_app, name="team")


_register_subcommands()
