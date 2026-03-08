"""Typer CLI application for the Akgentic catalog.

Provides the ``ak-catalog`` entry point with global options and subcommand
groups for managing catalog entries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import typer
from rich.console import Console

from akgentic.catalog.cli._output import OutputFormat

__all__ = ["app"]

logger = logging.getLogger(__name__)


@dataclass
class GlobalState:
    """Shared state passed through Typer context."""

    catalog_dir: Path = field(default_factory=lambda: Path("catalog"))
    format: OutputFormat = OutputFormat.table
    backend: str = "yaml"
    mongo_uri: str | None = None
    mongo_db: str | None = None


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
    backend: str = typer.Option(
        "yaml",
        "--backend",
        help="Storage backend: yaml or mongodb.",
    ),
    mongo_uri: str | None = typer.Option(
        None,
        "--mongo-uri",
        envvar="MONGO_URI",
        help="MongoDB connection URI (required when --backend=mongodb).",
    ),
    mongo_db: str | None = typer.Option(
        None,
        "--mongo-db",
        envvar="MONGO_DB",
        help="MongoDB database name (required when --backend=mongodb).",
    ),
) -> None:
    """Akgentic catalog CLI -- manage templates, tools, agents, and teams."""
    valid_backends = ("yaml", "mongodb")
    if backend not in valid_backends:
        err_console = Console(stderr=True)
        err_console.print(
            f"[red]Error:[/red] Invalid backend '{backend}'. "
            f"Must be one of: {', '.join(valid_backends)}"
        )
        logger.warning("Invalid backend value: %s", backend)
        raise typer.Exit(code=1)

    if backend == "mongodb":
        errors = _validate_mongodb_options(mongo_uri, mongo_db)
        if errors:
            err_console = Console(stderr=True)
            for err in errors:
                err_console.print(f"[red]Error:[/red] {err}")
            logger.warning("MongoDB validation failed: %s", errors)
            raise typer.Exit(code=1)

    ctx.ensure_object(dict)
    ctx.obj = GlobalState(
        catalog_dir=catalog_dir,
        format=fmt,
        backend=backend,
        mongo_uri=mongo_uri,
        mongo_db=mongo_db,
    )


def _validate_mongodb_options(
    mongo_uri: str | None,
    mongo_db: str | None,
) -> list[str]:
    """Validate MongoDB connection options and return error messages.

    Args:
        mongo_uri: MongoDB connection URI, or None if not provided.
        mongo_db: MongoDB database name, or None if not provided.

    Returns:
        List of error message strings (empty if valid).
    """
    errors: list[str] = []
    if not mongo_uri:
        errors.append("--mongo-uri (or MONGO_URI env var) is required when --backend=mongodb")
    elif not mongo_uri.startswith(("mongodb://", "mongodb+srv://")):
        errors.append("--mongo-uri must start with 'mongodb://' or 'mongodb+srv://'")
    if not mongo_db:
        errors.append("--mongo-db (or MONGO_DB env var) is required when --backend=mongodb")
    return errors


@app.command("import")
def import_cmd() -> None:
    """Import catalog entries from external sources."""
    typer.echo("Not yet implemented")


@app.command("validate")
def validate_cmd() -> None:
    """Validate catalog entries for consistency."""
    typer.echo("Not yet implemented")


def get_state(ctx: typer.Context) -> GlobalState:
    """Retrieve the global state from the Typer context.

    Falls back to default ``GlobalState`` if the context has no object set.

    Args:
        ctx: The current Typer command context.

    Returns:
        The ``GlobalState`` instance stored in ``ctx.obj``, or a default.
    """
    state: GlobalState = ctx.obj
    if state is None:
        state = GlobalState()
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
