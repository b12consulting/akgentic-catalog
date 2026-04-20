"""Typer CLI (v2) for the unified catalog — ``ak-catalog <kind> <verb>``.

This module is the Epic 17 Story 17.1 scaffold: a single flat CLI module
exposing a Typer ``app`` mounted on the ``ak-catalog`` console-script entry
point. Every verb is a thin dispatcher into :class:`akgentic.catalog.catalog.Catalog`
— no business logic lives here.

Coexists with ``cli/main.py`` (v1) until Epic 19 deletes v1 and renames
``cli/v2.py`` back to ``cli/main.py``.
"""

from __future__ import annotations

import builtins
import json
from pathlib import Path
from typing import Any, Literal

import typer
import yaml
from pydantic import BaseModel, ConfigDict, ValidationError
from rich.console import Console
from rich.table import Table

from akgentic.catalog.catalog import Catalog
from akgentic.catalog.models.entry import Entry, EntryKind
from akgentic.catalog.models.errors import CatalogValidationError, EntryNotFoundError
from akgentic.catalog.models.queries import EntryQuery
from akgentic.catalog.repositories.base import EntryRepository

__all__ = ["app"]

_V2_ENTRIES_COLLECTION = "catalog_entries"

_list = builtins.list  # Alias kept because ``list`` is a Typer verb name.

_stdout = Console()
err_console = Console(stderr=True)


# --------------------------------------------------------------------------- #
# CLI state
# --------------------------------------------------------------------------- #


class CliState(BaseModel):
    """Global CLI state stashed on ``ctx.obj`` by the root callback."""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    backend: Literal["yaml", "mongo"] = "yaml"
    root: Path = Path("./catalog")
    uri: str | None = None
    db: str | None = None
    output_format: Literal["table", "json", "yaml"] = "table"


# --------------------------------------------------------------------------- #
# Typer app + global callback
# --------------------------------------------------------------------------- #


app = typer.Typer(
    name="ak-catalog",
    help="Unified Akgentic catalog CLI (v2) — manage entries across kinds.",
    no_args_is_help=True,
)


def _validate_mongo_options(state: CliState) -> None:
    """Guard mongo-only options; raises ``typer.Exit(code=2)`` on failure."""
    if state.backend != "mongo":
        return
    if state.uri is None:
        err_console.print("--uri is required when --backend=mongo")
        raise typer.Exit(code=2)
    if state.db is None:
        err_console.print("--db is required when --backend=mongo")
        raise typer.Exit(code=2)
    if not (state.uri.startswith("mongodb://") or state.uri.startswith("mongodb+srv://")):
        err_console.print("--uri must start with 'mongodb://' or 'mongodb+srv://'")
        raise typer.Exit(code=2)


@app.callback()
def _root(
    ctx: typer.Context,
    backend: str = typer.Option(
        "yaml",
        "--backend",
        help="Storage backend: yaml or mongo.",
        case_sensitive=False,
    ),
    root: Path = typer.Option(
        Path("./catalog"),
        "--root",
        help="Root directory for YAML entries (only when --backend=yaml).",
    ),
    uri: str | None = typer.Option(
        None,
        "--uri",
        help="MongoDB connection URI (required when --backend=mongo).",
    ),
    db: str | None = typer.Option(
        None,
        "--db",
        help="MongoDB database name (required when --backend=mongo).",
    ),
    output_format: str = typer.Option(
        "table",
        "--format",
        help="Output format: table, json, or yaml.",
        case_sensitive=False,
    ),
) -> None:
    """Parse global options and stash a ``CliState`` on ``ctx.obj``."""
    if backend not in ("yaml", "mongo"):
        err_console.print(f"Invalid backend '{backend}'. Must be 'yaml' or 'mongo'.")
        raise typer.Exit(code=2)
    if output_format not in ("table", "json", "yaml"):
        err_console.print(f"Invalid format '{output_format}'. Must be 'table', 'json', or 'yaml'.")
        raise typer.Exit(code=2)

    state = CliState(
        backend=backend,  # type: ignore[arg-type]
        root=root,
        uri=uri,
        db=db,
        output_format=output_format,  # type: ignore[arg-type]
    )
    _validate_mongo_options(state)
    ctx.obj = state


# --------------------------------------------------------------------------- #
# Backend wiring
# --------------------------------------------------------------------------- #


def _build_catalog(state: CliState) -> Catalog:
    """Construct a ``Catalog`` from the active ``CliState``.

    YAML: creates ``state.root`` if absent and wraps a ``YamlEntryRepository``.

    Mongo: lazy-imports the Mongo repository so YAML-only users never trip a
    missing ``pymongo``. A missing extra is re-surfaced as an "optional extra"
    usage error (exit 2) rather than a cryptic ``ImportError``.
    """
    if state.backend == "yaml":
        from akgentic.catalog.repositories.yaml_entry_repo import YamlEntryRepository

        state.root.mkdir(parents=True, exist_ok=True)
        return Catalog(YamlEntryRepository(state.root))

    # state.backend == "mongo" — options pre-validated by the root callback.
    try:
        from akgentic.catalog.repositories.mongo._config import MongoCatalogConfig
        from akgentic.catalog.repositories.mongo_entry_repo import MongoEntryRepository
    except ImportError:
        err_console.print(
            "--backend=mongo requires the 'mongo' optional extra: "
            "pip install akgentic-catalog[mongo]"
        )
        raise typer.Exit(code=2) from None

    assert state.uri is not None  # guarded by _validate_mongo_options
    assert state.db is not None
    config = MongoCatalogConfig(connection_string=state.uri, database=state.db)
    client = config.create_client()
    collection = config.get_collection(client, _V2_ENTRIES_COLLECTION)
    return Catalog(MongoEntryRepository(collection))


def _repo_from_ctx(ctx: typer.Context) -> Catalog:
    """Helper: pull ``CliState`` off the context and build a Catalog."""
    state = ctx.obj
    assert isinstance(state, CliState), "CliState missing from ctx.obj"
    return _build_catalog(state)


def _state_from_ctx(ctx: typer.Context) -> CliState:
    state = ctx.obj
    assert isinstance(state, CliState), "CliState missing from ctx.obj"
    return state


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _truncate(text: str, max_len: int = 60) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _render_entries(entries: _list[Entry], fmt: str) -> None:
    """Render a list of entries to stdout per ``fmt``."""
    if fmt == "json":
        _stdout.print(json.dumps([e.model_dump(mode="json") for e in entries], indent=2))
        return
    if fmt == "yaml":
        payload = [e.model_dump(mode="json") for e in entries]
        _stdout.print(yaml.safe_dump(payload, sort_keys=False).rstrip("\n"))
        return
    # table
    if not entries:
        _stdout.print("(no entries)")
        return
    table = Table(title="entries")
    for col in ("id", "namespace", "kind", "user_id", "description"):
        table.add_column(col)
    for e in entries:
        table.add_row(
            e.id,
            e.namespace,
            e.kind,
            e.user_id or "",
            _truncate(e.description or ""),
        )
    _stdout.print(table)


def _render_entry(entry: Entry, fmt: str) -> None:
    """Render a single entry to stdout per ``fmt``."""
    if fmt == "json":
        _stdout.print(json.dumps(entry.model_dump(mode="json"), indent=2))
        return
    if fmt == "yaml":
        _stdout.print(yaml.safe_dump(entry.model_dump(mode="json"), sort_keys=False).rstrip("\n"))
        return
    table = Table(title=f"entry {entry.id}", show_header=False)
    table.add_column("field", style="bold")
    table.add_column("value")
    rows = (
        ("id", entry.id),
        ("kind", entry.kind),
        ("namespace", entry.namespace),
        ("user_id", entry.user_id or ""),
        ("parent_namespace", entry.parent_namespace or ""),
        ("parent_id", entry.parent_id or ""),
        ("model_type", entry.model_type),
        ("description", entry.description or ""),
    )
    for key, value in rows:
        table.add_row(key, value)
    _stdout.print(table)
    payload_json = json.dumps(entry.model_dump(mode="json")["payload"], indent=2)
    if len(payload_json) > 4096:
        kept = payload_json[:4096]
        suffix = len(payload_json) - 4096
        _stdout.print(f"{kept}… ({suffix} more chars)")
    else:
        _stdout.print(payload_json)


# --------------------------------------------------------------------------- #
# Verb helpers
# --------------------------------------------------------------------------- #


def _parse_tri_state(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    err_console.print(f"--user-id-set must be 'true' or 'false'; got {value!r}")
    raise typer.Exit(code=2)


def _list_impl(
    ctx: typer.Context,
    kind: EntryKind,
    namespace: str | None,
    user_id: str | None,
    user_id_set: str | None,
    parent_namespace: str | None,
    parent_id: str | None,
) -> None:
    tri = _parse_tri_state(user_id_set)
    catalog = _repo_from_ctx(ctx)
    state = _state_from_ctx(ctx)
    query = EntryQuery(
        kind=kind,
        namespace=namespace,
        user_id=user_id,
        user_id_set=tri,
        parent_namespace=parent_namespace,
        parent_id=parent_id,
    )
    entries = catalog.list(query)
    _render_entries(entries, state.output_format)


def _get_impl(ctx: typer.Context, kind: EntryKind, id: str, namespace: str) -> None:
    catalog = _repo_from_ctx(ctx)
    state = _state_from_ctx(ctx)
    try:
        entry = catalog.get(namespace, id)
    except EntryNotFoundError:
        err_console.print(f"Entry ({namespace}, {id}, {kind}) not found")
        raise typer.Exit(code=1) from None
    if entry.kind != kind:
        err_console.print(
            f"Entry at (namespace={namespace}, id={id}) has kind={entry.kind}, expected {kind}"
        )
        raise typer.Exit(code=1)
    _render_entry(entry, state.output_format)


def _load_entry_yaml(path: Path) -> Entry:
    if not path.is_file():
        err_console.print(f"file not found: {path}")
        raise typer.Exit(code=2)
    try:
        text = path.read_text()
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        err_console.print(f"YAML parse error: {exc}")
        raise typer.Exit(code=2) from None
    if not isinstance(data, dict):
        err_console.print(f"YAML file must contain a mapping; got {type(data).__name__}")
        raise typer.Exit(code=2)
    try:
        return Entry(**data)
    except ValidationError as exc:
        err_console.print(f"entry validation error: {exc}")
        raise typer.Exit(code=2) from None


def _create_impl(ctx: typer.Context, kind: EntryKind, yaml_file: Path) -> None:
    entry = _load_entry_yaml(yaml_file)
    if entry.kind != kind:
        err_console.print(f"entry kind={entry.kind} does not match sub-app kind={kind}")
        raise typer.Exit(code=2)
    catalog = _repo_from_ctx(ctx)
    state = _state_from_ctx(ctx)
    try:
        saved = catalog.create(entry)
    except CatalogValidationError as exc:
        for err in exc.errors:
            err_console.print(f"validation error: {err}")
        raise typer.Exit(code=1) from None
    _render_entry(saved, state.output_format)


def _update_impl(ctx: typer.Context, kind: EntryKind, yaml_file: Path) -> None:
    entry = _load_entry_yaml(yaml_file)
    if entry.kind != kind:
        err_console.print(f"entry kind={entry.kind} does not match sub-app kind={kind}")
        raise typer.Exit(code=2)
    catalog = _repo_from_ctx(ctx)
    state = _state_from_ctx(ctx)
    try:
        saved = catalog.update(entry)
    except EntryNotFoundError as exc:
        err_console.print(str(exc))
        raise typer.Exit(code=1) from None
    except CatalogValidationError as exc:
        for err in exc.errors:
            err_console.print(f"validation error: {err}")
        raise typer.Exit(code=1) from None
    _render_entry(saved, state.output_format)


def _delete_impl(ctx: typer.Context, kind: EntryKind, id: str, namespace: str) -> None:
    catalog = _repo_from_ctx(ctx)
    state = _state_from_ctx(ctx)
    try:
        catalog.delete(namespace, id)
    except EntryNotFoundError as exc:
        err_console.print(str(exc))
        raise typer.Exit(code=1) from None
    except CatalogValidationError as exc:
        for err in exc.errors:
            err_console.print(str(err))
        _surface_referrers_if_any(catalog, namespace, id, state.output_format)
        raise typer.Exit(code=1) from None
    err_console.print(f"deleted {kind} {id} from namespace {namespace}")


def _surface_referrers_if_any(catalog: Catalog, namespace: str, id: str, fmt: str) -> None:
    """Render referrers of (namespace, id) if any — best-effort, no raise."""
    try:
        referrers = catalog.find_references(namespace, id)
    except Exception:  # pragma: no cover - defensive
        return
    if referrers:
        _render_entries(referrers, fmt)


def _search_impl(
    ctx: typer.Context,
    kind: EntryKind,
    namespace: str | None,
    user_id: str | None,
    user_id_set: str | None,
    parent_namespace: str | None,
    parent_id: str | None,
    description_contains: str | None,
    id: str | None,
) -> None:
    tri = _parse_tri_state(user_id_set)
    catalog = _repo_from_ctx(ctx)
    state = _state_from_ctx(ctx)
    query = EntryQuery(
        kind=kind,
        namespace=namespace,
        user_id=user_id,
        user_id_set=tri,
        parent_namespace=parent_namespace,
        parent_id=parent_id,
        description_contains=description_contains,
        id=id,
    )
    entries = catalog.list(query)
    _render_entries(entries, state.output_format)


def _not_implemented_repo_handler(exc: Exception) -> None:  # pragma: no cover
    """Safety net for NotImplementedError from degraded repositories."""
    err_console.print(f"repository does not support this operation: {exc}")


def _ensure_repo(_: EntryRepository) -> None:  # pragma: no cover - hook placeholder
    """Placeholder helper to keep EntryRepository imported for type use."""


# --------------------------------------------------------------------------- #
# Per-kind sub-apps — one Typer() per EntryKind, registered on app
# --------------------------------------------------------------------------- #


def _make_kind_app(kind: EntryKind) -> typer.Typer:
    """Build a Typer sub-app with the six verbs bound to ``kind``."""
    sub = typer.Typer(name=kind, help=f"Manage '{kind}' entries.", no_args_is_help=True)

    @sub.command("list")
    def _list_cmd(
        ctx: typer.Context,
        namespace: str | None = typer.Option(None, "--namespace"),
        user_id: str | None = typer.Option(None, "--user-id"),
        user_id_set: str | None = typer.Option(None, "--user-id-set"),
        parent_namespace: str | None = typer.Option(None, "--parent-namespace"),
        parent_id: str | None = typer.Option(None, "--parent-id"),
    ) -> None:
        """List entries of this kind matching the provided filters."""
        _list_impl(
            ctx,
            kind,
            namespace=namespace,
            user_id=user_id,
            user_id_set=user_id_set,
            parent_namespace=parent_namespace,
            parent_id=parent_id,
        )

    @sub.command("get")
    def _get_cmd(
        ctx: typer.Context,
        id: str = typer.Argument(..., help="Entry id."),
        namespace: str = typer.Option(..., "--namespace", help="Entry namespace."),
    ) -> None:
        """Fetch a single entry by ``(namespace, id)``."""
        _get_impl(ctx, kind, id, namespace)

    @sub.command("create")
    def _create_cmd(
        ctx: typer.Context,
        yaml_file: Path = typer.Argument(..., help="Path to single-entry YAML file."),
    ) -> None:
        """Create an entry from a single-entry YAML file."""
        _create_impl(ctx, kind, yaml_file)

    @sub.command("update")
    def _update_cmd(
        ctx: typer.Context,
        yaml_file: Path = typer.Argument(..., help="Path to single-entry YAML file."),
    ) -> None:
        """Update an existing entry from a single-entry YAML file."""
        _update_impl(ctx, kind, yaml_file)

    @sub.command("delete")
    def _delete_cmd(
        ctx: typer.Context,
        id: str = typer.Argument(..., help="Entry id."),
        namespace: str = typer.Option(..., "--namespace", help="Entry namespace."),
    ) -> None:
        """Delete an entry; surfaces inbound referrers on failure."""
        _delete_impl(ctx, kind, id, namespace)

    @sub.command("search")
    def _search_cmd(
        ctx: typer.Context,
        namespace: str | None = typer.Option(None, "--namespace"),
        user_id: str | None = typer.Option(None, "--user-id"),
        user_id_set: str | None = typer.Option(None, "--user-id-set"),
        parent_namespace: str | None = typer.Option(None, "--parent-namespace"),
        parent_id: str | None = typer.Option(None, "--parent-id"),
        description_contains: str | None = typer.Option(None, "--description-contains"),
        id: str | None = typer.Option(None, "--id"),
    ) -> None:
        """Structured search across every ``EntryQuery`` filter."""
        _search_impl(
            ctx,
            kind,
            namespace=namespace,
            user_id=user_id,
            user_id_set=user_id_set,
            parent_namespace=parent_namespace,
            parent_id=parent_id,
            description_contains=description_contains,
            id=id,
        )

    return sub


_ENTRY_KINDS: tuple[EntryKind, ...] = ("team", "agent", "tool", "model", "prompt")
# Register one sub-app per EntryKind. The literal order matches EntryKind.
for _kind in _ENTRY_KINDS:
    app.add_typer(_make_kind_app(_kind), name=_kind)


# --------------------------------------------------------------------------- #
# Helper — used by tests & silencing "unused import" warnings at module level.
# --------------------------------------------------------------------------- #


def _unused(_e: Any) -> None:  # pragma: no cover - trivial
    """Retain references to imports used only for typing."""


_unused(EntryRepository)
