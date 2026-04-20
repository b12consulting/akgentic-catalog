"""Shared JSONB predicate builders for the Nagra-backed Postgres repositories.

Each builder returns a ``(sql_fragment, params)`` tuple where:

* ``sql_fragment`` is a raw SQL condition string using ``%s`` as the psycopg
  placeholder marker — the same marker Nagra's Transaction uses when it hands
  a statement to psycopg under the hood. No user-supplied value is ever
  interpolated into the fragment text; every variable appears as ``%s`` and is
  passed through ``params``.
* ``params`` is the list of bound values in argument order.

The module exposes per-field helpers (``template_id_predicate``, ``tool_name_predicate``,
etc.) plus two compose helpers (``build_template_where`` / ``build_tool_where``)
that walk the non-None fields of a query model and AND-combine the per-field
predicates. Story 15.3 will extend the same module with agent / team builders.

Implements ADR-006 §4 predicate table for Template and Tool queries. See the
ToolEntry nested-path note: because ``ToolEntry.model_dump()`` places ``name``
and ``description`` inside a nested ``tool`` object, the JSONB path uses
``data->'tool'->>'name'`` (not the top-level ``data->>'name'`` shown in the
ADR table).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from akgentic.catalog.models.queries import TemplateQuery, ToolQuery

__all__ = [
    "build_template_where",
    "build_tool_where",
    "decode_jsonb_column",
    "template_id_predicate",
    "template_placeholder_predicate",
    "tool_description_predicate",
    "tool_id_predicate",
    "tool_name_predicate",
    "tool_tool_class_predicate",
]


# --- JSONB column decoding (shared across repos) ---


def decode_jsonb_column(raw: object) -> dict[str, object]:
    """Normalise a JSONB column value to a Python dict.

    psycopg 3 decodes JSONB columns to native Python objects by default, but
    some driver configurations (or the plain ``JSON`` column type) return a
    JSON string. Handle both so hydration is robust regardless of the adapter
    wiring. Shared by ``template_repo`` and ``tool_repo`` (and by story 15.3's
    agent / team repos) to keep the JSONB↔dict contract in one place.
    """
    if isinstance(raw, str):
        return cast("dict[str, object]", json.loads(raw))
    return cast("dict[str, object]", raw)


# --- Per-field predicate builders (Template) ---


def template_id_predicate(value: str) -> tuple[str, list[object]]:
    """Build ``data->>'id' = %s`` predicate for an exact-id match."""
    return "data->>'id' = %s", [value]


def template_placeholder_predicate(value: str) -> tuple[str, list[object]]:
    """Build JSONB key-existence predicate on the ``placeholders`` array.

    Uses the Postgres ``?`` operator (does the right-hand text exist as a
    top-level key in the left-hand JSONB?). Against a JSON array of strings,
    this returns true when the value equals one of the array's string
    elements — matching ``placeholder in entry.placeholders`` semantics of the
    YAML and Mongo backends.

    The ``?`` operator is only defined for ``jsonb`` (not plain ``json``);
    the ``data`` column is declared as ``JSON`` in ``schema.toml`` (story
    15.1 invariant), so the value is cast to ``jsonb`` at query time.
    """
    return "(data->'placeholders')::jsonb ? %s", [value]


# --- Per-field predicate builders (Tool) ---


def tool_id_predicate(value: str) -> tuple[str, list[object]]:
    """Build ``data->>'id' = %s`` predicate for an exact-id match."""
    return "data->>'id' = %s", [value]


def tool_tool_class_predicate(value: str) -> tuple[str, list[object]]:
    """Build ``data->>'tool_class' = %s`` predicate for an exact class-path match."""
    return "data->>'tool_class' = %s", [value]


def tool_name_predicate(value: str) -> tuple[str, list[object]]:
    """Build a case-insensitive substring predicate on the tool's display name.

    The JSONB path is ``data->'tool'->>'name'`` because ``ToolEntry.model_dump()``
    nests ``name`` under a ``tool`` object — matching the Mongo backend's
    ``tool.name`` filter. The parameter is wrapped with ``%`` characters so
    the ILIKE operator performs substring matching.
    """
    return "data->'tool'->>'name' ILIKE %s", [f"%{value}%"]


def tool_description_predicate(value: str) -> tuple[str, list[object]]:
    """Build a case-insensitive substring predicate on the tool's description.

    Uses the same nested JSONB path convention as :func:`tool_name_predicate`.
    """
    return "data->'tool'->>'description' ILIKE %s", [f"%{value}%"]


# --- Compose helpers ---


def _combine(
    fragments: list[tuple[str, list[object]]],
) -> tuple[str | None, list[object]]:
    """AND-combine a list of ``(fragment, params)`` pairs.

    Returns ``(None, [])`` when the input list is empty so callers can fall
    back to a bare SELECT without an empty WHERE clause.
    """
    if not fragments:
        return None, []
    clauses = [frag for frag, _ in fragments]
    params: list[object] = []
    for _, param_list in fragments:
        params.extend(param_list)
    where_sql = " AND ".join(clauses)
    return where_sql, params


def build_template_where(query: TemplateQuery) -> tuple[str | None, list[object]]:
    """Build the WHERE-clause fragment for a :class:`TemplateQuery`.

    Returns ``(None, [])`` when no fields are set (callers issue a bare SELECT).
    Otherwise returns ``(sql_fragment, params)`` with all fields AND-combined.
    """
    fragments: list[tuple[str, list[object]]] = []
    if query.id is not None:
        fragments.append(template_id_predicate(query.id))
    if query.placeholder is not None:
        fragments.append(template_placeholder_predicate(query.placeholder))
    return _combine(fragments)


def build_tool_where(query: ToolQuery) -> tuple[str | None, list[object]]:
    """Build the WHERE-clause fragment for a :class:`ToolQuery`.

    Returns ``(None, [])`` when no fields are set (callers issue a bare SELECT).
    Otherwise returns ``(sql_fragment, params)`` with all fields AND-combined.
    """
    fragments: list[tuple[str, list[object]]] = []
    if query.id is not None:
        fragments.append(tool_id_predicate(query.id))
    if query.tool_class is not None:
        fragments.append(tool_tool_class_predicate(query.tool_class))
    if query.name is not None:
        fragments.append(tool_name_predicate(query.name))
    if query.description is not None:
        fragments.append(tool_description_predicate(query.description))
    return _combine(fragments)
