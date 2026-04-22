"""Namespace-bundle YAML serialization for the catalog v2 service.

This module owns the single-bundle YAML format defined in architecture shard 09:
an entire user or enterprise namespace serialised as one self-contained YAML
document with document-level ``namespace`` + ``user_id`` and per-entry fields
nested under ``entries.<id>``. The module exposes two pure functions:

* :func:`dump_namespace` — serialise ``list[Entry]`` to YAML ``str``.
* :func:`load_namespace` — parse YAML ``str`` into ``list[Entry]``.

The module is repository-agnostic: neither function performs repository I/O,
runs ``prepare_for_write``, or mutates any catalog state. The service-level
``Catalog.export_namespace_yaml`` / ``import_namespace_yaml`` methods own the
repository boundary; this module owns only the wire format.

``load_namespace`` is deliberately kept pure (no ``prepare_for_write``) so
Story 16.3's ``validate_namespace_yaml`` can reuse it in-process for dry-run
validation of proposed bundles.
"""

from __future__ import annotations

import logging
from typing import Any

import yaml
from pydantic import ValidationError

from akgentic.catalog.models.entry import Entry
from akgentic.catalog.models.errors import CatalogValidationError
from akgentic.catalog.repositories.yaml import _BlockScalarDumper

__all__ = ["dump_namespace", "load_namespace"]

logger = logging.getLogger(__name__)

# Kind emit order for bundle serialisation: team → agent → prompt → tool → model.
# Reading a bundle top-down then mirrors the consumption graph: teams consume
# agents; agents consume prompts, tools, and models. ``EntryKind`` is a closed
# ``Literal`` of exactly these five values, so indexing by ``e.kind`` is safe
# without a fallback.
_KIND_EMIT_ORDER: dict[str, int] = {
    "team": 0,
    "agent": 1,
    "prompt": 2,
    "tool": 3,
    "model": 4,
}


# --- dump_namespace ---------------------------------------------------------


def dump_namespace(entries: list[Entry]) -> str:
    """Serialise a uniform-namespace list of entries to bundle YAML.

    The output document has exactly three root keys in this order:
    ``namespace``, ``user_id``, ``entries``. Each value under ``entries``
    is keyed by the entry id and maps to six per-entry fields in declaration
    order: ``kind``, ``model_type``, ``parent_namespace``, ``parent_id``,
    ``description``, ``payload``. The ``id``, ``namespace`` and ``user_id``
    fields are NOT duplicated inside the per-entry maps — they are implied
    by the document context and the outer key.

    Ownership invariant: every entry in ``entries`` MUST share the same
    ``user_id`` (including the ``None`` case for enterprise bundles).
    Namespace invariant: every entry MUST share the same ``namespace``.
    Both invariants are checked together before emit; violations raise
    ``CatalogValidationError`` with one message per offender.

    ``payload`` values pass through verbatim — ref markers (``__ref__`` /
    ``__type__``) are preserved unchanged. ``dump_namespace`` does NOT
    re-resolve, re-validate, or re-reconcile; stored payloads are already
    intent-preserving.

    Entries are emitted in a stable order grouped by kind in consumption
    order — ``team`` → ``agent`` → ``prompt`` → ``tool`` → ``model`` — and
    within each kind sorted by ``id`` (lexicographic, unicode codepoint
    order). Reading a bundle top-down then matches the dependency tree:
    teams consume agents; agents consume prompts, tools, and models.

    Args:
        entries: Non-empty list of ``Entry`` instances sharing a single
            namespace and user_id. The list MUST include at least one
            entry — at import time a team entry is required, and
            ``dump_namespace`` fails fast on the empty-list case.

    Returns:
        A YAML document string produced via ``yaml.safe_dump`` with
        ``sort_keys=False``, ``allow_unicode=True``, and
        ``default_flow_style=False``.

    Raises:
        CatalogValidationError: When ``entries`` is empty, or when any
            entry's ``user_id`` / ``namespace`` disagrees with the first
            entry's values.
    """
    if not entries:
        raise CatalogValidationError(
            ["bundle must declare at least one entry, including a `kind=team` entry"]
        )

    errors: list[str] = []
    errors.extend(_check_uniform_owner(entries))
    errors.extend(_check_uniform_namespace(entries))
    if errors:
        raise CatalogValidationError(errors)

    sorted_entries = _sort_entries_for_emit(entries)
    doc: dict[str, Any] = {
        "namespace": entries[0].namespace,
        "user_id": entries[0].user_id,
        "entries": {e.id: _entry_to_map(e) for e in sorted_entries},
    }
    return yaml.dump(
        doc,
        Dumper=_BlockScalarDumper,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )


def _check_uniform_owner(entries: list[Entry]) -> list[str]:
    """Return one error message per entry whose ``user_id`` disagrees with the first."""
    expected = entries[0].user_id
    return [
        f"entry '{e.id}' has user_id={e.user_id!r} but bundle user_id is {expected!r}"
        for e in entries
        if e.user_id != expected
    ]


def _check_uniform_namespace(entries: list[Entry]) -> list[str]:
    """Return one error message per entry whose ``namespace`` disagrees with the first."""
    expected = entries[0].namespace
    return [
        f"entry '{e.id}' has namespace={e.namespace!r} but bundle namespace is {expected!r}"
        for e in entries
        if e.namespace != expected
    ]


def _sort_entries_for_emit(entries: list[Entry]) -> list[Entry]:
    """Return a new list sorted by (kind emit order, id).

    Kind order follows ``_KIND_EMIT_ORDER`` — ``team`` → ``agent`` →
    ``prompt`` → ``tool`` → ``model`` — the consumption graph. Within each
    kind, entries are sorted by ``id`` (lexicographic). ``EntryKind`` is a
    closed ``Literal`` of exactly these five values, so direct indexing
    (no ``.get`` fallback) is safe.
    """
    return sorted(entries, key=lambda e: (_KIND_EMIT_ORDER[e.kind], e.id))


def _entry_to_map(entry: Entry) -> dict[str, Any]:
    """Return the per-entry YAML map with the six pinned keys in declaration order."""
    return {
        "kind": entry.kind,
        "model_type": entry.model_type,
        "parent_namespace": entry.parent_namespace,
        "parent_id": entry.parent_id,
        "description": entry.description,
        "payload": entry.payload,
    }


# --- load_namespace ---------------------------------------------------------


def load_namespace(yaml_text: str) -> list[Entry]:
    """Parse a bundle YAML document and return the list of reconstructed entries.

    The parser is structurally strict: the root must be a ``dict`` with
    ``namespace`` (non-empty ``str``), ``user_id`` (``str | None``), and
    ``entries`` (non-empty ``dict``). Any missing or wrong-typed key accumulates
    into a single ``CatalogValidationError`` — the error list is NOT
    short-circuited after the first failure so frontends can render every
    issue in one pass.

    Each ``(entry_id, entry_map)`` pair becomes an ``Entry`` whose ``id`` is
    the outer key, and whose ``namespace`` and ``user_id`` come from the
    document context. The six per-entry keys (``kind``, ``model_type``,
    ``parent_namespace``, ``parent_id``, ``description``, ``payload``) are
    consumed verbatim. Any Pydantic ``ValidationError`` raised by ``Entry``
    construction is wrapped in ``CatalogValidationError`` with the substring
    ``"entry '<id>' is invalid"`` so the offending id surfaces in UI toasts.

    The function is parse-only: it does NOT call ``prepare_for_write``, does
    NOT touch a repository, and does NOT persist anything. Callers that want
    to persist parsed entries must feed them through the ``Catalog`` service.

    Args:
        yaml_text: The full bundle YAML document as a string.

    Returns:
        The list of parsed ``Entry`` instances in dict-iteration order
        (PyYAML yields keys in document order).

    Raises:
        CatalogValidationError: On malformed YAML, structural failures,
            empty entries dict, or per-entry construction failures.
    """
    doc = _parse_yaml(yaml_text)
    structural_errors = _validate_root_shape(doc)
    if structural_errors:
        raise CatalogValidationError(structural_errors)

    entries_map: dict[str, Any] = doc["entries"]
    if not entries_map:
        raise CatalogValidationError(
            ["bundle must declare at least one entry, including a `kind=team` entry"]
        )

    namespace: str = doc["namespace"]
    user_id: str | None = doc["user_id"]
    return [
        _build_entry(entry_id, entry_map, namespace, user_id)
        for entry_id, entry_map in entries_map.items()
    ]


def _parse_yaml(yaml_text: str) -> Any:
    """Return ``yaml.safe_load(yaml_text)`` or wrap failures as CatalogValidationError."""
    try:
        return yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        raise CatalogValidationError([f"Failed to parse bundle YAML: {exc}"]) from exc


def _validate_root_shape(doc: Any) -> list[str]:
    """Return a list of structural failures for the bundle root document.

    Accumulates every failure found; does not short-circuit on the first one.
    """
    errors: list[str] = []
    if not isinstance(doc, dict):
        return [f"bundle root must be a mapping, got {type(doc).__name__}"]

    if "namespace" not in doc:
        errors.append("bundle root missing required key 'namespace'")
    elif not isinstance(doc["namespace"], str) or not doc["namespace"]:
        errors.append("bundle 'namespace' must be a non-empty string")

    if "user_id" not in doc:
        errors.append("bundle root missing required key 'user_id'")
    elif doc["user_id"] is not None and not isinstance(doc["user_id"], str):
        errors.append("bundle 'user_id' must be a string or null")

    if "entries" not in doc:
        errors.append("bundle root missing required key 'entries'")
    elif not isinstance(doc["entries"], dict):
        errors.append(f"bundle 'entries' must be a mapping, got {type(doc['entries']).__name__}")
    return errors


def _build_entry(
    entry_id: str,
    entry_map: Any,
    namespace: str,
    user_id: str | None,
) -> Entry:
    """Build a single ``Entry`` from a per-entry YAML map.

    Missing required keys (``kind``, ``model_type``) surface through Pydantic
    validation; the caller wraps that into ``CatalogValidationError`` with the
    stable ``"entry '<id>' is invalid"`` substring.
    """
    if not isinstance(entry_map, dict):
        raise CatalogValidationError(
            [f"entry '{entry_id}' is invalid: expected a mapping, got {type(entry_map).__name__}"]
        )
    try:
        # Pydantic performs validation at construction; pass values through
        # ``model_validate`` so mypy stays out of the Literal / AllowlistedPath
        # type contract on ``Entry.kind`` / ``Entry.model_type``.
        return Entry.model_validate(
            {
                "id": entry_id,
                "namespace": namespace,
                "user_id": user_id,
                "kind": entry_map.get("kind"),
                "model_type": entry_map.get("model_type"),
                "parent_namespace": entry_map.get("parent_namespace"),
                "parent_id": entry_map.get("parent_id"),
                "description": entry_map.get("description", ""),
                "payload": entry_map.get("payload", {}),
            }
        )
    except ValidationError as exc:
        raise CatalogValidationError([f"entry '{entry_id}' is invalid: {exc}"]) from exc
