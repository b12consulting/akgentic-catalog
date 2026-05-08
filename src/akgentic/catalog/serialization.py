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
import re
from typing import Any

import yaml
from pydantic import ValidationError

from akgentic.catalog.models.entry import Entry
from akgentic.catalog.models.errors import CatalogValidationError
from akgentic.catalog.repositories.yaml import _BlockScalarDumper

# Local copies of the resolver-layer ref-marker sentinel keys, kept here so
# the serialization module remains independent of resolver imports (resolver
# in turn does NOT import serialization, preserving a clean acyclic shape).
# These values are the single-source-of-truth strings ``"__ref__"`` /
# ``"__namespace__"`` defined at module load time in ``catalog/resolver.py``.
_REF_KEY: str = "__ref__"
_NAMESPACE_KEY: str = "__namespace__"
_TYPE_KEY: str = "__type__"
_RESERVED_REF_KEYS: frozenset[str] = frozenset({_REF_KEY, _NAMESPACE_KEY, _TYPE_KEY})

__all__ = [
    "_iter_cross_ns_targets",
    "dump_namespace",
    "dump_namespace_v2",
    "load_namespace",
]

logger = logging.getLogger(__name__)

# Kind emit order for bundle serialisation: team → meta → agent → prompt → tool → model.
# Reading a bundle top-down then mirrors the consumption graph: teams consume
# agents; agents consume prompts, tools, and models. ``meta`` follows ``team``
# because both describe the namespace as a whole (ADR-008 §D1). ``EntryKind``
# is a closed ``Literal`` of exactly these six values, so indexing by ``e.kind``
# is safe without a fallback.
_KIND_EMIT_ORDER: dict[str, int] = {
    "team": 0,
    "meta": 1,
    "agent": 2,
    "prompt": 3,
    "tool": 4,
    "model": 5,
}

# Section-header comment strings for each kind, aligned to an 80-character visual
# width (Python string length) and bracketed with ``####`` markers at both ends so
# the section break is loudly visible in an editor. The character ─ is U+2500 (1
# Python char, 3 UTF-8 bytes). Pinned as frozen strings keyed by lowercase kind
# name — deliberately NOT computed from kind.capitalize() so a future EntryKind
# rename cannot silently shift header text. Shape: two-space indent + ``#### ``
# + ``─── `` + capitalized plural name + space + ``─`` fill up to column 75 +
# `` ####`` trailer → 80 columns total.
_KIND_HEADERS: dict[str, str] = {
    "team": "  #### ─── Teams ".ljust(75, "─") + " ####",
    "meta": "  #### ─── Meta ".ljust(75, "─") + " ####",
    "agent": "  #### ─── Agents ".ljust(75, "─") + " ####",
    "prompt": "  #### ─── Prompts ".ljust(75, "─") + " ####",
    "tool": "  #### ─── Tools ".ljust(75, "─") + " ####",
    "model": "  #### ─── Models ".ljust(75, "─") + " ####",
}

# Regex patterns used by the post-processor.
# Matches a top-level entry key: exactly 2 spaces + identifier + colon (nothing else).
_ENTRY_KEY_RE = re.compile(r"^  [A-Za-z0-9_\-]+:$")
# Matches the kind line of an entry: 4 spaces + "kind: " + kind value.
_KIND_LINE_RE = re.compile(r"^    kind: ([a-z]+)$")


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
    order — ``team`` → ``meta`` → ``agent`` → ``prompt`` → ``tool`` →
    ``model`` — and within each kind sorted by ``id`` (lexicographic,
    unicode codepoint order). Reading a bundle top-down then matches the
    dependency tree: teams describe the namespace, meta annotates it; then
    agents consume prompts, tools, and models.

    The rendered document includes a comment-header line per non-empty kind
    group and blank-line separation between entries; both are stripped by
    ``yaml.safe_load`` on the import path, so round-tripping is unaffected.

    Args:
        entries: Non-empty list of ``Entry`` instances sharing a single
            namespace and user_id. The list MUST include at least one
            entry — at import time a team entry is required, and
            ``dump_namespace`` fails fast on the empty-list case.

    Returns:
        A YAML document string produced via ``yaml.dump`` with
        ``sort_keys=False``, ``allow_unicode=True``, and
        ``default_flow_style=False``, post-processed to add section headers
        and blank-line separators.

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
    raw = yaml.dump(
        doc,
        Dumper=_BlockScalarDumper,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    return _format_bundle_sections(raw)


def _peek_kind(lines: list[str], i: int) -> str:
    """Return the kind value from the line immediately following entry key at index i."""
    for j in range(i + 1, len(lines)):
        m = _KIND_LINE_RE.match(lines[j])
        if m:
            return m.group(1)
    raise AssertionError("unreachable: every entry must start with `kind:`")


def _format_bundle_sections(yaml_text: str) -> str:
    """Post-process a raw PyYAML bundle string to add section headers and spacing.

    Inserts a kind-section comment header (from ``_KIND_HEADERS``) and a blank
    line at each kind transition inside the ``entries:`` block. Consecutive entries
    within the same kind are separated by exactly one blank line. The header line
    for a kind is preceded by one blank line (visual gap after the previous section
    or after the ``entries:`` key). No blank line is added after the last entry.

    The document ends with exactly one trailing newline.
    """
    lines = yaml_text.rstrip("\n").split("\n")
    output: list[str] = []
    last_kind: str | None = None
    for i, line in enumerate(lines):
        if _ENTRY_KEY_RE.match(line):
            kind = _peek_kind(lines, i)
            if kind != last_kind:
                output.append("")
                output.append(_KIND_HEADERS[kind])
                last_kind = kind
            else:
                output.append("")
            output.append(line)
        else:
            output.append(line)
    return "\n".join(output) + "\n"


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

    Kind order follows ``_KIND_EMIT_ORDER`` — ``team`` → ``meta`` →
    ``agent`` → ``prompt`` → ``tool`` → ``model`` — the consumption graph
    with the namespace-meta entry sandwiched between team and agents.
    Within each kind, entries are sorted by ``id`` (lexicographic).
    ``EntryKind`` is a closed ``Literal`` of exactly these six values, so
    direct indexing (no ``.get`` fallback) is safe.
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


# --- dump_namespace_v2 (Story 17.5) -----------------------------------------


def dump_namespace_v2(entries: list[Entry], external_refs: list[Entry]) -> str:
    """Serialise a namespace + cross-ns targets to the v2 bundle YAML.

    The v2 bundle wire format is a two-section mapping at the document root:

    * ``entries:`` — a **list** of full per-entry maps (one item per persisted
      entry in the namespace), in the same kind / id order produced by
      :func:`dump_namespace` (team → meta → agent → prompt → tool → model;
      sorted by ``id`` within each kind). The visual section-header comments
      from :func:`dump_namespace` are preserved inside this block.
    * ``external_refs:`` — a **list** of full per-entry maps from other
      namespaces, sorted by ``(namespace, kind, id)`` ascending. The frontend
      uses these to render readonly cards for cross-namespace targets without
      additional round-trips.

    Each list item carries the full ``Entry`` shape — ``id``, ``namespace``,
    ``user_id``, ``kind``, ``model_type``, ``parent_namespace``,
    ``parent_id``, ``description``, ``payload``. Lists are used (rather than
    a dict keyed by id) because ``external_refs`` items live in foreign
    namespaces and a dict-keyed shape would collapse same-id entries from
    different namespaces. Both lists use the same shape for symmetry.

    Args:
        entries: The non-empty namespace-bounded entry list (same uniform-
            owner / uniform-namespace invariants as :func:`dump_namespace`).
        external_refs: The deduplicated, sorted list of cross-namespace target
            entries reachable from ``entries`` (sourced from each target's
            own namespace via the catalog service). May be empty.

    Returns:
        A YAML document string emitting the two-section mapping with the
        same ``_BlockScalarDumper`` configuration as :func:`dump_namespace`,
        post-processed to add section headers inside the ``entries:`` block.

    Raises:
        CatalogValidationError: When ``entries`` is empty (same message as
            :func:`dump_namespace`), or when ``entries`` violates the
            uniform-owner / uniform-namespace invariants.
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
    # AC4 — sort externals lexicographically on (namespace, kind, id), NOT
    # by ``_KIND_EMIT_ORDER``. Lexicographic kind order keeps the diff
    # stable under future kind additions; the consumption-graph ordering
    # of ``entries:`` is a display projection that does not extend to
    # foreign-namespace entries.
    sorted_external = sorted(external_refs, key=lambda e: (e.namespace, e.kind, e.id))

    doc: dict[str, Any] = {
        "entries": [_full_entry_to_map(e) for e in sorted_entries],
        "external_refs": [_full_entry_to_map(e) for e in sorted_external],
    }
    raw = yaml.dump(
        doc,
        Dumper=_BlockScalarDumper,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    return _format_bundle_v2_sections(raw)


def _full_entry_to_map(entry: Entry) -> dict[str, Any]:
    """Return a per-entry YAML map carrying every persisted ``Entry`` field.

    Used by :func:`dump_namespace_v2` for both ``entries:`` and
    ``external_refs:`` items. Each item is self-contained — it carries its
    own ``id`` / ``namespace`` / ``user_id`` rather than relying on the
    document-level inheritance used by the legacy :func:`dump_namespace`
    shape.
    """
    return {
        "id": entry.id,
        "namespace": entry.namespace,
        "user_id": entry.user_id,
        "kind": entry.kind,
        "model_type": entry.model_type,
        "parent_namespace": entry.parent_namespace,
        "parent_id": entry.parent_id,
        "description": entry.description,
        "payload": entry.payload,
    }


# Regex matching the start of a list item under ``entries:`` in the v2 dump
# emitted by PyYAML with ``default_flow_style=False``: 2 spaces, dash, space,
# ``id:``. Items in ``external_refs:`` are formatted identically; the
# post-processor distinguishes the two blocks by walking from the
# ``entries:`` line to the ``external_refs:`` line.
_V2_ITEM_START_RE = re.compile(r"^- id: ")
# In v2 dumps each list item begins at column 0 (PyYAML emits ``- ...`` at
# the document-relative indent of its enclosing key, which for ``entries:``
# at the root is column 0). The ``id`` field is the first key of the item,
# so the very first line of every item starts with ``- id: ``.
_V2_ITEM_KIND_RE = re.compile(r"^  kind: ([a-z]+)$")


def _format_bundle_v2_sections(yaml_text: str) -> str:
    """Post-process v2 bundle YAML to insert section headers inside ``entries:``.

    Walks the document, locates the ``entries:`` block, and for each list
    item under it (a line beginning with ``- id: ``) inserts a kind-section
    header (from ``_KIND_HEADERS``) when the kind transitions, plus blank-
    line separators between items. The ``external_refs:`` block (and any
    other top-level block) is left untouched — section headers belong to
    the namespace's own entries, not to foreign-namespace targets.
    """
    lines = yaml_text.rstrip("\n").split("\n")
    entries_idx = _find_top_level_key_line(lines, "entries:")
    external_idx = _find_top_level_key_line(lines, "external_refs:")
    if entries_idx is None:
        # Defensive: dump_namespace_v2 always produces an entries: block.
        return yaml_text  # pragma: no cover

    end_idx = external_idx if external_idx is not None else len(lines)
    head = lines[: entries_idx + 1]
    body = lines[entries_idx + 1 : end_idx]
    tail = lines[end_idx:]

    formatted_body = _format_v2_entries_block(body)
    return "\n".join(head + formatted_body + tail) + "\n"


def _find_top_level_key_line(lines: list[str], key_line: str) -> int | None:
    """Return the index of the first line equal to ``key_line`` at column 0."""
    for i, line in enumerate(lines):
        if line == key_line:
            return i
    return None


def _format_v2_entries_block(body: list[str]) -> list[str]:
    """Insert kind-section headers and blank-line separators in the entries body.

    ``body`` is the slice of YAML lines between ``entries:`` and the next
    top-level key (``external_refs:`` or end of document). Items begin with
    a line matching ``_V2_ITEM_START_RE`` and continue until the next item
    or the end of the slice.
    """
    output: list[str] = []
    last_kind: str | None = None
    for i, line in enumerate(body):
        if _V2_ITEM_START_RE.match(line):
            kind = _peek_v2_kind(body, i)
            if kind != last_kind:
                output.append("")
                output.append(_KIND_HEADERS[kind])
                last_kind = kind
            else:
                output.append("")
        output.append(line)
    return output


def _peek_v2_kind(lines: list[str], i: int) -> str:
    """Return the kind value from the v2 list item starting at index ``i``.

    Each v2 item carries a ``kind:`` field on its own line at indent 2 (the
    item's interior indent). The header line is two-space indent + ``kind:
    <value>``.
    """
    for j in range(i + 1, len(lines)):
        m = _V2_ITEM_KIND_RE.match(lines[j])
        if m:
            return m.group(1)
    raise AssertionError("unreachable: every v2 item must declare `kind:`")


# --- Cross-namespace ref walker (Story 17.5 Task 1) -------------------------


def _iter_cross_ns_targets(payload: Any) -> list[tuple[str, str]]:
    """Return every ``(target_namespace, target_id)`` reachable through cross-ns ref markers.

    Walks the payload tree recursively and collects every dict node carrying
    a ``__ref__`` entry that resolves to a cross-namespace target. The
    walker recognises the same two cross-ns shapes as the resolver:

    * **Canonical** — a sibling ``__namespace__`` key on the ref marker.
    * **Shorthand** — a ``<ns>.<id>`` form in the ``__ref__`` value (split
      on the first dot only, matching :func:`_resolve_target_namespace` in
      the resolver).

    Same-namespace markers (no ``__namespace__``, no dot in ``__ref__``)
    are excluded — they do not belong in ``external_refs:``.

    The walker is **parse-only**:

    * No call to :func:`populate_refs`.
    * No Pydantic validation.
    * No repository access.
    * No allowlist or shared-flag check (the shared-flag gate fires later,
      when the catalog service decides whether to fetch the target).

    Sibling-override sub-payloads (architecture/03 — non-reserved keys
    alongside ``__ref__`` / ``__namespace__`` / ``__type__``) are walked
    recursively, so a cross-ns ref marker that ALSO carries an override
    sub-payload containing another cross-ns ref yields both targets.

    Args:
        payload: An ``Entry.payload`` tree — arbitrary nested ``dict`` /
            ``list`` of JSON-able primitives.

    Returns:
        A flat list of ``(target_namespace, target_id)`` tuples in
        depth-first traversal order. Duplicates are NOT removed at this
        layer (the caller — :meth:`Catalog._collect_external_refs` — owns
        deduplication so the worklist + visited-set algorithm operates on
        the raw walker output).
    """
    results: list[tuple[str, str]] = []
    _walk_for_cross_ns(payload, results)
    return results


def _walk_for_cross_ns(node: Any, out: list[tuple[str, str]]) -> None:
    """Recursive helper for :func:`_iter_cross_ns_targets`."""
    if isinstance(node, dict):
        if _REF_KEY in node:
            pair = _classify_cross_ns_marker(node)
            if pair is not None:
                out.append(pair)
            # Walk sibling-override values (keys other than the reserved set)
            # — they may themselves carry nested cross-ns refs. We do NOT
            # recurse into the reserved keys themselves (``__namespace__`` is
            # a string; ``__type__`` is a string; ``__ref__`` is a string or
            # the shorthand we already classified).
            for key, value in node.items():
                if key not in _RESERVED_REF_KEYS:
                    _walk_for_cross_ns(value, out)
            return
        for value in node.values():
            _walk_for_cross_ns(value, out)
        return
    if isinstance(node, list):
        for item in node:
            _walk_for_cross_ns(item, out)


def _classify_cross_ns_marker(node: dict[str, Any]) -> tuple[str, str] | None:
    """Return ``(target_namespace, target_id)`` for a cross-ns ref marker, else ``None``.

    The classification mirrors the resolver's
    :func:`_resolve_target_namespace`:

    * If ``__ref__`` is a string containing a dot, the first-dot split
      yields ``(namespace, id)`` — this is the shorthand form.
    * Else if ``__namespace__`` is set explicitly, the pair is
      ``(__namespace__, __ref__)``.
    * Otherwise the marker is same-namespace and ``None`` is returned.
    """
    raw_ref = node.get(_REF_KEY)
    if isinstance(raw_ref, str) and "." in raw_ref:
        ns, rid = raw_ref.split(".", 1)
        return ns, rid
    explicit_ns = node.get(_NAMESPACE_KEY)
    if isinstance(explicit_ns, str) and isinstance(raw_ref, str):
        return explicit_ns, raw_ref
    return None


# --- load_namespace ---------------------------------------------------------


def load_namespace(yaml_text: str) -> list[Entry]:
    """Parse a bundle YAML document and return the list of reconstructed entries.

    Two wire shapes are accepted:

    * **v2 (Story 17.5)** — root is a ``dict`` with at least an ``entries``
      key whose value is a ``list`` (full per-entry maps, each carrying its
      own ``id`` / ``namespace`` / ``user_id``); an optional
      ``external_refs`` key carries cross-namespace targets and is **ignored
      on import** (those entries belong to other namespaces and are not
      part of THIS namespace's atomic-replace contents).

    * **Legacy (Story 16.2 — pre-17.5)** — root is a ``dict`` with
      ``namespace`` (non-empty ``str``), ``user_id`` (``str | None``), and
      ``entries`` whose value is a ``dict`` keyed by id; ``namespace`` and
      ``user_id`` are inherited document-wide.

    Shape detection is unambiguous because the legacy ``entries`` is always
    a ``dict`` and the v2 ``entries`` is always a ``list``. When the root
    contains an ``external_refs`` key, the v2 shape is selected even if
    ``entries`` is missing (so a bundle with an empty namespace cannot
    masquerade as legacy). The v2 path validates each entry's map shape
    individually; the legacy path uses ``_validate_root_shape`` as before.

    Any Pydantic ``ValidationError`` raised by ``Entry`` construction is
    wrapped in ``CatalogValidationError`` with the substring ``"entry
    '<id>' is invalid"`` so the offending id surfaces in UI toasts.

    The function is parse-only: it does NOT call ``prepare_for_write``, does
    NOT touch a repository, and does NOT persist anything. Callers that want
    to persist parsed entries must feed them through the ``Catalog`` service.

    Args:
        yaml_text: The full bundle YAML document as a string.

    Returns:
        The list of parsed ``Entry`` instances in document order. For a v2
        bundle, only the ``entries:`` items are returned — ``external_refs:``
        items are silently dropped (they belong to other namespaces).

    Raises:
        CatalogValidationError: On malformed YAML, structural failures,
            empty entries collection, or per-entry construction failures.
    """
    doc = _parse_yaml(yaml_text)
    if not isinstance(doc, dict):
        raise CatalogValidationError([f"bundle root must be a mapping, got {type(doc).__name__}"])
    if _is_v2_bundle_shape(doc):
        return _load_v2_entries(doc)
    return _load_legacy_entries(doc)


def _is_v2_bundle_shape(doc: dict[str, Any]) -> bool:
    """Return ``True`` when ``doc`` is a v2 bundle (Story 17.5 wire shape).

    The discriminator is unambiguous:

    * The legacy shape's ``entries`` is always a ``dict`` (``namespace`` /
      ``user_id`` siblings sit alongside it).
    * The v2 shape's ``entries`` is always a ``list``; the document also
      may carry an ``external_refs`` list.

    Cases handled:

    * ``entries`` present and is a ``list`` ⇒ v2.
    * ``external_refs`` present (regardless of ``entries`` shape) ⇒ v2.
    * Otherwise ⇒ legacy (the legacy validator will report missing /
      wrong-typed keys as today).
    """
    entries = doc.get("entries")
    if isinstance(entries, list):
        return True
    if "external_refs" in doc:
        return True
    return False


def _load_v2_entries(doc: dict[str, Any]) -> list[Entry]:
    """Parse the v2 bundle shape into a list of ``Entry`` instances.

    ``doc["entries"]`` MUST be a ``list``. ``external_refs`` is ignored —
    those items belong to other namespaces and are not part of this
    namespace's atomic-replace contents. Each entry item carries its own
    ``id`` / ``namespace`` / ``user_id`` so the function does not need to
    inherit document-level fields.
    """
    entries_value = doc.get("entries")
    if not isinstance(entries_value, list):
        # External_refs-only document with no entries — surface as an empty
        # bundle, same shape as the legacy empty-entries error so callers
        # can rely on the substring.
        raise CatalogValidationError(
            ["bundle must declare at least one entry, including a `kind=team` entry"]
        )
    if not entries_value:
        raise CatalogValidationError(
            ["bundle must declare at least one entry, including a `kind=team` entry"]
        )

    return [_build_v2_entry(item) for item in entries_value]


def _build_v2_entry(item: Any) -> Entry:
    """Build a single ``Entry`` from a v2 list-item map.

    The v2 item carries every ``Entry`` field directly — ``id`` /
    ``namespace`` / ``user_id`` are NOT inherited from a document-level
    context (they sit on each item, which is what makes ``external_refs:``
    items self-contained when they live in foreign namespaces).
    """
    if not isinstance(item, dict):
        raise CatalogValidationError(
            [f"v2 bundle entry is invalid: expected a mapping, got {type(item).__name__}"]
        )
    entry_id = item.get("id", "<missing>")
    try:
        return Entry.model_validate(
            {
                "id": item.get("id"),
                "namespace": item.get("namespace"),
                "user_id": item.get("user_id"),
                "kind": item.get("kind"),
                "model_type": item.get("model_type"),
                "parent_namespace": item.get("parent_namespace"),
                "parent_id": item.get("parent_id"),
                "description": item.get("description", ""),
                "payload": item.get("payload", {}),
            }
        )
    except ValidationError as exc:
        raise CatalogValidationError([f"entry '{entry_id}' is invalid: {exc}"]) from exc


def _load_legacy_entries(doc: dict[str, Any]) -> list[Entry]:
    """Parse the legacy (Story 16.2) bundle shape into a list of ``Entry`` instances.

    The legacy shape is a root ``dict`` carrying ``namespace`` /
    ``user_id`` siblings and an ``entries`` dict keyed by id. Structural
    failures accumulate into a single ``CatalogValidationError``.
    """
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
