"""Namespace-level validation models and core ``validate_entries`` helper.

This module is the report-format source of truth for catalog namespace
validation (shard 05). It exposes two Pydantic models and two helper functions:

* :class:`EntryValidationIssue` — per-entry error payload.
* :class:`NamespaceValidationReport` — the structured report returned by
  :meth:`akgentic.catalog.catalog.Catalog.validate_namespace` and
  :meth:`akgentic.catalog.catalog.Catalog.validate_namespace_yaml`.
* :func:`validate_entries` — the collect-style validator shared by both the
  persisted-state flow and the dry-run bundle flow. Runs every check, collects
  every issue, returns a report. Never raises.
* :func:`check_global_invariants` — the bundle-wide checks alone, as a flat
  error list. The import path
  (``Catalog._validate_bundle_invariants``) calls it directly and raises on a
  non-empty result, so import and dry-run report the same defect in the same
  words (ADR-020 §D1).

The module is deliberately repository-agnostic on its per-entry and global
checks — it only calls ``repository.get`` indirectly through
:func:`akgentic.catalog.resolver.populate_refs` during transient validation.
It does NOT import from :mod:`akgentic.catalog.api`; the router consumes the
report format, not the other way around.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ValidationError, computed_field

from akgentic.catalog.models.entry import Entry, EntryKind, NonEmptyStr
from akgentic.catalog.models.errors import CatalogValidationError
from akgentic.catalog.refs import RefMarker, walk_payload
from akgentic.catalog.repositories.base import EntryRepository
from akgentic.catalog.resolver import (
    IsNamespaceShareableFn,
    load_model_type,
    populate_refs,
)
from akgentic.catalog.unknown_keys import UNKNOWN_KEY_MESSAGE, find_unknown_keys

__all__ = [
    "EntryValidationIssue",
    "NamespaceValidationReport",
    "check_global_invariants",
    "validate_entries",
]

logger = logging.getLogger(__name__)


class EntryValidationIssue(BaseModel):
    """Per-entry validation issue — carries every error collected for one entry."""

    entry_id: NonEmptyStr
    kind: EntryKind
    errors: list[str]


class NamespaceValidationReport(BaseModel):
    """Namespace-level validation report returned to callers verbatim on the wire.

    ``ok`` is computed from the two error lists rather than stored beside them:
    ``True`` if and only if ``global_errors`` is empty AND every
    ``entry_issues[].errors`` list is empty. It cannot disagree with them, so
    frontends can branch on ``ok`` alone without re-checking the two lists and
    no caller can construct a report whose ``ok`` contradicts its contents
    (ADR-020 §D7).

    Being a computed field, ``ok`` is part of the *serialization* schema and
    absent from the *validation* schema: it is emitted by ``model_dump`` and
    ``model_dump_json`` — after the declared fields — and a JSON body carrying
    ``"ok"`` still validates, the key being ignored on the way in and recomputed
    on the way out.
    """

    namespace: NonEmptyStr | None
    global_errors: list[str] = []
    entry_issues: list[EntryValidationIssue] = []

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ok(self) -> bool:
        """``True`` iff there are no global errors and no entry carries an error."""
        return not self.global_errors and all(not i.errors for i in self.entry_issues)


def validate_entries(
    entries: list[Entry],
    repository: EntryRepository,
    *,
    is_namespace_shareable: IsNamespaceShareableFn | None = None,
    has_header_meta: bool = False,
) -> NamespaceValidationReport:
    """Run every check against ``entries``; return a structured report.

    Collect-style: runs every check, collects every issue, never raises.
    Shared between ``Catalog.validate_namespace`` (which seeds ``entries`` from
    ``list_by_namespace``) and ``Catalog.validate_namespace_yaml`` (which seeds
    ``entries`` from ``load_namespace``).

    Performs zero repository writes — only calls ``repository.get`` indirectly
    via :func:`akgentic.catalog.resolver.populate_refs` during transient
    validation.

    Args:
        entries: The list of entries to validate. May be empty.
        repository: Entry repository used for transient-validation ref
            resolution (read-only).
        is_namespace_shareable: Forwarded to ``populate_refs`` so cross-ns
            ref errors (shareable-flag + ownership) surface as per-entry
            issues in the returned report (ADR-008 §D2 as updated
            2026-05-08).
        has_header_meta: ``True`` when the bundle's meta entry is hoisted into
            the document header rather than declared in ``entries``. Forwarded
            verbatim to :func:`check_global_invariants`, where it suppresses
            the *no anchor* error and nothing else. The persisted-state caller
            (``Catalog.validate_namespace``) leaves it ``False``: there the
            meta entry is a real row, so it is found in ``entries``. The
            bundle caller (``Catalog.validate_namespace_yaml``) MUST pass the
            parsed header's ``present`` — without it a header-anchored,
            team-less bundle (a tool-only shared namespace such as
            ``global``) is reported unanchored by the dry run while the import
            path accepts it, which is exactly the import/dry-run divergence
            this module exists to prevent.

    Returns:
        A :class:`NamespaceValidationReport` with ``ok`` derived from the
        collected errors.
    """
    if not entries:
        return NamespaceValidationReport(
            namespace=None,
            global_errors=["namespace has no entries"],
            entry_issues=[],
        )

    namespace = entries[0].namespace
    global_errors = check_global_invariants(entries, namespace, has_header_meta=has_header_meta)
    entry_issues = _collect_entry_issues(
        entries,
        repository,
        namespace,
        is_namespace_shareable=is_namespace_shareable,
    )

    return NamespaceValidationReport(
        namespace=namespace,
        global_errors=global_errors,
        entry_issues=entry_issues,
    )


def check_global_invariants(
    entries: list[Entry], namespace: str, *, has_header_meta: bool = False
) -> list[str]:
    """Return every bundle-wide error for ``entries`` as a flat list.

    The single collector for the bundle-wide rules, shared by the dry-run path
    (:func:`validate_entries`) and the import path
    (``Catalog._validate_bundle_invariants``) so both report the same defect in
    the same words.

    Every check always runs — there is deliberately no way to select a subset.
    A subset switch would let a caller reach the raise with an empty error list,
    which is the failure this collector exists to remove.

    Args:
        entries: The bundle's entries. May be empty; the anchor check then
            reports the missing anchor.
        namespace: The bundle namespace every entry is measured against —
            conventionally ``entries[0].namespace``.
        has_header_meta: ``True`` when the bundle's meta entry is hoisted into
            the document header rather than declared in ``entries``. It
            suppresses the *no anchor* error and nothing else.

    Returns:
        Every bundle-wide error, in check order: anchor, meta-singleton,
        uniform-namespace, uniform-user_id, duplicate-ids, dangling-refs.
    """
    errors: list[str] = []
    errors.extend(_check_namespace_anchor(entries, namespace, has_header_meta=has_header_meta))
    errors.extend(_check_meta_singleton(entries, namespace))
    errors.extend(_check_uniform_namespace(entries, namespace))
    errors.extend(_check_uniform_user_id(entries))
    errors.extend(_check_no_duplicate_ids(entries, namespace))
    errors.extend(_check_dangling_refs(entries, namespace))
    return errors


def _check_namespace_anchor(
    entries: list[Entry], namespace: str, *, has_header_meta: bool = False
) -> list[str]:
    """Require at least one anchor (team or meta); surface too-many teams.

    ``has_header_meta`` marks a bundle whose meta entry lives in the document
    header instead of ``entries`` — an anchor is present, just not in the list,
    so only the *no anchor* error is suppressed. The multiple-team error is
    unaffected.
    """
    teams = [e for e in entries if e.kind == "team"]
    metas = [e for e in entries if e.kind == "meta"]
    errors: list[str] = []
    if len(teams) == 0 and len(metas) == 0 and not has_header_meta:
        errors.append(
            f"namespace '{namespace}' has no team entry and no meta entry "
            f"— at least one anchor (team or meta) is required"
        )
    if len(teams) > 1:
        ids = sorted(t.id for t in teams)
        errors.append(f"namespace '{namespace}' has multiple team entries: {ids}")
    return errors


def _check_meta_singleton(entries: list[Entry], namespace: str) -> list[str]:
    """Allow zero or one ``kind=meta`` entry; flag two or more (ADR-008 §D1).

    A namespace MAY have a single ``kind=meta`` entry (convention id
    ``"_meta"``) carrying namespace-scoped metadata. Zero is valid (the route
    fallback in ``GET /catalog/namespaces`` covers the missing-meta case). Two
    or more produce a single error message naming every offending id so the
    UI can render every duplicate at once.
    """
    metas = [e for e in entries if e.kind == "meta"]
    if len(metas) <= 1:
        return []
    ids = sorted(m.id for m in metas)
    return [
        f"namespace '{namespace}' has multiple meta entries: {ids} — "
        f"exactly one kind=meta entry is allowed per namespace"
    ]


def _check_uniform_namespace(entries: list[Entry], namespace: str) -> list[str]:
    """One error message per entry whose ``namespace`` disagrees with the first entry's."""
    return [
        f"entry '{e.id}' has namespace '{e.namespace}' but bundle namespace is '{namespace}'"
        for e in entries
        if e.namespace != namespace
    ]


def _check_uniform_user_id(entries: list[Entry]) -> list[str]:
    """Anchored-by-the-team (or meta fallback) ownership check.

    Anchor resolution: if exactly one team, use team.user_id; else if exactly
    one meta, use meta.user_id; else skip (ambiguous or absent anchor — the
    absent case is caught by ``_check_namespace_anchor``).
    """
    teams = [e for e in entries if e.kind == "team"]
    if len(teams) == 1:
        anchor_user_id = teams[0].user_id
        anchor_kind = "team"
    else:
        metas = [e for e in entries if e.kind == "meta"]
        if len(teams) == 0 and len(metas) == 1:
            anchor_user_id = metas[0].user_id
            anchor_kind = "meta"
        else:
            return []
    return [
        f"entry '{e.id}' user_id '{e.user_id}' != {anchor_kind} user_id '{anchor_user_id}'"
        for e in entries
        if e.kind not in ("team", "meta") and e.user_id != anchor_user_id
    ]


def _check_no_duplicate_ids(entries: list[Entry], namespace: str) -> list[str]:
    """Surface each duplicate id exactly once."""
    seen: set[str] = set()
    dups: list[str] = []
    for e in entries:
        if e.id in seen and e.id not in dups:
            dups.append(e.id)
        seen.add(e.id)
    return [f"namespace '{namespace}' has duplicate entry id '{dup_id}'" for dup_id in dups]


def _check_dangling_refs(entries: list[Entry], namespace: str) -> list[str]:
    """Walk every payload; flag each ``__ref__`` target absent from the bundle's id set."""
    bundle_ids = {e.id for e in entries}
    errors: list[str] = []
    reported: set[tuple[str, str]] = set()
    for e in entries:
        for target_id in _iter_payload_ref_targets(e.payload):
            pair = (e.id, target_id)
            if target_id not in bundle_ids and pair not in reported:
                reported.add(pair)
                errors.append(
                    f"entry '{e.id}' has dangling ref to '{target_id}' in namespace '{namespace}'"
                )
    return errors


def _iter_payload_ref_targets(node: Any) -> Iterable[str]:
    """Return every same-namespace ``__ref__`` target id reachable under ``node``.

    Cross-ns markers (those carrying an explicit ``__namespace__`` key OR a
    ``<ns>.<id>`` shorthand in ``__ref__``) are external by design and are
    excluded from the bundle dangling-ref walker (ADR-008 §D2). Surfacing
    cross-ns errors is the resolver's job and they must appear in
    :class:`EntryValidationIssue.errors`, not in ``global_errors`` — see
    AC18 in story 17.3.

    Same-namespace is :meth:`~akgentic.catalog.refs.RefMarker.classify`
    reporting an empty ``target_namespace``.
    :func:`~akgentic.catalog.refs.walk_payload` has no yield channel, so the
    targets accumulate into a list through a closure; the return type stays
    ``Iterable[str]`` and the caller is unaffected.
    """
    found: list[str] = []

    def _visit(marker: dict[str, Any]) -> None:
        classified = RefMarker.classify(marker)
        if classified is not None and classified.target_namespace == "":
            found.append(classified.target_id)

    walk_payload(node, on_ref=_visit)
    return found


def _collect_entry_issues(
    entries: list[Entry],
    repository: EntryRepository,
    namespace: str,
    *,
    is_namespace_shareable: IsNamespaceShareableFn | None = None,
) -> list[EntryValidationIssue]:
    """Build per-entry issues; skip entries with empty error lists."""
    issues: list[EntryValidationIssue] = []
    for e in entries:
        errs = _per_entry_checks(
            e,
            repository,
            namespace,
            is_namespace_shareable=is_namespace_shareable,
        )
        if errs:
            issues.append(EntryValidationIssue(entry_id=e.id, kind=e.kind, errors=errs))
    return issues


def _per_entry_checks(
    entry: Entry,
    repository: EntryRepository,
    namespace: str,
    *,
    is_namespace_shareable: IsNamespaceShareableFn | None = None,
) -> list[str]:
    """Run model-type allowlist and transient-validation checks for ``entry``."""
    errors: list[str] = []
    cls: type[BaseModel] | None = None
    try:
        cls = load_model_type(entry.model_type)
    except CatalogValidationError as exc:
        errors.extend(exc.errors)
    if cls is not None:
        errors.extend(
            _check_transient_validation(
                entry,
                repository,
                namespace,
                cls,
                is_namespace_shareable=is_namespace_shareable,
            )
        )
    return errors


def _check_transient_validation(
    entry: Entry,
    repository: EntryRepository,
    namespace: str,
    cls: type[BaseModel],
    *,
    is_namespace_shareable: IsNamespaceShareableFn | None = None,
) -> list[str]:
    """Run ``populate_refs`` + ``cls.model_validate`` and collect every message.

    The unknown-key diff runs last, only once validation has succeeded: a
    wrong-typed value keeps producing its own message and nothing else.
    """
    try:
        populated = populate_refs(
            entry.payload,
            repository,
            namespace,
            is_namespace_shareable=is_namespace_shareable,
        )
    except CatalogValidationError as exc:
        return list(exc.errors)
    try:
        obj = cls.model_validate(populated)
    except ValidationError as exc:
        return [f"payload does not validate against {entry.model_type}: {exc}"]
    except CatalogValidationError as exc:  # pragma: no cover — defensive
        return list(exc.errors)
    return _check_unknown_keys(entry, obj)


def _check_unknown_keys(entry: Entry, obj: BaseModel) -> list[str]:
    """Return one finding per authored key ``entry.model_type`` never accepted.

    Mirrors ``prepare_for_write`` step 4b so Validate and Save agree by
    construction: the same helper reads the same dump, taken with the same
    flags, and the message comes from the same template.
    """
    # Same flags as the write path's step-4 dump, so the two paths compare the
    # same tree. ``exclude_unset=True`` is not what makes the check work — the
    # walk reports authored keys absent from the dump either way — it is carried
    # here only to keep this dump identical to the one the write path persists.
    dumped = obj.model_dump(mode="python", exclude_unset=True)
    try:
        paths = find_unknown_keys(entry.payload, dumped)
    except ValueError as exc:
        # A list-length mismatch under ``zip(strict=True)``. ``validate_entries``
        # never raises, so it becomes an ordinary finding here; the write path
        # lets it propagate, as ``reconcile_refs`` raises the same way on the
        # same trees one step later.
        return [f"cannot check unknown keys against {entry.model_type}: {exc}"]
    return [UNKNOWN_KEY_MESSAGE.format(path=p, model_type=entry.model_type) for p in paths]
