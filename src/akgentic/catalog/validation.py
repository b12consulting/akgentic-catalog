"""Namespace-level validation models and core ``validate_entries`` helper.

This module is the report-format source of truth for catalog namespace
validation (shard 05). It exposes two Pydantic models and one helper function:

* :class:`EntryValidationIssue` — per-entry error payload.
* :class:`NamespaceValidationReport` — the structured report returned by
  :meth:`akgentic.catalog.catalog.Catalog.validate_namespace` and
  :meth:`akgentic.catalog.catalog.Catalog.validate_namespace_yaml`.
* :func:`validate_entries` — the collect-style validator shared by both the
  persisted-state flow and the dry-run bundle flow. Runs every check, collects
  every issue, returns a report. Never raises.

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

from pydantic import BaseModel, ValidationError, model_validator

from akgentic.catalog.models.entry import Entry, EntryKind, NonEmptyStr
from akgentic.catalog.models.errors import CatalogValidationError
from akgentic.catalog.repositories.base import EntryRepository
from akgentic.catalog.resolver import REF_KEY, load_model_type, populate_refs

__all__ = ["EntryValidationIssue", "NamespaceValidationReport", "validate_entries"]

logger = logging.getLogger(__name__)


class EntryValidationIssue(BaseModel):
    """Per-entry validation issue — carries every error collected for one entry."""

    entry_id: NonEmptyStr
    kind: EntryKind
    errors: list[str]


class NamespaceValidationReport(BaseModel):
    """Namespace-level validation report returned to callers verbatim on the wire.

    ``ok`` is a derived invariant: ``True`` if and only if ``global_errors`` is
    empty AND every ``entry_issues[].errors`` list is empty. The post-init
    validator rejects inconsistent constructions so frontends can branch on
    ``ok`` alone without re-checking the two lists.
    """

    namespace: NonEmptyStr | None
    ok: bool
    global_errors: list[str] = []
    entry_issues: list[EntryValidationIssue] = []

    @model_validator(mode="after")
    def _check_ok_invariant(self) -> NamespaceValidationReport:
        """Ensure ``ok`` agrees with the error lists (both-empty iff ``ok``)."""
        expected = not self.global_errors and all(not i.errors for i in self.entry_issues)
        if self.ok != expected:
            raise ValueError(
                f"NamespaceValidationReport.ok={self.ok} but expected {expected} "
                f"(global_errors={self.global_errors!r}, entry_issues={self.entry_issues!r})"
            )
        return self


def validate_entries(
    entries: list[Entry], repository: EntryRepository
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

    Returns:
        A :class:`NamespaceValidationReport` with ``ok`` derived from the
        collected errors.
    """
    if not entries:
        return NamespaceValidationReport(
            namespace=None,
            ok=False,
            global_errors=["namespace has no entries"],
            entry_issues=[],
        )

    namespace = entries[0].namespace
    global_errors = _global_checks(entries, namespace)
    entry_issues = _collect_entry_issues(entries, repository, namespace)

    return NamespaceValidationReport(
        namespace=namespace,
        ok=not global_errors and not entry_issues,
        global_errors=global_errors,
        entry_issues=entry_issues,
    )


def _global_checks(entries: list[Entry], namespace: str) -> list[str]:
    """Return every bundle-wide error for ``entries`` as a flat list."""
    errors: list[str] = []
    errors.extend(_check_team_count(entries, namespace))
    errors.extend(_check_uniform_namespace(entries, namespace))
    errors.extend(_check_uniform_user_id(entries))
    errors.extend(_check_no_duplicate_ids(entries, namespace))
    errors.extend(_check_dangling_refs(entries, namespace))
    return errors


def _check_team_count(entries: list[Entry], namespace: str) -> list[str]:
    """Require exactly one ``kind=team`` entry; surface too-many / too-few."""
    teams = [e for e in entries if e.kind == "team"]
    if len(teams) == 0:
        return [f"namespace '{namespace}' has no team entry"]
    if len(teams) > 1:
        ids = sorted(t.id for t in teams)
        return [f"namespace '{namespace}' has multiple team entries: {ids}"]
    return []


def _check_uniform_namespace(entries: list[Entry], namespace: str) -> list[str]:
    """One error message per entry whose ``namespace`` disagrees with the first entry's."""
    return [
        f"entry '{e.id}' has namespace '{e.namespace}' but bundle namespace is '{namespace}'"
        for e in entries
        if e.namespace != namespace
    ]


def _check_uniform_user_id(entries: list[Entry]) -> list[str]:
    """Anchored-by-the-team ownership check; skipped when team count != 1."""
    teams = [e for e in entries if e.kind == "team"]
    if len(teams) != 1:
        return []
    team_user_id = teams[0].user_id
    return [
        f"entry '{e.id}' user_id '{e.user_id}' != team user_id '{team_user_id}'"
        for e in entries
        if e.kind != "team" and e.user_id != team_user_id
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
    """Yield every ``__ref__`` target id reachable under ``node`` (dict/list walk)."""
    if isinstance(node, dict):
        if REF_KEY in node:
            target = node[REF_KEY]
            if isinstance(target, str):
                yield target
            return
        for value in node.values():
            yield from _iter_payload_ref_targets(value)
        return
    if isinstance(node, list):
        for item in node:
            yield from _iter_payload_ref_targets(item)


def _collect_entry_issues(
    entries: list[Entry], repository: EntryRepository, namespace: str
) -> list[EntryValidationIssue]:
    """Build per-entry issues; skip entries with empty error lists."""
    issues: list[EntryValidationIssue] = []
    for e in entries:
        errs = _per_entry_checks(e, repository, namespace)
        if errs:
            issues.append(EntryValidationIssue(entry_id=e.id, kind=e.kind, errors=errs))
    return issues


def _per_entry_checks(entry: Entry, repository: EntryRepository, namespace: str) -> list[str]:
    """Run allowlist, lineage-pair, and transient-validation checks for ``entry``."""
    errors: list[str] = []
    cls: type[BaseModel] | None = None
    try:
        cls = load_model_type(entry.model_type)
    except CatalogValidationError as exc:
        errors.extend(exc.errors)
    errors.extend(_check_lineage_pair(entry))
    if cls is not None:
        errors.extend(_check_transient_validation(entry, repository, namespace, cls))
    return errors


def _check_lineage_pair(entry: Entry) -> list[str]:
    """Flag half-set lineage pairs (exactly one of ``parent_namespace`` / ``parent_id``)."""
    ns_set = entry.parent_namespace is not None
    id_set = entry.parent_id is not None
    if ns_set != id_set:
        return [
            f"entry '{entry.id}' has lineage pair half-set "
            f"(parent_namespace={entry.parent_namespace!r}, parent_id={entry.parent_id!r})"
        ]
    return []


def _check_transient_validation(
    entry: Entry, repository: EntryRepository, namespace: str, cls: type[BaseModel]
) -> list[str]:
    """Run ``populate_refs`` + ``cls.model_validate`` and collect every message."""
    try:
        populated = populate_refs(entry.payload, repository, namespace)
    except CatalogValidationError as exc:
        return list(exc.errors)
    try:
        cls.model_validate(populated)
    except ValidationError as exc:
        return [f"payload does not validate against {entry.model_type}: {exc}"]
    except CatalogValidationError as exc:  # pragma: no cover — defensive
        return list(exc.errors)
    return []
