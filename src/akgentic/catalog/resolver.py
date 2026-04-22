"""Ref-sentinel constants, allowlist loader, and the v2 resolver pipeline.

This module owns the v2 ref-sentinel sentinel keys (``REF_KEY``, ``TYPE_KEY``)
and the ``load_model_type(path)`` function that imports a Pydantic
``BaseModel`` class by dotted path, gated behind three defensive checks:

1. The path must start with one of ``_ALLOWED_PREFIXES`` (storage + runtime
   defence in depth — ``models.entry.AllowlistedPath`` enforces the same
   prefix at Pydantic construction time).
2. The resolved class must be a subclass of ``pydantic.BaseModel``.
3. The resolved class must not declare Pydantic fields named ``__ref__`` or
   ``__type__`` (the reserved keys used by the resolver's sentinel scheme —
   collisions would break round-tripping).

It also exposes the v2 resolver pipeline built on top of those primitives:

* :func:`populate_refs` — recursive ref replacement with cycle, missing-target,
  and ``__type__`` mismatch detection; namespace-bounded. Ref-marker positions
  resolve to typed Pydantic instances of the referenced entry's ``model_type``
  (Story 15.6) so polymorphic fields (``list[ToolCard]``, ``list[type]``,
  …) validate against concrete subclasses without authoring workarounds.
* :func:`reconcile_refs` — walks input + dumped trees in lockstep to preserve
  author-written ref markers against a Pydantic-dumped payload.
* :func:`resolve` — full hydration of an ``Entry`` into a runtime ``BaseModel``.
* :func:`prepare_for_write` — five-step write pipeline producing the
  intent-preserving, ref-preserving stored payload.
* :func:`validate_delete` — inbound-ref guard that returns a list of
  human-readable blocker messages (empty list = safe to delete).

``check_ownership`` is deliberately NOT part of this module — Story 15.5 is the
right place to decide whether ownership enforcement lives inside
``prepare_for_write`` or inside the ``Catalog`` service itself.
"""

from __future__ import annotations

import sys
from typing import Any, Final

from pydantic import BaseModel, ValidationError

from akgentic.core.utils.deserializer import import_class

from .models.entry import Entry
from .models.errors import CatalogValidationError
from .repositories.base import EntryRepository

__all__ = [
    "REF_KEY",
    "TYPE_KEY",
    "enumerate_allowlisted_model_types",
    "load_model_type",
    "populate_refs",
    "prepare_for_write",
    "reconcile_refs",
    "resolve",
    "validate_delete",
]


REF_KEY: Final[str] = "__ref__"
"""Sentinel dict key marking a ref placeholder inside a resolved payload.

A payload dict containing ``REF_KEY`` has been populated by the resolver and
must be hydrated (looked up in the repository) before use at runtime.
"""

TYPE_KEY: Final[str] = "__type__"
"""Sentinel dict key carrying the FQCN of a referenced entry's model type.

Emitted next to ``REF_KEY`` so the resolver can validate the target's type
without loading the target entry eagerly.
"""

# Runtime allowlist for ``load_model_type``. Duplicated intentionally in
# ``models.entry`` for the annotation-layer defence — two layers, two
# policies that only happen to agree today. See Story 15.1 Dev Notes.
_ALLOWED_PREFIXES: tuple[str, ...] = ("akgentic.",)

_RESERVED_KEYS: frozenset[str] = frozenset({REF_KEY, TYPE_KEY})


def load_model_type(path: str) -> type[BaseModel]:
    """Import and return a Pydantic ``BaseModel`` class by dotted path.

    Three checks run in order:

    1. ``path`` must start with one of ``_ALLOWED_PREFIXES``.
    2. The resolved object must be a subclass of ``pydantic.BaseModel``.
    3. The resolved class must not declare Pydantic fields named ``__ref__``
       or ``__type__``.

    Args:
        path: Dotted class path (e.g. ``"akgentic.core.agent_card.AgentCard"``).

    Returns:
        The imported class.

    Raises:
        CatalogValidationError: If any of the three checks fails. The error
            carries a single-element ``errors`` list with a substring-stable
            message (``"outside allowlist"``, ``"is not a Pydantic BaseModel
            subclass"``, or ``"reserved ref-sentinel fields"``) so callers
            can assert on behaviour without loading the exception chain.
    """
    if not any(path.startswith(prefix) for prefix in _ALLOWED_PREFIXES):
        raise CatalogValidationError([f"model_type '{path}' outside allowlist {_ALLOWED_PREFIXES}"])

    cls = import_class(path)

    if not (isinstance(cls, type) and issubclass(cls, BaseModel)):
        raise CatalogValidationError([f"model_type '{path}' is not a Pydantic BaseModel subclass"])

    collisions = sorted(_RESERVED_KEYS & set(cls.model_fields.keys()))
    if collisions:
        raise CatalogValidationError(
            [f"model_type '{path}' declares reserved ref-sentinel fields: {collisions}"]
        )

    return cls


def populate_refs(
    node: Any,
    repository: EntryRepository,
    namespace: str,
    _visiting: set[tuple[str, str]] | None = None,
) -> Any:
    """Recursively replace ref markers in ``node`` with their resolved payloads.

    Walks the payload tree. A dict with a ``REF_KEY`` (``"__ref__"``) entry is
    treated as a ref marker and replaced by a **typed Pydantic instance** built
    from the target entry's declared ``model_type`` (Story 15.6); ordinary
    dicts and lists recurse structurally; every other value is returned
    unchanged. The ``namespace`` is forwarded as-is to every recursive call —
    cross-namespace resolution is structurally impossible at runtime.

    Return-type note (Story 15.6): nodes that are **not** ref markers still
    return structurally as ``dict | list | leaf``. Only ref-marker positions
    become ``BaseModel`` instances. Downstream ``cls.model_validate(...)`` calls
    accept nested Pydantic instances where a field's declared annotation is
    the instance's base class (Pydantic v2 behaviour), so the return-type
    widening is compatible with every existing caller (``resolve``,
    ``prepare_for_write``, ``validation._check_transient_validation``).

    Args:
        node: Arbitrary payload subtree (dict, list, or leaf value).
        repository: Entry repository used to resolve refs in ``namespace``.
        namespace: The namespace used for every repository lookup in this call.
        _visiting: Internal cycle-detection set of ``(namespace, target_id)``
            pairs already traversed on the current ref chain. Callers pass
            ``None`` (the default); the function builds a fresh set per call.

    Returns:
        A new payload subtree with every ref marker replaced by a typed
        Pydantic instance of the referenced entry's ``model_type``. Dict and
        list inputs are never mutated — fresh containers are built for every
        recursion.

    Raises:
        CatalogValidationError: If a ref cycle is detected, the target id is
            absent from ``repository``, a ``TYPE_KEY`` hint does not match
            the target entry's ``model_type``, or the target entry's payload
            fails validation against its own ``model_type`` class. The error
            carries a single-element ``errors`` list with substring-stable
            messages (``"cycle"``, ``"not found"``, ``"expected X"`` +
            ``"got Y"``, or ``"Payload of '<id>' does not validate against
            <model_type>"``).
    """
    visiting: set[tuple[str, str]] = set() if _visiting is None else _visiting

    if isinstance(node, dict):
        if REF_KEY in node:
            return _populate_ref_marker(node, repository, namespace, visiting)
        return {k: populate_refs(v, repository, namespace, visiting) for k, v in node.items()}

    if isinstance(node, list):
        return [populate_refs(v, repository, namespace, visiting) for v in node]

    return node


def _populate_ref_marker(
    node: dict[str, Any],
    repository: EntryRepository,
    namespace: str,
    visiting: set[tuple[str, str]],
) -> Any:
    """Resolve a single ref-marker dict into a typed Pydantic instance.

    Checks run in order — cycle, missing target, ``__type__`` mismatch, then
    (Story 15.6) recursive population of nested refs inside the target's
    payload, ``load_model_type`` on the target's declared ``model_type``, and
    ``cls.model_validate`` to build a typed instance.

    Cycle comes first to avoid a redundant repository lookup when we already
    know we are looping; missing-target comes second because the ``__type__``
    check needs the fetched target in hand. Every pre-instantiation failure
    raises ``CatalogValidationError`` with the existing substring-stable
    messages (``"cycle"``, ``"not found"``, ``"expected X"`` + ``"got Y"``).
    A validation failure at instantiation time raises
    ``CatalogValidationError`` naming the referenced entry's id and
    ``model_type`` so authoring errors point at the offending entry, not at
    the parent that happens to reference it (AC #5).
    """
    target_id = node[REF_KEY]
    expected = node.get(TYPE_KEY)
    key = (namespace, target_id)

    if key in visiting:
        raise CatalogValidationError([f"Reference cycle detected at ({namespace}, {target_id})"])

    target = repository.get(namespace, target_id)
    if target is None:
        raise CatalogValidationError([f"Ref '{target_id}' not found in namespace '{namespace}'"])

    if expected is not None and target.model_type != expected:
        raise CatalogValidationError(
            [f"Ref '{target_id}' expected {expected}, got {target.model_type}"]
        )

    # Story 15.6: recurse into the target's payload (resolves nested refs),
    # then validate against the target's declared model_type so the splice
    # becomes a typed instance — Pydantic accepts a subclass instance where
    # the parent field is annotated with its base class.
    populated_payload = populate_refs(target.payload, repository, namespace, visiting | {key})
    cls = load_model_type(target.model_type)
    try:
        return cls.model_validate(populated_payload)
    except ValidationError as e:
        raise CatalogValidationError(
            [f"Payload of '{target.id}' does not validate against {target.model_type}: {e}"]
        ) from e


def resolve(entry: Entry, repository: EntryRepository) -> BaseModel:
    """Hydrate ``entry`` into an instance of its declared runtime Pydantic class.

    Composition: ``cls = load_model_type(entry.model_type)``; ``populated =
    populate_refs(entry.payload, repository, entry.namespace)``; finally
    ``cls.model_validate(populated)``. Errors from ``load_model_type`` and
    ``populate_refs`` propagate unchanged (per AC14/AC15). Only the
    Pydantic ``ValidationError`` from ``model_validate`` is converted to
    ``CatalogValidationError``, preserving the original traceback via
    ``raise ... from e`` (AC16).

    Story 15.6 note: ``populate_refs`` may now return a tree that contains
    nested typed Pydantic instances at ref-marker positions. The final
    ``cls.model_validate`` call accepts such trees — Pydantic v2 treats a
    concrete subclass instance as a valid value wherever the declared
    annotation is the instance's base class, so polymorphic fields
    (``list[ToolCard]``, ``list[type]``, …) validate without
    per-payload workarounds.

    Args:
        entry: The catalog entry to hydrate.
        repository: Entry repository used to resolve refs in ``entry.namespace``.

    Returns:
        A runtime instance of the class named by ``entry.model_type``.

    Raises:
        CatalogValidationError: From ``load_model_type`` (allowlist, BaseModel,
            reserved-key checks), from ``populate_refs`` (cycle, missing, type
            mismatch), or wrapping a ``pydantic.ValidationError`` from
            ``model_validate`` with the substring ``"Payload does not validate
            against"`` followed by the ``model_type`` string.
    """
    cls = load_model_type(entry.model_type)
    populated = populate_refs(entry.payload, repository, entry.namespace)
    try:
        return cls.model_validate(populated)
    except ValidationError as e:
        raise CatalogValidationError(
            [f"Payload does not validate against {entry.model_type}: {e}"]
        ) from e


def reconcile_refs(input_node: Any, dumped_node: Any) -> Any:
    """Restore ref markers from ``input_node`` against a dumped Pydantic tree.

    The dumped tree (from ``obj.model_dump(mode='python', exclude_unset=True)``)
    is the authority for every non-ref field — it carries validator-normalised
    values. Author-written ref markers in ``input_node`` must win verbatim so
    that the stored payload round-trips back to itself when re-resolved.

    Args:
        input_node: The original payload subtree the author wrote (may carry
            ``REF_KEY`` markers).
        dumped_node: The corresponding subtree from ``model_dump``.

    Returns:
        A reconciled subtree: ref markers preserved verbatim from ``input_node``,
        non-ref dict keys recursed pairwise, list elements zipped pairwise with
        strict length, and leaves taken from ``dumped_node``. Neither input
        tree is mutated.

    Raises:
        ValueError: If ``input_node`` and ``dumped_node`` are both lists but of
            mismatched lengths (``zip(..., strict=True)`` raises).
    """
    if isinstance(input_node, dict):
        if REF_KEY in input_node:
            return input_node
        return _reconcile_dict(input_node, dumped_node)

    if isinstance(input_node, list) and isinstance(dumped_node, list):
        return [reconcile_refs(i, d) for i, d in zip(input_node, dumped_node, strict=True)]

    return dumped_node


def _reconcile_dict(input_node: dict[str, Any], dumped_node: Any) -> dict[str, Any]:
    """Reconcile a non-ref input dict against the dumped counterpart.

    Iterates input keys in order. For each key: if it is present in
    ``dumped_node``, recurse pairwise; otherwise, if the input value is itself
    a ref-marker dict, preserve it verbatim (the "unset-but-refed" branch from
    AC19); otherwise drop the key (the dumped tree is authoritative for
    absent non-ref fields).
    """
    dumped_dict = dumped_node if isinstance(dumped_node, dict) else {}
    result: dict[str, Any] = {}
    for key, value in input_node.items():
        if key in dumped_dict:
            result[key] = reconcile_refs(value, dumped_dict[key])
        elif isinstance(value, dict) and REF_KEY in value:
            result[key] = value
    return result


def prepare_for_write(entry: Entry, repository: EntryRepository) -> Entry:
    """Run the five-step write pipeline and return a ref-preserving ``Entry``.

    Pipeline (each step's failure short-circuits the remainder, raising
    ``CatalogValidationError`` before the next step runs):

    1. ``resolved = populate_refs(entry.payload, repo, entry.namespace)``
       — surfaces missing target, ``__type__`` mismatch, and cycle errors.
    2. ``cls = load_model_type(entry.model_type)``
       — surfaces allowlist / reserved-key / non-BaseModel errors.
    3. ``obj = cls.model_validate(resolved)``
       — ``ValidationError`` is converted to ``CatalogValidationError`` with
       the substring ``"Payload does not validate against"`` and the
       ``model_type`` string, chained via ``from e``.
    4. ``dumped = obj.model_dump(mode="python", exclude_unset=True)``
       — intent-preserving dump.
    5. ``stored = reconcile_refs(entry.payload, dumped)``
       — restore the author's ref markers on top of the dumped tree.

    The returned ``Entry`` is ``entry.model_copy(update={"payload": stored})``
    — a shallow copy; the input ``entry`` is not mutated.

    Ownership enforcement (team-entry existence + ``user_id`` match) is
    currently the ``Catalog`` service's responsibility and will remain outside
    this function until Story 15.5 finalises the placement.

    Args:
        entry: Candidate entry to persist.
        repository: Entry repository for ref resolution in ``entry.namespace``.

    Returns:
        A new ``Entry`` with the reconciled, intent-preserving payload.

    Raises:
        CatalogValidationError: From any pipeline step that fails.
    """
    resolved = populate_refs(entry.payload, repository, entry.namespace)
    cls = load_model_type(entry.model_type)
    obj = _validate_payload(cls, resolved, entry.model_type)
    dumped = obj.model_dump(mode="python", exclude_unset=True)
    stored = reconcile_refs(entry.payload, dumped)
    return entry.model_copy(update={"payload": stored})


def _validate_payload(cls: type[BaseModel], resolved: Any, model_type: str) -> BaseModel:
    """Run ``cls.model_validate`` and convert Pydantic errors to catalog errors.

    Extracted so ``prepare_for_write`` stays short; the substring
    ``"Payload does not validate against"`` and the ``model_type`` string are
    pinned by AC23 step 3 and AC16.
    """
    try:
        return cls.model_validate(resolved)
    except ValidationError as e:
        raise CatalogValidationError(
            [f"Payload does not validate against {model_type}: {e}"]
        ) from e


def validate_delete(namespace: str, id: str, repository: EntryRepository) -> list[str]:
    """Return blocker messages for deleting ``(namespace, id)``; empty means safe.

    Two-phase check:

    1. If ``repository.get(namespace, id)`` returns ``None``, the target does
       not exist — surface a single-element list containing ``"not found"``
       and the offending ``(namespace, id)`` pair.
    2. Otherwise, call ``repository.find_references(namespace, id)`` and build
       one blocker message per inbound referrer, preserving repository order.

    No branching on ``user_id``, ``parent_namespace``, or ``kind`` — the rule
    is uniform across enterprise and user namespaces per ADR-007/ADR-008:
    any inbound ref within the passed namespace blocks the delete, and
    cross-namespace lineage is never policed by ``validate_delete``.

    Args:
        namespace: Namespace of the entry to delete.
        id: Id of the entry to delete within ``namespace``.
        repository: Entry repository providing ``get`` and ``find_references``.

    Returns:
        A list of human-readable blocker messages. An empty list is the
        "safe to delete" signal.
    """
    target = repository.get(namespace, id)
    if target is None:
        return [f"Entry ({namespace}, {id}) not found"]

    referrers = repository.find_references(namespace, id)
    return [
        f"Entry '{r.id}' (kind={r.kind}) in namespace '{namespace}' references '{id}'"
        for r in referrers
    ]


def enumerate_allowlisted_model_types() -> list[str]:
    """Enumerate allowlisted ``BaseModel`` subclasses loaded under ``akgentic.*``.

    Walks a snapshot of ``sys.modules`` to avoid mutation-during-iteration
    issues. Per-module introspection errors are swallowed — optional
    dependencies may be absent or partially imported. ``load_model_type``
    acts as the authoritative allowlist + ``BaseModel`` + reserved-key gate
    so enumeration never broadens the allowlist.

    Used by both the REST router (``GET /catalog/model_types``) and the
    ``ak-catalog model-types`` CLI verb.
    """
    results: set[str] = set()
    modules_snapshot = list(sys.modules.items())
    for module_name, module in modules_snapshot:
        if not module_name.startswith("akgentic.") or module is None:
            continue
        _collect_allowlisted(module, results)
    return sorted(results)


def _collect_allowlisted(module: Any, results: set[str]) -> None:
    """Add every allowlisted ``BaseModel`` subclass from ``module`` into ``results``."""
    try:
        items = list(vars(module).items())
    except Exception:  # noqa: BLE001 — defensive; partially imported modules
        return
    for _name, value in items:
        if not isinstance(value, type) or not issubclass(value, BaseModel):
            continue
        path = f"{value.__module__}.{value.__name__}"
        if not path.startswith("akgentic.") or path in results:
            continue
        try:
            load_model_type(path)
        except Exception:  # noqa: BLE001 — swallow reserved-key or import errors
            continue
        results.add(path)
