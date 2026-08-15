"""Structural diff detecting authored payload keys the model never accepted.

The invariant this module exists to make true: **nothing an author wrote is
silently discarded by the catalog**. Before it, an unknown key was
indistinguishable from an absent one — ``model_dump(exclude_unset=True)``
omits both, and :func:`akgentic.catalog.resolver._reconcile_dict` then drops
the key without a word. A misprinted ``temperatur`` was therefore accepted by
Validate, accepted by Save, and discovered weeks later as a field that had
never been stored.

Why a structural diff rather than ``extra="forbid"``: the payload models are
declared across five submodules and belong to their owners, and a Pydantic
``ConfigDict`` binds only at the class that declares it — a root-level
``extra="forbid"`` would leave every nested submodel untouched. The diff needs
no cooperation from the model tree: it compares the tree the author wrote
against the tree Pydantic accepted, so it reaches every depth of every model,
including third-party ones.

Both call sites dump with ``exclude_unset=True``
(:func:`akgentic.catalog.resolver.prepare_for_write` step 4 and
:func:`akgentic.catalog.validation._check_transient_validation`). That flag is
load-bearing for the **stored shape**, not for this diff: ``prepare_for_write``
persists the dump through ``reconcile_refs``, so dumping defaults would write
every unset field into storage and destroy intent preservation. The diff itself
is indifferent to it — this walk reports authored keys *absent* from the dump,
and dumping defaults only ever *adds* declared-field keys. A misprint is not a
declared field, so it stays absent and stays reported either way. Do not read
the flag as the thing that makes this check work; changing it silently changes
what gets stored, which is the reason to leave it alone.

Reported even though the model accepted the key: a ``mode="before"`` validator
that **renames** a key consumes it before field validation, so it never reaches
the dump and lands here as unknown. ``akgentic.llm.UsageLimits`` is the live
instance — it folds the pre-split ``request_limit`` onto ``run_request_limit``,
and a payload written with the old spelling is refused. The catalog therefore
requires the field name a model actually stores; a deprecation shim must use a
validation alias, which Pydantic records, rather than a renaming validator.

Deliberately **not** reported (each is a key the write path keeps, so nothing
is lost):

* the sentinel keys ``__ref__`` / ``__type__`` / ``__namespace__``, plus the
  ``__model__`` polymorphic tag emitted by ``akgentic-core``'s serializer —
  exempt at every depth, not only at the root;
* the interior of a ref marker (a dict carrying ``__ref__``), whose sibling
  overrides are checked at the resolver instead — see
  :func:`akgentic.catalog.resolver._reject_unknown_override_keys`, which needs
  the *target's* model and so cannot run inside this walk;
* an authored key absent from the dump whose value is itself a ref marker —
  ``_reconcile_dict``'s "unset-but-refed" branch preserves it verbatim.
"""

from __future__ import annotations

from typing import Any, Final

from .resolver import NAMESPACE_KEY, REF_KEY, TYPE_KEY

__all__ = [
    "EXEMPT_KEYS",
    "UNKNOWN_KEY_MESSAGE",
    "UNKNOWN_OVERRIDE_KEY_MESSAGE",
    "find_unknown_keys",
]


_MODEL_KEY: Final[str] = "__model__"
"""Polymorphic FQCN tag stamped by ``akgentic.core``'s ``SerializableBaseModel``.

Stripped by that package's before-validator, so it never reaches
``__pydantic_fields_set__`` and would otherwise read as an unknown key on any
payload pasted from a serialized dump. Declared here rather than imported:
``akgentic-catalog`` does not participate in the ``__model__`` protocol and
must not grow a dependency on it for the sake of one exemption.
"""

EXEMPT_KEYS: Final[frozenset[str]] = frozenset({REF_KEY, TYPE_KEY, NAMESPACE_KEY, _MODEL_KEY})
"""The one notion of "a key we never report", at any depth of any authored tree.

Read by two callers: the walk below, which skips these keys at every level of
the payload, and
:func:`akgentic.catalog.resolver._reject_unknown_override_keys`, which skips
them among a ref marker's siblings. Sharing the set is what keeps a key exempt
on one path from being a finding on the other.
"""

UNKNOWN_KEY_MESSAGE: Final[str] = "unknown key '{path}' — not a field of {model_type}"
"""One-per-path error template, shared verbatim by the validate and write paths.

Both paths format this same string so a finding reads identically whether it
surfaced from ``validate_namespace_yaml`` or from a rejected ``create`` —
which is the point of the pair being detected by one helper.
"""

UNKNOWN_OVERRIDE_KEY_MESSAGE: Final[str] = (
    "unknown override key '{key}' on ref to '{target_id}' — not a field of {model_type}"
)
"""One-per-key template for a misprinted sibling of a ``__ref__`` marker.

Lives beside :data:`UNKNOWN_KEY_MESSAGE` so the two wordings cannot drift
apart: the override case names the **target's** ``model_type`` (the model that
would have to accept the override) and the target id, because the referring
entry's own model never sees the key at all.
"""


def find_unknown_keys(authored: Any, dumped: Any, *, path: str = "") -> list[str]:
    """Return the dotted paths present in ``authored`` but absent from ``dumped``.

    The inverse traversal of
    :func:`akgentic.catalog.resolver.reconcile_refs`: that function walks the
    two trees in lockstep and *drops* every authored key the dump does not
    carry; this one walks them the same way and *names* those keys instead.
    The two therefore agree case for case on which keys are at stake.

    Pure — neither input tree is mutated, no I/O is performed, and nothing is
    imported beyond stdlib typing and the resolver's sentinel constants.

    Args:
        authored: The payload subtree the author wrote.
        dumped: The corresponding subtree of
            ``obj.model_dump(mode="python", exclude_unset=True)``.
        path: Dotted path of the current position, used to prefix the reported
            paths. Callers pass the default (``""``), which reports a top-level
            key as its bare name.

    Returns:
        Dotted paths in document order — the order the keys appear in
        ``authored``. Dict keys are dot-joined, list indices are rendered
        ``[i]``: ``temperatur``, ``llm.temperatur``, ``tools[1].nmae``. An
        empty list means every authored key survived validation.

    Raises:
        ValueError: If ``authored`` and ``dumped`` are both lists of mismatched
            lengths (``zip(..., strict=True)`` raises).
            :func:`akgentic.catalog.resolver.reconcile_refs` raises the same
            way on the same trees; the validate path converts it into an
            ordinary finding to keep its never-raises contract.
    """
    if isinstance(authored, dict):
        if REF_KEY in authored:
            # A ref marker's interior belongs to the referenced entry, not to
            # this payload. Sibling overrides on it are checked against the
            # TARGET's model at the resolver, which is the only place that
            # model is known.
            return []
        return _find_in_dict(authored, dumped, path)

    if isinstance(authored, list) and isinstance(dumped, list):
        found: list[str] = []
        for i, (a, d) in enumerate(zip(authored, dumped, strict=True)):
            found.extend(find_unknown_keys(a, d, path=f"{path}[{i}]"))
        return found

    return []


def _find_in_dict(authored: dict[str, Any], dumped: Any, path: str) -> list[str]:
    """Report the unknown keys of one authored dict level, then recurse.

    A non-dict ``dumped`` counterpart is read as an empty dict, so every
    authored key at that level is unknown — the same authority rule
    :func:`akgentic.catalog.resolver._reconcile_dict` applies when it drops
    them.
    """
    dumped_dict = dumped if isinstance(dumped, dict) else {}
    found: list[str] = []
    for key, value in authored.items():
        if key in EXEMPT_KEYS:
            continue
        child = f"{path}.{key}" if path else key
        if key in dumped_dict:
            found.extend(find_unknown_keys(value, dumped_dict[key], path=child))
        elif isinstance(value, dict) and REF_KEY in value:
            # Unset-but-refed: `_reconcile_dict` keeps this key verbatim, so
            # the author loses nothing and there is nothing to report.
            continue
        else:
            found.append(child)
    return found
