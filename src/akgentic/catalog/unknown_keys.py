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

The comparison rests on ``exclude_unset=True`` at **both** call sites
(:func:`akgentic.catalog.resolver.prepare_for_write` step 4 and
:func:`akgentic.catalog.validation._check_transient_validation`). Pydantic
records ``__pydantic_fields_set__`` per submodel as exactly the keys it
accepted, so an ``exclude_unset`` dump emits the accepted keys and nothing
else, at every level. That is what turns "absent from the dump" into "the
model never accepted it". Switch the flag off and every defaulted field
appears in the dump, the diff finds nothing, and the check is inert.

Deliberately **not** reported (each is a key the write path keeps, so nothing
is lost):

* the sentinel keys ``__ref__`` / ``__type__`` / ``__namespace__``, plus the
  ``__model__`` polymorphic tag emitted by ``akgentic-core``'s serializer —
  exempt at every depth, not only at the root;
* the interior of a ref marker (a dict carrying ``__ref__``), whose sibling
  overrides are story 29.2's subject;
* an authored key absent from the dump whose value is itself a ref marker —
  ``_reconcile_dict``'s "unset-but-refed" branch preserves it verbatim.
"""

from __future__ import annotations

from typing import Any, Final

from .resolver import NAMESPACE_KEY, REF_KEY, TYPE_KEY

__all__ = ["UNKNOWN_KEY_MESSAGE", "find_unknown_keys"]


_MODEL_KEY: Final[str] = "__model__"
"""Polymorphic FQCN tag stamped by ``akgentic.core``'s ``SerializableBaseModel``.

Stripped by that package's before-validator, so it never reaches
``__pydantic_fields_set__`` and would otherwise read as an unknown key on any
payload pasted from a serialized dump. Declared here rather than imported:
``akgentic-catalog`` does not participate in the ``__model__`` protocol and
must not grow a dependency on it for the sake of one exemption.
"""

_EXEMPT_KEYS: Final[frozenset[str]] = frozenset({REF_KEY, TYPE_KEY, NAMESPACE_KEY, _MODEL_KEY})
"""Keys never reported as unknown, at any depth of the authored tree."""

UNKNOWN_KEY_MESSAGE: Final[str] = "unknown key '{path}' — not a field of {model_type}"
"""One-per-path error template, shared verbatim by the validate and write paths.

Both paths format this same string so a finding reads identically whether it
surfaced from ``validate_namespace_yaml`` or from a rejected ``create`` —
which is the point of the pair being detected by one helper.
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
            # this payload. Sibling overrides on it are story 29.2's subject.
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
        if key in _EXEMPT_KEYS:
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
