"""The ref-marker shape and the payload walk, owned in one place.

Implements ADR-019 §D1. Before this module the marker shape was re-derived by
five hand-rolled parsers and the recursive payload walk was copy-pasted across
nine walkers, each of which decided for itself whether a malformed marker
should raise or be skipped. This module owns both, once.

**Why this module is a leaf.** It imports nothing from ``akgentic.catalog``
beyond :class:`~akgentic.catalog.models.errors.CatalogValidationError`, and
``models/errors.py`` itself imports nothing at all. That gives ``resolver`` and
``unknown_keys`` a common dependency that points back at neither, which is what
lets the package's one import cycle dissolve. A test in
``tests/v2/test_refs.py`` parses this file's AST and fails on any other
first-party import; do not add one.

**Two entry points, deliberately.** :meth:`RefMarker.parse` raises on a
malformed marker — the resolver must refuse to hydrate one. :meth:`RefMarker.
classify` returns ``None`` instead — a guard walking arbitrary stored payloads
must not raise mid-scan over data it did not author. Same derivation, two
policies.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, ValidationError

from .models.errors import CatalogValidationError

__all__ = [
    "NAMESPACE_KEY",
    "REF_KEY",
    "RESERVED_REF_KEYS",
    "TYPE_KEY",
    "RefMarker",
    "walk_payload",
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

NAMESPACE_KEY: Final[str] = "__namespace__"
"""Sentinel dict key carrying the target namespace of a cross-namespace ref.

Implements ADR-008 §D2 — the canonical cross-ns sentinel. A ref-marker dict
may carry ``NAMESPACE_KEY`` next to ``REF_KEY`` (and optionally ``TYPE_KEY``)
to address an entry in a different namespace; the resolver gates the lookup
on the data-driven shareable-flag (the target namespace's ``_meta`` entry has
``payload["shareable"] is True`` — ADR-008 §D2 as updated 2026-05-08 rev 2).
The shorthand ``{"__ref__": "<ns>.<id>"}`` is parsed equivalently — the
resolver splits on the first ``.``. Same-namespace refs (no
``NAMESPACE_KEY``, no dot in ``__ref__``) bypass the gate entirely.
"""

RESERVED_REF_KEYS: Final[frozenset[str]] = frozenset({REF_KEY, TYPE_KEY, NAMESPACE_KEY})
"""The three sentinel keys a ref marker may carry, and no others.

A marker is a pure pointer: any key outside this set sitting next to
``REF_KEY`` is a validation error (ADR-018 §D4).
"""

_DISAGREEMENT_MESSAGE: Final[str] = (
    "Ref marker has both shorthand 'ns.id' and explicit __namespace__ — these "
    "disagree: '{shorthand_ns}' vs '{explicit_ns}'"
)

_NON_STRING_REF_MESSAGE: Final[str] = (
    "Ref marker '__ref__' must be a string naming the target entry, got {type_name}"
)


class _DerivationError(BaseModel):
    """Why a node could not be derived into a :class:`RefMarker`.

    Returned rather than raised so the single derivation can serve both a
    raising and a non-raising entry point.
    """

    model_config = ConfigDict(frozen=True)

    message: str


class RefMarker(BaseModel):
    """A resolved ref marker: which entry, in which namespace, of which type.

    Frozen — a marker is a value, so ``==`` and hashing come for free.
    """

    model_config = ConfigDict(frozen=True)

    target_namespace: str
    target_id: str
    expected_type: str | None = None

    @classmethod
    def parse(cls, node: dict[str, Any], current_namespace: str) -> RefMarker:
        """Return the marker ``node`` denotes, raising when it is malformed.

        Reproduces ``resolver._resolve_target_namespace``: the canonical form
        carries an explicit ``__namespace__``; the shorthand encodes the
        namespace as a ``<ns>.<id>`` prefix on ``__ref__`` and is split on the
        FIRST dot only, so ids may continue to contain ``.``. The shorthand
        branch is checked first, so ``{"__ref__": "A.x", "__namespace__": "A"}``
        resolves to ``("A", "x")``.

        Args:
            node: The ref-marker dict. Must contain ``REF_KEY``.
            current_namespace: The enclosing namespace, used when the marker
                carries neither a shorthand nor an explicit ``__namespace__``.

        Returns:
            The marker.

        Raises:
            CatalogValidationError: When shorthand and explicit
                ``__namespace__`` both appear and disagree, or when
                ``__ref__`` is not a string.
            KeyError: When ``node`` carries no ``__ref__``.
            ValidationError: When ``__namespace__`` or ``__type__`` carries a
                non-string — it reaches model construction unguarded.
                :meth:`classify` returns ``None`` there instead.
        """
        derived = _derive(
            raw_ref=node[REF_KEY],
            explicit_ns=node.get(NAMESPACE_KEY),
            expected_type=node.get(TYPE_KEY),
            default_namespace=current_namespace,
        )
        if isinstance(derived, _DerivationError):
            raise CatalogValidationError([derived.message])
        return derived

    @classmethod
    def classify(cls, node: Any) -> RefMarker | None:
        """Return the marker ``node`` denotes, or ``None`` — never raising.

        For scanning walkers over arbitrary stored payloads, where a malformed
        node is data to skip rather than an error to report. ``None`` comes
        back for a non-dict, a dict without ``__ref__``, a non-string
        ``__ref__``, and a shorthand that disagrees with an explicit
        ``__namespace__``.

        Unlike :meth:`parse`, ``classify`` takes no ``current_namespace`` — a
        scan has no enclosing namespace to default to. **A marker carrying
        neither a shorthand nor an explicit ``__namespace__`` therefore
        classifies with ``target_namespace == ""``**, which is how a
        same-namespace marker looks coming out of this method. Callers
        comparing full ``(namespace, id)`` pairs must account for it.

        Args:
            node: Any node encountered during a payload walk.

        Returns:
            The marker, or ``None`` when ``node`` does not denote a
            well-formed one.
        """
        if not isinstance(node, dict) or REF_KEY not in node:
            return None
        try:
            derived = _derive(
                raw_ref=node[REF_KEY],
                explicit_ns=node.get(NAMESPACE_KEY),
                expected_type=node.get(TYPE_KEY),
                default_namespace="",
            )
        except ValidationError:
            # A non-string sibling sentinel (``__namespace__``, ``__type__``)
            # in a payload this scan did not author. Skip it; do not raise.
            return None
        return derived if isinstance(derived, RefMarker) else None


def _derive(
    *,
    raw_ref: Any,
    explicit_ns: Any,
    expected_type: Any,
    default_namespace: str,
) -> RefMarker | _DerivationError:
    """Derive a marker from a node's sentinel values, or say why it cannot.

    The single algorithm behind both :meth:`RefMarker.parse` and
    :meth:`RefMarker.classify`; the two differ only in what they do with a
    :class:`_DerivationError`.
    """
    if not isinstance(raw_ref, str):
        return _DerivationError(
            message=_NON_STRING_REF_MESSAGE.format(type_name=type(raw_ref).__name__)
        )
    # Shorthand parsing — first-dot split. Ids may continue to contain dots.
    if "." in raw_ref:
        shorthand_ns, shorthand_id = raw_ref.split(".", 1)
        if explicit_ns is not None and explicit_ns != shorthand_ns:
            return _DerivationError(
                message=_DISAGREEMENT_MESSAGE.format(
                    shorthand_ns=shorthand_ns, explicit_ns=explicit_ns
                )
            )
        return RefMarker(
            target_namespace=shorthand_ns,
            target_id=shorthand_id,
            expected_type=expected_type,
        )
    if explicit_ns is not None:
        return RefMarker(
            target_namespace=explicit_ns,
            target_id=raw_ref,
            expected_type=expected_type,
        )
    return RefMarker(
        target_namespace=default_namespace,
        target_id=raw_ref,
        expected_type=expected_type,
    )


def walk_payload(
    node: Any,
    *,
    on_ref: Callable[[dict[str, Any]], None],
    on_leaf: Callable[[Any], None] | None = None,
) -> None:
    """Walk a payload tree depth-first, left-to-right, visiting every node.

    A dict carrying ``REF_KEY`` is handed to ``on_ref`` and **not descended
    into** — a marker is a pure pointer with no interior (ADR-018 §D4). Making
    that structural here is the point of the function: every walker used to
    have to remember it.

    Containers themselves never reach ``on_leaf``; only scalars and other
    non-container values do. There is no short-circuit — a callback's return
    value is ignored and the whole tree is always visited. The walk returns
    nothing and mutates nothing; callers accumulate through a closure.

    Args:
        node: Any payload node — dict, list, or scalar.
        on_ref: Called with each ref-marker dict encountered.
        on_leaf: Called with each non-container value, when given.
    """
    if isinstance(node, dict):
        if REF_KEY in node:
            on_ref(node)
            return
        for value in node.values():
            walk_payload(value, on_ref=on_ref, on_leaf=on_leaf)
        return
    if isinstance(node, list):
        for item in node:
            walk_payload(item, on_ref=on_ref, on_leaf=on_leaf)
        return
    if on_leaf is not None:
        on_leaf(node)
