"""Process-wide policy for the ``Entry.model_type`` class-path prefix allowlist.

This module is the single source of truth for which dotted class paths a
catalog entry may name in ``Entry.model_type``. Two enforcement points consult
it — the annotation-layer check in ``akgentic.catalog.models.entry`` (fires at
``Entry(...)`` construction) and the runtime check in
``akgentic.catalog.resolver.load_model_type`` (fires before ``import_class``) —
plus the two enumeration helpers that power the model-type picker. Because
there is one policy, the enforcement points cannot drift apart.

``akgentic.`` is always allowed and is never removable. A deployment widens the
set by exporting ``AKGENTIC_CATALOG_MODEL_TYPE_PREFIXES`` (comma-separated, or
a JSON list) or by calling :func:`set_allowed_prefixes` during startup wiring.
Implements ADR-016 §D1-D7.

Operational contract:

* **Mutation is startup-only.** Call :func:`set_allowed_prefixes` before the
  first ``Entry`` is constructed or resolved. The resolved policy is cached in
  a module global; changing it after entries are in flight means two entries
  in the same process were validated against different policies.
* **The policy is process-wide.** It is not per-request, per-tenant, or
  per-namespace. Every ``Entry`` in the process — whatever its namespace or
  owner — is checked against the same tuple.
* **It is never reachable from the HTTP surface.** No route, request body, or
  catalog entry can change it. The policy comes from the trusted environment
  or from deployment wiring code, never from catalog data.
* **Widening a prefix widens the blast radius.** Every module under an allowed
  prefix becomes something a catalog entry can cause to be imported. Prefer the
  narrowest prefix that covers the models you need — ``acme.core.models.``
  rather than ``acme.``.

Example:
    Widening the policy from deployment wiring::

        from akgentic.catalog import set_allowed_prefixes

        set_allowed_prefixes(["acme.core.models."])
        # -> allowed_prefixes() == ("akgentic.", "acme.core.models.")

    Or equivalently, with no wiring code at all::

        AKGENTIC_CATALOG_MODEL_TYPE_PREFIXES=acme.core.models.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Final

__all__ = [
    "BASE_PREFIX",
    "ENV_VAR",
    "allowed_prefixes",
    "parse_prefixes",
    "reset_allowed_prefixes",
    "set_allowed_prefixes",
]


ENV_VAR: Final[str] = "AKGENTIC_CATALOG_MODEL_TYPE_PREFIXES"
"""Environment variable naming the extra prefixes a deployment authorises.

Two formats are accepted and parse identically: a comma-separated list
(``acme.,contoso.models.``) or a JSON array (``["acme.","contoso.models."]``).
A value starting with ``[`` is read as JSON; anything else is split on ``,``.
"""

BASE_PREFIX: Final[str] = "akgentic."
"""The framework prefix. Always allowed, always first, never removable."""

# Shared substring carried by every configuration error this module raises, so
# callers can assert on one grep-stable marker rather than on phrasing.
_ERROR_MARKER: Final[str] = "invalid model_type prefix"

# Cached parse result. ``None`` means "not yet resolved" — the next
# ``allowed_prefixes()`` call reads the environment. An empty tuple means
# "explicitly resolved to no extra prefixes" and the environment stays unread.
_configured: tuple[str, ...] | None = None


def allowed_prefixes() -> tuple[str, ...]:
    """Return the active prefix policy, resolving it from the environment once.

    On the first call (and on the first call after
    :func:`reset_allowed_prefixes`) the value of :data:`ENV_VAR` is read
    through :func:`parse_prefixes` and cached. Later calls reuse the cache.

    Returns:
        ``(BASE_PREFIX, *configured)`` — :data:`BASE_PREFIX` first, then the
        configured prefixes in first-seen order with any redundant copy of
        :data:`BASE_PREFIX` removed.

    Raises:
        ValueError: If :data:`ENV_VAR` is set to a malformed value. The message
            carries the substring ``invalid model_type prefix``.
    """
    global _configured
    if _configured is None:
        _configured = parse_prefixes(os.environ.get(ENV_VAR))
    return (BASE_PREFIX, *(prefix for prefix in _configured if prefix != BASE_PREFIX))


def set_allowed_prefixes(prefixes: Sequence[str] | str | None) -> None:
    """Replace the cached policy and stop the environment being consulted.

    Startup-only — see the module docstring. Passing ``None`` or an empty
    sequence means "explicitly no extra prefixes"; it still marks the policy
    resolved, so a later :func:`allowed_prefixes` call does **not** fall back
    to reading :data:`ENV_VAR`.

    Args:
        prefixes: Prefixes to authorise, as a sequence of strings or as a raw
            string in either supported environment format. Normalised and
            validated by :func:`parse_prefixes`.

    Raises:
        ValueError: If any prefix is malformed. The message carries the
            substring ``invalid model_type prefix``.
    """
    global _configured
    _configured = parse_prefixes(prefixes)


def reset_allowed_prefixes() -> None:
    """Return the module to its unresolved state.

    The only supported way to make :data:`ENV_VAR` be read again. Tests use it
    (via the autouse fixture in ``tests/v2/conftest.py``) to keep the module
    global from leaking between test functions.
    """
    global _configured
    _configured = None


def parse_prefixes(raw: Sequence[str] | str | None) -> tuple[str, ...]:
    """Parse, normalise, validate, and deduplicate a prefix specification.

    A pure function — it neither reads nor writes the cached policy, and it
    does **not** strip :data:`BASE_PREFIX` (composing the final policy is
    :func:`allowed_prefixes`'s job). Deployment wiring may call it directly to
    validate operator input before handing it to :func:`set_allowed_prefixes`.

    Args:
        raw: ``None`` or a blank string (meaning "unset"); a raw string in
            either environment format; or an already-split sequence of
            prefixes.

    Returns:
        The normalised prefixes in first-seen order, duplicates removed. Each
        one ends with ``.``.

    Raises:
        ValueError: If the JSON form is malformed or does not decode to a list
            of strings, or if any individual prefix fails normalisation. Every
            message carries the substring ``invalid model_type prefix``.
    """
    if raw is None:
        return ()
    tokens = _split_raw(raw) if isinstance(raw, str) else list(raw)
    normalised: dict[str, None] = {}
    for token in tokens:
        normalised[_normalise_prefix(token)] = None
    return tuple(normalised)


def _split_raw(raw: str) -> list[str]:
    """Split a raw environment-style value into unnormalised prefix tokens.

    A wholly blank value is "unset" and yields no tokens — an operator writing
    ``AKGENTIC_CATALOG_MODEL_TYPE_PREFIXES=`` in a compose file or ``.env``
    means "no extra prefixes", and must not brick the process.
    """
    text = raw.strip()
    if not text:
        return []
    if text.startswith("["):
        return _decode_json_tokens(text)
    return text.split(",")


def _decode_json_tokens(text: str) -> list[str]:
    """Decode the JSON-array form, never falling back to comma splitting.

    A value that opens with ``[`` was meant to be JSON; silently re-reading a
    broken array as a comma-separated list would turn a typo into a policy the
    operator did not write.
    """
    try:
        decoded = json.loads(text)
    except ValueError as exc:  # json.JSONDecodeError is a ValueError subclass
        raise ValueError(f"{_ERROR_MARKER}: {text!r} is not valid JSON") from exc
    if not isinstance(decoded, list | tuple) or not all(isinstance(i, str) for i in decoded):
        raise ValueError(f"{_ERROR_MARKER}: {text!r} must be a JSON list of strings")
    return [str(item) for item in decoded]


def _normalise_prefix(token: str) -> str:
    """Strip, append the trailing dot, and shape-check a single prefix.

    Validation is shape-only: every dot-separated segment must be a Python
    identifier. There is deliberately no denylist of dangerous module roots —
    the prefix set comes from the trusted environment, so an operator who
    writes ``os.`` has authorised ``os.``.
    """
    stripped = token.strip()
    if not stripped:
        # An empty prefix matches every importable path and would silently
        # disable the gate. A wholly-empty raw *value* is "unset" (see
        # _split_raw); an empty *token* is a configuration error.
        raise ValueError(f"{_ERROR_MARKER}: empty prefix")
    prefix = stripped if stripped.endswith(".") else f"{stripped}."
    segments = prefix.split(".")[:-1]  # drop the empty tail left by the trailing dot
    if not segments or not all(segment.isidentifier() for segment in segments):
        raise ValueError(f"{_ERROR_MARKER}: {token!r} is not a dotted module path")
    return prefix
