"""Unit tests for the ref-marker leaf module.

``refs`` is a pure module — no repository, no ``Entry``, no fixture from
``tests/v2/conftest.py``. Its tests are pure too, which is what makes them a
faithful check of the module rather than of the machinery around it.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import akgentic.catalog
from akgentic.catalog import refs, resolver
from akgentic.catalog.models.errors import CatalogValidationError
from akgentic.catalog.refs import (
    NAMESPACE_KEY,
    REF_KEY,
    RESERVED_REF_KEYS,
    TYPE_KEY,
    RefMarker,
    walk_payload,
)

# The package holding refs.py, used to resolve relative imports to absolute.
_PACKAGE_PARTS = ("akgentic", "catalog")

# The single first-party import refs.py is allowed to carry. models/errors.py
# imports nothing at all, so it cannot drag a cycle in behind it.
_ALLOWED_FIRST_PARTY = frozenset({"akgentic.catalog.models.errors"})


class TestSentinelConstants:
    """The three sentinel keys and the reserved set built from them."""

    def test_sentinels_are_the_documented_literals(self) -> None:
        assert REF_KEY == "__ref__"
        assert TYPE_KEY == "__type__"
        assert NAMESPACE_KEY == "__namespace__"

    def test_reserved_keys_hold_exactly_the_three_sentinels(self) -> None:
        assert RESERVED_REF_KEYS == frozenset({"__ref__", "__type__", "__namespace__"})


class TestParse:
    """``parse`` resolves a marker, and refuses one it cannot resolve."""

    def test_bare_id_resolves_against_the_current_namespace(self) -> None:
        marker = RefMarker.parse({REF_KEY: "widget"}, "tenant-A")
        assert marker == RefMarker(target_namespace="tenant-A", target_id="widget")

    def test_shorthand_splits_the_namespace_off_the_id(self) -> None:
        marker = RefMarker.parse({REF_KEY: "global.widget"}, "tenant-A")
        assert marker == RefMarker(target_namespace="global", target_id="widget")

    def test_shorthand_splits_on_the_first_dot_only(self) -> None:
        """An id may itself contain dots — only the leading segment is the ns."""
        marker = RefMarker.parse({REF_KEY: "global.a.b.c"}, "tenant-A")
        assert marker == RefMarker(target_namespace="global", target_id="a.b.c")

    def test_explicit_namespace_with_a_dotless_id(self) -> None:
        marker = RefMarker.parse({REF_KEY: "widget", NAMESPACE_KEY: "global"}, "tenant-A")
        assert marker == RefMarker(target_namespace="global", target_id="widget")

    def test_agreeing_shorthand_and_explicit_namespace_still_splits(self) -> None:
        """The shorthand branch is checked first, so the id loses its prefix."""
        marker = RefMarker.parse({REF_KEY: "A.x", NAMESPACE_KEY: "A"}, "tenant-A")
        assert marker == RefMarker(target_namespace="A", target_id="x")

    def test_disagreeing_shorthand_and_explicit_namespace_is_rejected(self) -> None:
        with pytest.raises(CatalogValidationError) as exc_info:
            RefMarker.parse({REF_KEY: "A.x", NAMESPACE_KEY: "B"}, "tenant-A")
        assert exc_info.value.errors == [
            "Ref marker has both shorthand 'ns.id' and explicit __namespace__ — "
            "these disagree: 'A' vs 'B'"
        ]

    def test_type_sentinel_is_captured_as_the_expected_type(self) -> None:
        marker = RefMarker.parse({REF_KEY: "widget", TYPE_KEY: "acme.models.Widget"}, "tenant-A")
        assert marker.expected_type == "acme.models.Widget"

    def test_expected_type_is_none_when_the_type_sentinel_is_absent(self) -> None:
        marker = RefMarker.parse({REF_KEY: "widget"}, "tenant-A")
        assert marker.expected_type is None

    @pytest.mark.parametrize("raw_ref", [42, None, ["global", "widget"], {"nested": 1}])
    def test_non_string_ref_is_rejected(self, raw_ref: Any) -> None:
        with pytest.raises(CatalogValidationError) as exc_info:
            RefMarker.parse({REF_KEY: raw_ref}, "tenant-A")
        assert type(raw_ref).__name__ in exc_info.value.errors[0]

    def test_a_node_carrying_no_ref_sentinel_raises_key_error(self) -> None:
        """The documented KeyError; ``classify`` returns None on the same node."""
        with pytest.raises(KeyError):
            RefMarker.parse({NAMESPACE_KEY: "global"}, "tenant-A")

    @pytest.mark.parametrize("sentinel", [NAMESPACE_KEY, TYPE_KEY])
    def test_a_non_string_sibling_sentinel_is_not_swallowed(self, sentinel: str) -> None:
        """``parse`` refuses what ``classify`` skips — the split, on this input too.

        A non-string ``__namespace__`` / ``__type__`` reaches model
        construction unguarded, so the error out of ``parse`` is Pydantic's,
        not ``CatalogValidationError``. Pinned because story 33.2 swaps
        ``parse`` in behind callers that today catch the latter.
        """
        with pytest.raises(ValidationError):
            RefMarker.parse({REF_KEY: "widget", sentinel: 42}, "tenant-A")


class TestClassify:
    """``classify`` answers the same question without ever raising."""

    @pytest.mark.parametrize("node", ["a string", 42, None, ["a", "list"]])
    def test_a_non_dict_classifies_as_none(self, node: Any) -> None:
        assert RefMarker.classify(node) is None

    def test_a_dict_without_the_ref_sentinel_classifies_as_none(self) -> None:
        assert RefMarker.classify({"name": "widget", NAMESPACE_KEY: "global"}) is None

    def test_a_non_string_ref_classifies_as_none(self) -> None:
        assert RefMarker.classify({REF_KEY: 42}) is None

    @pytest.mark.parametrize("sentinel", [NAMESPACE_KEY, TYPE_KEY])
    def test_a_non_string_sibling_sentinel_classifies_as_none(self, sentinel: str) -> None:
        """A scan reads payloads it did not author — nothing may raise mid-walk."""
        assert RefMarker.classify({REF_KEY: "widget", sentinel: 42}) is None

    def test_a_disagreement_classifies_as_none_where_parse_raises(self) -> None:
        """The same malformed node, both entry points — the split, pinned."""
        node = {REF_KEY: "A.x", NAMESPACE_KEY: "B"}
        assert RefMarker.classify(node) is None
        with pytest.raises(CatalogValidationError):
            RefMarker.parse(node, "tenant-A")

    def test_shorthand_classifies_to_the_same_marker_parse_returns(self) -> None:
        node = {REF_KEY: "global.widget", TYPE_KEY: "acme.models.Widget"}
        assert RefMarker.classify(node) == RefMarker(
            target_namespace="global",
            target_id="widget",
            expected_type="acme.models.Widget",
        )

    def test_a_same_namespace_marker_classifies_with_an_empty_namespace(self) -> None:
        """A scan has no enclosing namespace, so there is nothing to default to."""
        assert RefMarker.classify({REF_KEY: "widget"}) == RefMarker(
            target_namespace="", target_id="widget"
        )


class _Recorder:
    """Accumulates the callback sequence a walk produces, in order."""

    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []

    def on_ref(self, node: dict[str, Any]) -> None:
        self.events.append(("ref", node[REF_KEY]))

    def on_leaf(self, value: Any) -> None:
        self.events.append(("leaf", value))


class TestWalkPayload:
    """One walk, depth-first, with a marker as a leaf by construction."""

    def test_a_marker_in_a_flat_dict_is_visited_once(self) -> None:
        marker = {REF_KEY: "global.widget"}
        seen: list[dict[str, Any]] = []
        walk_payload({"part": marker}, on_ref=seen.append)
        assert seen == [marker]

    def test_every_marker_in_a_list_is_visited_in_order(self) -> None:
        recorder = _Recorder()
        walk_payload(
            [{REF_KEY: "one"}, {REF_KEY: "two"}, {REF_KEY: "three"}],
            on_ref=recorder.on_ref,
            on_leaf=recorder.on_leaf,
        )
        assert recorder.events == [("ref", "one"), ("ref", "two"), ("ref", "three")]

    def test_a_nested_payload_is_walked_depth_first_left_to_right(self) -> None:
        recorder = _Recorder()
        walk_payload(
            {
                "a": {REF_KEY: "ns.one"},
                "b": ["scalar-1", {"c": {REF_KEY: "two"}}, [{REF_KEY: "three"}, "scalar-2"]],
                "d": "scalar-3",
            },
            on_ref=recorder.on_ref,
            on_leaf=recorder.on_leaf,
        )
        assert recorder.events == [
            ("ref", "ns.one"),
            ("leaf", "scalar-1"),
            ("ref", "two"),
            ("ref", "three"),
            ("leaf", "scalar-2"),
            ("leaf", "scalar-3"),
        ]

    def test_a_scalar_reaches_on_leaf_and_never_on_ref(self) -> None:
        recorder = _Recorder()
        walk_payload("just a value", on_ref=recorder.on_ref, on_leaf=recorder.on_leaf)
        assert recorder.events == [("leaf", "just a value")]

    @pytest.mark.parametrize("empty", [{}, []])
    def test_an_empty_container_fires_no_callback(self, empty: Any) -> None:
        recorder = _Recorder()
        walk_payload(empty, on_ref=recorder.on_ref, on_leaf=recorder.on_leaf)
        assert recorder.events == []

    def test_a_marker_is_a_leaf_and_the_walk_does_not_descend_into_it(self) -> None:
        """A marker is a pure pointer — whatever sits under it is not walked."""
        recorder = _Recorder()
        walk_payload(
            {REF_KEY: "global.widget", "nested": {REF_KEY: "never.seen"}},
            on_ref=recorder.on_ref,
            on_leaf=recorder.on_leaf,
        )
        assert recorder.events == [("ref", "global.widget")]

    def test_on_leaf_is_optional_and_the_walk_still_completes(self) -> None:
        seen: list[dict[str, Any]] = []
        walk_payload({"a": ["scalar", {REF_KEY: "widget"}]}, on_ref=seen.append)
        assert seen == [{REF_KEY: "widget"}]


def _import_targets(source: str) -> list[str]:
    """Return every module path the import statements in ``source`` name.

    Relative and absolute spellings both come back absolute, so one membership
    check covers ``from .resolver import X``, ``from . import catalog``,
    ``from ..catalog import x``, ``from akgentic.catalog.catalog import X`` and
    ``import akgentic.catalog.catalog`` alike.
    """
    targets: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            targets.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _absolute_base(node)
            if node.module is None:
                # ``from . import catalog`` — each alias names a submodule.
                targets.extend(f"{base}.{alias.name}" for alias in node.names)
            else:
                targets.append(base)
    return targets


def _absolute_base(node: ast.ImportFrom) -> str:
    """Return the absolute package path an ``ImportFrom`` node reads from."""
    if node.level == 0:
        return node.module or ""
    parts = list(_PACKAGE_PARTS)
    stripped = node.level - 1
    if stripped:
        parts = parts[:-stripped]
    if node.module:
        parts.append(node.module)
    return ".".join(parts)


class TestImportTargetExtraction:
    """The guard's own AST walk, spelling by spelling.

    The leaf guard below was verified once, by hand, by mutating ``refs.py``.
    That evidence does not survive into the suite: ``refs.py`` carries a single
    level-1 ``ImportFrom``, so parsing it exercises one branch of
    ``_absolute_base`` and leaves the rest unproven. Without these cases a
    refactor could stop resolving relative imports altogether and the leaf
    guard would go on reporting green.
    """

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("import akgentic.catalog.catalog", ["akgentic.catalog.catalog"]),
            ("import os, ast", ["os", "ast"]),
            ("from akgentic.catalog.resolver import REF_KEY", ["akgentic.catalog.resolver"]),
            ("from .resolver import REF_KEY", ["akgentic.catalog.resolver"]),
            ("from . import catalog", ["akgentic.catalog.catalog"]),
            ("from .models.errors import X", ["akgentic.catalog.models.errors"]),
            ("from ..catalog import x", ["akgentic.catalog"]),
            ("from .. import catalog", ["akgentic.catalog"]),
            (
                "def f() -> None:\n    from .resolver import REF_KEY\n",
                ["akgentic.catalog.resolver"],
            ),
        ],
    )
    def test_every_spelling_resolves_to_an_absolute_module_path(
        self, source: str, expected: list[str]
    ) -> None:
        assert _import_targets(source) == expected


class TestModuleIsALeaf:
    """The property the whole module exists to have, and the re-export it keeps."""

    def test_every_allowlisted_import_is_itself_import_free(self) -> None:
        """Why the allowlist has one entry — checked, not asserted in prose.

        ``refs.py`` may import ``models/errors.py`` because that module imports
        nothing that could carry a cycle back. Both the module docstring and
        the comment on ``_ALLOWED_FIRST_PARTY`` say so and nothing checks it:
        an import added there would weaken ``refs.py``'s leaf property with the
        guard below still green. Iterating the allowlist also means a second
        entry has to clear the same bar the first one did.
        """
        for allowed in sorted(_ALLOWED_FIRST_PARTY):
            source = Path(str(importlib.import_module(allowed).__file__)).read_text(
                encoding="utf-8"
            )
            # Asked of the statements directly, not of _import_targets: that
            # helper resolves relative levels against refs.py's own package,
            # which is the wrong base for a module at a different depth.
            statements = sorted(
                statement
                for statement in (
                    ast.unparse(node)
                    for node in ast.walk(ast.parse(source))
                    if isinstance(node, ast.Import | ast.ImportFrom)
                )
                if statement != "from __future__ import annotations"
            )
            assert statements == [], (
                f"{allowed} is allowlisted into refs.py because it imports nothing "
                f"that could carry a cycle back. It now carries {statements} — either "
                "refs.py is no longer a leaf, or this allowlist needs a different "
                "justification."
            )

    def test_refs_imports_no_first_party_module_but_the_error_model(self) -> None:
        source = Path(str(refs.__file__)).read_text(encoding="utf-8")
        offenders = sorted(
            target
            for target in _import_targets(source)
            if (target == "akgentic.catalog" or target.startswith("akgentic.catalog."))
            and target not in _ALLOWED_FIRST_PARTY
        )
        assert offenders == [], (
            "refs.py must stay a leaf of akgentic.catalog: the whole package's "
            "import cycle dissolves into it. Allowed first-party imports are "
            f"{sorted(_ALLOWED_FIRST_PARTY)}; found {offenders}."
        )

    def test_sentinels_stay_importable_from_every_path_that_carries_them(self) -> None:
        assert akgentic.catalog.REF_KEY == resolver.REF_KEY == refs.REF_KEY
        assert akgentic.catalog.TYPE_KEY == resolver.TYPE_KEY == refs.TYPE_KEY
        assert akgentic.catalog.NAMESPACE_KEY == resolver.NAMESPACE_KEY == refs.NAMESPACE_KEY
        assert {"NAMESPACE_KEY", "REF_KEY", "TYPE_KEY"} <= set(resolver.__all__)
        assert "RefMarker" not in akgentic.catalog.__all__
        assert "walk_payload" not in akgentic.catalog.__all__
