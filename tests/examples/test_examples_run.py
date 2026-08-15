"""Execute every script in ``examples/`` and assert the directory stays well-formed.

The previous generation of examples was deleted after two epics of raising
``ImportError`` on load, because nothing ever ran it. This module is the fix: it
discovers the examples by glob at collection time, calls each ``main()``, and turns
the build red the day an API change breaks one of them.

Discovery is deliberately mechanical. Adding ``07_something.py`` requires no edit to
this file — a hardcoded list would rot exactly the way the old examples did. Three
guard tests protect the discovery itself, because a parametrisation over an empty
list is green, which is precisely how a harness quietly becomes a no-op.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import pytest

# parents[0] = tests/examples, parents[1] = tests, parents[2] = the package root.
EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"

EXAMPLE_GLOB = "[0-9][0-9]_*.py"
"""Numbered scripts only — a bare ``*.py`` would sweep up support files."""


def _discover_examples() -> list[Path]:
    """Return every example script, sorted, so parametrisation ids are stable."""
    return sorted(EXAMPLES_DIR.glob(EXAMPLE_GLOB))


EXAMPLE_PATHS = _discover_examples()


@contextmanager
def _loaded_example(path: Path) -> Iterator[ModuleType]:
    """Import ``path`` under a unique name, yield the module, then unregister it.

    Digit-prefixed stems are not importable names (``import 00_hello_catalog`` is a
    syntax error), so the module is loaded by path and given a prefixed synthetic
    name. Registering it in ``sys.modules`` *before* execution matters: a module
    absent from ``sys.modules`` breaks Pydantic's forward-reference and
    ``__module__`` resolution for any model the example declares. The entry is
    removed afterwards so two examples can never collide.
    """
    module_name = f"akgentic_catalog_example_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None, f"cannot build a spec for {path.name}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(module_name, None)


def _required_modules(module: ModuleType, path: Path) -> tuple[str, ...]:
    """Return the example's declared ``REQUIRES`` tuple, defaulting to empty.

    ``REQUIRES`` is read off the *already imported* module, which fixes one rule for
    example authors: **import the optional package inside ``main()``**, never at the
    top of the file. A module-level ``import pymongo`` raises while the module is
    still loading — before this function can read the declaration — so a developer
    without that package gets a red test instead of the clean skip the mechanism
    exists to give them.
    """
    requires = getattr(module, "REQUIRES", ())
    assert isinstance(requires, tuple), (
        f"{path.name}: REQUIRES must be a tuple of importable module names, got {requires!r}"
    )
    return requires


@pytest.mark.parametrize("example_path", EXAMPLE_PATHS, ids=lambda p: p.stem)
def test_example_main_runs(
    example_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run one example's ``main()`` and let its own assertions do the judging.

    The working directory is a throwaway ``tmp_path`` so an accidental relative-path
    write lands somewhere disposable instead of in the checkout. Optional
    dependencies declared in ``REQUIRES`` are skipped here rather than inside the
    example, which has to stay runnable with no pytest in sight.
    """
    monkeypatch.chdir(tmp_path)
    with _loaded_example(example_path) as module:
        for dependency in _required_modules(module, example_path):
            pytest.importorskip(dependency)
        main = getattr(module, "main", None)
        assert callable(main), (
            f"{example_path.name} exposes no callable main() — every example must "
            f"define `def main() -> None` taking no required arguments"
        )
        main()


def test_examples_are_discovered() -> None:
    """The directory exists and the glob found something.

    Not parametrised, on purpose: a parametrisation over an empty list passes
    silently, so a directory rename, a wrong ``parents[n]`` index or a tightened
    glob would leave the suite green while running nothing at all. This test is the
    only thing standing between that mistake and a build that looks fine.
    """
    assert EXAMPLES_DIR.is_dir(), f"examples directory not found at {EXAMPLES_DIR}"
    assert EXAMPLE_PATHS, (
        f"no example matched {EXAMPLE_GLOB!r} in {EXAMPLES_DIR} — the harness would "
        f"have run nothing and still passed"
    )


def test_every_example_script_is_discovered() -> None:
    """No script in ``examples/`` sits outside the discovered set.

    Guards the other direction from :func:`test_examples_are_discovered`: an example
    added as ``7_probe.py`` or ``demo.py`` would never be executed, and nothing else
    would notice. Support files are exempt by the leading-underscore convention.
    """
    on_disk = {
        p for p in EXAMPLES_DIR.iterdir() if p.suffix == ".py" and not p.name.startswith("_")
    }
    missed = sorted(p.name for p in on_disk - set(EXAMPLE_PATHS))
    assert not missed, (
        f"these scripts are not executed by the harness: {missed} — an example must be "
        f"named NN_snake_name.py to match {EXAMPLE_GLOB!r}"
    )


@pytest.mark.parametrize("example_path", EXAMPLE_PATHS, ids=lambda p: p.stem)
def test_example_has_a_narrative(example_path: Path) -> None:
    """Every ``NN_name.py`` has a companion ``NN-*.md`` beside it.

    Structural only — it says nothing about what either file contains. The pair is
    the unit: a script whose narrative half went missing teaches nothing.
    """
    number = example_path.stem.split("_", 1)[0]
    companions = sorted(EXAMPLES_DIR.glob(f"{number}-*.md"))
    assert companions, (
        f"{example_path.name} has no companion {number}-*.md — an example is a pair, "
        f"NN_snake_name.py alongside NN-kebab-name.md"
    )
