"""Tests for YamlTemplateCatalogRepository."""

from pathlib import Path

import pytest
import yaml

from akgentic.catalog.models.errors import CatalogValidationError, EntryNotFoundError
from akgentic.catalog.models.queries import TemplateQuery
from akgentic.catalog.models.template import TemplateEntry
from akgentic.catalog.repositories.yaml.template_repo import YamlTemplateCatalogRepository


def _write_yaml(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


# ---- Loading ----


def test_load_single_yaml_file(tmp_path: Path) -> None:
    """AC #1: Load entries from a single YAML file containing list of dicts."""
    _write_yaml(
        tmp_path / "templates.yaml",
        [
            {"id": "t1", "template": "Hello {name}"},
            {"id": "t2", "template": "Bye {name}"},
        ],
    )
    repo = YamlTemplateCatalogRepository(tmp_path)
    entries = repo.list()
    assert len(entries) == 2
    assert all(isinstance(e, TemplateEntry) for e in entries)


def test_load_multiple_yaml_files(tmp_path: Path) -> None:
    """AC #1: Multiple YAML files merged into single registry."""
    _write_yaml(tmp_path / "coordinator.yaml", [{"id": "c1", "template": "Coord {role}"}])
    _write_yaml(tmp_path / "specialist.yaml", [{"id": "s1", "template": "Spec {skill}"}])
    repo = YamlTemplateCatalogRepository(tmp_path)
    entries = repo.list()
    assert len(entries) == 2
    ids = {e.id for e in entries}
    assert ids == {"c1", "s1"}


def test_duplicate_id_across_files_raises(tmp_path: Path) -> None:
    """AC #2: Duplicate id across files raises CatalogValidationError with file paths."""
    _write_yaml(tmp_path / "file_a.yaml", [{"id": "dup", "template": "A {x}"}])
    _write_yaml(tmp_path / "file_b.yaml", [{"id": "dup", "template": "B {y}"}])
    repo = YamlTemplateCatalogRepository(tmp_path)
    with pytest.raises(CatalogValidationError) as exc_info:
        repo.list()
    assert "dup" in str(exc_info.value)
    assert "file_a.yaml" in str(exc_info.value)
    assert "file_b.yaml" in str(exc_info.value)


def test_empty_directory(tmp_path: Path) -> None:
    """Edge case: empty directory returns empty list."""
    repo = YamlTemplateCatalogRepository(tmp_path)
    assert repo.list() == []


def test_nonexistent_directory(tmp_path: Path) -> None:
    """Edge case: nonexistent directory returns empty list."""
    repo = YamlTemplateCatalogRepository(tmp_path / "nonexistent")
    assert repo.list() == []


def test_empty_yaml_file(tmp_path: Path) -> None:
    """Edge case: YAML file with None content (empty file) is skipped."""
    (tmp_path / "empty.yaml").write_text("", encoding="utf-8")
    repo = YamlTemplateCatalogRepository(tmp_path)
    assert repo.list() == []


# ---- get() ----


def test_get_returns_entry_by_id(tmp_path: Path) -> None:
    """AC #4: get() returns entry by id."""
    _write_yaml(tmp_path / "t.yaml", [{"id": "t1", "template": "Hello {name}"}])
    repo = YamlTemplateCatalogRepository(tmp_path)
    entry = repo.get("t1")
    assert entry is not None
    assert entry.id == "t1"


def test_get_returns_none_for_missing(tmp_path: Path) -> None:
    """get() returns None for non-existent id."""
    _write_yaml(tmp_path / "t.yaml", [{"id": "t1", "template": "Hello {name}"}])
    repo = YamlTemplateCatalogRepository(tmp_path)
    assert repo.get("missing") is None


# ---- list() ----


def test_list_returns_all_entries(tmp_path: Path) -> None:
    """list() returns all cached entries."""
    _write_yaml(
        tmp_path / "t.yaml",
        [
            {"id": "a", "template": "A {x}"},
            {"id": "b", "template": "B {y}"},
            {"id": "c", "template": "C {z}"},
        ],
    )
    repo = YamlTemplateCatalogRepository(tmp_path)
    assert len(repo.list()) == 3


# ---- search() ----


def test_search_by_placeholder(tmp_path: Path) -> None:
    """AC #5: search(TemplateQuery(placeholder='role')) returns only matching templates."""
    _write_yaml(
        tmp_path / "t.yaml",
        [
            {"id": "t1", "template": "Hello {role} and {name}"},
            {"id": "t2", "template": "Only {name} here"},
            {"id": "t3", "template": "Agent {role}"},
        ],
    )
    repo = YamlTemplateCatalogRepository(tmp_path)
    results = repo.search(TemplateQuery(placeholder="role"))
    assert len(results) == 2
    ids = {e.id for e in results}
    assert ids == {"t1", "t3"}


def test_search_by_id(tmp_path: Path) -> None:
    """search(TemplateQuery(id='foo')) exact id match."""
    _write_yaml(
        tmp_path / "t.yaml",
        [
            {"id": "foo", "template": "Foo {x}"},
            {"id": "bar", "template": "Bar {y}"},
        ],
    )
    repo = YamlTemplateCatalogRepository(tmp_path)
    results = repo.search(TemplateQuery(id="foo"))
    assert len(results) == 1
    assert results[0].id == "foo"


def test_search_and_semantics(tmp_path: Path) -> None:
    """search(TemplateQuery(id='foo', placeholder='role')) AND semantics."""
    _write_yaml(
        tmp_path / "t.yaml",
        [
            {"id": "foo", "template": "Foo {role}"},
            {"id": "bar", "template": "Bar {role}"},
            {"id": "foo2", "template": "Foo2 {name}"},
        ],
    )
    repo = YamlTemplateCatalogRepository(tmp_path)
    results = repo.search(TemplateQuery(id="foo", placeholder="role"))
    assert len(results) == 1
    assert results[0].id == "foo"


def test_search_no_filters_returns_all(tmp_path: Path) -> None:
    """search with all None fields returns all entries."""
    _write_yaml(
        tmp_path / "t.yaml",
        [
            {"id": "a", "template": "A {x}"},
            {"id": "b", "template": "B {y}"},
        ],
    )
    repo = YamlTemplateCatalogRepository(tmp_path)
    results = repo.search(TemplateQuery())
    assert len(results) == 2


# ---- Caching ----


def test_caching_no_reread(tmp_path: Path) -> None:
    """AC #4: second list() doesn't re-read files."""
    _write_yaml(tmp_path / "t.yaml", [{"id": "t1", "template": "Hello {name}"}])
    repo = YamlTemplateCatalogRepository(tmp_path)
    entries1 = repo.list()
    assert len(entries1) == 1

    # Modify file on disk
    _write_yaml(
        tmp_path / "t.yaml",
        [
            {"id": "t1", "template": "Hello {name}"},
            {"id": "t2", "template": "New {item}"},
        ],
    )
    # Cache should return stale data
    entries2 = repo.list()
    assert len(entries2) == 1


def test_reload_forces_rescan(tmp_path: Path) -> None:
    """AC #4: reload() forces re-scan from disk."""
    _write_yaml(tmp_path / "t.yaml", [{"id": "t1", "template": "Hello {name}"}])
    repo = YamlTemplateCatalogRepository(tmp_path)
    assert len(repo.list()) == 1

    # Modify file on disk
    _write_yaml(
        tmp_path / "t.yaml",
        [
            {"id": "t1", "template": "Hello {name}"},
            {"id": "t2", "template": "New {item}"},
        ],
    )
    repo.reload()
    assert len(repo.list()) == 2


# ---- create() ----


def test_create_persists_entry(tmp_path: Path) -> None:
    """AC #3: create() persists entry to YAML file."""
    repo = YamlTemplateCatalogRepository(tmp_path)
    entry = TemplateEntry(id="new-t", template="New {placeholder}")
    result_id = repo.create(entry)
    assert result_id == "new-t"

    # Verify persisted on disk
    repo2 = YamlTemplateCatalogRepository(tmp_path)
    loaded = repo2.get("new-t")
    assert loaded is not None
    assert loaded.template == "New {placeholder}"


# ---- update() ----


def test_update_modifies_entry(tmp_path: Path) -> None:
    """AC #3: update() modifies entry in file."""
    _write_yaml(tmp_path / "t1.yaml", [{"id": "t1", "template": "Old {x}"}])
    repo = YamlTemplateCatalogRepository(tmp_path)
    updated = TemplateEntry(id="t1", template="Updated {y}")
    repo.update("t1", updated)

    entry = repo.get("t1")
    assert entry is not None
    assert entry.template == "Updated {y}"


def test_update_raises_for_missing_id(tmp_path: Path) -> None:
    """update() raises EntryNotFoundError for missing id."""
    _write_yaml(tmp_path / "t.yaml", [{"id": "t1", "template": "Hello {x}"}])
    repo = YamlTemplateCatalogRepository(tmp_path)
    with pytest.raises(EntryNotFoundError):
        repo.update("nonexistent", TemplateEntry(id="nonexistent", template="T {x}"))


# ---- delete() ----


def test_delete_removes_entry_and_deletes_file_if_empty(tmp_path: Path) -> None:
    """AC #3: delete() removes entry, deletes file if empty."""
    _write_yaml(tmp_path / "t1.yaml", [{"id": "t1", "template": "Hello {x}"}])
    repo = YamlTemplateCatalogRepository(tmp_path)
    repo.delete("t1")
    assert repo.get("t1") is None
    assert not (tmp_path / "t1.yaml").exists()


def test_delete_keeps_file_with_remaining_entries(tmp_path: Path) -> None:
    """delete() removes entry but keeps file if other entries remain."""
    _write_yaml(
        tmp_path / "multi.yaml",
        [
            {"id": "a", "template": "A {x}"},
            {"id": "b", "template": "B {y}"},
        ],
    )
    repo = YamlTemplateCatalogRepository(tmp_path)
    repo.delete("a")
    assert repo.get("a") is None
    assert repo.get("b") is not None
    assert (tmp_path / "multi.yaml").exists()


def test_delete_raises_for_missing_id(tmp_path: Path) -> None:
    """delete() raises EntryNotFoundError for missing id."""
    _write_yaml(tmp_path / "t.yaml", [{"id": "t1", "template": "Hello {x}"}])
    repo = YamlTemplateCatalogRepository(tmp_path)
    with pytest.raises(EntryNotFoundError):
        repo.delete("nonexistent")


# ---- Public API export ----


def test_public_api_export() -> None:
    """YamlTemplateCatalogRepository is importable from public API."""
    from akgentic.catalog import YamlTemplateCatalogRepository as Exported

    assert Exported is YamlTemplateCatalogRepository
