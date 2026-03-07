"""Tests for YamlAgentCatalogRepository."""

from pathlib import Path

import pytest
import yaml

from akgentic.catalog.models.agent import AgentEntry
from akgentic.catalog.models.errors import CatalogValidationError, EntryNotFoundError
from akgentic.catalog.models.queries import AgentQuery
from akgentic.catalog.repositories.yaml.agent_repo import YamlAgentCatalogRepository

AGENT_CLASS = "akgentic.agent.BaseAgent"


def _agent_dict(
    id: str,
    *,
    role: str = "Expert",
    description: str = "A test agent",
    skills: list[str] | None = None,
) -> dict[str, object]:
    if skills is None:
        skills = ["testing"]
    return {
        "id": id,
        "tool_ids": [],
        "card": {
            "role": role,
            "description": description,
            "skills": skills,
            "agent_class": AGENT_CLASS,
            "config": {
                "name": f"@{role}",
                "role": role,
                "prompt": {"template": f"You are a {role}.", "params": {}},
                "model_cfg": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "temperature": 0.3,
                },
            },
            "routes_to": [],
        },
    }


def _write_yaml(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


# ---- Loading (AC #1) ----


def test_load_agent_entry_from_yaml(tmp_path: Path) -> None:
    """AC #1: Load AgentEntry from YAML with agent_class resolution."""
    _write_yaml(
        tmp_path / "agents.yaml",
        [_agent_dict("mgr", role="Manager", description="Coordinates team work")],
    )
    repo = YamlAgentCatalogRepository(tmp_path)
    entries = repo.list()
    assert len(entries) == 1
    assert isinstance(entries[0], AgentEntry)
    assert entries[0].card.role == "Manager"


# ---- CRUD (AC #1) ----


def test_create_persists_agent(tmp_path: Path) -> None:
    """AC #1: create() persists agent entry to YAML file."""
    repo = YamlAgentCatalogRepository(tmp_path)
    data = _agent_dict("new-agent", role="Tester")
    entry = AgentEntry.model_validate(data)
    result_id = repo.create(entry)
    assert result_id == "new-agent"

    repo2 = YamlAgentCatalogRepository(tmp_path)
    loaded = repo2.get("new-agent")
    assert loaded is not None
    assert loaded.card.role == "Tester"


def test_create_duplicate_id_raises(tmp_path: Path) -> None:
    """create() raises CatalogValidationError for duplicate id."""
    _write_yaml(tmp_path / "a.yaml", [_agent_dict("dup")])
    repo = YamlAgentCatalogRepository(tmp_path)
    entry = AgentEntry.model_validate(_agent_dict("dup", role="Manager"))
    with pytest.raises(CatalogValidationError):
        repo.create(entry)


def test_get_returns_entry_by_id(tmp_path: Path) -> None:
    """get() returns agent by id."""
    _write_yaml(tmp_path / "a.yaml", [_agent_dict("a1")])
    repo = YamlAgentCatalogRepository(tmp_path)
    entry = repo.get("a1")
    assert entry is not None
    assert entry.id == "a1"


def test_get_returns_none_for_missing(tmp_path: Path) -> None:
    """get() returns None for non-existent id."""
    _write_yaml(tmp_path / "a.yaml", [_agent_dict("a1")])
    repo = YamlAgentCatalogRepository(tmp_path)
    assert repo.get("missing") is None


def test_list_returns_all_entries(tmp_path: Path) -> None:
    """list() returns all cached entries."""
    _write_yaml(
        tmp_path / "a.yaml",
        [_agent_dict("a1"), _agent_dict("a2", role="Manager")],
    )
    repo = YamlAgentCatalogRepository(tmp_path)
    assert len(repo.list()) == 2


def test_update_modifies_agent(tmp_path: Path) -> None:
    """update() modifies agent entry in file."""
    _write_yaml(tmp_path / "a.yaml", [_agent_dict("a1", role="Expert")])
    repo = YamlAgentCatalogRepository(tmp_path)
    updated = AgentEntry.model_validate(_agent_dict("a1", role="Manager"))
    repo.update("a1", updated)
    entry = repo.get("a1")
    assert entry is not None
    assert entry.card.role == "Manager"


def test_update_raises_for_missing_id(tmp_path: Path) -> None:
    """update() raises EntryNotFoundError for missing id."""
    _write_yaml(tmp_path / "a.yaml", [_agent_dict("a1")])
    repo = YamlAgentCatalogRepository(tmp_path)
    entry = AgentEntry.model_validate(_agent_dict("nope"))
    with pytest.raises(EntryNotFoundError):
        repo.update("nope", entry)


def test_delete_removes_agent(tmp_path: Path) -> None:
    """delete() removes agent entry and deletes file if empty."""
    _write_yaml(tmp_path / "a1.yaml", [_agent_dict("a1")])
    repo = YamlAgentCatalogRepository(tmp_path)
    repo.delete("a1")
    assert repo.get("a1") is None
    assert not (tmp_path / "a1.yaml").exists()


def test_delete_raises_for_missing_id(tmp_path: Path) -> None:
    """delete() raises EntryNotFoundError for missing id."""
    _write_yaml(tmp_path / "a.yaml", [_agent_dict("a1")])
    repo = YamlAgentCatalogRepository(tmp_path)
    with pytest.raises(EntryNotFoundError):
        repo.delete("nonexistent")


def test_duplicate_id_across_files_raises(tmp_path: Path) -> None:
    """Duplicate id across files raises CatalogValidationError."""
    _write_yaml(tmp_path / "file_a.yaml", [_agent_dict("dup")])
    _write_yaml(tmp_path / "file_b.yaml", [_agent_dict("dup", role="Manager")])
    repo = YamlAgentCatalogRepository(tmp_path)
    with pytest.raises(CatalogValidationError) as exc_info:
        repo.list()
    assert "dup" in str(exc_info.value)
    assert "file_a.yaml" in str(exc_info.value)
    assert "file_b.yaml" in str(exc_info.value)


# ---- search() (AC #3) ----


def test_search_by_id(tmp_path: Path) -> None:
    """search(AgentQuery(id='a1')) exact id match."""
    _write_yaml(
        tmp_path / "agents.yaml",
        [_agent_dict("a1", role="Expert"), _agent_dict("a2", role="Manager")],
    )
    repo = YamlAgentCatalogRepository(tmp_path)
    results = repo.search(AgentQuery(id="a1"))
    assert len(results) == 1
    assert results[0].id == "a1"


def test_search_by_id_no_match(tmp_path: Path) -> None:
    """search(AgentQuery(id='missing')) returns empty."""
    _write_yaml(tmp_path / "agents.yaml", [_agent_dict("a1")])
    repo = YamlAgentCatalogRepository(tmp_path)
    assert repo.search(AgentQuery(id="missing")) == []


def test_search_by_role(tmp_path: Path) -> None:
    """AC #3: search(AgentQuery(role='Manager')) exact role match."""
    _write_yaml(
        tmp_path / "agents.yaml",
        [
            _agent_dict("m1", role="Manager", description="Coordinates team work"),
            _agent_dict("e1", role="Expert", description="Domain expert"),
        ],
    )
    repo = YamlAgentCatalogRepository(tmp_path)
    results = repo.search(AgentQuery(role="Manager"))
    assert len(results) == 1
    assert results[0].id == "m1"


def test_search_by_skills_match_any(tmp_path: Path) -> None:
    """AC #3: search(AgentQuery(skills=['research'])) match-any semantics."""
    _write_yaml(
        tmp_path / "agents.yaml",
        [
            _agent_dict("a1", skills=["research", "writing"]),
            _agent_dict("a2", skills=["coding"]),
        ],
    )
    repo = YamlAgentCatalogRepository(tmp_path)
    results = repo.search(AgentQuery(skills=["research"]))
    assert len(results) == 1
    assert results[0].id == "a1"


def test_search_by_skills_no_match(tmp_path: Path) -> None:
    """search(AgentQuery(skills=['unknown'])) returns empty when no match."""
    _write_yaml(
        tmp_path / "agents.yaml",
        [_agent_dict("a1", skills=["research", "writing"])],
    )
    repo = YamlAgentCatalogRepository(tmp_path)
    results = repo.search(AgentQuery(skills=["unknown"]))
    assert results == []


def test_search_by_description(tmp_path: Path) -> None:
    """search(AgentQuery(description='coord')) case-insensitive substring match."""
    _write_yaml(
        tmp_path / "agents.yaml",
        [
            _agent_dict("m1", description="Coordinates team work"),
            _agent_dict("e1", description="Domain expert"),
        ],
    )
    repo = YamlAgentCatalogRepository(tmp_path)
    results = repo.search(AgentQuery(description="coord"))
    assert len(results) == 1
    assert results[0].id == "m1"


def test_search_and_semantics(tmp_path: Path) -> None:
    """search(AgentQuery(role='Manager', description='coord')) AND semantics."""
    _write_yaml(
        tmp_path / "agents.yaml",
        [
            _agent_dict("m1", role="Manager", description="Coordinates team work"),
            _agent_dict("m2", role="Manager", description="Manages budget"),
            _agent_dict("e1", role="Expert", description="Coordinates research"),
        ],
    )
    repo = YamlAgentCatalogRepository(tmp_path)
    results = repo.search(AgentQuery(role="Manager", description="coord"))
    assert len(results) == 1
    assert results[0].id == "m1"


# ---- Caching ----


def test_caching_no_reread(tmp_path: Path) -> None:
    """Second list() call doesn't re-read files."""
    _write_yaml(tmp_path / "a.yaml", [_agent_dict("a1")])
    repo = YamlAgentCatalogRepository(tmp_path)
    assert len(repo.list()) == 1

    # Modify file on disk
    _write_yaml(
        tmp_path / "a.yaml",
        [_agent_dict("a1"), _agent_dict("a2", role="Manager")],
    )
    # Cache should return stale data
    assert len(repo.list()) == 1


def test_reload_forces_rescan(tmp_path: Path) -> None:
    """reload() forces re-scan from disk."""
    _write_yaml(tmp_path / "a.yaml", [_agent_dict("a1")])
    repo = YamlAgentCatalogRepository(tmp_path)
    assert len(repo.list()) == 1

    _write_yaml(
        tmp_path / "a.yaml",
        [_agent_dict("a1"), _agent_dict("a2", role="Manager")],
    )
    repo.reload()
    assert len(repo.list()) == 2


# ---- Public API export ----


def test_public_api_export() -> None:
    """YamlAgentCatalogRepository is importable from public API."""
    from akgentic.catalog import YamlAgentCatalogRepository as Exported

    assert Exported is YamlAgentCatalogRepository
