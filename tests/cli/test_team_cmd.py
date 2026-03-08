"""Tests for the team CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from typer.testing import CliRunner

from akgentic.catalog.cli.main import app

runner = CliRunner()

AGENT_CLASS = "akgentic.agent.BaseAgent"


def _agent_data(agent_id: str, role: str = "engineer") -> dict[str, Any]:
    """Build an agent entry dict suitable for YAML serialization."""
    return {
        "id": agent_id,
        "tool_ids": [],
        "card": {
            "role": role,
            "description": f"Test {role} agent",
            "skills": ["coding"],
            "agent_class": AGENT_CLASS,
            "config": {"name": f"@{role}", "role": role},
            "routes_to": [],
        },
    }


def _team_data(
    team_id: str,
    name: str = "Test Team",
    entry_point: str = "eng-mgr",
    members: list[dict[str, Any]] | None = None,
    description: str = "A test team",
) -> dict[str, Any]:
    """Build a team spec dict suitable for YAML serialization."""
    return {
        "id": team_id,
        "name": name,
        "entry_point": entry_point,
        "message_types": ["akgentic.core.messages.UserMessage"],
        "members": members or [{"agent_id": entry_point, "headcount": 1}],
        "profiles": [],
        "description": description,
    }


def _seed_agent(catalog_dir: Path, agent_id: str, role: str = "engineer") -> None:
    """Seed an agent entry directly in the catalog directory."""
    data = _agent_data(agent_id, role=role)
    (catalog_dir / "agents" / f"{agent_id}.yaml").write_text(
        yaml.dump(data, default_flow_style=False)
    )


def _write_team_yaml(path: Path, team_id: str, **kwargs: Any) -> Path:
    """Write a team YAML file and return its path."""
    data = _team_data(team_id, **kwargs)
    file = path / f"{team_id}.yaml"
    file.write_text(yaml.dump(data, default_flow_style=False))
    return file


def _seed_team(catalog_dir: Path, team_id: str, **kwargs: Any) -> None:
    """Seed a team entry directly in the catalog directory."""
    data = _team_data(team_id, **kwargs)
    (catalog_dir / "teams" / f"{team_id}.yaml").write_text(
        yaml.dump(data, default_flow_style=False)
    )


def _make_dirs(tmp_path: Path) -> None:
    for name in ("templates", "tools", "agents", "teams"):
        (tmp_path / name).mkdir(exist_ok=True)


class TestTeamList:
    """team list command."""

    def test_list_empty(self, tmp_path: Path) -> None:
        _make_dirs(tmp_path)
        result = runner.invoke(app, ["--catalog-dir", str(tmp_path), "team", "list"])
        assert result.exit_code == 0
        assert "No entries found" in result.output

    def test_list_with_entries(self, tmp_path: Path) -> None:
        _make_dirs(tmp_path)
        _seed_agent(tmp_path, "eng-mgr", role="Manager")
        _seed_agent(tmp_path, "dev-lead", role="Lead")
        _seed_team(tmp_path, "team1", name="Engineering", entry_point="eng-mgr")
        _seed_team(tmp_path, "team2", name="Dev", entry_point="dev-lead")
        result = runner.invoke(app, ["--catalog-dir", str(tmp_path), "team", "list"])
        assert result.exit_code == 0
        assert "team1" in result.output
        assert "team2" in result.output


class TestTeamGet:
    """team get command."""

    def test_get_existing(self, tmp_path: Path) -> None:
        _make_dirs(tmp_path)
        _seed_agent(tmp_path, "eng-mgr", role="Manager")
        _seed_team(tmp_path, "team1", name="Engineering", entry_point="eng-mgr")
        result = runner.invoke(app, ["--catalog-dir", str(tmp_path), "team", "get", "team1"])
        assert result.exit_code == 0
        assert "team1" in result.output

    def test_get_nonexistent(self, tmp_path: Path) -> None:
        _make_dirs(tmp_path)
        result = runner.invoke(app, ["--catalog-dir", str(tmp_path), "team", "get", "nope"])
        assert result.exit_code == 1
        assert "Not found" in result.output


class TestTeamCreate:
    """team create command."""

    def test_create_from_yaml(self, tmp_path: Path) -> None:
        _make_dirs(tmp_path)
        _seed_agent(tmp_path, "eng-mgr", role="Manager")
        yaml_file = _write_team_yaml(tmp_path, "new-team", name="New Team", entry_point="eng-mgr")
        result = runner.invoke(
            app, ["--catalog-dir", str(tmp_path), "team", "create", str(yaml_file)]
        )
        assert result.exit_code == 0
        # Verify it was actually created
        get_result = runner.invoke(app, ["--catalog-dir", str(tmp_path), "team", "get", "new-team"])
        assert get_result.exit_code == 0
        assert "new-team" in get_result.output

    def test_create_duplicate(self, tmp_path: Path) -> None:
        _make_dirs(tmp_path)
        _seed_agent(tmp_path, "eng-mgr", role="Manager")
        _seed_team(tmp_path, "dup", name="Dup Team", entry_point="eng-mgr")
        yaml_file = _write_team_yaml(tmp_path, "dup", name="Dup Team", entry_point="eng-mgr")
        result = runner.invoke(
            app, ["--catalog-dir", str(tmp_path), "team", "create", str(yaml_file)]
        )
        assert result.exit_code == 1

    def test_create_with_missing_agent(self, tmp_path: Path) -> None:
        _make_dirs(tmp_path)
        yaml_file = _write_team_yaml(
            tmp_path,
            "bad-team",
            name="Bad Team",
            entry_point="ghost-agent",
            members=[{"agent_id": "ghost-agent", "headcount": 1}],
        )
        result = runner.invoke(
            app, ["--catalog-dir", str(tmp_path), "team", "create", str(yaml_file)]
        )
        assert result.exit_code == 1
        assert "ghost-agent" in result.output


class TestTeamUpdate:
    """team update command."""

    def test_update_existing(self, tmp_path: Path) -> None:
        _make_dirs(tmp_path)
        _seed_agent(tmp_path, "eng-mgr", role="Manager")
        _seed_team(tmp_path, "team1", name="Old Name", entry_point="eng-mgr")
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        yaml_file = _write_team_yaml(input_dir, "team1", name="New Name", entry_point="eng-mgr")
        result = runner.invoke(
            app, ["--catalog-dir", str(tmp_path), "team", "update", "team1", str(yaml_file)]
        )
        assert result.exit_code == 0

    def test_update_nonexistent(self, tmp_path: Path) -> None:
        _make_dirs(tmp_path)
        _seed_agent(tmp_path, "eng-mgr", role="Manager")
        yaml_file = _write_team_yaml(tmp_path, "nope", name="Ghost Team", entry_point="eng-mgr")
        result = runner.invoke(
            app, ["--catalog-dir", str(tmp_path), "team", "update", "nope", str(yaml_file)]
        )
        assert result.exit_code == 1


class TestTeamDelete:
    """team delete command."""

    def test_delete_existing(self, tmp_path: Path) -> None:
        _make_dirs(tmp_path)
        _seed_agent(tmp_path, "eng-mgr", role="Manager")
        _seed_team(tmp_path, "team1", entry_point="eng-mgr")
        result = runner.invoke(app, ["--catalog-dir", str(tmp_path), "team", "delete", "team1"])
        assert result.exit_code == 0

    def test_delete_nonexistent(self, tmp_path: Path) -> None:
        _make_dirs(tmp_path)
        result = runner.invoke(app, ["--catalog-dir", str(tmp_path), "team", "delete", "nope"])
        assert result.exit_code == 1
        assert "Not found" in result.output


class TestTeamSearch:
    """team search command."""

    def test_search_by_name(self, tmp_path: Path) -> None:
        _make_dirs(tmp_path)
        _seed_agent(tmp_path, "eng-mgr", role="Manager")
        _seed_team(tmp_path, "team1", name="Engineering Team", entry_point="eng-mgr")
        _seed_team(tmp_path, "team2", name="Marketing Team", entry_point="eng-mgr")
        result = runner.invoke(
            app, ["--catalog-dir", str(tmp_path), "team", "search", "--name", "Engineering"]
        )
        assert result.exit_code == 0
        assert "team1" in result.output

    def test_search_by_agent_id(self, tmp_path: Path) -> None:
        _make_dirs(tmp_path)
        _seed_agent(tmp_path, "eng-mgr", role="Manager")
        _seed_agent(tmp_path, "dev-lead", role="Lead")
        _seed_team(tmp_path, "team1", name="Eng Team", entry_point="eng-mgr")
        _seed_team(tmp_path, "team2", name="Dev Team", entry_point="dev-lead")
        result = runner.invoke(
            app, ["--catalog-dir", str(tmp_path), "team", "search", "--agent-id", "eng-mgr"]
        )
        assert result.exit_code == 0
        assert "team1" in result.output

    def test_search_no_results(self, tmp_path: Path) -> None:
        _make_dirs(tmp_path)
        _seed_agent(tmp_path, "eng-mgr", role="Manager")
        _seed_team(tmp_path, "team1", name="Engineering", entry_point="eng-mgr")
        result = runner.invoke(
            app, ["--catalog-dir", str(tmp_path), "team", "search", "--name", "Ghost"]
        )
        assert result.exit_code == 0
        assert "No entries found" in result.output
