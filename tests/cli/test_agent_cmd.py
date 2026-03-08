"""Tests for the agent CLI commands."""

from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from akgentic.catalog.cli.main import app

runner = CliRunner()

AGENT_CLASS = "akgentic.agent.BaseAgent"


def _agent_data(
    agent_id: str,
    role: str = "engineer",
    description: str = "Test agent",
    skills: list[str] | None = None,
) -> dict:
    """Build an agent entry dict suitable for YAML serialization."""
    return {
        "id": agent_id,
        "tool_ids": [],
        "card": {
            "role": role,
            "description": description,
            "skills": skills or ["coding"],
            "agent_class": AGENT_CLASS,
            "config": {"name": f"@{role}", "role": role},
            "routes_to": [],
        },
    }


def _write_agent_yaml(path: Path, agent_id: str, **kwargs: str | list[str]) -> Path:
    """Write an agent YAML file and return its path."""
    data = _agent_data(agent_id, **kwargs)  # type: ignore[arg-type]
    file = path / f"{agent_id}.yaml"
    file.write_text(yaml.dump(data, default_flow_style=False))
    return file


def _seed_agent(
    catalog_dir: Path,
    agent_id: str,
    **kwargs: str | list[str],
) -> None:
    """Seed an agent entry directly in the catalog directory."""
    data = _agent_data(agent_id, **kwargs)  # type: ignore[arg-type]
    (catalog_dir / "agents" / f"{agent_id}.yaml").write_text(
        yaml.dump(data, default_flow_style=False)
    )


def _seed_team(
    catalog_dir: Path,
    team_id: str,
    entry_point: str,
    members: list[dict[str, str | int]] | None = None,
) -> None:
    """Seed a team entry directly in the catalog directory."""
    data = {
        "id": team_id,
        "name": f"Team {team_id}",
        "entry_point": entry_point,
        "message_types": ["akgentic.core.messages.UserMessage"],
        "members": members or [{"agent_id": entry_point, "headcount": 1}],
        "profiles": [],
        "description": f"Test team {team_id}",
    }
    (catalog_dir / "teams" / f"{team_id}.yaml").write_text(
        yaml.dump(data, default_flow_style=False)
    )


def _make_dirs(tmp_path: Path) -> None:
    for name in ("templates", "tools", "agents", "teams"):
        (tmp_path / name).mkdir(exist_ok=True)


class TestAgentList:
    """agent list command."""

    def test_list_empty(self, tmp_path: Path) -> None:
        _make_dirs(tmp_path)
        result = runner.invoke(app, ["--catalog-dir", str(tmp_path), "agent", "list"])
        assert result.exit_code == 0
        assert "No entries found" in result.output

    def test_list_with_entries(self, tmp_path: Path) -> None:
        _make_dirs(tmp_path)
        _seed_agent(tmp_path, "a1", role="Manager")
        _seed_agent(tmp_path, "a2", role="Engineer")
        result = runner.invoke(app, ["--catalog-dir", str(tmp_path), "agent", "list"])
        assert result.exit_code == 0
        assert "a1" in result.output
        assert "a2" in result.output


class TestAgentGet:
    """agent get command."""

    def test_get_existing(self, tmp_path: Path) -> None:
        _make_dirs(tmp_path)
        _seed_agent(tmp_path, "a1", role="Manager")
        result = runner.invoke(app, ["--catalog-dir", str(tmp_path), "agent", "get", "a1"])
        assert result.exit_code == 0
        assert "a1" in result.output

    def test_get_nonexistent(self, tmp_path: Path) -> None:
        _make_dirs(tmp_path)
        result = runner.invoke(app, ["--catalog-dir", str(tmp_path), "agent", "get", "nope"])
        assert result.exit_code == 1
        assert "Not found" in result.output


class TestAgentCreate:
    """agent create command."""

    def test_create_from_yaml(self, tmp_path: Path) -> None:
        _make_dirs(tmp_path)
        yaml_file = _write_agent_yaml(tmp_path, "new-agent", role="Researcher")
        result = runner.invoke(
            app, ["--catalog-dir", str(tmp_path), "agent", "create", str(yaml_file)]
        )
        assert result.exit_code == 0
        # Verify it was actually created
        get_result = runner.invoke(
            app, ["--catalog-dir", str(tmp_path), "agent", "get", "new-agent"]
        )
        assert get_result.exit_code == 0
        assert "new-agent" in get_result.output

    def test_create_duplicate(self, tmp_path: Path) -> None:
        _make_dirs(tmp_path)
        _seed_agent(tmp_path, "dup", role="Manager")
        yaml_file = _write_agent_yaml(tmp_path, "dup", role="Manager")
        result = runner.invoke(
            app, ["--catalog-dir", str(tmp_path), "agent", "create", str(yaml_file)]
        )
        assert result.exit_code == 1

    def test_create_with_invalid_yaml(self, tmp_path: Path) -> None:
        _make_dirs(tmp_path)
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("id: bad\n")  # Missing required card field
        result = runner.invoke(
            app, ["--catalog-dir", str(tmp_path), "agent", "create", str(bad_file)]
        )
        assert result.exit_code == 1
        assert "Validation error" in result.output


class TestAgentUpdate:
    """agent update command."""

    def test_update_existing(self, tmp_path: Path) -> None:
        _make_dirs(tmp_path)
        _seed_agent(tmp_path, "a1", role="Manager")
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        yaml_file = _write_agent_yaml(input_dir, "a1", role="Senior Manager")
        result = runner.invoke(
            app, ["--catalog-dir", str(tmp_path), "agent", "update", "a1", str(yaml_file)]
        )
        assert result.exit_code == 0

    def test_update_nonexistent(self, tmp_path: Path) -> None:
        _make_dirs(tmp_path)
        yaml_file = _write_agent_yaml(tmp_path, "nope", role="Ghost")
        result = runner.invoke(
            app, ["--catalog-dir", str(tmp_path), "agent", "update", "nope", str(yaml_file)]
        )
        assert result.exit_code == 1


class TestAgentDelete:
    """agent delete command."""

    def test_delete_existing(self, tmp_path: Path) -> None:
        _make_dirs(tmp_path)
        _seed_agent(tmp_path, "a1", role="Manager")
        result = runner.invoke(app, ["--catalog-dir", str(tmp_path), "agent", "delete", "a1"])
        assert result.exit_code == 0

    def test_delete_nonexistent(self, tmp_path: Path) -> None:
        _make_dirs(tmp_path)
        result = runner.invoke(app, ["--catalog-dir", str(tmp_path), "agent", "delete", "nope"])
        assert result.exit_code == 1
        assert "Not found" in result.output

    def test_delete_protected_by_team(self, tmp_path: Path) -> None:
        _make_dirs(tmp_path)
        _seed_agent(tmp_path, "eng-mgr", role="Manager")
        _seed_team(
            tmp_path,
            "eng-team",
            entry_point="eng-mgr",
            members=[{"agent_id": "eng-mgr", "headcount": 1}],
        )
        result = runner.invoke(app, ["--catalog-dir", str(tmp_path), "agent", "delete", "eng-mgr"])
        assert result.exit_code == 1
        assert "eng-team" in result.output


class TestAgentSearch:
    """agent search command."""

    def test_search_by_role(self, tmp_path: Path) -> None:
        _make_dirs(tmp_path)
        _seed_agent(tmp_path, "a1", role="Manager")
        _seed_agent(tmp_path, "a2", role="Engineer")
        result = runner.invoke(
            app, ["--catalog-dir", str(tmp_path), "agent", "search", "--role", "Manager"]
        )
        assert result.exit_code == 0
        assert "a1" in result.output

    def test_search_by_skill(self, tmp_path: Path) -> None:
        _make_dirs(tmp_path)
        _seed_agent(tmp_path, "a1", role="Researcher", skills=["research", "analysis"])
        _seed_agent(tmp_path, "a2", role="Engineer", skills=["coding"])
        result = runner.invoke(
            app, ["--catalog-dir", str(tmp_path), "agent", "search", "--skill", "research"]
        )
        assert result.exit_code == 0
        assert "a1" in result.output

    def test_search_no_results(self, tmp_path: Path) -> None:
        _make_dirs(tmp_path)
        _seed_agent(tmp_path, "a1", role="Manager")
        result = runner.invoke(
            app, ["--catalog-dir", str(tmp_path), "agent", "search", "--role", "Ghost"]
        )
        assert result.exit_code == 0
        assert "No entries found" in result.output
