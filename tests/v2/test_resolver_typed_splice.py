"""Story 15.6 integration — end-to-end typed-instance splice against real fixtures.

Epic 15 Story 15.6 made ``populate_refs`` return a typed Pydantic instance at
every ref-marker position (built from the referenced entry's declared
``model_type``) so polymorphic parent fields (``AgentConfig.tools: list[ToolCard]``,
``TeamCard.message_types: list[type]``, …) validate against concrete subclasses
without requiring authors to repeat the FQCN inside payloads via ``__model__``
markers.

This module exercises the change against the trimmed ``agent-team-v1`` bundle
under ``packages/akgentic-catalog/tests/fixtures/`` — real ``ToolCard`` /
``AgentCard`` / ``TeamCard`` shapes with real ref chains (tool-ref → agent-ref
→ team-ref, plus a shared model-ref). AC #6 (``resolve`` returns a typed
``TeamCard``), AC #8 (``_check_transient_validation`` mirrors the resolve path),
AC #9 (end-to-end ``validate_namespace`` reports ``ok=True``), and the
polymorphic-field invariants (AC #1) are covered here; unit-level splice
semantics remain in :mod:`.test_populate_refs`.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

# These imports exercise real ToolCard / AgentCard / TeamCard classes so the
# allowlisted model_types in the fixture resolve to live code.
import akgentic.tool  # noqa: F401 — load real ToolCard subclasses (search, planning, workspace)
import pytest

from akgentic.catalog.catalog import Catalog
from akgentic.catalog.repositories.yaml import YamlEntryRepository
from akgentic.catalog.resolver import populate_refs, resolve

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures"
"""Directory holding the trimmed ``agent-team-v1`` bundle used by this module."""


@pytest.fixture
def agent_team_v1_repo(tmp_path: Path) -> YamlEntryRepository:
    """Copy the fixture bundle into a tmp path and return a ``YamlEntryRepository``.

    Copying into ``tmp_path`` isolates the test from any repository write that
    might happen inside the assertion (``put`` would otherwise mutate the
    checked-in fixture). The copy is per-test via pytest's function-scoped
    ``tmp_path``.
    """
    dest = tmp_path / "catalog"
    shutil.copytree(FIXTURE_ROOT, dest)
    return YamlEntryRepository(dest)


class TestResolveTeamFromFixture:
    """AC #6 — ``resolve()`` on the team entry returns a typed ``TeamCard``."""

    def test_team_entry_resolves_to_team_card_with_typed_tools(
        self, agent_team_v1_repo: YamlEntryRepository
    ) -> None:
        from akgentic.team.models import TeamCard
        from akgentic.tool.core import ToolCard

        team_entry = agent_team_v1_repo.get("agent-team-v1", "team")
        assert team_entry is not None

        result = resolve(team_entry, agent_team_v1_repo)

        assert isinstance(result, TeamCard)
        # The manager's nested tools must round-trip as ToolCard instances —
        # this is the failure mode Story 15.6 fixes: before the change, the
        # tool refs spliced as bare dicts and AgentConfig.tools: list[ToolCard]
        # failed with "Can't instantiate abstract class ToolCard".
        _assert_tools_are_typed_instances(result, ToolCard)


def _assert_tools_are_typed_instances(team_card: Any, tool_base: type) -> None:
    """Walk the team's member tree and assert every ``tools`` entry is a ``tool_base``.

    Extracted to keep the test body declarative — the walk depends on the
    TeamCard + TeamMemberSpec + AgentCard shapes from akgentic-core / team /
    agent, so the assertion can't be a single one-liner without pulling every
    import into the test.
    """
    # entry_point + top-level members cover the whole tree per TeamCard's shape.
    queue: list[Any] = [team_card.entry_point, *team_card.members]
    seen_at_least_one_tool = False
    while queue:
        member = queue.pop()
        card = member.card
        config = getattr(card, "config", None)
        tools = getattr(config, "tools", None) if config is not None else None
        if tools:
            for tool in tools:
                assert isinstance(tool, tool_base), f"Expected {tool_base}, got {type(tool)}"
                seen_at_least_one_tool = True
        if member.members:
            queue.extend(member.members)
    assert seen_at_least_one_tool, "Fixture must exercise at least one tools ref for AC #1"


class TestValidateNamespaceFromFixture:
    """AC #9 — full-namespace validation reports ``ok=True`` on the fixture bundle."""

    def test_validate_namespace_ok(self, agent_team_v1_repo: YamlEntryRepository) -> None:
        catalog = Catalog(agent_team_v1_repo)
        report = catalog.validate_namespace("agent-team-v1")
        # A green report means every Epic 15 + Story 15.6 check (allowlist,
        # per-entry model validation, cycle, type mismatch, lineage, ownership,
        # transient validation via populate_refs + model_validate) passed with
        # zero entry issues and zero global errors.
        assert report.ok, f"Namespace report was not ok: {report}"
        assert report.global_errors == []
        assert report.entry_issues == []


class TestPopulateRefsTypedOnFixture:
    """AC #1 — tool refs resolve to typed ``SearchTool``/``PlanningTool``/``WorkspaceTool``."""

    def test_assistant_agent_tools_are_typed(
        self, agent_team_v1_repo: YamlEntryRepository
    ) -> None:
        from akgentic.tool.planning import PlanningTool
        from akgentic.tool.search import SearchTool
        from akgentic.tool.workspace import WorkspaceTool

        assistant_entry = agent_team_v1_repo.get("agent-team-v1", "assistant")
        assert assistant_entry is not None

        populated = populate_refs(
            assistant_entry.payload, agent_team_v1_repo, assistant_entry.namespace
        )
        tool_instances = populated["config"]["tools"]
        # Order matches the fixture: web_search, planning, workspace.
        assert isinstance(tool_instances[0], SearchTool)
        assert isinstance(tool_instances[1], PlanningTool)
        assert isinstance(tool_instances[2], WorkspaceTool)
