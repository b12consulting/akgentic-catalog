"""Example 07 — Python-First Workflows.

Purpose
-------
Demonstrate the D10 Python-first principle: every catalog entry can be built
and registered entirely in Python using Pydantic constructors.  YAML is just
one persistence interface — the service layer accepts models directly and
writes YAML as a side effect.

What you'll learn
-----------------
* **All-Python construction** — build ``TemplateEntry``, ``ToolEntry``,
  ``AgentEntry``, and ``TeamSpec`` entirely via Pydantic constructors.
* **Identical validation** — entries go through the same ``create()``
  pipeline regardless of origin (Python, YAML, REST, CLI).
* **YAML as side effect** — after ``create()``, YAML files appear on disk
  without the developer ever writing them.
* **Prototyping workflow** — modify an entry in Python, call
  ``catalog.update()``, and assert the change persists.
* **Testing workflow** — create entries in a fixture-style block, exercise
  ``get`` / ``list`` / ``search``, then tear down the temp directory.
* **Programmatic generation** — build agent entries in a loop from a roles
  list, register each via ``catalog.create()``, and assert all appear in
  ``catalog.list()``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from akgentic.agent.config import AgentConfig
from akgentic.core import AgentCard
from akgentic.llm.config import ModelConfig
from akgentic.llm.prompts import PromptTemplate
from akgentic.tool.planning.planning import PlanningTool, UpdatePlanning
from akgentic.tool.search.search import SearchTool, WebSearch

from akgentic.catalog import (
    AgentCatalog,
    AgentEntry,
    AgentQuery,
    TeamCatalog,
    TeamMemberSpec,
    TeamSpec,
    TemplateCatalog,
    TemplateEntry,
    ToolCatalog,
    ToolEntry,
    YamlAgentCatalogRepository,
    YamlTeamCatalogRepository,
    YamlTemplateCatalogRepository,
    YamlToolCatalogRepository,
)

# --- Helpers ---


def _wire_catalogs(
    template_dir: str,
    tool_dir: str,
    agent_dir: str,
    team_dir: str,
) -> tuple[TemplateCatalog, ToolCatalog, AgentCatalog, TeamCatalog]:
    """Wire four catalogs with upstream dependencies using temp directories."""
    template_repo = YamlTemplateCatalogRepository(Path(template_dir))
    tool_repo = YamlToolCatalogRepository(Path(tool_dir))
    agent_repo = YamlAgentCatalogRepository(Path(agent_dir))
    team_repo = YamlTeamCatalogRepository(Path(team_dir))

    template_catalog = TemplateCatalog(template_repo)
    tool_catalog = ToolCatalog(tool_repo)
    agent_catalog = AgentCatalog(agent_repo, template_catalog, tool_catalog)
    team_catalog = TeamCatalog(team_repo, agent_catalog)
    return template_catalog, tool_catalog, agent_catalog, team_catalog


def _make_agent(
    agent_id: str,
    role: str,
    skills: list[str],
    description: str,
    tool_ids: list[str] | None = None,
    routes_to: list[str] | None = None,
) -> AgentEntry:
    """Build an AgentEntry entirely in Python — no YAML involved."""
    return AgentEntry(
        id=agent_id,
        tool_ids=tool_ids or [],
        card=AgentCard(
            role=role,
            description=description,
            skills=skills,
            agent_class="akgentic.agent.agent.BaseAgent",
            config=AgentConfig(
                name=f"@{role}",
                role=role,
                prompt=PromptTemplate(
                    template=f"You are a {role.lower()}.",
                    params={},
                ),
                model_cfg=ModelConfig(
                    provider="openai",
                    model="gpt-4.1",
                    temperature=0.3,
                ),
            ),
            routes_to=routes_to or [],
        ),
    )


def main() -> None:
    """Run example demonstrating Python-first catalog workflows."""
    with (
        tempfile.TemporaryDirectory() as template_dir,
        tempfile.TemporaryDirectory() as tool_dir,
        tempfile.TemporaryDirectory() as agent_dir,
        tempfile.TemporaryDirectory() as team_dir,
    ):
        template_catalog, tool_catalog, agent_catalog, team_catalog = _wire_catalogs(
            template_dir, tool_dir, agent_dir, team_dir,
        )

        # =============================================================
        # Section 1: All-Python construction — build entries via Pydantic
        # =============================================================
        print("=== Section 1: All-Python Construction ===\n")

        # TemplateEntry — built entirely in Python
        template = TemplateEntry(
            id="researcher-prompt",
            template="You are a {role} researching {topic}. Focus on {depth} analysis.",
        )
        template_catalog.create(template)
        print(f"  TemplateEntry created: id={template.id!r}  placeholders={template.placeholders}")
        assert len(template.placeholders) == 3, (
            f"Expected 3 placeholders, got {template.placeholders}"
        )
        assert set(template.placeholders) == {"role", "topic", "depth"}, (
            f"Unexpected placeholders: {template.placeholders}"
        )

        # ToolEntry — built entirely in Python
        search_tool = ToolEntry(
            id="web-search",
            tool_class="akgentic.tool.search.search.SearchTool",
            tool=SearchTool(
                name="Web Search",
                description="Search the web for current information",
                web_search=WebSearch(max_results=5),
                web_crawl=True,
                web_fetch=True,
            ),
        )
        tool_catalog.create(search_tool)
        print(f"  ToolEntry created: id={search_tool.id!r}  name={search_tool.tool.name!r}")

        planner_tool = ToolEntry(
            id="task-planner",
            tool_class="akgentic.tool.planning.planning.PlanningTool",
            tool=PlanningTool(
                name="Task Planner",
                description="Task planning and tracking tool",
                update_planning=UpdatePlanning(
                    instructions="CRITICAL: Always keep the plan updated.",
                ),
            ),
        )
        tool_catalog.create(planner_tool)
        print(f"  ToolEntry created: id={planner_tool.id!r}  name={planner_tool.tool.name!r}")

        # AgentEntry — built entirely in Python
        researcher = _make_agent(
            "researcher", "Researcher", ["research", "analysis"],
            "Researches topics using web search",
            tool_ids=["web-search"],
        )
        agent_catalog.create(researcher)
        print(f"  AgentEntry created: id={researcher.id!r}  role={researcher.card.role!r}")

        analyst = _make_agent(
            "analyst", "Analyst", ["analysis", "data-processing"],
            "Analyzes research findings in depth",
        )
        agent_catalog.create(analyst)
        print(f"  AgentEntry created: id={analyst.id!r}  role={analyst.card.role!r}")

        # TeamSpec — built entirely in Python
        team = TeamSpec(
            id="research-team",
            name="Research Team",
            description="Research and analysis team",
            entry_point="researcher",
            message_types=["akgentic.agent.messages.AgentMessage"],
            members=[
                TeamMemberSpec(
                    agent_id="researcher",
                    headcount=1,
                    members=[TeamMemberSpec(agent_id="analyst", headcount=1)],
                ),
            ],
        )
        team_catalog.create(team)
        print(f"  TeamSpec created: id={team.id!r}  name={team.name!r}")
        print()

        # =============================================================
        # Section 2: Validation pipeline — identical to YAML-loaded
        # =============================================================
        print("=== Section 2: Validation Pipeline (Same as YAML) ===\n")

        # Verify entries are retrievable via get()
        retrieved_template = template_catalog.get("researcher-prompt")
        assert retrieved_template.id == template.id, "Template round-trip failed"
        assert retrieved_template.placeholders == template.placeholders, (
            "Template placeholders mismatch"
        )
        print(f"  template_catalog.get('researcher-prompt') → id={retrieved_template.id!r}")

        retrieved_tool = tool_catalog.get("web-search")
        assert retrieved_tool.id == search_tool.id, "Tool round-trip failed"
        assert retrieved_tool.tool.name == search_tool.tool.name, "Tool name mismatch"
        print(f"  tool_catalog.get('web-search') → id={retrieved_tool.id!r}")

        retrieved_agent = agent_catalog.get("researcher")
        assert retrieved_agent.id == researcher.id, "Agent round-trip failed"
        assert retrieved_agent.card.role == researcher.card.role, "Agent role mismatch"
        print(f"  agent_catalog.get('researcher') → id={retrieved_agent.id!r}")

        retrieved_team = team_catalog.get("research-team")
        assert retrieved_team.id == team.id, "Team round-trip failed"
        assert retrieved_team.name == team.name, "Team name mismatch"
        print(f"  team_catalog.get('research-team') → id={retrieved_team.id!r}")

        # Verify list() returns all entries
        assert len(template_catalog.list()) == 1, "Expected 1 template"
        assert len(tool_catalog.list()) == 2, "Expected 2 tools"
        assert len(agent_catalog.list()) == 2, "Expected 2 agents"
        assert len(team_catalog.list()) == 1, "Expected 1 team"
        print("  list() counts: templates=1, tools=2, agents=2, teams=1")
        print()

        # =============================================================
        # Section 3: YAML as side effect — files appear without writing
        # =============================================================
        print("=== Section 3: YAML as Side Effect ===\n")

        # After create(), YAML files should exist on disk
        template_files = list(Path(template_dir).glob("*.yaml"))
        assert len(template_files) >= 1, f"Expected YAML files in {template_dir}, found none"
        print(f"  Template YAML files: {[f.name for f in template_files]}")

        tool_files = list(Path(tool_dir).glob("*.yaml"))
        assert len(tool_files) >= 1, f"Expected YAML files in {tool_dir}, found none"
        print(f"  Tool YAML files: {[f.name for f in tool_files]}")

        agent_files = list(Path(agent_dir).glob("*.yaml"))
        assert len(agent_files) >= 1, f"Expected YAML files in {agent_dir}, found none"
        print(f"  Agent YAML files: {[f.name for f in agent_files]}")

        team_files = list(Path(team_dir).glob("*.yaml"))
        assert len(team_files) >= 1, f"Expected YAML files in {team_dir}, found none"
        print(f"  Team YAML files: {[f.name for f in team_files]}")

        print("  Developer never wrote YAML — files are a side effect of create()")
        print()

        # =============================================================
        # Section 4: Prototyping workflow — modify, update, assert
        # =============================================================
        print("=== Section 4: Prototyping Workflow ===\n")

        # Modify the template in Python and update via catalog
        updated_template = TemplateEntry(
            id="researcher-prompt",
            template="You are a senior {role} researching {topic}. {instructions}",
        )
        template_catalog.update("researcher-prompt", updated_template)

        # Verify the change persists
        refreshed = template_catalog.get("researcher-prompt")
        assert "senior" in refreshed.template, "Update did not persist"
        assert set(refreshed.placeholders) == {"role", "topic", "instructions"}, (
            f"Updated placeholders wrong: {refreshed.placeholders}"
        )
        print(f"  Updated template: placeholders changed to {refreshed.placeholders}")

        # Modify an agent — change skills and description
        updated_researcher = _make_agent(
            "researcher", "Researcher", ["research", "analysis", "synthesis"],
            "Senior researcher with synthesis capabilities",
            tool_ids=["web-search", "task-planner"],
        )
        agent_catalog.update("researcher", updated_researcher)

        refreshed_agent = agent_catalog.get("researcher")
        assert "synthesis" in refreshed_agent.card.skills, "Agent skill update did not persist"
        assert len(refreshed_agent.tool_ids) == 2, "Agent tool_ids update did not persist"
        print(f"  Updated agent: skills={refreshed_agent.card.skills}")
        print(f"  Updated agent: tool_ids={refreshed_agent.tool_ids}")
        print("  No manual YAML editing required — just Python objects and update()")
        print()

        # =============================================================
        # Section 5: Testing workflow — fixture-style create/exercise/teardown
        # =============================================================
        print("=== Section 5: Testing Workflow ===\n")

        # Simulate a test fixture: create a fresh, isolated catalog set
        with (
            tempfile.TemporaryDirectory() as test_tpl_dir,
            tempfile.TemporaryDirectory() as test_tool_dir,
            tempfile.TemporaryDirectory() as test_agent_dir,
            tempfile.TemporaryDirectory() as test_team_dir,
        ):
            test_tpl_cat, test_tool_cat, test_agent_cat, test_team_cat = _wire_catalogs(
                test_tpl_dir, test_tool_dir, test_agent_dir, test_team_dir,
            )

            # Create test fixtures in Python — no YAML fixtures needed
            test_tpl_cat.create(TemplateEntry(
                id="test-prompt",
                template="Test {scenario} with {expected_result}.",
            ))
            test_tool_cat.create(ToolEntry(
                id="test-search",
                tool_class="akgentic.tool.search.search.SearchTool",
                tool=SearchTool(
                    name="Test Search",
                    description="Search tool for testing",
                    web_search=WebSearch(max_results=3),
                ),
            ))
            test_agent_cat.create(_make_agent(
                "test-agent", "Tester", ["testing", "validation"],
                "Agent for test scenarios",
                tool_ids=["test-search"],
            ))
            test_team_cat.create(TeamSpec(
                id="test-team",
                name="Test Team",
                description="Team for testing",
                entry_point="test-agent",
                message_types=["akgentic.agent.messages.AgentMessage"],
                members=[TeamMemberSpec(agent_id="test-agent", headcount=1)],
            ))

            # Exercise: get, list, search
            assert test_tpl_cat.get("test-prompt").id == "test-prompt"
            assert len(test_tpl_cat.list()) == 1
            assert len(test_tool_cat.list()) == 1
            assert len(test_agent_cat.list()) == 1
            assert len(test_team_cat.list()) == 1

            # Search works on Python-created entries
            search_results = test_agent_cat.search(AgentQuery(skills=["testing"]))
            assert len(search_results) == 1, "Search should find test agent"
            assert search_results[0].id == "test-agent"

            print("  Fixture created: 1 template, 1 tool, 1 agent, 1 team")
            print("  get(), list(), search() all work on Python-created entries")
            print("  No YAML fixture files authored — entries built in code")

        # After the with block, temp dirs are cleaned up automatically
        print("  Temp directories cleaned up — full isolation achieved")
        print()

        # =============================================================
        # Section 6: Programmatic generation — build agents from a list
        # =============================================================
        print("=== Section 6: Programmatic Generation ===\n")

        # Define roles — same pattern as agent_team.py
        roles = [
            {"id": "gen-planner", "role": "Planner", "skills": ["planning", "strategy"],
             "description": "Plans and strategizes project tasks"},
            {"id": "gen-coder", "role": "Coder", "skills": ["coding", "debugging"],
             "description": "Writes and debugs code"},
            {"id": "gen-reviewer", "role": "Reviewer", "skills": ["review", "quality"],
             "description": "Reviews code and ensures quality"},
            {"id": "gen-writer", "role": "Writer", "skills": ["documentation", "writing"],
             "description": "Writes documentation and reports"},
        ]

        print(f"  Generating {len(roles)} agents from roles list...\n")

        for role_def in roles:
            agent_entry = _make_agent(
                role_def["id"],
                role_def["role"],
                role_def["skills"],
                role_def["description"],
            )
            agent_catalog.create(agent_entry)
            print(f"    Created: id={agent_entry.id!r}  role={agent_entry.card.role!r}")

        # Assert all generated agents appear in catalog.list()
        all_agents = agent_catalog.list()
        generated_ids = {r["id"] for r in roles}
        catalog_ids = {a.id for a in all_agents}
        assert generated_ids.issubset(catalog_ids), (
            f"Not all generated agents found in catalog: "
            f"missing={generated_ids - catalog_ids}"
        )
        print(f"\n  catalog.list() contains {len(all_agents)} agents total")
        print(f"  All {len(roles)} generated agents confirmed in catalog")
        print()

        print("=== Example 07 complete ===")


if __name__ == "__main__":
    main()
