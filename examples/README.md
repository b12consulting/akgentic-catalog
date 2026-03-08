# akgentic-catalog Examples

Progressive examples demonstrating the akgentic-catalog module capabilities.
Each example is a self-contained script paired with a companion `.md` file
explaining concepts.

## Running the Examples

From the **project root**:

```bash
uv run python packages/akgentic-catalog/examples/01_catalog_entries.py
```

From the **akgentic-catalog directory**:

```bash
cd packages/akgentic-catalog
uv run python examples/01_catalog_entries.py
```

## Available Examples

### [01_catalog_entries.py](01_catalog_entries.py) — Template & Tool Entry Basics

First contact with the catalog API: create, retrieve, and list `TemplateEntry`
and `ToolEntry` — the two leaf entity types with no cross-catalog dependencies.
Demonstrates placeholder auto-parsing, FQCN class resolution, and duplicate-ID
error handling.

Companion docs: [01-catalog-entries.md](01-catalog-entries.md)

### [02_agent_entries.py](02_agent_entries.py) — Agent Entries & Cross-Validation

How `AgentCatalog` validates cross-catalog references (`tool_ids`,
`@`-template, `routes_to`) at registration time, and how `resolve_tools()` /
`resolve_template()` bridge catalog-form to runtime objects.

Companion docs: [02-agent-entries.md](02-agent-entries.md)

### [03_team_specs.py](03_team_specs.py) — Team Composition, Member Trees & Profiles

All four catalogs wired together. `TeamSpec` with nested `TeamMemberSpec`
hierarchies, `entry_point` validation, `headcount`, members vs profiles,
`message_types` FQCN validation, and recursive `TeamQuery` search.

Companion docs: [03-team-specs.md](03-team-specs.md)

## Prerequisites

All examples require:

- Python 3.12+
- `akgentic` (akgentic-core)
- `akgentic-agent`
- `akgentic-llm`
- `akgentic-tool`

Install with:

```bash
uv sync
```
