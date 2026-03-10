# Example 03 — Team Composition, Member Trees & Profiles

## Concepts Covered

### TeamEntry with Nested TeamMemberSpec

A `TeamEntry` defines a team as a hierarchical tree of `TeamMemberSpec` nodes.
Each node carries an `agent_id` (referencing an `AgentEntry` in the catalog),
a `headcount` (defaulting to 1), and optional nested `members`. The tree can
nest to arbitrary depth, mirroring real delegation chains.

### entry_point Validation

Every team has an `entry_point` — the agent that receives external messages
(the "front door"). The catalog enforces that this agent_id appears somewhere
in the `members` tree. An agent that only exists in `profiles` does not
satisfy this check.

### headcount for Multi-Instance Agents

Setting `headcount=2` on a member tells the orchestrator to create two
instances of that agent at startup. This is useful for parallel workers
(e.g. two researchers processing different queries simultaneously).

### members vs profiles

- **members** — agents instantiated at startup as part of the tree; they
  receive routed messages according to the hierarchy.
- **profiles** — agent IDs registered with the orchestrator for runtime
  hiring (like an Expert who joins on demand) but **not** part of the
  initial tree.

### message_types FQCN Validation

`message_types` stores fully qualified class name strings (e.g.
`"akgentic.agent.messages.AgentMessage"`). The catalog validates each FQCN
at `create()` time using `import_class()`, catching typos and missing
modules before runtime.

### Recursive TeamQuery Search

`TeamQuery(agent_id="researcher")` searches the entire nested member tree
recursively using `agent_in_members()`. This finds teams containing that
agent at any depth — not just at the top level.
`TeamQuery(name="research")` performs a case-insensitive substring match
on the team name.

## Key API Patterns

### TeamCatalog Construction with Upstream Wiring

```python
team_catalog = TeamCatalog(
    repository=team_repo,
    agent_catalog=agent_catalog,  # Required for member/profile validation
)
```

`TeamCatalog` is the fourth catalog in the wiring chain. It requires
`AgentCatalog` so it can validate that every `agent_id` in the members
tree and profiles list actually exists.

### TeamEntry Construction

```python
TeamEntry(
    id="research-team",
    name="Research Team",
    entry_point="manager",
    message_types=["akgentic.agent.messages.AgentMessage"],
    members=[
        TeamMemberSpec(
            agent_id="manager",
            members=[
                TeamMemberSpec(agent_id="researcher", headcount=2,
                               members=[TeamMemberSpec(agent_id="analyst")]),
                TeamMemberSpec(agent_id="reviewer"),
            ],
        ),
    ],
    profiles=["specialist"],
)
```

### resolve_entry_point() and resolve_message_types()

These are methods on `TeamEntry` (not on the catalog service):

```python
team = team_catalog.get("research-team")
entry_agent = team.resolve_entry_point(agent_catalog)   # → AgentEntry
msg_types = team.resolve_message_types()                # → list[type]
```

`resolve_entry_point()` looks up the entry point agent_id in the catalog
and returns the full `AgentEntry`. `resolve_message_types()` imports each
FQCN string and returns the actual Python class objects.

### Recursive TeamQuery

```python
# Finds team because "researcher" is nested inside the members tree
team_catalog.search(TeamQuery(agent_id="researcher"))

# Case-insensitive substring match on team name
team_catalog.search(TeamQuery(name="research"))
```

## Common Pitfalls

- **entry_point must be in the members tree, not just any agent.** An agent
  that only appears in `profiles` will fail validation. The entry point is
  the team's front door — it must be part of the startup hierarchy.

- **profiles are not instantiated at startup.** They are available for
  runtime hiring only. Do not expect a profiles-only agent to receive
  messages in the initial team setup.

- **message_types uses FQCN strings, not class objects.** Pass
  `"akgentic.agent.messages.AgentMessage"` (a string), not the imported
  class. The catalog resolves these strings at registration time.

- **headcount defaults to 1.** You only need to set it explicitly when you
  want multiple instances of the same agent.

- **agent_id in TeamQuery does a recursive walk.** It searches the entire
  nested tree, not just top-level members. This is useful for finding which
  team a deeply nested agent belongs to.

- **Agent creation order matters.** Because `routes_to` validation checks
  against existing agents' `config.name` values, create leaf agents first,
  then agents that route to them.

## Related Examples

- [Example 02 — Agent Entries & Cross-Validation](02-agent-entries.md):
  How `AgentEntry` cross-validates tool_ids, templates, and routes_to
- [Example 04 — YAML Repository Round-Trip](04-yaml-persistence.md):
  Persisting and loading catalog entries from YAML files
