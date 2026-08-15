"""Sharing one value between entries — ``NativeValue`` and the single unwrap site.

Run it directly::

    python examples/03_sharing_values.py

Example ``02`` established that a ``__ref__`` marker is a *pure pointer*: it carries
``__ref__`` and optionally ``__type__`` / ``__namespace__``, and nothing else. That
leaves a question open — if a marker cannot carry a patch, how do two entries share
one value? The answer is a ``NativeValue`` entry, and this is where it is met.

The namespace built here mirrors the shipped ``data/catalog/agent-team/``: one prompt
body stored once, three agents whose per-agent variation lives in their own inline
payloads. It **builds its own copy in a temporary directory and never reads ``data/``**
— go and read the real files if you want to compare. The shipped agents also carry
``model_cfg`` and ``tools`` refs into a second namespace; those are deliberately left
out, because cross-namespace addressing is a separate topic.

The narrative half is ``03-sharing-values.md``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from akgentic.agent import AgentConfig
from akgentic.core import AgentCard
from akgentic.llm import PromptTemplate

from akgentic.catalog import (
    UNSET_NAMESPACE,
    Catalog,
    CatalogValidationError,
    Entry,
    YamlEntryRepository,
)

REQUIRES: tuple[str, ...] = ()
"""Importable module names this example needs beyond the base install — none.

The test harness calls ``pytest.importorskip`` on every name here before invoking
``main()``. An example that ever needs an optional package must import it *inside*
``main()``: the harness reads this declaration off the already-imported module, so
a module-level import raises before the tuple can be read.
"""

TEAM_ID = "agent-team"
TEMPLATE_ID = "id_team_template"
AGENT_IDS = ("assistant", "expert", "human_proxy")
OWNER = "u1"

# Stored once, in its own entry. The placeholders are filled in per agent.
SHARED_BODY = "You are a helpful {role}. \n{instructions}"

# What the shared body is edited to, to prove the three agents really do share it.
REVISED_BODY = "You are a rigorous {role}, and you cite your sources.\n{instructions}"

# The marker every agent carries at ``config.prompt.template`` — a pure pointer at
# the one entry holding the body.
TEMPLATE_REF: dict[str, Any] = {"__ref__": TEMPLATE_ID}

# The half that is NOT shared: each agent writes its own params inline. This is the
# idiom refusal 2 of example 02 points at — inline your own payload, ref the shared part.
AGENT_PARAMS: dict[str, dict[str, str]] = {
    "assistant": {"role": "assistant", "instructions": "Answer the user directly."},
    "expert": {"role": "expert", "instructions": "Provide deep specialized knowledge."},
    "human_proxy": {"role": "human proxy", "instructions": "Relay the human's intent."},
}

# A team entry anchors its namespace, so its card is inline: at that moment there is
# nothing in the namespace to point at.
ANCHOR_CARD: dict[str, Any] = {
    "description": "Placeholder while the namespace bootstraps",
    "skills": [],
    "agent_class": "akgentic.agent.BaseAgent",
    "config": {"name": "@Bootstrap", "role": "Bootstrap"},
}


def main() -> None:
    """Build an ``agent-team``-shaped namespace in a temp dir and prove the body is shared."""
    with tempfile.TemporaryDirectory() as tmpdir:
        catalog = Catalog(YamlEntryRepository(Path(tmpdir)))

        namespace = catalog.create(_team_entry()).namespace
        catalog.create(_template_entry(namespace))
        for agent_id in AGENT_IDS:
            catalog.create(_agent_entry(namespace, agent_id))
        print(f"0. one prompt body, {len(AGENT_IDS)} agents pointing at it")

        _assert_consumer_receives_a_bare_str(catalog, namespace)
        print("1. the consumer's typed `str` field received the bare scalar")

        _assert_one_body_many_renderings(catalog, namespace)
        print("2. one stored template, three different rendered prompts")

        _assert_update_propagates(catalog, namespace)
        print("3. editing the single NativeValue changed all three agents")

        _assert_storage_is_not_flattened(catalog, namespace)
        print("4. the marker is still stored, and a direct fetch is unwrapped by nobody")

        _assert_shared_entry_is_guarded(catalog, namespace)
        print("5. all three referrers are found, and the delete is refused")


# --- Building the namespace -------------------------------------------------------


def _team_entry() -> Entry:
    """Build the anchor team entry, asking the catalog to mint its namespace."""
    return Entry(
        id=TEAM_ID,
        kind="team",  # plain string: EntryKind is a Literal alias, not an enum
        namespace=UNSET_NAMESPACE,  # replaced by a fresh UUID on create
        user_id=OWNER,
        model_type="akgentic.team.models.TeamCard",
        payload={
            "name": "Agent Team",
            "entry_point": {"card": ANCHOR_CARD, "headcount": 1, "members": []},
            "members": [],
        },
    )


def _template_entry(namespace: str) -> Entry:
    """Build the one entry that holds the shared body.

    ``kind`` stays *semantic*: a shared prompt body is ``kind="prompt"``. There is no
    ``kind="native"`` — ``model_type`` says how the payload is shaped, ``kind`` says
    what the entry is for, and ``NativeValue`` is a shape, not a purpose.
    """
    return Entry(
        id=TEMPLATE_ID,
        kind="prompt",
        namespace=namespace,
        user_id=OWNER,
        model_type="akgentic.catalog.NativeValue",
        description="Shared system-prompt template body for team members",
        payload={"value": SHARED_BODY},
    )


def _agent_entry(namespace: str, agent_id: str) -> Entry:
    """Build one agent: an inline ``PromptTemplate`` payload whose ``template`` is a marker.

    Note where the marker sits — at ``config.prompt.template``, the ``str`` field, not
    at ``config.prompt``. Refing the whole ``PromptTemplate`` would share the params
    too, and the params are exactly what each agent needs to vary.
    """
    return Entry(
        id=agent_id,
        kind="agent",
        namespace=namespace,
        user_id=OWNER,  # sub-entries must match the anchor entry's owner
        model_type="akgentic.core.AgentCard",
        description=f"Team member {agent_id}",
        payload={
            "description": f"Team member {agent_id}",
            "agent_class": "akgentic.agent.BaseAgent",
            "skills": [],
            "config": {
                "name": f"@{agent_id}",
                "role": agent_id,
                "prompt": {
                    "template": dict(TEMPLATE_REF),  # shared, by pointer
                    "params": dict(AGENT_PARAMS[agent_id]),  # inline, per agent
                },
            },
        },
    )


# --- Assertions -------------------------------------------------------------------


def _resolved_prompt(catalog: Catalog, namespace: str, agent_id: str) -> PromptTemplate:
    """Resolve ``agent_id`` and return its ``config.prompt``, narrowed twice.

    Both narrowings are needed under ``mypy --strict``: ``resolve_by_id`` is declared
    ``-> BaseModel``, and ``AgentCard.config`` is declared ``BaseConfig``, which has no
    ``prompt`` field even though the runtime value is an ``AgentConfig``.

    ``AgentConfig`` must be imported from ``akgentic.agent``. ``akgentic.core.agent_config``
    exports a same-named alias of ``BaseConfig``; importing that one type-checks cleanly
    and asserts something weaker than it appears to.
    """
    resolved = catalog.resolve_by_id(namespace, agent_id)
    assert isinstance(resolved, AgentCard), f"resolved to {type(resolved).__name__}"
    assert isinstance(resolved.config, AgentConfig), f"config is {type(resolved.config).__name__}"
    return resolved.config.prompt


def _assert_consumer_receives_a_bare_str(catalog: Catalog, namespace: str) -> None:
    """Require the consumer's typed ``str`` field to hold the shared body itself.

    The value was written into one entry (``id_team_template``) and is read back off a
    different one (``expert``), through a marker — an end-to-end claim about the splice,
    not a restatement of an input.

    The ``isinstance`` check guards the case where ``PromptTemplate.template`` is one day
    widened (to ``Any``, or to a union admitting the wrapper): today a regressed unwrap
    fails earlier, inside ``resolve_by_id``, because a ``str`` field rejects a wrapper.
    """
    template = _resolved_prompt(catalog, namespace, "expert").template
    assert isinstance(template, str), f"expected a bare str, got {type(template).__name__}"
    assert template == SHARED_BODY, repr(template)


def _assert_one_body_many_renderings(catalog: Catalog, namespace: str) -> None:
    """Require one shared template, three distinct params maps, three distinct renderings.

    This is the payoff: the body is stored once, and the per-consumer variation lives in
    the consumer. Cardinalities are asserted rather than pairwise equalities so the claim
    stays exactly "all the same" / "all different".
    """
    prompts = [_resolved_prompt(catalog, namespace, a) for a in AGENT_IDS]

    assert {p.template for p in prompts} == {SHARED_BODY}, [p.template for p in prompts]

    distinct_params = {tuple(sorted(p.params.items())) for p in prompts}
    assert len(distinct_params) == len(AGENT_IDS), distinct_params

    distinct_renderings = {p.render() for p in prompts}
    assert len(distinct_renderings) == len(AGENT_IDS), distinct_renderings


def _assert_update_propagates(catalog: Catalog, namespace: str) -> None:
    """Edit the single stored body and require all three consumers to follow.

    **This is the assertion that cannot pass for the wrong reason.** Three agents holding
    inlined copies of the body would still resolve to the old string; only genuine
    sharing propagates. No agent entry is touched between the two reads.

    The new entry is derived with ``model_copy(update=...)`` rather than rebuilt field by
    field, and the payload is replaced outright — nothing the assertion below claims to
    have changed is quietly re-sent by the caller.
    """
    before = {_resolved_prompt(catalog, namespace, a).template for a in AGENT_IDS}
    assert before == {SHARED_BODY}, before

    stored = catalog.get(namespace, TEMPLATE_ID)
    catalog.update(stored.model_copy(update={"payload": {"value": REVISED_BODY}}))

    after = {_resolved_prompt(catalog, namespace, a).template for a in AGENT_IDS}
    assert after == {REVISED_BODY}, after


def _assert_storage_is_not_flattened(catalog: Catalog, namespace: str) -> None:
    """Require the stored marker to survive, and a direct fetch to stay wrapped.

    Resolution happens on the way out. The update above rewrote one entry and left the
    three agents' stored payloads alone — what is written down is still a pointer, which
    is why editing the target moved all three.

    The second assertion is the other half of the rule: the resolver unwraps **only** at
    the ref-splice site. ``get`` on the shared entry returns an ordinary entry whose
    payload is the ``NativeValue`` wrapper, ``{"value": ...}``, not the bare scalar.
    """
    stored_marker = catalog.get(namespace, "expert").payload["config"]["prompt"]["template"]
    assert stored_marker == TEMPLATE_REF, f"the stored payload was flattened: {stored_marker}"

    shared = catalog.get(namespace, TEMPLATE_ID)
    assert shared.payload == {"value": REVISED_BODY}, shared.payload


def _assert_shared_entry_is_guarded(catalog: Catalog, namespace: str) -> None:
    """Require every referrer to be findable, and the delete to be refused naming each.

    Sharing does not weaken the delete guard: a widely-referenced entry is guarded like
    any other, and every referrer is reported in one pass rather than one per attempt.
    """
    referrers = catalog.find_references(namespace, TEMPLATE_ID)
    assert {e.id for e in referrers} == set(AGENT_IDS), sorted(e.id for e in referrers)

    try:
        catalog.delete(namespace, TEMPLATE_ID)
    except CatalogValidationError as exc:
        expected = {
            f"Entry '{agent_id}' (kind=agent) in namespace '{namespace}' references '{TEMPLATE_ID}'"
            for agent_id in AGENT_IDS
        }
        assert set(exc.errors) == expected, exc.errors
    else:
        raise AssertionError("the shared entry was deleted out from under three referrers")

    # A refused delete changes nothing: the body is still there, for all three.
    assert catalog.get(namespace, TEMPLATE_ID).payload == {"value": REVISED_BODY}


if __name__ == "__main__":
    main()
