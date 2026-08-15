# 03 — Sharing a value between entries

```bash
python examples/03_sharing_values.py
```

It prints six lines and exits 0. It also runs as part of `pytest tests/`.

Example `02` ended on a rule and an unanswered question. The rule: **a `__ref__` marker is
a pure pointer and takes no other keys.** The question it creates: if a marker cannot
carry a patch, how do two entries share one value and still differ?

This is the answer.

The namespace the script builds mirrors one that ships with the framework:
`data/catalog/agent-team/` — one prompt body stored once, three agents pointing at it.
Read those files beside this page if you want to see the real thing. The script builds its
own copy in a temporary directory and **never reads `data/`**, and it stops at the parts
this example is about: the shipped team wires its `entry_point.card` and a `members` tree
to its agents by marker (example `02`'s subject, not this one), and the shipped agents
carry `model_cfg` and `tools` refs into a second namespace, which is cross-namespace
addressing and a separate topic again.

## The question, concretely

Three agents want the same system-prompt body and their own parameters. The shape that
suggests itself is one shared `PromptTemplate` entry with each agent overriding a piece
of it:

```yaml
# What you cannot write
config:
  prompt:
    __ref__: id_team_prompt
    params:                          # ← refused
      role: expert
```

```
ref marker to 'id_team_prompt' carries key 'params' — a ref marker is a pure pointer
and takes no other keys. Inline the payload and reference shared values via a
NativeValue entry.
```

The error names the replacement. **Turn it inside out**: instead of sharing the whole
model and patching it, inline the model and share only the part that is genuinely
common.

```yaml
# What you write instead
config:
  prompt:                            # an inline PromptTemplate payload
    template:
      __ref__: id_team_template      # the shared half — a pointer
    params:                          # the varying half — written here
      role: expert
      instructions: Provide deep specialized knowledge.
```

The marker moved *down*, from the whole `PromptTemplate` to its `template` field. That
field is a `str`, which is the whole difficulty: a `__ref__` marker resolves to the
target entry's model, and there is no entry whose model is a bare `str`.

## `NativeValue` — an entry whose payload is one value

```yaml
id: id_team_template
kind: prompt
model_type: akgentic.catalog.NativeValue
payload:
  value: "You are a helpful {role}. \n{instructions}"
```

That is the whole model: a single field, `value`, spanning `str | int | float | bool |
list | dict`. It exists so a scalar can *be an entry* — because being an entry is what
makes something addressable by a marker.

**The resolver unwraps it in exactly one place.** At a ref-splice site, after building the
target's model, if that model is a `NativeValue` the resolver returns `instance.value` —
the bare scalar — rather than the wrapper. So the consumer's typed `str` field receives a
`str`. Everywhere else, a `NativeValue` entry behaves like any other entry: `get` returns
an ordinary `Entry` whose payload is `{"value": ...}`, and the script asserts both halves.

## Sharing, proved by changing

Three agents resolving to the same string proves very little — three inlined copies of the
same literal would too. So the script does the thing that discriminates: it edits the
**single** `NativeValue` entry, touches no agent, and re-resolves all three.

```python
stored = catalog.get(namespace, "id_team_template")
catalog.update(stored.model_copy(update={"payload": {"value": REVISED_BODY}}))
# ... all three agents now resolve to REVISED_BODY
```

Inlined copies would still carry the old body. This is the assertion the example exists
for, and it is the one worth mutating if you want to watch the build go red.

The payoff is the other assertion beside it: one stored template, three *different*
rendered prompts, because the params never left the consumers. The body is stored once;
the variation lives where it varies.

## Why the shared entry is not privileged

It is guarded like any other target. `find_references(namespace, "id_team_template")`
returns all three agents, and deleting it while they point at it is refused — one message
per referrer, in a single pass:

```
Entry 'expert' (kind=agent) in namespace '<ns>' references 'id_team_template'
```

Being shared by three entries makes an entry harder to delete, not easier.

## `kind` stays semantic

There is no `kind="native"`. The shared prompt body above is `kind="prompt"`, because
that is what it is *for*. `model_type` describes how a payload is shaped; `kind` describes
what the entry is. `NativeValue` is a shape, and a shape is not a purpose — a shared
temperature would be `kind="model"`, a shared tool argument `kind="tool"`.

## The anti-pattern

`value` includes a `dict` arm. **It is not an untyped escape hatch, and using it as one is
the mistake this section exists to name.**

```yaml
# Don't
model_type: akgentic.catalog.NativeValue
payload:
  value:
    retries: 3
    endpoint: https://example.invalid
    prompt: {__ref__: id_team_template}     # resolved anyway — see below
```

The `dict` arm is for boundary-crossing JSON literals: a blob the splice site consumes as
opaque data. If you need structured content with typed fields, **write a real Pydantic
model and store it as a normal entry.** You get validation, an allowlisted `model_type`,
and refs that work.

Nothing stops you. `NativeValue` declares one field and says nothing about the interior of
a dict handed to it, so a dict of any shape is accepted and stored verbatim. The catalog
does **not** block the misuse mechanically.

**But the dict is not inert, and that is the part that bites.** Ref resolution is a
generic structural walk over the entire payload, and it does not stop at `value`. A
`__ref__` marker buried anywhere inside that dict is resolved like any other marker —
*before* the unwrap hands the value back — so the consumer receives a dict whose contents
have been substituted out from under it, and a marker pointing at a missing entry raises
from somewhere you never intended to be resolving at all.

That is the argument for the real model, and it is stronger than "the shortcut is untidy":
the shortcut buys you the resolver's semantics without the typed model that would say what
the shape was supposed to be. The discipline is yours, which is precisely why it is written
down here rather than left to be discovered.

## Where to go next

`04_namespace_bundles` moves a whole namespace at once — export, import, and the
round-trip guarantee. See `README.md` in this directory for the full learning path and
the contract every example here satisfies.
