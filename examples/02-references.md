# 02 — References between entries

```bash
python examples/02_references.py
```

It prints five lines and exits 0. It also runs as part of `pytest tests/`.

Example `01` built entries one at a time. This one wires two together: a team whose
entry point *points at* an agent entry instead of carrying a copy of it. Everything
here happens inside a single namespace.

## The marker

A reference is a dict standing where a nested model would otherwise be written:

```yaml
entry_point:
  card:
    __ref__: lead-agent
    __type__: akgentic.core.AgentCard
```

Three keys, and only these three:

- **`__ref__`** — the target's `id`. Required; it is what makes the dict a marker.
- **`__type__`** — optional. The target's declared `model_type`, pinned. If the target
  is not that class, the write fails.
- **`__namespace__`** — optional. Addresses an entry in a *different* namespace, which
  is a topic of its own and not this example's.

Note where the marker sits: at `card`, not at `entry_point`. `entry_point` is a
`TeamCardMember` — a card plus a headcount plus subordinates — and only its `card` field
is an `AgentCard`. A marker replaces the value of exactly the field whose type matches
the target.

**A marker is a pure pointer, and therefore a leaf.** It has no interior: every payload
walker in the package — the resolver, the write-path reconciler, the unknown-key check,
the delete guard's scanner — stops at a dict carrying `__ref__` rather than descending
into it. That agreement is not a coincidence; it is the property the whole design turns
on, and the rule below that forbids extra keys is what makes it true.

## Reading: two phases

`resolve` (and `resolve_by_id`, and `load_team`) is two steps stacked:

1. **`populate_refs`** walks the payload and replaces every marker with a *typed*
   instance built from the target entry's own `model_type` — recursing into the target's
   payload, so a chain of refs resolves in one pass.
2. **`model_validate`** then builds the entry's own model from that populated tree.

Because step 1 produces a real instance of the target's class, a polymorphic field
(`list[ToolCard]`, `list[type]`, …) validates against the concrete subclass without any
per-payload workaround.

Use the typed door where one exists: `catalog.load_team(namespace)` is declared
`-> TeamCard`, while `resolve_by_id` returns a bare `BaseModel` that you have to narrow
yourself.

## Writing: the marker survives

Resolution happens on the way *out*. What is written down stays a pointer — the script
asserts that a fresh `get` still shows `{"__ref__": "lead-agent", "__type__": ...}` at
`entry_point.card`, not a flattened copy of the agent.

That is the whole benefit: editing `lead-agent` changes every entry pointing at it, and
nothing goes stale. It also means the stored payload re-resolves to the same in-memory
shape every time, because the write path re-applies the author's markers over the tree
Pydantic dumped.

## The four refusals

The script exercises each one and asserts the message.

**1. A `__type__` that disagrees with the target.**

```
Ref 'lead-agent' expected akgentic.llm.PromptTemplate, got akgentic.core.AgentCard
```

`__type__` is optional, and this is what declaring it buys: the ref fails where it is
*written* rather than where it is eventually *used*, which may be a deployment away.

**2. Any other key beside `__ref__`.**

```
ref marker to 'lead-agent' carries key 'description' — a ref marker is a pure pointer
and takes no other keys. Inline the payload and reference shared values via a
NativeValue entry.
```

A marker used to accept siblings and shallow-merge them onto the target before
validation. That interior is exactly what made the package's walkers disagree about
whether to descend into a marker, and the disagreement cost real defects. So the rule is
a hard error, not a lint.

The consequence in practice: **a consumer that needs to vary something inlines its own
payload and refs only the shared part.** Sharing a bare value — a prompt body, a role
string, a number — goes through a `NativeValue` entry, which resolves to the value
itself at the splice site. Example `03` covers that.

**3. A reference cycle.**

```
Reference cycle detected at (<namespace>, agent-b)
```

The timing here is the lesson, and the script shows it honestly. Two agents are made to
point at each other through their `metadata` dicts, and **the write that closes the loop
is accepted** — each write resolves against the *stored* state, and at that moment the
loop does not exist yet. The cycle surfaces on the next resolve. An example that implied
the catalog refuses the closing write would be teaching something false.

**4. Deleting something that is referenced.**

```
Entry 'research-team' (kind=team) in namespace '<ns>' references 'lead-agent'
```

One message per referrer. The question can also be asked *before* the attempt:
`catalog.find_references(namespace, target_id)` returns the referring entries, so a
caller can show them rather than discovering them in an exception. The legal order is
the obvious one — remove the ref (or delete the referrer) first, then delete the target,
which the script does by inlining a card again.

## Where to go next

`03_sharing_values` covers `NativeValue`, the sanctioned way to share a bare value
between entries — the idiom refusal 2 above points at. See `README.md` in this directory
for the full learning path and the contract every example here satisfies.
