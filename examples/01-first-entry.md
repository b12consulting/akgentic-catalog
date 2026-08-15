# 01 — The first entry

```bash
python examples/01_first_entry.py
```

It prints eight lines and exits 0. It also runs as part of `pytest tests/`.

## What this adds beyond `00`

`00_hello_catalog` already showed a namespace being minted, a payload round-tripping,
a misprinted key refused and a deleted entry staying gone. This example recaps the
minting in a single asserted line — it needs a namespace to work in — and spends the
rest on what `00` never reaches: **how an entry is addressed**, **why `model_type` is
constrained**, **what a namespace needs before it will accept anything**, and the rest
of the CRUD surface (`update`, listing, `delete`). Its rejected write is deliberately a
different one: nested rather than flat, so you can see the report *locate* the misprint.

## What an entry is

Three parts, and it is worth keeping them separate in your head:

- **the envelope** — `id`, `kind`, `namespace`, `user_id`, `description`. This is what
  the catalog itself understands and queries.
- **`model_type`** — the dotted path of the Pydantic class that gives the payload its
  shape. `akgentic.core.AgentCard`, `akgentic.team.models.TeamCard`, and so on.
- **the payload** — a plain dict, validated against that class on every write. Nothing
  the class does not accept is stored.

The catalog is not a document store with a schema bolted on the side. The payload is
*only* ever meaningful through `model_type`, which is why that field is not free-form.

## The key: what actually scopes what

An entry is addressed by `kind`, `namespace` and `id` — but those three do not play the
same role, and reading "compound key" as "ids are unique per `(namespace, kind)`" is the
mistake this example exists to prevent. Four facts, each asserted in the script:

1. **`kind` selects the entry's semantics and its partition.** The YAML backend files
   entries at `<root>/<namespace>/<kind>/<id>.yaml`, and `kind` is a first-class filter
   on `EntryQuery`.
2. **An id is unique per *namespace*, not per `(namespace, kind)`.** Creating an `agent`
   and then a `model` under the same id in one namespace is refused with
   `Entry (<namespace>, <id>) already exists`.
3. **`catalog.get(namespace, id)` takes no `kind`.** One id, one entry — whatever kind
   it turns out to be. That signature is the same fact stated from the read side.
4. **The namespace is the boundary.** The same id in a different namespace is a
   different entry, and both are fine.

So: `kind` addresses and filters; the *namespace alone* scopes uniqueness.

## Why `model_type` goes through an allowlist

`model_type` is a class the catalog will import in order to validate a payload. An entry
is therefore not just data — it is a name of something that will be loaded into the
process that reads it. The allowlist is the answer to "what may a catalog entry cause to
be imported": a blast radius, not a gate.

Two properties are worth noticing in the script:

- The check runs at **`Entry(...)` construction** and raises `pydantic.ValidationError`
  with `outside allowlist` in the message. No `Catalog`, no repository, no import — a
  disallowed path never becomes an `Entry` in the first place.
- `akgentic.` is always allowed. Widening the set is a deployment decision, and example
  `06` is where that is shown.

## The anchor rule

A namespace will not accept an ordinary entry until something anchors it — a `team`
entry or a `_meta` entry. Write a lone agent into an empty namespace and you get:

```
Namespace 'unanchored' has no team entry and no meta entry
— create at least one anchor entry first (team OR meta)
```

This is the reason the team entries in `00` and `01` carry their `entry_point` card
*inline*: an anchor bootstraps its namespace, so at the moment it is written there is
nothing in that namespace to point at. Example `02` is where the pointing starts.

## What the rejected write guarantees

`00` misprints a top-level key. This one misprints a key one level down, inside
`config`, and the message locates it:

```
unknown key 'config.temperatur' — not a field of akgentic.core.AgentCard
```

Two details make this useful rather than merely strict. The path is dot-joined (list
indices render as `[i]`), so on a large payload you are told *where* rather than only
*that*. And the model named is the entry's **declared** `model_type`, not the nested
class that owns the field — the check compares the tree you wrote against the tree
Pydantic accepted, and that comparison knows paths, not owners.

The other half of the guarantee is the part that keeps it from being a nuisance:
**leaving a key out is not a misprint.** The script writes the same payload without
`temperatur` and it is accepted with no findings. The rule keys on what the author
*wrote*, so omitting a declared field remains the ordinary way to let it take its
default — `squad_id` is absent from what is stored and comes back as `None` when the
entry is resolved.

## Deriving an update

`update` re-runs the whole write pipeline on the entry it is handed, so the entry you
pass has to be complete. Derive it from the stored one:

```python
stored = catalog.get(namespace, "lead-agent")
catalog.update(stored.model_copy(update={"description": "..."}))
```

Not by rebuilding it field by field. A hand-written reconstruction is correct on the day
it is written and silently drops whatever field is added to `Entry` afterwards.

The script asserts the difference rather than asserting round the edge of it. The payload
keys it re-sends survive because it re-sent them; the *envelope* — `kind`, `user_id`,
`model_type` — survives because `model_copy` carried it, and the update never named it.
That is the half a field-by-field rebuild would lose.

## Where to go next

`02_references` replaces that inline card with a `__ref__` marker, and covers what
happens when entries start pointing at one another. See `README.md` in this directory
for the full learning path and the contract every example here satisfies.
