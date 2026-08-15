# 04 — A namespace as one document

```bash
python examples/04_namespace_bundles.py
```

It prints four lines and exits 0. It also runs as part of `pytest tests/`.

Examples `00`–`03` moved one entry at a time. A **bundle** is the unit in which a whole
namespace leaves and re-enters the catalog: one YAML document carrying every entry, plus
a header describing the namespace itself.

Everything here stays inside a single namespace.

## When to reach for a bundle

Single-entry CRUD is the right tool when you know which entry you are changing. A bundle
is the right tool when the *set* is the thing:

- handing a namespace to someone else, or to another environment;
- reviewing a namespace as a diff, in one file, instead of a dozen;
- editing several entries as one change, where a half-applied result would be worse than
  no change at all;
- keeping a namespace in version control beside the code that consumes it.

The trade is deliberate. CRUD gives you precision; a bundle gives you atomicity and a
single artefact to move around.

## The shape

```yaml
namespace: 3fa85f64-...      # the namespace's identifier
user_id: u1                  # every entry in a bundle shares one owner
name: Greeting Team          # ─┐
description: A one-agent...  #  │
properties:                  #  │ the header — projected from the `_meta` entry
  tier: demo                 #  │
shareable: false             #  │
public: true                 # ─┘
entries:

  #### ─── Teams ───────────────────────────────────────────── ####

  greeting-team:
    kind: team
    model_type: akgentic.team.models.TeamCard
    description: ''
    payload: {...}

  #### ─── Agents ──────────────────────────────────────────── ####

  greeter:
    kind: agent
    ...
```

`entries:` is a **mapping keyed by id**, grouped into commented sections by kind in
consumption order — teams, then agents, prompts, tools, models — and sorted by id within
each. `id`, `namespace` and `user_id` are not repeated inside each entry: they are implied
by the document.

**Both key sets are closed.** Eight keys at the root, four per entry, and anything else is
an error — see *Two misprints* below.

## The header, and the entry it comes from

The header is a projection of the namespace's `_meta` entry — a `kind="meta"` entry with
the canonical id `_meta`, at most one per namespace, carrying `name`, `description`, a
free-form `properties` map, and the `shareable` / `public` booleans.

**`_meta` never appears under `entries:`.** It is hoisted into the header on export and
rebuilt from the header on import. The script asserts both directions: the header fields
equal the stored `_meta` payload, `_meta` is absent from `entries:`, and after the import
the fresh catalog has a `_meta` entry again with the same payload.

One practical consequence: the header is emitted **only when something forces it** — a
non-empty `name`, `description` or `properties`, or a boolean set to `true`. A namespace
whose meta is entirely default exports the older three-key shape (`namespace`, `user_id`,
`entries`) and there is no header to read. If you want a header, give the namespace real
metadata.

## Import is an atomic replace

Not a merge. **Every entry absent from the bundle is deleted from the namespace.** A
bundle is a statement about what the namespace *is*, not a patch describing what to add,
so importing a partial bundle into a populated namespace removes the rest. The example
imports into an empty catalog, where this is invisible — which is exactly why it is worth
saying here.

Atomic in the other sense too: **every check runs before any write.** The document is
parsed, each entry is put through the full write pipeline, the bundle invariants are
checked, and the refs are checked — and only then does anything reach the repository. A
bundle that fails validation leaves the namespace byte-identical. The script proves that
directly: after a refused import and a failed dry run, it re-exports and compares against
the good document.

### Why bundle-internal refs resolve here

Create the agent in this example on its own, before the entry it points at exists, and the
write is refused — the ref target is not there. Import the same two entries as a bundle
and it resolves fine.

The bundle is staged into an overlay in front of the repository before validation, so a
ref to a sibling entry *declared in the same document* finds its target even though
neither has been written yet. That is what makes a bundle self-contained: a namespace can
be re-created from one document in any order, without a bootstrapping dance. The dry-run
validator uses the same overlay, so it agrees with the import rather than reporting
failures the import would not have.

## The round-trip guarantee

Export a namespace, import that document into a **different** catalog, export again from
there — and get the same document.

```python
document = catalog.export_namespace_yaml(namespace)
fresh.import_namespace_yaml(document)
assert fresh.export_namespace_yaml(namespace) == document
```

That is worth more than it looks. It means the document is the whole truth about the
namespace: nothing is dropped on the way out, nothing is invented on the way in, and no
information lives only in the repository's own storage layout. In practice it is what lets
you treat an exported bundle as a reviewable artefact — diff two exports and every
difference is a real difference.

**The example does not assert that equality on its own**, because an equality is a weak
claim: two empty documents are equal, and so are two identically broken ones. Before
comparing, it pins the document down — non-empty, carrying the ids the namespace holds,
and still carrying the `__ref__` marker intact. Then string identity means what it looks
like it means.

Note what survives: the marker, verbatim. The bundle stores pointers, not the values they
resolve to, so a round trip preserves the *structure* of the namespace and not merely its
rendered contents.

## Two misprints, refused together

Take the good document, add an unknown key at the root and an unknown key on one entry,
and import it:

```
bundle root has unknown key 'sharable' — expected one of: description, entries, name,
namespace, properties, public, shareable, user_id

entry 'greeter' has unknown key 'descriptin' — expected one of: description, kind,
model_type, payload
```

Both, in one pass. The script asserts on the error *list* rather than on the exception's
string, because "reported together" is the claim being made — a parser that stopped at the
first problem would satisfy a substring check across two separate attempts.

`sharable` is why the root key set is closed at all. It reads as correct. Under a tolerant
parser it would be ignored, `shareable` would stay `false`, and the namespace would
silently not be shareable — a misprint that costs an afternoon. Same story one level down:
`descriptin` would leave the entry's description quietly reset to `''`.

## Validating without writing

`validate_namespace_yaml(text)` answers *would this import cleanly?* and **never raises**.
The findings come back in a report, so a caller can render all of them at once:

```python
report = catalog.validate_namespace_yaml(bad_document)
report.ok            # False
report.namespace     # None — the document did not parse into a namespace
report.global_errors # both misprints
```

`report.ok` is the thing to assert. The script runs it twice — once over the bad document
and once over the good one, which comes back `ok is True` — because a validator only ever
seen failing has not been shown to discriminate.

## Where to go next

`05_backends` runs one walkthrough against YAML, Mongo and Postgres alike. See `README.md`
in this directory for the full learning path and the contract every example here
satisfies.
