# 05 — One protocol, three backends

```bash
python examples/05_backends.py
```

It prints one line per backend it could not run, then two lines, and exits 0. It also runs
as part of `pytest tests/`.

Examples `00`–`04` all used `YamlEntryRepository`. The package ships three backends —
YAML, MongoDB, PostgreSQL — behind a single `EntryRepository` protocol, and the README
says that *which database is behind it is a deployment choice rather than a change to any
calling code*. This example is that claim, executable.

**One walkthrough function, run unchanged against every backend present, with its
observations compared for equality.** The identity is the whole lesson. A walkthrough
rewritten per backend would show only that three pieces of code can each be made to pass.

## What `EntryRepository` is, and what it deliberately is not

It is a narrow data plane. Eight methods: `get`, `put`, `delete`, `list`,
`list_by_namespace`, `get_by_kind`, `find_references`, `find_references_global`. Rows in,
rows out.

Everything that makes the catalog a catalog lives **above** it, in `Catalog`:

| Concern | Lives in |
|---|---|
| Ref resolution (`__ref__` markers, cycles, `NativeValue` unwrapping) | `Catalog` / `resolver` |
| Payload validation against `model_type` | `Catalog` / `resolver` |
| The delete guard | `Catalog` |
| Namespace anchoring and ownership | `Catalog` |
| Namespace minting | `Catalog` |
| Storing and fetching a row | the repository |

That division is what makes swapping the backend cheap, and the example demonstrates it
directly: the Postgres arm cleans up after itself through `repository.delete(...)`, which
removes a referenced entry without complaint, because the guard that would refuse is a
`Catalog` policy and the store underneath has never heard of it.

## Conformance is proved twice, in two different ways

`EntryRepository` is a **structural** `typing.Protocol` with no `@runtime_checkable`
decorator. `isinstance(repo, EntryRepository)` does not return `False` — it raises
`TypeError`. Never write it.

What proves conformance is the annotation:

```python
backends: list[tuple[str, EntryRepository]] = [(YAML, YamlEntryRepository(yaml_root))]
```

`mypy --strict` — which CI runs over this directory — checks that every repository handed
to that list has the right shape. The runtime assertions check that they have the same
**semantics**. Mypy proves the shape; the example proves the behaviour. Neither one alone
would be worth much.

## How each backend is obtained here, and how it is wired for real

### YAML — always

```python
YamlEntryRepository(Path(tmpdir))    # a Path, not a str
```

No probe and no skip path. If YAML is ever absent the example must fail, not degrade.

### Mongo — when `pymongo` and `mongomock` import

The example builds an in-memory `mongomock` client and hands the repository a collection
off it. That is the mechanism the package's own suite uses (`tests/v2/conftest.py`, the
`entries_collection` fixture): no Docker, no server, and CI has no Mongo service, so a
live-server requirement would mean this arm never ran anywhere.

**Say the trade out loud: the repository code exercised here is the real one, the server
is not.** A `mongomock` collection is standing in for a live one, and anything the driver
does differently against a real `mongod` is outside what this example can see.

In a deployment the stand-in is replaced by three lines, and `MongoCatalogConfig` is the
thing that produces the collection — the repository owns neither the client nor the
collection lifecycle:

```python
cfg = MongoCatalogConfig(
    connection_string="mongodb://localhost:27017",
    database="akgentic",
)
client = cfg.create_client()
collection = cfg.get_collection(client, cfg.catalog_entries_collection)
catalog = Catalog(MongoEntryRepository(collection))
```

### Postgres — when the environment names a reachable database

Three conditions, all of them: `nagra` and `psycopg` import, `DB_CONN_STRING_PERSISTENCE`
is set, and `init_db` against that DSN succeeds. The env var is the supply channel the
README already documents and the one CI sets; it is **optional**, so the example still
honours the "no required environment variable" contract that every example here satisfies.

```python
init_db(PostgresCatalogConfig(connection_string=dsn))   # idempotent; creates the table
catalog = Catalog(PostgresEntryRepository(dsn))         # a bare DSN string, not a config
```

Two shapes worth memorising, because both are commonly got wrong: the repository takes a
**bare DSN string**, and the constructor never creates the schema. `PostgresCatalogConfig`
exists to validate the DSN and to feed `init_db`.

To exercise this arm locally, export `DB_CONN_STRING_PERSISTENCE` at a reachable database
before running. Without it the arm prints its reason and the example still exits 0.

### The probe may be forgiving; the walkthrough may not

A broad `except Exception` wraps the Postgres availability probe, and only that. An
unreachable database is an arm that does not run. The walkthrough itself is never wrapped
— a genuine contract break has to stay red.

## What the walkthrough asserts

One namespace, built the same way on every backend: a `_meta` anchor, one `NativeValue`
entry holding a number, and two agents whose payloads point at it. Then, per backend:

- the number the resolver spliced in, read back off an agent's `metadata`;
- the ids `find_references` returned — compared as a **set**, not a length;
- the blocker messages the refused `delete` raised, one per referrer;
- the ids present in the namespace afterwards.

The marker sits **three levels below the payload root**, at
`payload.metadata.limits.batch_size`. A marker at the root would be found by a walker that
never recursed, and the recursion is precisely the part all three backends share.

### Two halves of `find_references`, and only one of them is under test

Mongo and Postgres both import `_payload_has_ref` — the walker the **YAML** backend
defines. There is no parallel implementation and no JSONB containment query.

So the walker half is parity *by construction*: three backends calling one function cannot
disagree. What differs is the **fetch** underneath it — a directory scan, a Mongo `find`,
a `SELECT` — and whether the rows that come back out of each one are the same rows that
went in. That is what this example actually tests, and it is the half where a divergence
could plausibly hide.

### Order is never asserted

`PostgresEntryRepository.list` emits no `ORDER BY`. Every collection field on the
observations model is a **sorted tuple**, and comparisons on sets. Asserting a list order
would make this example red for a reason that has nothing to do with the contract — the
same mistake the package's own audit found in five existing tests.

### One known divergence, deliberately not exercised

`EntryQuery(description_contains=...)` is **not** in the walkthrough. It is
case-**in**sensitive on Postgres (an `ILIKE`) and case-**sensitive** on YAML and Mongo (a
plain `in`). Same user-facing flag, two semantics.

Asserting parity over it would turn this example red for a live divergence — a real bug,
but one that needs a source fix in its own story, and an example is not the place to
smuggle one in.

So read the parity claim for exactly what it covers: **the operations this example
exercises**. It is not a blanket guarantee that every method of every backend behaves
identically, and `description_contains` is the counter-example that proves it needs saying.

## The parity assertion, and where it is honestly weak

```python
divergent = {label: report for label, report in reports.items() if report != baseline}
assert not divergent, divergent
```

Whole-model equality on the observations, not a field-by-field walk — a field added to the
model later is compared without anyone having to remember to add it here.

But an equality on its own is a weak claim: three identical *empty* reports satisfy one
just as happily as three correct ones. So the YAML report is pinned down first — the value
really came through the marker, both referrers really were found, the delete really was
refused naming each of them, and the namespace holds the ids it should.

**And there is a limit worth stating rather than hiding.** On a machine with neither
optional backend, the equality compares one report with itself and proves nothing there.
That is not a flaw in the assertion; it is the honest consequence of degrading per backend
instead of skipping. It is also why CI matters: the pipeline installs
`[dev,api,cli,mongo,postgres]` and runs a `postgres:16-alpine` service, so **all three arms
execute on every push** and the comparison has three reports to work with.

The example prints which backends it ran, and one line per backend it skipped saying why.
Read that line before reading anything into a green run.

## Postgres leaves nothing behind

The Postgres arm shares one `catalog_entries` table with the rest of the test suite. It
therefore clears its own namespace before it starts — an earlier run may have been
interrupted — and again in a `finally` when it ends.

Both sweeps go through `repository.delete(...)`, never `catalog.delete(...)`, for the
reason given at the top: the delete guard would rightly refuse to remove a referenced
entry. And no `TRUNCATE` — the table is not this example's to empty.

## Choosing between the three

| | Reach for it when | The cost |
|---|---|---|
| **YAML** | The catalog is version-controlled beside the code that consumes it; a single process; local development; a shipped default namespace. | One file per entry on a filesystem. No concurrent writers, no server, no query engine. |
| **MongoDB** | Several processes share a catalog; the deployment already runs Mongo; entries are written at runtime by a UI or an API. | A server to operate. Payloads are documents, so a schema change is invisible until something resolves. |
| **PostgreSQL** | The deployment already runs Postgres; you want one durable store, transactions, and ordinary backups. | Requires `init_db` before first use. `list` returns rows unordered, and `description_contains` is case-insensitive here and nowhere else. |

None of these choices reaches any calling code. That is the point the example exists to
make: pick the one that matches the deployment, and the walkthrough above still passes.

## Where to go next

`06_extending` widens the `model_type` allowlist and stores a deployment's own Pydantic
models — including a custom `ToolCard`. See `README.md` in this directory for the full
learning path and the contract every example here satisfies.
