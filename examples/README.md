# akgentic-catalog examples

Runnable walkthroughs of the catalog API. Each example is a pair — a script and the
narrative beside it:

```
NN_snake_name.py     the code, runnable on its own
NN-kebab-name.md     what it teaches and why
```

**The test suite executes every script in this directory.** `tests/examples/` discovers
them by glob, calls each `main()`, and fails the build when one of them fails. CI runs
that suite on every push. This is the whole point of the directory: the previous
generation of examples was deleted after two epics of quietly raising `ImportError`,
because nothing ever ran it. An example that no longer works is now a red build on the
day the API changes, not a discovery months later.

## Running them

```bash
python examples/00_hello_catalog.py     # any single example, standalone
pytest tests/examples/ -v               # all of them, the way CI does
```

No network, no API key, no LLM, no environment variable to set. Nothing needs Docker.

## The learning path

| Example | Teaches | Status |
|---|---|---|
| `00_hello_catalog` | Mint a namespace, read it back, meet the misprint guard, delete | **available** |
| `01_first_entry` | The full CRUD surface, the `(kind, namespace, id)` key, the `model_type` allowlist | planned — story 35.3 |
| `02_references` | `__ref__` / `__type__`, resolution, cycles, the delete guard | planned — story 35.3 |
| `03_sharing_values` | `NativeValue` as the sanctioned way to share a bare scalar | planned — story 35.4 |
| `04_namespace_bundles` | Export / import round-trip, bundle headers, dry-run validation | planned — story 35.4 |
| `05_backends` | One walkthrough against YAML, Mongo and Postgres alike | planned — story 35.5 |
| `06_extending` | Custom models outside the `akgentic.` prefix, custom `ToolCard`s | planned — story 35.5 |

Rows marked *planned* have no file yet, and this table deliberately carries no link to
one. A README promising files that do not exist is the same rot in a different place.

## The contract every example satisfies

1. **A module-level `main() -> None`** taking no required arguments. The harness calls
   it; a discovered script without a callable `main` fails the build by name.

2. **It asserts its own outcomes.** Every step whose result matters is checked with an
   `assert` or an explicit `raise`. An example that silently produces the wrong answer
   must fail, not print. Narrative `print()` calls are welcome *alongside* the
   assertions, never instead of them.

3. **All storage goes to a `tempfile.TemporaryDirectory()`** created inside `main()`. No
   example writes to `data/`, to the package tree, to the working directory, or to any
   fixed path — so editing demo data can never redden CI, and running an example twice
   behaves the same as running it once.

4. **It runs standalone.** `python examples/NN_name.py` exits 0, via an
   `if __name__ == "__main__": main()` guard. That is why an example never calls
   `pytest.skip` or `pytest.raises`: it has to work with no pytest in sight.

5. **No network, no API key, no LLM, no required environment variable.** The catalog is
   a persistence and validation layer; the examples exercise it without a model
   provider.

6. **It is self-contained.** A reader can copy one `.py` file and run it. Shared
   assertion helpers factored into a support module would save a few lines and defeat
   the purpose.

7. **It passes `ruff` and `mypy --strict`.** CI runs both over this directory, so an
   example is held to the same static standard as `src/`. That is deliberate: the
   harness catches a break on the path an example executes, and mypy catches one it
   never reaches.

## Examples that need an optional dependency

An example that needs a package outside the base install declares it as a module-level
tuple of importable module names:

```python
REQUIRES: tuple[str, ...] = ("pymongo",)
```

The harness calls `pytest.importorskip(name)` for each entry **before** invoking
`main()`, so the example itself stays free of pytest imports (contract item 4). Declare
`REQUIRES: tuple[str, ...] = ()` when there is nothing to require — `00_hello_catalog`
does, which keeps the mechanism exercised on the happy path.

The harness reads `REQUIRES` off the imported module, so **import the optional package
inside `main()`**, not at the top of the file. A module-level `import pymongo` fails
while the module is still loading, before the declaration can be read, and the developer
without that package gets a red test instead of a skip.

`REQUIRES` covers *importability*, not a running service. The existing test suite already
solves the service half, and examples should reuse that machinery rather than invent a
second one:

- **Mongo** — `mongomock` provides an in-memory `MongoClient`, ships in the `dev` extra,
  and needs no Docker (`tests/v2/conftest.py`, the `entries_collection` fixture).
  `pytest.importorskip("pymongo")` guards the real driver.
- **Postgres** — the session-scoped `postgres_dsn` fixture in `tests/conftest.py` skips
  cleanly when `nagra` / `psycopg` are missing, and otherwise takes
  `DB_CONN_STRING_PERSISTENCE` if set or starts a testcontainer if Docker is available.

A developer with no Docker and no database still gets a fully green run.

## Adding an example

Drop `NN_name.py` and `NN-name.md` into this directory. That is the entire procedure —
the harness globs `[0-9][0-9]_*.py`, so a new example is picked up with no edit to any
test file, and a guard test fails the build if the `.md` half is missing. Add a row to
the table above and move it out of *planned*.

The number prefix is not cosmetic. A guard test fails the build for any `.py` here that
the glob would miss — `7_probe.py` or `demo.py` would sit in the directory and never
run, which is the rot this harness exists to prevent. The one exemption is a leading
underscore (`_something.py`), reserved for tooling files; it is **not** a licence for the
shared helper module contract item 6 rules out.
