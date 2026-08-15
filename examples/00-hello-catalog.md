# 00 — Hello, catalog

The 30-second tour. One namespace is created, read back, protected from a typo, and
deleted — enough to see what the catalog is for before any of the interesting parts.

```bash
python examples/00_hello_catalog.py
```

It prints four lines and exits 0. It also runs as part of `pytest tests/`.

## What it demonstrates

A **catalog** is a `Catalog` service wrapped around an `EntryRepository`. Here the
repository is a directory of YAML files in a temporary directory; swapping it for Mongo
or Postgres changes nothing else in the script.

Everything the catalog stores is an `Entry`: an envelope carrying `(kind, namespace, id)`
plus a `model_type` — the dotted path of the Pydantic class that gives the `payload` its
shape. Nothing is stored that the named model does not accept.

A **team** entry is special: it anchors its namespace. Create one with
`namespace=UNSET_NAMESPACE` and the catalog mints a fresh UUID for it. Because it is the
anchor, its `entry_point` card is written inline — there is nothing to point at yet.
Example `01` starts pointing.

## The four guarantees it asserts

1. **The namespace is minted, not echoed.** Creating the team entry with the
   `UNSET_NAMESPACE` sentinel returns an entry whose namespace is a fresh, non-empty
   UUID.
2. **The payload round-trips.** Reading the entry back yields the same name, the same
   agent description and skills, and the same role — through Pydantic and through YAML,
   unchanged.
3. **A misprint is an error, not a silent drop.** Writing an agent payload with `skils`
   instead of `skills` raises `CatalogValidationError`, and the message names the
   offending key. This is the guarantee most worth knowing about: a bare Pydantic model
   would have ignored the key, and the author would never have learned the skills list
   was thrown away.
4. **A deleted entry is gone.** `get` raises `EntryNotFoundError` afterwards; it does not
   return `None`. Every read path in the catalog raises rather than handing back a
   sentinel you might forget to check.

## Where to go next

`01_first_entry` covers the rest of the CRUD surface and the `model_type` prefix
allowlist. See `README.md` in this directory for the full learning path and the contract
every example here satisfies.
