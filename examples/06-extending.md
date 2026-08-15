# 06 — Your own models in the catalog

```bash
python examples/06_extending.py
```

It prints five lines and exits 0. It also runs as part of `pytest tests/`.

Every `model_type` in examples `00`–`05` began with `akgentic.`. This one stores a
deployment's own Pydantic models — a plain configuration model and a custom `ToolCard` —
under `acme.core.models.`, and puts the policy back when it is done.

`acme` is a placeholder. Substitute your own package name everywhere below.

## Why there is an allowlist at all

`model_type` is a dotted class path, and resolving an entry **imports the module it
names**. That is not a lookup in a registry the framework controls; it is an import,
executing whatever that module executes at import time.

So the set of paths an entry may name is a **security boundary**, not an ergonomics
setting. A catalog entry is data — it may arrive over HTTP, out of a bundle file, from a
clone of somebody else's namespace — and without a gate, "data" would mean "a list of
modules to import".

`akgentic.` is always allowed and is never removable. Everything else is opt-in.

Three properties follow, and all three matter operationally:

- **Startup-only.** Set the policy before the first `Entry` is constructed or resolved.
  It is cached in a module global; changing it mid-flight means two entries in one process
  were validated against different policies.
- **Process-wide.** Not per request, not per tenant, not per namespace. Every `Entry` in
  the process is checked against the same tuple.
- **Never reachable from the HTTP surface.** No route, no request body, and no catalog
  entry can change it. The policy comes from the trusted environment or from deployment
  wiring code — never from catalog data.

## The two supply channels

```bash
export AKGENTIC_CATALOG_MODEL_TYPE_PREFIXES=acme.core.models.
export AKGENTIC_CATALOG_MODEL_TYPE_PREFIXES='["acme.core.models."]'   # equivalent
```

```python
from akgentic.catalog import set_allowed_prefixes

set_allowed_prefixes(["acme.core.models."])   # startup wiring; what this example uses
```

They are equivalent in effect, with one difference worth knowing: `set_allowed_prefixes`
replaces the cache **and stops the environment variable being consulted afterwards**. That
is what makes the example's assertion deterministic on any machine:

```python
assert allowed_prefixes() == ("akgentic.", "acme.core.models.")
```

**Give every process the same value.** Server, worker, CLI, init container — all of them.
A worker with a narrower policy than the server will refuse to resolve an entry the server
happily accepted, and the failure will surface far from its cause.

### Prefer the narrowest prefix

`acme.core.models.`, not `acme.`. Every module under an allowed prefix becomes something a
catalog entry can cause to be imported, so **a prefix is a blast radius**, not a gate. Widen
it by exactly what you need to name and no more.

There is deliberately no denylist of dangerous module roots. Shape validation only: each
dot-separated segment must be a Python identifier. An operator who writes `os.` has
authorised `os.`, and the framework takes them at their word — which is the other half of
the reason to keep the prefix tight.

## Widening changes what may be *named*, never what may be *resolved*

This is the part that most often gets misread. The prefix check is one of **three** checks
that run before a class is used, and the other two do not move:

1. the path starts with an allowed prefix;
2. the resolved object is a `pydantic.BaseModel` subclass;
3. the class declares no field named `__ref__`, `__type__` or `__namespace__` — those are
   the resolver's reserved sentinels, and a model that declared one would be ambiguous.

Checks 2 and 3 run for every path that passes check 1. Adding `acme.core.models.` to the
policy does not let an entry name `acme.core.models.some_function`, or a dataclass, or a
model with a `__ref__` field. It lets it name *your Pydantic models*, and nothing else.

The two enforcement points differ only in which exception they raise, because they fire in
different places:

| Where | When | Raises |
|---|---|---|
| `Entry.model_type` annotation validator | `Entry(...)` construction, before any import | Pydantic's `ValidationError` |
| `resolver.load_model_type` | resolve time, before `import_class` | `CatalogValidationError` |

The example hits the first one, twice — once before widening and once after restoring —
which is why it catches `ValidationError` and not `CatalogValidationError`.

## Making the module importable

```python
module = types.ModuleType("acme.core.models")
for cls in (EngagementProfile, ArchiveSearchTool):
    cls.__module__ = "acme.core.models"
    setattr(module, cls.__name__, cls)
sys.modules["acme.core.models"] = module
```

Both halves are needed, and for different reasons:

- `import_class` resolves `acme.core.models.X` out of `sys.modules`, so without the module
  the resolver simply cannot find the class.
- `serialize_type` reads `__module__` to stamp `__model__` on every dump, so without the
  reassignment the FQCN written into a payload would read `__main__.ArchiveSearchTool`
  standalone and `akgentic_catalog_example_06_extending.ArchiveSearchTool` under pytest.

**None of this exists in a real deployment.** There, `acme-core` is an installed package,
`import acme.core.models` works, and the classes report their own module. The scaffolding
is here only because an example cannot ship a second distribution.

## A customer model, stored and resolved

The shape is the README's, and the pieces are the ones examples `01`–`03` introduced: a
`_meta` entry anchoring the namespace, a `NativeValue` entry holding one number, and an
entry whose `model_type` is the customer's class pulling that number in through a marker.

```python
catalog.create(Entry(
    id="case-ingestion",
    kind="model",
    namespace="acme-prod",
    user_id="u1",
    model_type="acme.core.models.EngagementProfile",
    payload={"source": "sftp", "batch_size": {"__ref__": "default-batch-size"}},
))

resolved = catalog.resolve_by_id("acme-prod", "case-ingestion")
resolved.batch_size    # 200 — an EngagementProfile, with the ref spliced in
```

Three things are asserted, and none of them restates an input. What came back is an
`EngagementProfile` — the catalog imported and built *the customer's class*, not a dict.
`batch_size` holds a value written into a **different entry**, and because the field is
declared `int` the `{"value": 200}` wrapper could never have been assigned to it. And what
is written down is still the marker: resolution happens on the way out, which is what keeps
the two entries independent.

Note the anchor is a `kind="meta"` entry. A meta **is** an anchor — it skips the
initialisation check exactly as a team entry does — so a namespace of plain configuration
models needs no team.

## A custom `ToolCard`, and the three rules it obeys

A `ToolCard` is configuration *and* a callable factory in one class. It has to round-trip
through Pydantic serialisation cleanly, because it is stored as a payload and rebuilt from
one. Three rules:

**Only serialisable fields.** Plain `str` / `int` / `bool` / `list[str]`, other
`BaseModel`s, enums, containers of those. A field holding a live connection is not
configuration and does not belong in the model.

**No `model_config = ConfigDict(arbitrary_types_allowed=True)` on the subclass.** `ToolCard`
already inherits it from `SerializableBaseModel`, so grepping for it and finding it on the
base is not a contradiction — the rule forbids a *subclass* adding it. Reaching for it is
the signal that a non-serialisable type has leaked into a field, and the config silences
the complaint instead of fixing the leak.

**Runtime state goes in a `PrivateAttr`.** `ToolCard` does this itself, for its
`_observer_ref`:

```python
class ArchiveSearchTool(ToolCard):
    endpoint: str
    max_results: int = 20
    include_attachments: bool = False
    regions: list[str] = Field(default_factory=list)

    _client: _SearchClient | None = PrivateAttr(default=None)

    def get_tools(self) -> list[Callable[..., Any]]:
        return []
```

`get_tools` is abstract on `ToolCard`. A subclass that does not implement it cannot be
instantiated at all — which is a better failure than one discovered when an agent asks the
card for its tools.

### The round trip drops runtime state, and that is the assertion

```python
card.connect(_SearchClient())          # attach something live
dumped = card.model_dump()

assert set(dumped) == {"__model__", "endpoint", "max_results",
                       "include_attachments", "regions"}
assert dumped["__model__"] == "acme.core.models.ArchiveSearchTool"

# ... store, resolve ...
assert restored.client is None         # back at its default
assert restored.regions == card.regions
```

The dump is checked against a **closed** key set rather than against "`_client` is absent".
Promoting the runtime attribute to a Pydantic field — the exact change the rule exists to
prevent — would rename it, and a check for one spelling would sail straight past the
regression. A closed set cannot.

`__model__` is what makes the round trip work at all: it carries the concrete FQCN, which
is how a polymorphic field recovers `ArchiveSearchTool` rather than some base class. The
catalog's unknown-key check exempts that key by name, so a payload pasted straight out of a
dump is accepted rather than reported as a misprint.

## Putting the policy back — and why you will need to do the same

The widening and the `sys.modules` registration are undone in a `finally`, so a failure
mid-walkthrough still leaves the process as it was found:

```python
finally:
    reset_allowed_prefixes()
    sys.modules.pop("acme.core.models", None)
```

**Know what `reset_allowed_prefixes` does.** It returns the module to its *unresolved*
state, so the next read consults `AKGENTIC_CATALOG_MODEL_TYPE_PREFIXES` again. It does
**not** restore a previously-set tuple, and there is no API that does. If you need the old
value back, you have to have kept it.

The restoration is then asserted behaviourally rather than assumed: `acme.core.models.` is
gone from `allowed_prefixes()`, and the very construction that opened the example is
refused again with the same message.

**This is a rule for your own tests, not housekeeping for this file.** The policy is a
module-level global. The example harness runs every example in one pytest process, and the
autouse fixture that isolates the policy covers `tests/v2/` only. A test that widens the
allowlist and does not reset it will not fail — it will make some *later*, unrelated test
pass or fail for reasons that have nothing to do with what it is testing, which is the
worst kind of failure to track down. Reset it in a fixture, in a `finally`, or both.

## Where to go next

This is the last example. See `README.md` in this directory for the full learning path and
the contract every example here satisfies, and the package README for the CLI and REST
surfaces that sit on top of everything shown here.
