# akgentic-catalog Examples

Progressive examples demonstrating the akgentic-catalog module capabilities.
Each example is a self-contained script paired with a companion `.md` file
explaining concepts.

## Running the Examples

From the **project root**:

```bash
uv run python packages/akgentic-catalog/examples/01_catalog_entries.py
```

From the **akgentic-catalog directory**:

```bash
cd packages/akgentic-catalog
uv run python examples/01_catalog_entries.py
```

## Available Examples

### [01_catalog_entries.py](01_catalog_entries.py) — Template & Tool Entry Basics

First contact with the catalog API: create, retrieve, and list `TemplateEntry`
and `ToolEntry` — the two leaf entity types with no cross-catalog dependencies.
Demonstrates placeholder auto-parsing, FQCN class resolution, and duplicate-ID
error handling.

Companion docs: [01-catalog-entries.md](01-catalog-entries.md)

## Prerequisites

All examples require:

- Python 3.12+
- `akgentic` (akgentic-core)
- `akgentic-tool`

Install with:

```bash
uv sync
```
