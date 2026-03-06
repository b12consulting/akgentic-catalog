# akgentic-catalog

## Installation

### Workspace Installation (Recommended)

This package is designed for use within the Akgentic monorepo workspace:

```bash
# From workspace root
git clone git@github.com:b12consulting/akgentic-quick-start.git
cd akgentic-quick-start

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install all workspace packages
uv sync --all-packages --all-extras
```

All dependencies (`akgentic-core`, `akgentic-llm`, `akgentic-tool`) are automatically
resolved via workspace configuration.

## Architecture

## Features

## Dependencies

- `pydantic` — Data validation and configuration models
- `akgentic` (`akgentic-core`) — Actor system, messaging, orchestration
- `akgentic-tool` — Tool abstractions: `TeamTool`, `PlanningTool`, `ToolCard`

## Quick Example

## Documentation

## Examples

See the [examples/](examples/) directory for usage patterns (coming soon).
