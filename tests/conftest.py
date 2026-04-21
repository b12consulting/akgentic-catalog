"""Shared test configuration for akgentic-catalog.

Intentionally minimal after Epic 19 removed v1 models, repositories, and
services. The v2 test surface lives under ``tests/v2/`` and defines its own
fixtures in ``tests/v2/conftest.py``. Model-layer and top-level tests rely
on pytest's built-in fixtures only.
"""

from __future__ import annotations
