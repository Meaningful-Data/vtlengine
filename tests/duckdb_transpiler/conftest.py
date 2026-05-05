"""
Pytest configuration for duckdb_transpiler tests.

Provides a timeout mechanism to skip slow tests.
"""

import os

import pytest

_skip_reason = "DuckDB transpiler tests require VTL_ENGINE_BACKEND=duckdb"
_should_skip = os.environ.get("VTL_ENGINE_BACKEND", "duckdb") != "duckdb"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip all duckdb_transpiler tests when VTL_ENGINE_BACKEND is not duckdb."""
    if not _should_skip:
        return
    skip_marker = pytest.mark.skip(reason=_skip_reason)
    for item in items:
        if "duckdb_transpiler" in str(item.fspath):
            item.add_marker(skip_marker)
