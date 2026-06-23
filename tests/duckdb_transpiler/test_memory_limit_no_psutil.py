"""
Memory-limit configuration no longer depends on psutil.

A percentage VTL_MEMORY_LIMIT (e.g. the default "80%") is left unset so DuckDB
applies its own default (80% of physical RAM); an absolute limit (e.g. "8GB")
is passed straight through to DuckDB, which parses the units itself.
"""

import os
from unittest import mock

import duckdb

from vtlengine.duckdb_transpiler.Config import config as cfg


def test_config_does_not_depend_on_psutil() -> None:
    # psutil was removed as a dependency; guard against reintroduction.
    assert not hasattr(cfg, "psutil")


def test_duckdb_memory_limit_defers_percentages() -> None:
    for pct in ("80%", "50%", " 80% "):
        with mock.patch.dict(os.environ, {"VTL_MEMORY_LIMIT": pct}, clear=False):
            assert cfg._duckdb_memory_limit() is None


def test_duckdb_memory_limit_passes_absolute_through() -> None:
    with mock.patch.dict(os.environ, {"VTL_MEMORY_LIMIT": "8GB"}, clear=False):
        assert cfg._duckdb_memory_limit() == "8GB"
    with mock.patch.dict(os.environ, {"VTL_MEMORY_LIMIT": "512MB"}, clear=False):
        assert cfg._duckdb_memory_limit() == "512MB"


def test_duckdb_memory_limit_bare_integer_gets_byte_unit() -> None:
    # A bare integer is a byte count; DuckDB needs an explicit unit ("B").
    with mock.patch.dict(os.environ, {"VTL_MEMORY_LIMIT": "1073741824"}, clear=False):
        assert cfg._duckdb_memory_limit() == "1073741824B"


def test_percentage_limit_leaves_duckdb_default() -> None:
    """A percentage limit must not override DuckDB's built-in memory_limit."""
    default_conn = duckdb.connect(":memory:")
    duckdb_default = default_conn.execute("SELECT current_setting('memory_limit')").fetchone()[0]
    default_conn.close()

    env = {k: v for k, v in os.environ.items() if k != "VTL_MEMORY_LIMIT"}
    with mock.patch.dict(os.environ, env, clear=True):  # no VTL_MEMORY_LIMIT -> default "80%"
        conn = duckdb.connect(":memory:")
        try:
            cfg.configure_duckdb_connection(conn)
            configured = conn.execute("SELECT current_setting('memory_limit')").fetchone()[0]
        finally:
            conn.close()

    assert configured == duckdb_default
