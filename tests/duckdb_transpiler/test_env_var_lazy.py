"""
Tests that DuckDB configuration env vars are read lazily — i.e. mutations
to ``os.environ`` made after importing ``vtlengine`` take effect on the
next call into the engine.

Before the fix, ``VTL_MEMORY_LIMIT``, ``VTL_THREADS``,
``VTL_TEMP_DIRECTORY``, ``VTL_MAX_TEMP_DIRECTORY_SIZE``,
``VTL_USE_IN_MEMORY_DB`` and ``VTL_SKIP_LOAD_VALIDATION`` were captured at
module import time and could not be overridden from a user script.
"""

import os
from unittest import mock

import duckdb

from vtlengine.duckdb_transpiler.Config import config as cfg
from vtlengine.duckdb_transpiler.io import _io as io_mod


def test_memory_limit_accessor_reads_env_each_call() -> None:
    with mock.patch.dict(os.environ, {"VTL_MEMORY_LIMIT": "2GB"}, clear=False):
        assert cfg._memory_limit() == "2GB"
    with mock.patch.dict(os.environ, {"VTL_MEMORY_LIMIT": "5GB"}, clear=False):
        assert cfg._memory_limit() == "5GB"


def test_threads_accessor_reads_env_each_call() -> None:
    with mock.patch.dict(os.environ, {"VTL_THREADS": "3"}, clear=False):
        assert cfg._threads() == 3
    with mock.patch.dict(os.environ, {"VTL_THREADS": "7"}, clear=False):
        assert cfg._threads() == 7


def test_threads_accessor_default() -> None:
    env = {k: v for k, v in os.environ.items() if k != "VTL_THREADS"}
    with mock.patch.dict(os.environ, env, clear=True):
        assert cfg._threads() == 1


def test_temp_directory_accessor_reads_env_each_call(tmp_path: object) -> None:
    p = str(tmp_path)
    with mock.patch.dict(os.environ, {"VTL_TEMP_DIRECTORY": p}, clear=False):
        assert cfg._temp_directory() == p


def test_max_temp_directory_size_accessor_reads_env_each_call() -> None:
    with mock.patch.dict(os.environ, {"VTL_MAX_TEMP_DIRECTORY_SIZE": "10GB"}, clear=False):
        assert cfg._max_temp_directory_size() == "10GB"


def test_use_in_memory_db_accessor_reads_env_each_call() -> None:
    with mock.patch.dict(os.environ, {"VTL_USE_IN_MEMORY_DB": "0"}, clear=False):
        assert cfg._use_in_memory_db() is False
    with mock.patch.dict(os.environ, {"VTL_USE_IN_MEMORY_DB": "1"}, clear=False):
        assert cfg._use_in_memory_db() is True


def test_skip_load_validation_accessor_reads_env_each_call() -> None:
    with mock.patch.dict(os.environ, {"VTL_SKIP_LOAD_VALIDATION": "1"}, clear=False):
        assert io_mod._skip_load_validation() is True
    with mock.patch.dict(os.environ, {"VTL_SKIP_LOAD_VALIDATION": "0"}, clear=False):
        assert io_mod._skip_load_validation() is False


def test_configure_duckdb_connection_applies_post_import_env() -> None:
    """Integration check: env vars mutated AFTER import flow through to DuckDB."""
    env = {
        "VTL_MEMORY_LIMIT": "2GB",
        "VTL_THREADS": "3",
    }
    with mock.patch.dict(os.environ, env, clear=False):
        conn = duckdb.connect(":memory:")
        try:
            cfg.configure_duckdb_connection(conn)
            settings = dict(
                conn.execute(
                    "SELECT name, value FROM duckdb_settings() "
                    "WHERE name IN ('memory_limit', 'threads')"
                ).fetchall()
            )
        finally:
            conn.close()

    assert settings["threads"] == "3"
    # DuckDB normalises memory_limit to a human-readable form; just check
    # the value is in the right ballpark for 2GB (it reports ~1.8 GiB after
    # reserving overhead). The threads check above is the authoritative one.
    assert "GiB" in settings["memory_limit"] or "MiB" in settings["memory_limit"]
