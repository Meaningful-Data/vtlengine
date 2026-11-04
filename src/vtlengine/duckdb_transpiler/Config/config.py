"""
DuckDB Transpiler Configuration.

Configuration values can be set via environment variables:
- VTL_MEMORY_LIMIT: Max memory for DuckDB as an absolute size (e.g. "8GB", "512MB").
  A percentage (e.g. "80%", the default) defers to DuckDB's own default of 80% of RAM.
- VTL_THREADS: Number of threads for DuckDB (default: 1)
- VTL_TEMP_DIRECTORY: Directory for spill-to-disk (default: system temp)
- VTL_MAX_TEMP_DIRECTORY_SIZE: Max size for temp directory spill
  (e.g., "100GB") (default: available disk space)
- VTL_USE_IN_MEMORY_DB: Use in-memory database (default: "1"; set to "0" for file-backed)

Number columns are stored as DOUBLE (IEEE 754 float64), the same representation
the pandas engine uses. Arithmetic and output precision are controlled by
OUTPUT_NUMBER_SIGNIFICANT_DIGITS (see vtlengine.Utils._number_config).

Example:
    export VTL_MEMORY_LIMIT=16GB
    export VTL_THREADS=4
    export VTL_USE_IN_MEMORY_DB=0
"""

import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Union

import duckdb

from vtlengine.duckdb_transpiler.Transpiler.operators import (
    register_regex_functions,
    register_time_functions,
)

# =============================================================================
# Memory & Performance Configuration
# =============================================================================

# Accessor functions read os.environ on every call so user scripts can mutate
# os.environ after `import vtlengine` and have those values take effect on the
# next run(). Capturing them in module-level constants would freeze the values
# at import time.


def _memory_limit() -> str:
    return os.getenv("VTL_MEMORY_LIMIT", "80%")


def _threads() -> int:
    return int(os.getenv("VTL_THREADS", "1"))


def _temp_directory() -> str:
    return os.getenv("VTL_TEMP_DIRECTORY", tempfile.gettempdir())


def _max_temp_directory_size() -> str:
    return os.getenv("VTL_MAX_TEMP_DIRECTORY_SIZE", "")


def _use_in_memory_db() -> bool:
    return os.getenv("VTL_USE_IN_MEMORY_DB", "1").lower() in ("1", "true")


# Minimum storage version required by the transpiler (typed macro parameters need >= v1.4.0).
# DuckDB defaults to an older on-disk format for portability, so it must be set explicitly.
STORAGE_COMPATIBILITY_VERSION: str = "v1.4.0"


def _duckdb_memory_limit() -> Optional[str]:
    """
    Value for DuckDB's ``memory_limit`` setting, or ``None`` to leave it unset.

    DuckDB cannot parse percentage strings (e.g. "80%") and its own default is
    already 80% of physical RAM, so percentage limits are deferred to that
    default rather than computed here. Absolute limits (e.g. "8GB", "512MB") are
    passed straight through — DuckDB parses the units itself. Deferring on
    percentages is what lets vtlengine run without psutil (e.g. under
    Pyodide/Emscripten, where psutil cannot be installed).
    """
    limit = _memory_limit().strip()
    if not limit or limit.endswith("%"):
        return None
    if limit.isdigit():
        # A bare integer is a byte count; DuckDB requires an explicit unit.
        return f"{limit}B"
    return limit


def configure_duckdb_connection(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Apply memory and performance settings to a DuckDB connection.

    Statements:
    - Set memory limit (absolute VTL_MEMORY_LIMIT only; a percentage defers to DuckDB's
        own default of 80% of RAM)
    - Set temp directory: configure where DuckDB can spill to disk when memory is exceeded
    - Set max temp directory size (if configured): limit how much disk space DuckDB can use for
        spill-to-disk
    - Set thread count: configure how many CPU threads DuckDB can use for query execution
    - Set preserve_insertion_order to false for performance: DuckDB can reorder data for better
        performance
    - Set max_expression_depth to 10000 to avoid issues with complex queries: DuckDB has a default
        expression depth limit which can be too low for complex VTL queries
    - Enable object cache for better performance on repeated queries: DuckDB can cache query plans
        and data structures to speed up repeated queries
    - Disable the CTE filter pusher: a MATERIALIZED CTE holds an operand that must be computed
        over all its data points, so an operation failing at a data point that a later filter
        drops still raises its error, as in the pandas engine.
    """
    _set(conn, _connection_settings())
    _register_python_functions(conn)


# Settings that only an open connection takes: DuckDB 1.4 refuses them in duckdb.connect
_SESSION_SETTINGS = ("max_expression_depth",)


def _connection_settings() -> dict[str, Union[str, int, bool]]:
    """The settings of a connection (see configure_duckdb_connection), by DuckDB option name."""
    settings: dict[str, Union[str, int, bool]] = {}
    memory_limit = _duckdb_memory_limit()
    if memory_limit is not None:
        settings["memory_limit"] = memory_limit
    settings.update(
        temp_directory=_temp_directory(),
        preserve_insertion_order=False,
        max_expression_depth=10000,
        enable_object_cache=True,
        disabled_optimizers="cte_filter_pusher",
        threads=_threads(),
    )
    max_temp_dir_size = _max_temp_directory_size()
    if max_temp_dir_size:
        settings["max_temp_directory_size"] = max_temp_dir_size
    return settings


def _set(conn: duckdb.DuckDBPyConnection, settings: dict[str, Union[str, int, bool]]) -> None:
    conn.execute(
        ";\n".join(f"SET {name} = {_sql_literal(value)}" for name, value in settings.items())
    )


def _sql_literal(value: Union[str, int, bool]) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return "'" + value.replace("'", "''") + "'"


def _register_python_functions(conn: duckdb.DuckDBPyConnection) -> None:
    """Register Python UDFs (regex fallback for patterns RE2 cannot compile, time checks)."""
    register_regex_functions(conn)
    register_time_functions(conn)


def create_configured_connection(database: str = ":memory:") -> duckdb.DuckDBPyConnection:
    """
    Create a new DuckDB connection with configured limits.

    The settings are given to duckdb.connect rather than set afterwards: a database opened
    with the default settings starts one thread per CPU core, which ``SET threads`` then stops.

    Args:
        database: Database path or ":memory:" for in-memory

    Returns:
        Configured DuckDB connection
    """
    settings = _connection_settings()
    session = {name: settings.pop(name) for name in _SESSION_SETTINGS}
    conn = duckdb.connect(
        database,
        config={"storage_compatibility_version": STORAGE_COMPATIBILITY_VERSION, **settings},
    )
    _set(conn, session)
    _register_python_functions(conn)
    return conn


@contextmanager
def configured_connection(database: str = ":memory:") -> Iterator[duckdb.DuckDBPyConnection]:
    """Context manager that yields a configured DuckDB connection."""
    temp_dir = _temp_directory()
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    session_dir = Path(temp_dir) / f"duckdb_tmp_{uuid.uuid4().hex}"
    session_dir.mkdir(exist_ok=True)

    if database == ":memory:" and not _use_in_memory_db():
        database = str(session_dir / "session.duckdb")

    conn = create_configured_connection(database)
    conn.execute(f"SET temp_directory = '{session_dir}'")
    try:
        yield conn
    finally:
        try:
            conn.close()
        finally:
            shutil.rmtree(session_dir, ignore_errors=True)


def get_system_info() -> dict[str, Union[float, int, str, None]]:
    """
    Get the DuckDB memory/thread configuration.

    Live system-RAM figures are not reported: probing them required psutil,
    which is no longer a dependency. When VTL_MEMORY_LIMIT is a percentage,
    DuckDB's own default (80% of physical RAM) applies.
    """
    return {
        "configured_limit": _memory_limit(),
        "threads": _threads() or os.cpu_count(),
        "temp_directory": _temp_directory(),
    }
