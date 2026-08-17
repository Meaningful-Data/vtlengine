"""
DuckDB Transpiler Configuration.

Configuration values can be set via environment variables:
- VTL_MEMORY_LIMIT: Max memory for DuckDB (e.g., "8GB", "80%") (default: "80%")
- VTL_THREADS: Number of threads for DuckDB (default: 1)
- VTL_TEMP_DIRECTORY: Directory for spill-to-disk (default: system temp)
- VTL_MAX_TEMP_DIRECTORY_SIZE: Max size for temp directory spill
  (e.g., "100GB") (default: available disk space)
- VTL_USE_IN_MEMORY_DB: Use in-memory database (default: "1"; set to "0" for file-backed)

Number columns are stored as DOUBLE (IEEE 754 float64), the same representation
the pandas engine uses. Arithmetic and output precision are controlled by
OUTPUT_NUMBER_SIGNIFICANT_DIGITS (see vtlengine.Utils._number_config). The
legacy VTL_DUCKDB_DECIMAL_WIDTH variable is ignored and only triggers a
DeprecationWarning.

Example:
    export VTL_MEMORY_LIMIT=16GB
    export VTL_THREADS=4
    export VTL_USE_IN_MEMORY_DB=0
"""

import os
import shutil
import tempfile
import uuid
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Union

import duckdb
import psutil  # type: ignore[import-untyped]

from vtlengine.duckdb_transpiler.Transpiler.operators import register_regex_functions

# =============================================================================
# Deprecated Decimal Configuration
# =============================================================================

# Number columns were stored as DECIMAL(width, scale) until 1.9.2; they are now
# DOUBLE so both engines agree to 15 significant digits (issue #985).
DECIMAL_WIDTH_ENV_VAR = "VTL_DUCKDB_DECIMAL_WIDTH"


def _warn_deprecated_decimal_width() -> None:
    if os.environ.get(DECIMAL_WIDTH_ENV_VAR):
        warnings.warn(
            f"{DECIMAL_WIDTH_ENV_VAR} is deprecated and ignored: Number columns are "
            "stored as DOUBLE. Use OUTPUT_NUMBER_SIGNIFICANT_DIGITS to control "
            "arithmetic precision.",
            DeprecationWarning,
            stacklevel=2,
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


def get_memory_limit_bytes() -> int:
    """
    Parse memory limit and return bytes.

    Supports formats:
    - "80%" - percentage of system RAM
    - "8GB" - absolute size in GB
    - "8192MB" - absolute size in MB

    Returns:
        Memory limit in bytes
    """
    limit = _memory_limit().strip().upper()

    total_ram = psutil.virtual_memory().total

    if limit.endswith("%"):
        pct = float(limit[:-1]) / 100.0
        return int(total_ram * pct)
    elif limit.endswith("GB"):
        return int(float(limit[:-2]) * 1024 * 1024 * 1024)
    elif limit.endswith("MB"):
        return int(float(limit[:-2]) * 1024 * 1024)
    elif limit.endswith("KB"):
        return int(float(limit[:-2]) * 1024)
    else:
        # Assume bytes
        return int(limit)


def get_memory_limit_str() -> str:
    """
    Get memory limit as a human-readable string for DuckDB.

    Returns:
        Memory limit string (e.g., "8GB")
    """
    bytes_limit = get_memory_limit_bytes()
    gb = bytes_limit / (1024**3)
    if gb >= 1:
        return f"{gb:.1f}GB"
    else:
        mb = bytes_limit / (1024**2)
        return f"{mb:.0f}MB"


def configure_duckdb_connection(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Apply memory and performance settings to a DuckDB connection.

    Statements:
    - Set memory limit: set the maximum memory DuckDB can use based on configuration
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
    """
    max_temp_dir_size = _max_temp_directory_size()
    statements = [
        f"SET memory_limit = '{get_memory_limit_str()}'",
        f"SET temp_directory = '{_temp_directory()}'",
        "SET preserve_insertion_order = false",
        "SET max_expression_depth TO 10000",
        "SET enable_object_cache = true",
        f"SET threads = {_threads()}",
    ]
    if max_temp_dir_size:
        statements.append(f"SET max_temp_directory_size = '{max_temp_dir_size}'")

    conn.execute(";\n".join(statements))

    # Register Python UDFs (regex fallback for patterns RE2 cannot compile).
    register_regex_functions(conn)

    # Legacy decimal storage configuration (Number is DOUBLE since issue #985)
    _warn_deprecated_decimal_width()


def create_configured_connection(database: str = ":memory:") -> duckdb.DuckDBPyConnection:
    """
    Create a new DuckDB connection with configured limits.

    Args:
        database: Database path or ":memory:" for in-memory

    Returns:
        Configured DuckDB connection
    """
    conn = duckdb.connect(
        database, config={"storage_compatibility_version": STORAGE_COMPATIBILITY_VERSION}
    )
    configure_duckdb_connection(conn)
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
    Get system memory information.

    Returns:
        Dict with total_ram, available_ram, memory_limit (all in GB)
    """
    mem = psutil.virtual_memory()
    return {
        "total_ram_gb": mem.total / (1024**3),
        "available_ram_gb": mem.available / (1024**3),
        "used_percent": mem.percent,
        "configured_limit_gb": get_memory_limit_bytes() / (1024**3),
        "configured_limit_str": get_memory_limit_str(),
        "threads": _threads() or os.cpu_count(),
        "temp_directory": _temp_directory(),
    }
