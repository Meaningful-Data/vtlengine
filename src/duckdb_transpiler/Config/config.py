"""
DuckDB Transpiler Configuration.

Configuration values can be set via environment variables:
- VTL_MEMORY_LIMIT: Max memory for DuckDB (e.g., "8GB", "80%") (default: "80%")
- VTL_THREADS: Number of threads for DuckDB (default: system cores)
- VTL_TEMP_DIRECTORY: Directory for spill-to-disk (default: system temp)
"""

import os
import tempfile

import duckdb
import psutil

# =============================================================================
# Memory & Performance Configuration
# =============================================================================

# Default memory limit (80% of system RAM)
MEMORY_LIMIT: str = os.getenv("VTL_MEMORY_LIMIT", "80%")

# Default thread count (default = 1)
THREADS: int = int(os.getenv("VTL_THREADS", "1"))

# Temp directory for spill-to-disk
TEMP_DIRECTORY: str = os.getenv("VTL_TEMP_DIRECTORY", tempfile.gettempdir())


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
    limit = MEMORY_LIMIT.strip().upper()

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
    mb = bytes_limit / (1024**2)
    return f"{mb:.0f}MB"


def configure_duckdb_connection(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Apply memory and performance settings to a DuckDB connection.

    Args:
        conn: DuckDB connection to configure
    """
    memory_limit = get_memory_limit_str()

    # Set memory limit
    conn.execute(f"SET memory_limit = '{memory_limit}'")

    # Set temp directory for spill-to-disk
    conn.execute(f"SET temp_directory = '{TEMP_DIRECTORY}'")

    # Set thread count if specified
    if THREADS is not None:
        conn.execute(f"SET threads = {THREADS}")

    # Disable insertion order preservation for better memory efficiency
    conn.execute("SET preserve_insertion_order = false")

    # Enable progress bar for long operations
    conn.execute("SET enable_progress_bar = true")

    # Performance optimizations for large data loads
    # Enable object cache for repeated query patterns
    conn.execute("SET enable_object_cache = true")


def create_configured_connection(database: str = ":memory:") -> duckdb.DuckDBPyConnection:
    """
    Create a new DuckDB connection with configured limits.

    Args:
        database: Database path or ":memory:" for in-memory

    Returns:
        Configured DuckDB connection
    """
    conn = duckdb.connect(database)
    configure_duckdb_connection(conn)
    return conn


def get_system_info() -> dict:
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
        "threads": THREADS or os.cpu_count(),
        "temp_directory": TEMP_DIRECTORY,
    }
