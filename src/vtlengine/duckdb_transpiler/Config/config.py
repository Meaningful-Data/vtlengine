"""
DuckDB Transpiler Configuration.

Configuration values can be set via environment variables:
- VTL_DECIMAL_PRECISION: Total number of digits for DECIMAL type (default: 18, -1 to disable)
- VTL_DECIMAL_SCALE: Number of decimal places for DECIMAL type (default: 8, -1 to disable)
- VTL_MEMORY_LIMIT: Max memory for DuckDB (e.g., "8GB", "80%") (default: "80%")
- VTL_THREADS: Number of threads for DuckDB (default: system cores)
- VTL_TEMP_DIRECTORY: Directory for spill-to-disk (default: system temp)
- VTL_MAX_TEMP_DIRECTORY_SIZE: Max size for temp directory spill
  (e.g., "100GB") (default: available disk space)

Example:
    export VTL_DECIMAL_PRECISION=28
    export VTL_DECIMAL_SCALE=8
    export VTL_MEMORY_LIMIT=16GB
    export VTL_THREADS=4
"""

import os
import tempfile
from typing import Tuple, Union

import duckdb
import psutil  # type: ignore[import-untyped]

# =============================================================================
# Decimal Configuration
# =============================================================================

DECIMAL_SCALE_ENV_VAR = "OUTPUT_NUMBER_SIGNIFICANT_DIGITS"

DEFAULT_DECIMAL_PRECISION = 28
DEFAULT_DECIMAL_SCALE = 8

MAX_DECIMAL_SCALE = 15
MIN_DECIMAL_SCALE = 6

DISABLED_VALUE = -1

DECIMAL_PRECISION: int = DEFAULT_DECIMAL_PRECISION
DECIMAL_SCALE: int = int(os.getenv(DECIMAL_SCALE_ENV_VAR, MIN_DECIMAL_SCALE))


def get_decimal_type() -> str:
    """
    Get the DuckDB type string for Number columns.

    Returns:
        "DOUBLE" if disabled (scale or precision is -1),
        otherwise DECIMAL type string, e.g., "DECIMAL(28,15)"
    """
    return f"DECIMAL({DECIMAL_PRECISION},{DECIMAL_SCALE})"


def get_decimal_config() -> Tuple[int, int]:
    """
    Get the current decimal precision and scale configuration.

    Returns:
        Tuple of (precision, scale)
    """
    return (DECIMAL_PRECISION, DECIMAL_SCALE)


def set_decimal_config() -> None:
    """
    Set decimal precision and scale at runtime.

    Args:
        precision: Total number of digits
        scale: Number of decimal places
    """
    global DECIMAL_PRECISION, DECIMAL_SCALE

    if DECIMAL_SCALE == DISABLED_VALUE:
        return
    if DECIMAL_SCALE < MIN_DECIMAL_SCALE or DECIMAL_SCALE > MAX_DECIMAL_SCALE:
        raise ValueError(
            f"Invalid value for {DECIMAL_SCALE_ENV_VAR}: {DECIMAL_SCALE}. "
            f"Expected an integer between {MIN_DECIMAL_SCALE} and {MAX_DECIMAL_SCALE}, "
            f"or {DISABLED_VALUE} to disable."
        )


set_decimal_config()


# =============================================================================
# Memory & Performance Configuration
# =============================================================================

# Default memory limit (80% of system RAM)
MEMORY_LIMIT: str = os.getenv("VTL_MEMORY_LIMIT", "80%")

# Default thread count (default = 1)
THREADS: int = int(os.getenv("VTL_THREADS", "1"))

# Temp directory for spill-to-disk
TEMP_DIRECTORY: str = os.getenv("VTL_TEMP_DIRECTORY", tempfile.gettempdir())

# Max temp directory size for spill-to-disk (empty = use available disk space)
MAX_TEMP_DIRECTORY_SIZE: str = os.getenv("VTL_MAX_TEMP_DIRECTORY_SIZE", "")

# Use file-backed database instead of in-memory (better for large datasets)
USE_FILE_DATABASE: bool = os.getenv("VTL_USE_FILE_DATABASE", "").lower() in ("1", "true", "yes")


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

    Args:
        conn: DuckDB connection to configure
    """
    memory_limit = get_memory_limit_str()

    # Set memory limit
    conn.execute(f"SET memory_limit = '{memory_limit}'")

    # Set temp directory for spill-to-disk
    conn.execute(f"SET temp_directory = '{TEMP_DIRECTORY}'")

    # Set max temp directory size if explicitly configured
    if MAX_TEMP_DIRECTORY_SIZE:
        conn.execute(f"SET max_temp_directory_size = '{MAX_TEMP_DIRECTORY_SIZE}'")

    # Set thread count if specified
    if THREADS is not None:
        conn.execute(f"SET threads = {THREADS}")

    # Disable insertion order preservation for better memory efficiency
    conn.execute("SET preserve_insertion_order = false")

    # Enable progress bar for long operations
    conn.execute("SET enable_progress_bar = true")

    # Increase max expression depth for deeply nested SQL (e.g. 225+ operand chains)
    conn.execute("SET max_expression_depth TO 10000")

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
        "threads": THREADS or os.cpu_count(),
        "temp_directory": TEMP_DIRECTORY,
    }
