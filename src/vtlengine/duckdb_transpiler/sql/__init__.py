"""SQL initialization for VTL time types in DuckDB."""

import weakref
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import duckdb

_SQL_DIR = Path(__file__).parent
_INIT_SQL = _SQL_DIR / "init.sql"
_TIME_OPERATORS_SQL = _SQL_DIR / "time_operators.sql"

# Use WeakSet to track initialized connections - entries are automatically
# removed when the connection is garbage collected, preventing false positives
# from ID reuse.
_initialized_connections: "weakref.WeakSet[duckdb.DuckDBPyConnection]" = weakref.WeakSet()


@lru_cache(maxsize=1)
def _read_init_sql() -> str:
    """Read and cache the SQL script that defines VTL time types."""
    if not _INIT_SQL.exists():
        raise FileNotFoundError(f"SQL init file not found: {_INIT_SQL}")
    return _INIT_SQL.read_text()


@lru_cache(maxsize=1)
def _read_time_operators_sql() -> str:
    """Read and cache optional SQL operators for time handling."""
    if not _TIME_OPERATORS_SQL.exists():
        return ""
    return _TIME_OPERATORS_SQL.read_text()


def initialize_time_types(conn: "duckdb.DuckDBPyConnection") -> None:
    """
    Initialize VTL time types and functions in a DuckDB connection.

    This function is idempotent - it tracks which connections have been
    initialized and skips if already done. Uses weak references so that
    when a connection is closed/garbage collected, it's removed from tracking.

    Args:
        conn: DuckDB connection to initialize
    """
    if conn in _initialized_connections:
        return

    conn.execute(_read_init_sql())

    time_operators_sql = _read_time_operators_sql()
    if time_operators_sql:
        conn.execute(time_operators_sql)

    _initialized_connections.add(conn)


def get_init_sql() -> str:
    """
    Get the raw SQL for initializing time types.

    Useful for debugging or manual initialization.

    Returns:
        SQL string containing all type and function definitions
    """
    return _read_init_sql()
