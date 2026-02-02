"""SQL initialization for VTL time types in DuckDB."""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import duckdb

_SQL_DIR = Path(__file__).parent
_INIT_SQL = _SQL_DIR / "init.sql"

_initialized_connections: set[int] = set()


def initialize_time_types(conn: "duckdb.DuckDBPyConnection") -> None:
    """
    Initialize VTL time types and functions in a DuckDB connection.

    This function is idempotent - it tracks which connections have been
    initialized and skips if already done.

    Args:
        conn: DuckDB connection to initialize
    """
    conn_id = id(conn)
    if conn_id in _initialized_connections:
        return

    if not _INIT_SQL.exists():
        raise FileNotFoundError(f"SQL init file not found: {_INIT_SQL}")

    conn.execute(_INIT_SQL.read_text())
    _initialized_connections.add(conn_id)


def get_init_sql() -> str:
    """
    Get the raw SQL for initializing time types.

    Useful for debugging or manual initialization.

    Returns:
        SQL string containing all type and function definitions
    """
    return _INIT_SQL.read_text()
