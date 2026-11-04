"""The DuckDB database the pandas engine's operators query their frames in."""

import os
import threading
from contextlib import contextmanager
from typing import Iterator, Optional, Tuple

import duckdb


class _ProcessDatabase:
    """One in-memory database per process, opened when first needed."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # The process that opened the database: a child made by os.fork() opens its own
        self._pid: Optional[int] = None
        self.connection: Optional[duckdb.DuckDBPyConnection] = None

    def cursor(self) -> Tuple[duckdb.DuckDBPyConnection, duckdb.DuckDBPyConnection]:
        """A new connection to the database, with the database's own connection."""
        with self._lock:
            if self.connection is None or self._pid != os.getpid():
                self._pid = os.getpid()
                self.connection = duckdb.connect(database=":memory:")
            return self.connection, self.connection.cursor()

    def discard(self, database: duckdb.DuckDBPyConnection) -> None:
        """Forget the database, unless another one has replaced it already."""
        with self._lock:
            if self.connection is database:
                self.connection = None


_database = _ProcessDatabase()


@contextmanager
def query_connection() -> Iterator[duckdb.DuckDBPyConnection]:
    """
    A connection of its own to the in-memory database of the process, for one query.

    Opening a database costs about 10 ms, a connection to an open one a hundredth of a
    millisecond. The frames registered on the connection are private to it and go away
    when it closes, so threads can query at the same time.
    """
    database, conn = _database.cursor()
    try:
        yield conn
    except duckdb.FatalException:
        # DuckDB invalidates a database after a fatal error: the next query opens another
        _database.discard(database)
        raise
    finally:
        conn.close()
