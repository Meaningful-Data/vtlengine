"""The DuckDB database the pandas engine's operators query their frames in."""

import os
import threading
from contextlib import contextmanager
from typing import Iterator, Optional, Tuple

import duckdb

_lock = threading.Lock()
# The database, with the process that opened it: a child made by os.fork() opens its own
_database: Optional[Tuple[int, duckdb.DuckDBPyConnection]] = None


@contextmanager
def query_connection() -> Iterator[duckdb.DuckDBPyConnection]:
    """
    A connection of its own to the in-memory database of the process, for one query.

    Opening a database costs about 10 ms, a connection to an open one a hundredth of a
    millisecond. The frames registered on the connection are private to it and go away
    when it closes, so threads can query at the same time.
    """
    global _database
    with _lock:
        if _database is None or _database[0] != os.getpid():
            _database = (os.getpid(), duckdb.connect(database=":memory:"))
        database = _database
        conn = database[1].cursor()
    try:
        yield conn
    except duckdb.FatalException:
        # DuckDB invalidates a database after a fatal error: the next query opens another
        with _lock:
            if _database is database:
                _database = None
        raise
    finally:
        conn.close()
