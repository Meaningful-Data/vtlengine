import contextlib
import os
from typing import Optional

import duckdb

# import psutil

BASE_DATABASE = os.getenv("DUCKDB_DATABASE", ":memory:")
BASE_MEMORY_LIMIT = "1GB"
# TODO: uncomment the following line to use the memory limit by env-var
# total_memory = psutil.virtual_memory().total
# memory_limit = f"{total_memory * 0.8 / (1024 ** 3):.0f}GB"
# BASE_MEMORY_LIMIT = os.getenv("DUCKDB_MEMORY_LIMIT", memory_limit)


class ConnectionManager:
    _connection = None
    _database = BASE_DATABASE
    _memory_limit = BASE_MEMORY_LIMIT
    _threads = None

    @classmethod
    def configure(
        cls,
        database: str = BASE_DATABASE,
        memory_limit: str = BASE_MEMORY_LIMIT,
        threads: Optional[int] = None,
    ) -> None:
        """
        Configures the database path and memory limit for DuckDB.
        """
        cls._database = database
        cls._memory_limit = memory_limit
        if threads is not None:
            cls._threads = threads

    @classmethod
    def get_connection(cls) -> duckdb.DuckDBPyConnection:
        """
        Returns a local DuckDB connection. Creates one if it doesn't exist.
        """
        if cls._connection is None:
            cls._connection = duckdb.connect(database=cls._database)
            cls._connection.execute(f"SET memory_limit='{cls._memory_limit}'")
            if cls._threads is not None:
                cls._connection.execute(f"SET threads={cls._threads}")
        return cls._connection

    @classmethod
    def close_connection(cls) -> None:
        """
        Closes the thread-local DuckDB connection.
        """
        if cls._connection:
            cls._connection.close()
            cls._connection = None

    @classmethod
    def clean_connection(cls) -> None:
        """
        Cleans the connection by closing it and resetting the class variables.
        """
        try:
            if cls._connection:
                cls._connection.rollback()
        except Exception as e:
            # No rollback needed
            contextlib.suppress(e)  # type: ignore[arg-type]
