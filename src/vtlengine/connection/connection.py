import contextlib
import inspect
import os
from typing import Optional

import duckdb
from duckdb.functional import FunctionNullHandling

import vtlengine.duckdb.duckdb_custom_functions as custom_functions

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
            cls.register_functions()
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

    @classmethod
    def register_functions(cls) -> None:
        """
        Registers custom functions with the DuckDB connection.
        """
        if cls._connection is None:
            cls.get_connection()
        else:
            # Register custom functions here, definitions can be
            # found in duckdb_custom_functions.py:
            for func_name in dir(custom_functions):
                func_ref = getattr(custom_functions, func_name)
                if func_name.startswith("__") or not inspect.isfunction(func_ref):
                    continue
                cls._connection.create_function(
                    func_name,
                    func_ref,  # type: ignore[arg-type]
                    null_handling=FunctionNullHandling.SPECIAL,
                )
                # duckdb.create_function expects a function,
                # we are using FunctionType which works the same
