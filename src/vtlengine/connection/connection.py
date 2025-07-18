import contextlib
import os
from pathlib import Path
from typing import Optional

import duckdb

# import psutil

BASE_PATH = Path(__file__).resolve().parents[3]
# BASE_DATABASE = str(Path(os.getenv("DUCKDB_DATABASE", BASE_PATH / "vtl_duckdb.db")).resolve())
BASE_DATABASE = os.getenv("DUCKDB_DATABASE", ":memory:")
BASE_TEMP_DIRECTORY = str(Path(os.getenv("DUCKDB_TEMP_DIRECTORY", BASE_PATH / ".tmp")))
BASE_MEMORY_LIMIT = "4GB"
# TODO: uncomment the following line to use the memory limit by env-var
# total_memory = psutil.virtual_memory().total
# memory_limit = f"{total_memory * 0.8 / (1024 ** 3):.0f}GB"
# BASE_MEMORY_LIMIT = os.getenv("DUCKDB_MEMORY_LIMIT", memory_limit)
PLAN_FORMAT = "optimized_only"


class ConnectionManager:
    _connection = None
    _database = BASE_DATABASE
    _memory_limit = BASE_MEMORY_LIMIT
    _plan_format = PLAN_FORMAT
    _temp_directory: str = BASE_TEMP_DIRECTORY
    _threads = None

    @classmethod
    def configure(
        cls,
        database: str = BASE_DATABASE,
        memory_limit: str = BASE_MEMORY_LIMIT,
        plan_format: str = PLAN_FORMAT,
        temp_directory: str = BASE_TEMP_DIRECTORY,
        threads: Optional[int] = None,
    ) -> None:
        """
        Configures the database path and memory limit for DuckDB.
        """
        cls._database = database
        cls._memory_limit = memory_limit
        cls._plan_format = plan_format
        cls._temp_directory = temp_directory
        if threads is not None:
            cls._threads = threads

    @classmethod
    def get_connection(cls) -> duckdb.DuckDBPyConnection:
        """
        Returns a local DuckDB connection. Creates one if it doesn't exist.
        """
        if cls._connection is None:
            config_dict = {
                "memory_limit": cls._memory_limit,
                "temp_directory": cls._temp_directory,
            }
            cls._connection = duckdb.connect(database=cls._database, config=config_dict)
            cls._connection.execute(f"SET explain_output={cls._plan_format};")
            # cls._connection.execute("SET preserve_insertion_order=false;")
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
