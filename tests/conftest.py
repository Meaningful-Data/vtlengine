import pytest

from src.vtlengine.connection.connection import ConnectionManager


@pytest.fixture(autouse=True)
def clean_duckdb_connection():
    """
    Fixture to clean the DuckDB connection before and after each test.
    """
    ConnectionManager.clean_connection()
