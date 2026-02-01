"""
DuckDB-based CSV IO optimized for out-of-core processing.

Public functions:
- load_datapoints_duckdb: Load CSV data into DuckDB table with validation
- save_datapoints_duckdb: Save DuckDB table to CSV file
- execute_queries: Execute transpiled SQL queries with DAG scheduling
"""

from ._execution import execute_queries
from ._io import load_datapoints_duckdb, save_datapoints_duckdb

__all__ = [
    "load_datapoints_duckdb",
    "save_datapoints_duckdb",
    "execute_queries",
]
