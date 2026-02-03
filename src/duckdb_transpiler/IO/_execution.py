"""
Execution helpers for DuckDB transpiler.

This module contains helper functions for executing VTL scripts with DuckDB,
handling dataset loading/saving with DAG scheduling for memory efficiency.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import pandas as pd

from duckdb_transpiler.AST.DAG._words import DELETE, GLOBAL, INSERT, PERSISTENT
from duckdb_transpiler.IO._model import Query
from duckdb_transpiler.Utils.sql import initialize_time_types
from vtlengine.Model import Dataset, Scalar

from ._io import load_datapoints_duckdb, register_dataframes, save_datapoints_duckdb


def load_scheduled_datasets(
    conn: duckdb.DuckDBPyConnection,
    statement_num: int,
    ds_analysis: Dict[str, Any],
    path_dict: Optional[Dict[str, Path]],
    dataframe_dict: Dict[str, pd.DataFrame],
    input_datasets: Dict[str, Dataset],
) -> None:
    """
    Load datasets scheduled for a given statement using DAG analysis.

    Args:
        conn: DuckDB connection
        statement_num: Current statement number (1-indexed)
        ds_analysis: DAG analysis dict with insertion schedule
        path_dict: Dict mapping dataset names to CSV paths
        dataframe_dict: Dict mapping dataset names to DataFrames
        input_datasets: Dict of input dataset structures
    """
    if statement_num not in ds_analysis.get(INSERT, {}):
        return

    for ds_name in ds_analysis[INSERT][statement_num]:
        if ds_name not in input_datasets:
            continue

        if path_dict and ds_name in path_dict:
            # Load from CSV using DuckDB's native read_csv
            load_datapoints_duckdb(
                conn=conn,
                components=input_datasets[ds_name].components,
                dataset_name=ds_name,
                csv_path=path_dict[ds_name],
            )
        elif ds_name in dataframe_dict:
            # Register DataFrame directly with proper schema
            register_dataframes(conn, {ds_name: dataframe_dict[ds_name]}, input_datasets)


def cleanup_scheduled_datasets(
    conn: duckdb.DuckDBPyConnection,
    statement_num: int,
    ds_analysis: Dict[str, Any],
    output_folder: Optional[Path],
    queries: List[Query],
    results: List[Query],
    return_only_persistent: bool,
) -> None:
    """
    Clean up datasets scheduled for deletion at a given statement.

    Args:
        conn: DuckDB connection
        statement_num: Current statement number (1-indexed)
        ds_analysis: DAG analysis dict with deletion schedule
        output_folder: Path to save CSVs (None for in-memory mode)
        queries: List of all Query objects
        results: List to store completed Query results
        return_only_persistent: Only return persistent assignments
    """
    if statement_num not in ds_analysis.get(DELETE, {}):
        return

    global_inputs = ds_analysis.get(GLOBAL, [])
    persistent_datasets = ds_analysis.get(PERSISTENT, [])

    # Create a mapping of query names to Query objects
    query_map = {q.name: q for q in queries}

    for ds_name in ds_analysis[DELETE][statement_num]:
        if ds_name in global_inputs:
            # Drop global inputs without saving
            conn.execute(f'DROP TABLE IF EXISTS "{ds_name}"')
        elif not return_only_persistent or ds_name in persistent_datasets:
            # Find the corresponding query
            query = query_map.get(ds_name)
            if query:
                if output_folder:
                    # Save to CSV and drop table
                    save_datapoints_duckdb(conn, ds_name, output_folder)
                    results.append(query)
                else:
                    # Fetch data before dropping table
                    result_df = conn.execute(f'SELECT * FROM "{ds_name}"').fetchdf()
                    query.structure.data = result_df
                    results.append(query)
                    conn.execute(f'DROP TABLE IF EXISTS "{ds_name}"')
        else:
            # Drop non-persistent intermediate results
            conn.execute(f'DROP TABLE IF EXISTS "{ds_name}"')


def fetch_result(
    conn: duckdb.DuckDBPyConnection,
    query: Query,
    output_folder: Optional[Path],
) -> Query:
    """
    Fetch a result from DuckDB and populate the Query's structure data field.

    Args:
        conn: DuckDB connection
        query: Query object to populate with result data
        output_folder: Path to save CSV (None for in-memory mode)

    Returns:
        Query object with structure.data populated
    """
    if output_folder:
        # Save to CSV
        save_datapoints_duckdb(conn, query.name, output_folder)
        # Structure data remains None when saved to file
        return query

    # Fetch as DataFrame
    result_df = conn.execute(f'SELECT * FROM "{query.name}"').fetchdf()

    # Populate the structure's data field
    if isinstance(query.structure, Scalar):
        if len(result_df) == 1 and len(result_df.columns) == 1:
            query.structure.value = result_df.iloc[0, 0]
        else:
            # If scalar query returned multiple rows/cols, treat as dataset
            query.structure.data = result_df
    else:
        # Dataset
        query.structure.data = result_df

    return query


def execute_queries(
    conn: duckdb.DuckDBPyConnection,
    queries: List[Query],
    ds_analysis: Dict[str, Any],
    path_dict: Optional[Dict[str, Path]],
    dataframe_dict: Dict[str, pd.DataFrame],
    input_datasets: Dict[str, Dataset],
    output_folder: Optional[Path],
    return_only_persistent: bool,
) -> List[Query]:
    """
    Execute transpiled SQL queries with DAG-scheduled dataset loading/saving.

    Args:
        conn: DuckDB connection
        queries: List of Query objects with name, sql, structure, and is_persistent
        ds_analysis: DAG analysis dict
        path_dict: Dict mapping dataset names to CSV paths
        dataframe_dict: Dict mapping dataset names to DataFrames
        input_datasets: Dict of input dataset structures
        output_folder: Path to save CSVs (None for in-memory mode)
        return_only_persistent: Only return persistent assignments

    Returns:
        List of Query objects with structure.data populated
    """
    results: List[Query] = []
    result_names: set = set()

    # Initialize VTL time type functions (idempotent - safe to call multiple times)
    initialize_time_types(conn)

    # Ensure output folder exists if provided
    if output_folder:
        output_folder.mkdir(parents=True, exist_ok=True)

    # Execute each query with DAG scheduling
    for statement_num, query in enumerate(queries, start=1):
        # Load datasets scheduled for this statement
        load_scheduled_datasets(
            conn=conn,
            statement_num=statement_num,
            ds_analysis=ds_analysis,
            path_dict=path_dict,
            dataframe_dict=dataframe_dict,
            input_datasets=input_datasets,
        )

        # Execute query and create table
        conn.execute(f'CREATE TABLE "{query.name}" AS {query.sql}')

        # Clean up datasets scheduled for deletion
        cleanup_scheduled_datasets(
            conn=conn,
            statement_num=statement_num,
            ds_analysis=ds_analysis,
            output_folder=output_folder,
            queries=queries,
            results=results,
            return_only_persistent=return_only_persistent,
        )

        # Track which results were added
        result_names.update(q.name for q in results)

    # Handle final results not yet processed
    for query in queries:
        if query.name in result_names:
            continue

        should_include = not return_only_persistent or query.is_persistent
        if not should_include:
            continue

        results.append(fetch_result(conn=conn, query=query, output_folder=output_folder))

    return results
