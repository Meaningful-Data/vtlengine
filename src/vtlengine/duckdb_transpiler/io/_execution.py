"""
Execution helpers for DuckDB transpiler.

This module contains helper functions for executing VTL scripts with DuckDB,
handling dataset loading/saving with DAG scheduling for memory efficiency.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import duckdb
import pandas as pd

from vtlengine.duckdb_transpiler.io._io import (
    load_datapoints_duckdb,
    register_dataframes,
    save_datapoints_duckdb,
)
from vtlengine.duckdb_transpiler.sql import initialize_time_types
from vtlengine.Model import Dataset, Scalar


def load_scheduled_datasets(
    conn: duckdb.DuckDBPyConnection,
    statement_num: int,
    ds_analysis: Dict[str, Any],
    path_dict: Optional[Dict[str, Path]],
    dataframe_dict: Dict[str, pd.DataFrame],
    input_datasets: Dict[str, Dataset],
    insert_key: str,
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
        insert_key: Key in ds_analysis for insertion schedule (e.g., 'insertion')
    """
    if statement_num not in ds_analysis.get(insert_key, {}):
        return

    for ds_name in ds_analysis[insert_key][statement_num]:
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
    output_datasets: Dict[str, Dataset],
    results: Dict[str, Union[Dataset, Scalar]],
    return_only_persistent: bool,
    delete_key: str,
    global_key: str,
    persistent_key: str,
) -> None:
    """
    Clean up datasets scheduled for deletion at a given statement.

    Args:
        conn: DuckDB connection
        statement_num: Current statement number (1-indexed)
        ds_analysis: DAG analysis dict with deletion schedule
        output_folder: Path to save CSVs (None for in-memory mode)
        output_datasets: Dict of output dataset structures
        results: Dict to store results
        return_only_persistent: Only return persistent assignments
        delete_key: Key in ds_analysis for deletion schedule
        global_key: Key in ds_analysis for global inputs
        persistent_key: Key in ds_analysis for persistent outputs
    """
    if statement_num not in ds_analysis.get(delete_key, {}):
        return

    global_inputs = ds_analysis.get(global_key, [])
    persistent_datasets = ds_analysis.get(persistent_key, [])

    for ds_name in ds_analysis[delete_key][statement_num]:
        if ds_name in global_inputs:
            # Drop global inputs without saving
            conn.execute(f'DROP TABLE IF EXISTS "{ds_name}"')
        elif not return_only_persistent or ds_name in persistent_datasets:
            if output_folder:
                # Save to CSV and drop table
                save_datapoints_duckdb(conn, ds_name, output_folder)
                ds = output_datasets.get(ds_name, Dataset(name=ds_name, components={}, data=None))
                results[ds_name] = ds
            else:
                # Fetch data before dropping table
                result_df = conn.execute(f'SELECT * FROM "{ds_name}"').fetchdf()
                ds = output_datasets.get(ds_name, Dataset(name=ds_name, components={}, data=None))
                ds.data = result_df
                results[ds_name] = ds
                conn.execute(f'DROP TABLE IF EXISTS "{ds_name}"')
        else:
            # Drop non-persistent intermediate results
            conn.execute(f'DROP TABLE IF EXISTS "{ds_name}"')


def fetch_result(
    conn: duckdb.DuckDBPyConnection,
    result_name: str,
    output_folder: Optional[Path],
    output_datasets: Dict[str, Dataset],
    output_scalars: Dict[str, Scalar],
) -> Union[Dataset, Scalar]:
    """
    Fetch a result from DuckDB and return as Dataset or Scalar.

    Args:
        conn: DuckDB connection
        result_name: Name of the result table
        output_folder: Path to save CSV (None for in-memory mode)
        output_datasets: Dict of output dataset structures
        output_scalars: Dict of output scalar structures

    Returns:
        Dataset or Scalar with result data
    """
    if output_folder:
        # Save to CSV
        save_datapoints_duckdb(conn, result_name, output_folder)
        return output_datasets.get(result_name, Dataset(name=result_name, components={}, data=None))

    # Fetch as DataFrame
    result_df = conn.execute(f'SELECT * FROM "{result_name}"').fetchdf()

    if result_name in output_scalars:
        if len(result_df) == 1 and len(result_df.columns) == 1:
            scalar = output_scalars[result_name]
            scalar.value = result_df.iloc[0, 0]
            return scalar
        return Dataset(name=result_name, components={}, data=result_df)

    ds = output_datasets.get(result_name, Dataset(name=result_name, components={}, data=None))
    ds.data = result_df
    return ds


def execute_queries(
    conn: duckdb.DuckDBPyConnection,
    queries: List[Tuple[str, str, bool]],
    ds_analysis: Dict[str, Any],
    path_dict: Optional[Dict[str, Path]],
    dataframe_dict: Dict[str, pd.DataFrame],
    input_datasets: Dict[str, Dataset],
    output_datasets: Dict[str, Dataset],
    output_scalars: Dict[str, Scalar],
    output_folder: Optional[Path],
    return_only_persistent: bool,
    insert_key: str,
    delete_key: str,
    global_key: str,
    persistent_key: str,
) -> Dict[str, Union[Dataset, Scalar]]:
    """
    Execute transpiled SQL queries with DAG-scheduled dataset loading/saving.

    Args:
        conn: DuckDB connection
        queries: List of (result_name, sql_query, is_persistent) tuples
        ds_analysis: DAG analysis dict
        path_dict: Dict mapping dataset names to CSV paths
        dataframe_dict: Dict mapping dataset names to DataFrames
        input_datasets: Dict of input dataset structures
        output_datasets: Dict of output dataset structures
        output_scalars: Dict of output scalar structures
        output_folder: Path to save CSVs (None for in-memory mode)
        return_only_persistent: Only return persistent assignments
        insert_key: Key in ds_analysis for insertion schedule
        delete_key: Key in ds_analysis for deletion schedule
        global_key: Key in ds_analysis for global inputs
        persistent_key: Key in ds_analysis for persistent outputs

    Returns:
        Dict of result_name -> Dataset or Scalar
    """
    results: Dict[str, Union[Dataset, Scalar]] = {}

    # Initialize VTL time type functions (idempotent - safe to call multiple times)
    initialize_time_types(conn)

    # Ensure output folder exists if provided
    if output_folder:
        output_folder.mkdir(parents=True, exist_ok=True)

    # Execute each query with DAG scheduling
    for statement_num, (result_name, sql_query, _) in enumerate(queries, start=1):
        # Load datasets scheduled for this statement
        load_scheduled_datasets(
            conn=conn,
            statement_num=statement_num,
            ds_analysis=ds_analysis,
            path_dict=path_dict,
            dataframe_dict=dataframe_dict,
            input_datasets=input_datasets,
            insert_key=insert_key,
        )

        # Execute query and create table
        conn.execute(f'CREATE TABLE "{result_name}" AS {sql_query}')

        # Clean up datasets scheduled for deletion
        cleanup_scheduled_datasets(
            conn=conn,
            statement_num=statement_num,
            ds_analysis=ds_analysis,
            output_folder=output_folder,
            output_datasets=output_datasets,
            results=results,
            return_only_persistent=return_only_persistent,
            delete_key=delete_key,
            global_key=global_key,
            persistent_key=persistent_key,
        )

    # Handle final results not yet processed
    for result_name, _, is_persistent in queries:
        if result_name in results:
            continue

        should_include = not return_only_persistent or is_persistent
        if not should_include:
            continue

        results[result_name] = fetch_result(
            conn=conn,
            result_name=result_name,
            output_folder=output_folder,
            output_datasets=output_datasets,
            output_scalars=output_scalars,
        )

    return results
