"""
Execution helpers for DuckDB transpiler.

This module contains helper functions for executing VTL scripts with DuckDB,
handling dataset loading/saving with DAG scheduling for memory efficiency.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import duckdb
import pandas as pd

from vtlengine.AST.DAG._models import DatasetSchedule
from vtlengine.DataTypes import (
    Date,
    TimeInterval,
    TimePeriod,
)
from vtlengine.duckdb_transpiler.io._io import (
    load_datapoints_duckdb,
    register_dataframes,
    save_datapoints_duckdb,
    save_scalars_duckdb,
)
from vtlengine.duckdb_transpiler.io._time_handling import (
    apply_time_period_representation,
    format_time_period_scalar,
)
from vtlengine.duckdb_transpiler.sql import initialize_time_types
from vtlengine.files.output._time_period_representation import TimePeriodRepresentation
from vtlengine.Model import Dataset, Scalar


def _format_timestamp(ts: Any) -> str:
    """Format a pandas Timestamp / datetime to a VTL date string.

    Preserves time components when present:
    - ``2020-01-15 00:00:00`` → ``'2020-01-15'``
    - ``2020-01-15 10:30:00`` → ``'2020-01-15 10:30:00'``
    - ``2020-01-15 10:30:00.123456`` → ``'2020-01-15 10:30:00.123456'``
    """
    if hasattr(ts, "microsecond") and ts.microsecond:
        return ts.strftime("%Y-%m-%d %H:%M:%S.%f")
    if hasattr(ts, "hour") and (ts.hour or ts.minute or ts.second):
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    return ts.strftime("%Y-%m-%d")


def _format_timestamp_with_time(ts: Any) -> str:
    """Format a timestamp always including time (for columns with mixed values).

    - ``2020-01-15 00:00:00`` → ``'2020-01-15 00:00:00'``
    - ``2020-01-15 10:30:00`` → ``'2020-01-15 10:30:00'``
    - ``2020-01-15 10:30:00.123456`` → ``'2020-01-15 10:30:00.123456'``
    """
    if hasattr(ts, "microsecond") and ts.microsecond:
        return ts.strftime("%Y-%m-%d %H:%M:%S.%f")
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def _normalize_scalar_value(raw_value: Any) -> Any:
    """Convert pandas/numpy types to plain Python values.

    DuckDB's ``fetchdf()`` may return ``pd.NA``, ``pd.NaT`` or
    ``numpy.nan`` for SQL NULLs.  The rest of the engine expects
    plain ``None``.  Timestamps are converted to VTL date strings.
    """
    if hasattr(raw_value, "item"):
        raw_value = raw_value.item()
    if pd.isna(raw_value):
        return None
    # Convert datetime/Timestamp to VTL date string
    if isinstance(raw_value, pd.Timestamp):
        return _format_timestamp(raw_value)
    import datetime

    if isinstance(raw_value, (datetime.datetime, datetime.date)):
        return _format_timestamp(raw_value)
    return raw_value


def _project_columns(ds: Dataset) -> None:
    """Project DataFrame columns to match the dataset's component structure.

    DuckDB tables may retain extra columns from upstream operations (e.g. filter
    preserves all columns from the source table).  The semantic analysis already
    determines the correct components, so we just select those columns.
    """
    if ds.components and ds.data is not None:
        expected_cols = [c for c in ds.components if c in ds.data.columns]
        if expected_cols and set(expected_cols) != set(ds.data.columns):
            ds.data = ds.data[expected_cols]


def _convert_date_columns(ds: Dataset) -> None:
    """Convert DuckDB datetime columns to VTL string format.

    DuckDB returns Timestamp/NaT for date columns but the VTL engine
    (Pandas backend) uses string dates and None for nulls.
    Preserves time components when present (e.g. '2020-01-15 10:30:00').
    If any non-null value has a non-midnight time, all values in the column
    are formatted with time to preserve consistency.
    Only converts columns that actually have datetime dtype (not already strings).
    """
    if ds.components and ds.data is not None:
        for comp_name, comp in ds.components.items():
            if (
                comp.data_type in (Date, TimePeriod, TimeInterval)
                and comp_name in ds.data.columns
                and pd.api.types.is_datetime64_any_dtype(ds.data[comp_name])
            ):
                col = ds.data[comp_name]
                non_null = col.dropna()
                has_time = False
                if len(non_null) > 0:
                    has_time = bool(
                        any(v.hour or v.minute or v.second or v.microsecond for v in non_null)
                    )
                if has_time:
                    ds.data[comp_name] = col.apply(
                        lambda x: _format_timestamp_with_time(x) if pd.notna(x) else None  # type: ignore[redundant-expr,unused-ignore]
                    )
                else:
                    ds.data[comp_name] = col.apply(
                        lambda x: x.strftime("%Y-%m-%d") if pd.notna(x) else None  # type: ignore[redundant-expr,unused-ignore]
                    )


def load_scheduled_datasets(
    conn: duckdb.DuckDBPyConnection,
    statement_num: int,
    ds_analysis: DatasetSchedule,
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
        insert_key: Key in ds_analysis for insertion schedule (e.g., 'insertion')
    """
    if statement_num not in ds_analysis.insertion:
        return

    for ds_name in ds_analysis.insertion[statement_num]:
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
        else:
            # No data provided - create empty table with proper schema
            load_datapoints_duckdb(
                conn=conn,
                components=input_datasets[ds_name].components,
                dataset_name=ds_name,
                csv_path=None,
            )


def cleanup_scheduled_datasets(
    conn: duckdb.DuckDBPyConnection,
    statement_num: int,
    ds_analysis: DatasetSchedule,
    output_folder: Optional[Path],
    output_datasets: Dict[str, Dataset],
    output_scalars: Dict[str, Scalar],
    results: Dict[str, Union[Dataset, Scalar]],
    return_only_persistent: bool,
    representation: Optional[TimePeriodRepresentation] = None,
) -> None:
    """
    Clean up datasets scheduled for deletion at a given statement.

    Args:
        conn: DuckDB connection
        statement_num: Current statement number (1-indexed)
        ds_analysis: DAG analysis dict with deletion schedule
        output_folder: Path to save CSVs (None for in-memory mode)
        output_datasets: Dict of output dataset structures
        output_scalars: Dict of output scalar structures
        results: Dict to store results
        return_only_persistent: Only return persistent assignments
        representation: TimePeriod output format
    """
    if statement_num not in ds_analysis.deletion:
        return

    global_inputs = ds_analysis.global_inputs
    persistent_datasets = ds_analysis.persistent

    for ds_name in ds_analysis.deletion[statement_num]:
        if ds_name in global_inputs:
            # Drop global inputs without saving
            conn.execute(f'DROP TABLE IF EXISTS "{ds_name}"')
        elif not return_only_persistent or ds_name in persistent_datasets:
            results[ds_name] = fetch_result(
                conn,
                ds_name,
                output_folder,
                output_datasets,
                output_scalars,
                representation,
            )
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
    representation: Optional[TimePeriodRepresentation] = None,
) -> Union[Dataset, Scalar]:
    """
    Fetch a result from DuckDB and return as Dataset or Scalar.

    Args:
        conn: DuckDB connection
        result_name: Name of the result table
        output_folder: Path to save CSV (None for in-memory mode)
        output_datasets: Dict of output dataset structures
        output_scalars: Dict of output scalar structures
        representation: TimePeriod output format (applied before save/fetch)

    Returns:
        Dataset or Scalar with result data
    """
    # Apply time period representation before saving/fetching
    apply_time_period_representation(
        conn, result_name, output_datasets, output_scalars, representation
    )

    # Scalars are always fetched in-memory (never saved to CSV)
    if result_name in output_scalars:
        result_df = conn.execute(f'SELECT * FROM "{result_name}"').fetchdf()
        if len(result_df) == 1 and len(result_df.columns) == 1:
            scalar = output_scalars[result_name]
            raw_value = _normalize_scalar_value(result_df.iloc[0, 0])
            scalar.value = raw_value
            format_time_period_scalar(scalar, representation)
            return scalar
        return Dataset(name=result_name, components={}, data=result_df)

    # Save to CSV if output folder provided (table kept alive for fetch)
    if output_folder:
        save_datapoints_duckdb(conn, result_name, output_folder, delete_after_save=False)

    # Fetch as DataFrame
    result_df = conn.execute(f'SELECT * FROM "{result_name}"').fetchdf()
    ds = output_datasets.get(result_name, Dataset(name=result_name, components={}, data=None))
    ds.data = result_df
    _project_columns(ds)
    _convert_date_columns(ds)

    return ds


def execute_queries(
    conn: duckdb.DuckDBPyConnection,
    queries: List[Tuple[str, str, bool]],
    ds_analysis: DatasetSchedule,
    path_dict: Optional[Dict[str, Path]],
    dataframe_dict: Dict[str, pd.DataFrame],
    input_datasets: Dict[str, Dataset],
    output_datasets: Dict[str, Dataset],
    output_scalars: Dict[str, Scalar],
    output_folder: Optional[Path],
    return_only_persistent: bool,
    time_period_output_format: str = "vtl",
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
        time_period_output_format: Output format for TimePeriod columns
    Returns:
        Dict of result_name -> Dataset or Scalar
    """
    results: Dict[str, Union[Dataset, Scalar]] = {}
    representation = TimePeriodRepresentation.check_value(time_period_output_format)

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
        )

        # Execute query and create table
        try:
            conn.execute(f'CREATE TABLE "{result_name}" AS {sql_query}')
        except Exception:
            import sys

            print(f"FAILED at query {statement_num}: {result_name}", file=sys.stderr)
            print(f"SQL: {str(sql_query)[:2000]}", file=sys.stderr)
            raise

        # Clean up datasets scheduled for deletion
        cleanup_scheduled_datasets(
            conn=conn,
            statement_num=statement_num,
            ds_analysis=ds_analysis,
            output_folder=output_folder,
            output_datasets=output_datasets,
            output_scalars=output_scalars,
            results=results,
            return_only_persistent=return_only_persistent,
            representation=representation,
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
            representation=representation,
        )

    # Save scalars to CSV when output_folder is provided
    if output_folder:
        result_scalars = {k: v for k, v in results.items() if isinstance(v, Scalar)}
        save_scalars_duckdb(result_scalars, output_folder)

    return results
