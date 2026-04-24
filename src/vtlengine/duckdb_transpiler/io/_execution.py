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
    _DUCKDB_TYPE_TO_VTL,
    Duration,
    Null,
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
from vtlengine.Exceptions import RunTimeError
from vtlengine.files.output._time_period_representation import TimePeriodRepresentation
from vtlengine.Model import Dataset, Scalar
from vtlengine.Utils._number_config import get_effective_numeric_digits

_TIME_SQL_MARKERS = ("vtl_",)


def _contains_time_components(datasets: Dict[str, Dataset]) -> bool:
    """Return True when any dataset contains VTL time-related components."""
    for ds in datasets.values():
        for comp in ds.components.values():
            if comp.data_type in (TimePeriod, TimeInterval, Duration):
                return True
    return False


def _requires_time_types_initialization(
    queries: List[Tuple[str, str, bool]],
    input_datasets: Dict[str, Dataset],
    output_datasets: Dict[str, Dataset],
    output_scalars: Dict[str, Scalar],
) -> bool:
    """Return True when execution paths may need VTL time macros/types."""
    if _contains_time_components(input_datasets) or _contains_time_components(output_datasets):
        return True

    for scalar in output_scalars.values():
        if scalar.data_type in (TimePeriod, TimeInterval, Duration):
            return True

    for _, sql_query, _ in queries:
        sql_lower = sql_query.lower()
        if any(marker in sql_lower for marker in _TIME_SQL_MARKERS):
            return True

    return False


def _map_time_agg_error(msg: str, msg_lower: str) -> RunTimeError:
    """Extract source indicator and target from a vtl error 2-1-19-1 message."""
    value = "unknown"
    new_indicator = "unknown"
    if "period indicator " in msg_lower:
        parts = msg.split("period indicator ")
        if len(parts) >= 2:
            value = parts[1].split(" ")[0]
    if "target " in msg_lower:
        parts = msg.split("target ")
        if len(parts) >= 2:
            new_indicator = parts[-1].strip()
    return RunTimeError("2-1-19-1", value=value, new_indicator=new_indicator)


def _map_query_error(error: duckdb.Error, sql_query: str) -> Exception:
    """Map a DuckDB query execution error to a VTL exception.

    Patterns:
    - Conversion errors on timestamp/date → RunTimeError 2-1-19-8
    - Division by zero → RunTimeError 2-1-3-1
    - Cast errors → SemanticError 1-1-5-1
    """
    msg = str(error)
    msg_lower = msg.lower()

    # VTL macro: TimePeriod aggregation with mixed indicators (max/min)
    if "vtl error 2-1-19-20" in msg_lower:
        agg_op = "min" if "unable to get the min" in msg_lower else "max"
        return RunTimeError("2-1-19-20", op=agg_op)

    # VTL macro: TimePeriod comparison with different period indicators
    if "vtl error 2-1-19-19" in msg_lower:
        indicators = ""
        if "different indicators:" in msg_lower:
            indicators = msg.split("different indicators:")[-1].strip()
        parts = indicators.split(" vs ") if " vs " in indicators else ["", ""]
        return RunTimeError(
            "2-1-19-19", value1=parts[0].strip(), op="comparison", value2=parts[1].strip()
        )

    # daytoyear / daytomonth: negative input value (check before 2-1-19-1 prefix match)
    if "vtl error 2-1-19-16" in msg_lower:
        op = "daytoyear" if "daytoyear" in msg_lower else "daytomonth"
        return RunTimeError("2-1-19-16", op=op)

    # time_agg: period indicator too coarse for target
    if "vtl error 2-1-19-1" in msg_lower:
        return _map_time_agg_error(msg, msg_lower)

    # Custom VTL macro errors: non-daily TimePeriod → Date cast
    if "cannot cast non-daily timeperiod to date" in msg_lower:
        value = msg.split(": ", 1)[-1] if ": " in msg else "unknown"
        return RunTimeError("2-1-5-1", value=value, type_1="Time_Period", type_2="Date")

    # Custom VTL macro errors: TimeInterval → Date with different dates
    if "cannot cast timeinterval to date" in msg_lower:
        value = msg.split(": ", 1)[-1] if ": " in msg else "unknown"
        return RunTimeError("2-1-5-1", value=value, type_1="Time", type_2="Date")

    # Custom VTL macro errors: cannot determine period
    if "cannot determine period for interval" in msg_lower:
        value = msg.split(": ", 1)[-1] if ": " in msg else "unknown"
        return RunTimeError("2-1-5-1", value=value, type_1="Time", type_2="Time_Period")

    # Invalid date/timestamp format (e.g. casting interval string to timestamp)
    if "conversion" in msg_lower and ("timestamp" in msg_lower or "date" in msg_lower):
        date_val = "unknown"
        if '"' in msg:
            parts = msg.split('"')
            if len(parts) >= 2:
                date_val = parts[1]
        return RunTimeError("2-1-19-8", date=date_val)

    # VTL macro vtl_div: denominator was 0 (mirrors Python engine error 2-1-15-6)
    if "vtl 2-1-15-6" in msg_lower:
        return RunTimeError("2-1-15-6", op="/")

    # Division by zero (explicit DuckDB error or VTL error from ratio_to_report)
    if "division by zero" in msg_lower or "divide by zero" in msg_lower:
        return RunTimeError("2-1-3-1", op="division")
    if "vtl error 2-1-3-1" in msg_lower:
        return RunTimeError("2-1-3-1", op="ratio_to_report")

    # Math domain error (e.g. log(0))
    if "logarithm of zero" in msg_lower or "logarithm of negative" in msg_lower:
        return ValueError("math domain error")

    # Logarithm of a negative number (log(x, negative_base))
    if "cannot take logarithm of a negative number" in msg_lower:
        return RunTimeError("2-1-15-3", op="log", value="negative")

    # Return original error if no mapping found
    return error


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


def _infer_scalar_type_from_duckdb(col_description: Any) -> Any:
    """Infer VTL data type from DuckDB column description when semantic type is Null."""
    if col_description is None:
        return None
    type_str = str(col_description).upper()
    for prefix, vtl_type in _DUCKDB_TYPE_TO_VTL.items():
        if type_str.startswith(prefix):
            return vtl_type
    return None


def _round_significant(value: float, sig_digits: int) -> float:
    """Round a float to a given number of significant digits."""
    import math

    if value == 0.0:
        return 0.0
    d = math.ceil(math.log10(abs(value)))
    return round(value, sig_digits - d)


def _normalize_scalar_value(raw_value: Any) -> Any:
    """Convert pandas/numpy types to plain Python values.

    DuckDB's ``fetchdf()`` may return ``pd.NA``, ``pd.NaT`` or
    ``numpy.nan`` for SQL NULLs.  The rest of the engine expects
    plain ``None``.  Timestamps are converted to VTL date strings.
    Float results are rounded to match the Decimal precision used by
    the core engine (OUTPUT_NUMBER_SIGNIFICANT_DIGITS, default 15).
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
    if isinstance(raw_value, float):
        precision = get_effective_numeric_digits()
        if precision is not None:
            raw_value = _round_significant(raw_value, precision)
    return raw_value


def _build_dataset_fetch_select(
    conn: duckdb.DuckDBPyConnection,
    result_name: str,
    ds: Dataset,
) -> str:
    """Build a SELECT query with column projection and in-SQL date/timestamp formatting.

    Moves all post-fetch pandas processing into DuckDB SQL so that fetchdf()
    receives data already in the correct shape and format:
    - Column projection: only the columns declared in ds.components (in order)
    - DATE columns → strftime('%Y-%m-%d', col) → 'YYYY-MM-DD' strings
    - TIMESTAMP with any non-midnight value → formatted with time component
    - TIMESTAMP with all-midnight values → formatted as date-only
    - Other columns → passed through unchanged

    The non-midnight check uses LIMIT 1 so DuckDB stops at the first match.
    """
    # Inspect schema without fetching data
    schema_rel = conn.execute(f'SELECT * FROM "{result_name}" LIMIT 0')
    col_types: Dict[str, str] = {}
    if schema_rel.description:
        for col_desc in schema_rel.description:
            col_types[col_desc[0]] = str(col_desc[1]).upper()

    # Column projection: follow component order, or all columns when unspecified
    if ds.components:
        ordered_cols = [c for c in ds.components if c in col_types]
    else:
        ordered_cols = list(col_types.keys())

    if not ordered_cols:
        return f'SELECT * FROM "{result_name}"'

    exprs = []
    for col in ordered_cols:
        col_type = col_types.get(col, "")

        if "TIMESTAMP" in col_type:
            has_time = (
                conn.execute(
                    f'SELECT 1 FROM "{result_name}" '
                    f'WHERE "{col}" IS NOT NULL '
                    f'AND (hour("{col}") != 0 OR minute("{col}") != 0 '
                    f'OR second("{col}") != 0 OR microsecond("{col}") % 1000000 != 0) '
                    f"LIMIT 1"
                ).fetchone()
                is not None
            )
            if has_time:
                exprs.append(
                    f'CASE WHEN "{col}" IS NULL THEN NULL'
                    f' WHEN microsecond("{col}") % 1000000 != 0'
                    f" THEN strftime('%Y-%m-%d %H:%M:%S', \"{col}\")"
                    f" || '.' || printf('%06d', microsecond(\"{col}\") % 1000000)"
                    f" ELSE strftime('%Y-%m-%d %H:%M:%S', \"{col}\")"
                    f' END AS "{col}"'
                )
            else:
                exprs.append(f'strftime(\'%Y-%m-%d\', "{col}") AS "{col}"')
        elif col_type == "DATE":
            exprs.append(f'strftime(\'%Y-%m-%d\', "{col}") AS "{col}"')
        else:
            exprs.append(f'"{col}"')

    return f'SELECT {", ".join(exprs)} FROM "{result_name}"'


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
        rel = conn.execute(f'SELECT * FROM "{result_name}"')
        result_df = rel.fetchdf()
        if len(result_df) == 1 and len(result_df.columns) == 1:
            scalar = output_scalars[result_name]
            raw_value = _normalize_scalar_value(result_df.iloc[0, 0])
            scalar.value = raw_value
            # When semantic analysis produced Null type but DuckDB resolved a concrete
            # type (e.g. nvl(null, 3) → INTEGER), override with the DuckDB type.
            # Only override when the actual value is non-null (DuckDB defaults NULL
            # expressions to INTEGER even when the result is NULL).
            if scalar.data_type is Null and raw_value is not None and rel.description:
                inferred = _infer_scalar_type_from_duckdb(rel.description[0][1])
                if inferred is not None:
                    scalar.data_type = inferred
            format_time_period_scalar(scalar, representation)
            return scalar
        return Dataset(name=result_name, components={}, data=result_df)

    # Save to CSV if output folder provided (table kept alive for fetch)
    if output_folder:
        save_datapoints_duckdb(conn, result_name, output_folder, delete_after_save=False)

    # Build fetch query: column projection + date/timestamp formatting inside DuckDB
    ds = output_datasets.get(result_name, Dataset(name=result_name, components={}, data=None))
    fetch_sql = _build_dataset_fetch_select(conn, result_name, ds)
    ds.data = conn.execute(fetch_sql).fetchdf()

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

    # Initialize VTL time types/macros only when required by inputs/outputs/SQL.
    if _requires_time_types_initialization(
        queries,
        input_datasets,
        output_datasets,
        output_scalars,
    ):
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
        except duckdb.Error as e:
            mapped = _map_query_error(e, sql_query)
            if mapped is not e:
                raise mapped from e
            raise
        except Exception:
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
