"""
Internal IO functions for DuckDB-based CSV loading and saving.

This module contains the core load/save implementations to avoid circular imports.
"""

import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import duckdb
import pandas as pd

from vtlengine.DataTypes import Date, TimePeriod
from vtlengine.duckdb_transpiler.io._validation import (
    build_create_table_sql,
    build_csv_column_types,
    build_select_columns,
    check_missing_identifiers,
    get_column_sql_type,
    handle_sdmx_columns,
    map_duckdb_error,
    validate_csv_path,
    validate_no_duplicates,
    validate_temporal_columns,
)
from vtlengine.Exceptions import DataLoadError, InputValidationException
from vtlengine.files.sdmx_handler import (
    extract_sdmx_dataset_name,
    is_sdmx_datapoint_file,
    load_sdmx_datapoints,
)
from vtlengine.Model import Component, Dataset, Role, Scalar

# Environment variable to skip post-load validations (for benchmarking)
SKIP_LOAD_VALIDATION = os.environ.get("VTL_SKIP_LOAD_VALIDATION", "").lower() in (
    "1",
    "true",
    "yes",
)


def _validate_loaded_table(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    components: Dict[str, Component],
) -> None:
    """Validate a loaded DuckDB table after data insertion.

    Runs the shared post-load validation checks:
    1. TimePeriod normalization to canonical format
    2. DWI check (no identifiers → max 1 row)
    3. Duplicate identifier check via GROUP BY HAVING
    4. Temporal type regex validation (TimePeriod, TimeInterval, Duration)

    On validation failure, drops the table and re-raises DataLoadError.
    Respects VTL_SKIP_LOAD_VALIDATION (skips checks 2-4 when set).
    """
    # Normalize TimePeriod columns to canonical internal representation
    _normalize_time_period_columns(conn, table_name, components)

    if SKIP_LOAD_VALIDATION:
        return

    try:
        id_columns = [n for n, c in components.items() if c.role == Role.IDENTIFIER]

        # DWI: no identifiers → max 1 row
        if not id_columns:
            result = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()
            if result and result[0] > 1:
                raise DataLoadError("0-3-1-4", name=table_name)

        # Duplicate check (GROUP BY HAVING)
        validate_no_duplicates(conn, table_name, id_columns)

        # Temporal type validation
        validate_temporal_columns(conn, table_name, components)

    except DataLoadError:
        conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        raise


def _normalize_time_period_columns(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    components: Dict[str, Component],
) -> None:
    """Normalize TimePeriod columns to the canonical internal representation.

    Converts all accepted input formats (#505) to the canonical format
    from TimePeriodHandler.__str__ using the vtl_period_normalize() macro.
    """
    for comp_name, comp in components.items():
        if comp.data_type == TimePeriod:
            conn.execute(
                f'UPDATE "{table_name}" SET "{comp_name}" = '
                f'vtl_period_normalize("{comp_name}") '
                f'WHERE "{comp_name}" IS NOT NULL'
            )


def _detect_csv_format(conn: duckdb.DuckDBPyConnection, csv_path: Path) -> str:
    """Detect CSV delimiter, quote and escape using sniff_csv.

    Returns a string of read_csv format options (e.g. "delim=',', quote='\"', escape='\"'").
    Falls back to defaults if sniffing fails or produces unreliable results.
    """
    try:
        sniff_result = conn.sql(
            f'SELECT "Delimiter", "Quote", "Escape" FROM sniff_csv(\'{csv_path}\')'
        ).fetchone()
    except duckdb.Error:
        return "delim=','"

    if not sniff_result:
        return "delim=','"

    csv_delimiter = sniff_result[0] or ","
    csv_quote = sniff_result[1] or ""
    csv_escape = sniff_result[2] or ""

    # Validate: read header with sniffed delimiter and compare to auto_detect
    try:
        auto_cols = conn.sql(
            f"SELECT * FROM read_csv('{csv_path}', header=true, auto_detect=true,"
            f" null_padding=true) LIMIT 0"
        ).columns

        sniff_cols = conn.sql(
            f"SELECT * FROM read_csv('{csv_path}', header=true, auto_detect=true,"
            f" delim='{csv_delimiter}', null_padding=true) LIMIT 0"
        ).columns

        if list(sniff_cols) != list(auto_cols):
            # Sniffed delimiter disagrees with auto_detect — fall back to auto_detect delimiter
            csv_delimiter = ","
    except duckdb.Error:
        csv_delimiter = ","

    fmt_parts = [f"delim='{csv_delimiter}'"]
    if csv_quote and csv_quote != "(empty)":
        esc_quote = csv_quote.replace("'", "\\'")
        fmt_parts.append(f"quote='{esc_quote}'")
    if csv_escape and csv_escape != "(empty)":
        esc_escape = csv_escape.replace("'", "\\'")
        fmt_parts.append(f"escape='{esc_escape}'")
    return ", ".join(fmt_parts)


def load_datapoints_duckdb(
    conn: duckdb.DuckDBPyConnection,
    components: Dict[str, Component],
    dataset_name: str,
    csv_path: Optional[Union[Path, str]] = None,
) -> duckdb.DuckDBPyRelation:
    """
    Load CSV data into DuckDB table with optimized validation.

    Validation Strategy:
    1. CREATE TABLE with NOT NULL constraints (no PRIMARY KEY for memory efficiency)
    2. Load CSV with explicit types → DuckDB validates types on load
    3. Post-hoc duplicate check via GROUP BY HAVING COUNT > 1
    4. Temporal types validated via regex (TimePeriod, TimeInterval, Duration)
    5. DWI check (no identifiers → max 1 row)

    Args:
        conn: DuckDB connection
        components: Dataset component definitions
        dataset_name: Name for the table
        csv_path: Path to CSV file (None for empty table)

    Returns:
        DuckDB relation pointing to the created table

    Raises:
        DataLoadError: If validation fails
    """
    # Handle empty dataset
    if csv_path is None:
        return _create_empty_table(conn, components, dataset_name)

    csv_path = Path(csv_path) if isinstance(csv_path, str) else csv_path
    if not csv_path.exists():
        return _create_empty_table(conn, components, dataset_name)

    validate_csv_path(csv_path)

    # Get identifier columns (needed for duplicate validation)
    id_columns = [n for n, c in components.items() if c.role == Role.IDENTIFIER]

    # For CSV, Date columns use TIMESTAMP as safe default (can't inspect values cheaply)
    csv_date_overrides = {n: "TIMESTAMP" for n, c in components.items() if c.data_type == Date}

    # 1. Create table (NOT NULL only, no PRIMARY KEY)
    conn.execute(build_create_table_sql(dataset_name, components, csv_date_overrides))

    try:
        # 2. Detect CSV format (delimiter, quote, escape) using sniff_csv
        _sniffed_fmt = _detect_csv_format(conn, csv_path)

        # 3. Read CSV header and check for duplicate columns
        sniffed_delim = _sniffed_fmt.split("'")[1] if "delim=" in _sniffed_fmt else ","
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=sniffed_delim)
            csv_columns = next(reader, [])

        if len(set(csv_columns)) != len(csv_columns):
            duplicates = list({item for item in csv_columns if csv_columns.count(item) > 1})
            raise InputValidationException(
                code="0-1-2-3",
                element_type="Columns",
                element=f"{', '.join(duplicates)}",
            )

        # 4. Handle SDMX-CSV special columns
        keep_columns = handle_sdmx_columns(csv_columns, components)

        # Check required identifier columns exist
        check_missing_identifiers(id_columns, keep_columns, csv_path)

        # 5. Build column type mapping and SELECT expressions
        csv_dtypes = build_csv_column_types(components, keep_columns)
        select_cols = build_select_columns(
            components, keep_columns, csv_dtypes, dataset_name, csv_date_overrides
        )

        # 6. Build type string for read_csv (must include ALL CSV columns)
        # Include extra SDMX columns (DATAFLOW, ACTION, etc.) as VARCHAR so
        # the columns parameter matches the actual CSV column count.
        all_csv_dtypes = dict(csv_dtypes)
        for col in csv_columns:
            if col not in all_csv_dtypes:
                all_csv_dtypes[col] = "VARCHAR"
        # Preserve original CSV column order for read_csv
        ordered_dtypes = {col: all_csv_dtypes[col] for col in csv_columns if col in all_csv_dtypes}
        type_str = ", ".join(f"'{k}': '{v}'" for k, v in ordered_dtypes.items())

        # 7. Build filter for SDMX ACTION column
        action_filter = ""
        if "ACTION" in csv_columns and "ACTION" not in components:
            action_filter = 'WHERE "ACTION" != \'D\' OR "ACTION" IS NULL'

        # 8. Execute INSERT
        insert_sql = f"""
            INSERT INTO "{dataset_name}"
            SELECT {", ".join(select_cols)}
            FROM read_csv(
                '{csv_path}',
                header=true,
                columns={{{type_str}}},
                auto_detect=false,
                {_sniffed_fmt},
                null_padding=true,
                parallel=true,
                ignore_errors=false
            )
            {action_filter}
        """
        conn.execute(insert_sql)

    except duckdb.Error as e:
        conn.execute(f'DROP TABLE IF EXISTS "{dataset_name}"')
        raise map_duckdb_error(e, dataset_name, components)

    # Post-load: normalize TimePeriod + validate constraints
    _validate_loaded_table(conn, dataset_name, components)

    return conn.table(dataset_name)


def _create_empty_table(
    conn: duckdb.DuckDBPyConnection,
    components: Dict[str, Component],
    table_name: str,
) -> duckdb.DuckDBPyRelation:
    """Create empty table with proper schema."""
    conn.execute(build_create_table_sql(table_name, components))
    return conn.table(table_name)


def save_datapoints_duckdb(
    conn: duckdb.DuckDBPyConnection,
    dataset_name: str,
    output_path: Union[Path, str],
    delete_after_save: bool = True,
) -> None:
    """
    Save dataset to CSV using DuckDB's COPY TO.

    Args:
        conn: DuckDB connection
        dataset_name: Name of the table to save
        output_path: Directory path where CSV will be saved
        delete_after_save: If True, drop table after saving to free memory

    The CSV is saved with:
    - Header row present
    - No index column
    - Comma delimiter
    """
    output_path = Path(output_path) if isinstance(output_path, str) else output_path
    output_file = output_path / f"{dataset_name}.csv"

    copy_sql = f"""
        COPY "{dataset_name}"
        TO '{output_file}'
        WITH (HEADER true, DELIMITER ',')
    """
    conn.execute(copy_sql)

    if delete_after_save:
        conn.execute(f'DROP TABLE IF EXISTS "{dataset_name}"')


def save_scalars_duckdb(
    scalars: Dict[str, Scalar],
    output_path: Union[Path, str],
) -> None:
    """Save scalar results to a _scalars.csv file.

    Args:
        scalars: Dict mapping scalar names to Scalar objects
        output_path: Directory path where _scalars.csv will be saved
    """
    if not scalars:
        return
    output_path = Path(output_path) if isinstance(output_path, str) else output_path
    file_path = output_path / "_scalars.csv"
    with open(file_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["name", "value"])
        for name, scalar in sorted(scalars.items(), key=lambda item: item[0]):
            value_to_write = "" if scalar.value is None else scalar.value
            writer.writerow([name, str(value_to_write)])


def extract_datapoint_paths(
    datapoints: Optional[
        Union[Dict[str, Union[pd.DataFrame, str, Path]], List[Union[str, Path]], str, Path]
    ],
    input_datasets: Dict[str, Dataset],
) -> Tuple[Optional[Dict[str, Path]], Dict[str, pd.DataFrame]]:
    """
    Extract CSV paths and DataFrames from datapoints without pandas validation.

    This function is optimized for DuckDB execution - it only extracts paths
    without loading or validating data. DuckDB will validate during its native CSV load.

    Args:
        datapoints: Dict of DataFrames/paths, list of paths, or single path
        input_datasets: Dict of input dataset structures (for validation)

    Returns:
        Tuple of (path_dict, dataframe_dict):
        - path_dict: Dict mapping dataset names to CSV Paths (None if no paths)
        - dataframe_dict: Dict mapping dataset names to DataFrames (for direct registration)

    Raises:
        InputValidationException: If dataset name not found in structures
    """
    if datapoints is None:
        return None, {}

    path_dict: Dict[str, Path] = {}
    df_dict: Dict[str, pd.DataFrame] = {}

    # Handle dictionary input
    if isinstance(datapoints, dict):
        for name, value in datapoints.items():
            if name not in input_datasets:
                raise InputValidationException(f"Not found dataset {name} in datastructures.")

            if value is None:
                # No datapoints for this dataset (e.g. semantic-only test)
                continue
            elif isinstance(value, pd.DataFrame):
                # Store DataFrame for direct DuckDB registration
                df_dict[name] = value
            elif isinstance(value, (str, Path)):
                path = Path(value) if isinstance(value, str) else value
                # Check if this is an SDMX file — load via pysdmx into DataFrame
                if is_sdmx_datapoint_file(path):
                    try:
                        components = input_datasets[name].components
                        sdmx_df = load_sdmx_datapoints(components, name, path)
                        df_dict[name] = sdmx_df
                        continue
                    except Exception:  # noqa: S110
                        pass  # Fall through to treat as regular file
                path_dict[name] = path
            else:
                raise InputValidationException(
                    f"Invalid datapoint for {name}. Must be DataFrame, Path, or string."
                )
        return path_dict if path_dict else None, df_dict

    # Handle list of paths
    if isinstance(datapoints, list):
        for item in datapoints:
            path = Path(item) if isinstance(item, str) else item
            # Check if this is an SDMX file — load via pysdmx into DataFrame
            if is_sdmx_datapoint_file(path):
                try:
                    sdmx_name = extract_sdmx_dataset_name(path)
                    if sdmx_name in input_datasets:
                        components = input_datasets[sdmx_name].components
                        sdmx_df = load_sdmx_datapoints(components, sdmx_name, path)
                        df_dict[sdmx_name] = sdmx_df
                        continue
                except Exception:  # noqa: S110
                    pass  # Fall through to treat as regular file
            # Extract dataset name from filename (without extension)
            name = path.stem
            if name in input_datasets:
                path_dict[name] = path
        return path_dict if path_dict else None, df_dict

    # Handle single path
    path = Path(datapoints) if isinstance(datapoints, str) else datapoints
    # Check if this is an SDMX file — load via pysdmx into DataFrame
    if is_sdmx_datapoint_file(path):
        try:
            sdmx_name = extract_sdmx_dataset_name(path)
            if sdmx_name in input_datasets:
                components = input_datasets[sdmx_name].components
                sdmx_df = load_sdmx_datapoints(components, sdmx_name, path)
                df_dict[sdmx_name] = sdmx_df
                return None, df_dict
        except Exception:  # noqa: S110
            pass  # Fall through to treat as regular file
    name = path.stem
    if name in input_datasets:
        path_dict[name] = path
    return path_dict if path_dict else None, df_dict


def _detect_date_type_overrides(
    df: pd.DataFrame, components: Dict[str, Component]
) -> Dict[str, str]:
    """Determine which Date columns need TIMESTAMP instead of DATE.

    Inspects actual string values: if any value in a Date column has a time
    component (length > 10 with 'T' or ' ' separator), the column is stored
    as TIMESTAMP to preserve the time part. Otherwise DATE is used.
    """
    overrides: Dict[str, str] = {}
    for comp_name, comp in components.items():
        if comp.data_type != Date or comp_name not in df.columns:
            continue
        for val in df[comp_name].dropna():
            if isinstance(val, str) and len(val) > 10 and val[10] in ("T", " "):
                overrides[comp_name] = "TIMESTAMP"
                break
    return overrides


def _build_dataframe_select_columns(
    components: Dict[str, Component],
    df_columns: Optional[List[str]] = None,
    type_overrides: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Build SELECT expressions with explicit CAST for DataFrame → DuckDB table insertion.

    Ensures type enforcement matches the CSV loading path (load_datapoints_duckdb).
    Columns missing from the DataFrame are filled with NULL.
    """
    df_col_set = set(df_columns) if df_columns is not None else None
    overrides = type_overrides or {}
    exprs: List[str] = []
    for comp_name, comp in components.items():
        target_type = overrides.get(comp_name, get_column_sql_type(comp))
        if df_col_set is not None and comp_name not in df_col_set:
            exprs.append(f'CAST(NULL AS {target_type}) AS "{comp_name}"')
        else:
            exprs.append(f'CAST("{comp_name}" AS {target_type}) AS "{comp_name}"')
    return exprs


def register_dataframes(
    conn: duckdb.DuckDBPyConnection,
    dataframes: Dict[str, pd.DataFrame],
    input_datasets: Dict[str, Dataset],
) -> None:
    """
    Register DataFrames directly with DuckDB connection.

    Creates tables from DataFrames with proper schema based on dataset components.

    Args:
        conn: DuckDB connection
        dataframes: Dict mapping dataset names to DataFrames
        input_datasets: Dict of input dataset structures
    """
    for name, df in dataframes.items():
        if name not in input_datasets:
            continue

        components = input_datasets[name].components

        # Detect Date columns that contain time values → TIMESTAMP instead of DATE
        type_overrides = _detect_date_type_overrides(df, components)

        # Create table with proper schema
        conn.execute(build_create_table_sql(name, components, type_overrides))

        # Register DataFrame and insert data with explicit type casting
        temp_view = f"_temp_{name}"
        conn.register(temp_view, df)
        try:
            select_exprs = _build_dataframe_select_columns(
                components, list(df.columns), type_overrides
            )
            col_list = ", ".join(f'"{c}"' for c in components)
            conn.execute(
                f'INSERT INTO "{name}" ({col_list}) '
                f'SELECT {", ".join(select_exprs)} FROM "{temp_view}"'
            )
        except duckdb.Error as e:
            conn.execute(f'DROP TABLE IF EXISTS "{name}"')
            raise map_duckdb_error(e, name, components)
        finally:
            conn.unregister(temp_view)

        # Post-load: normalize TimePeriod + validate constraints
        _validate_loaded_table(conn, name, components)
