"""
Internal IO functions for DuckDB-based CSV loading and saving.

This module contains the core load/save implementations to avoid circular imports.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import duckdb
import pandas as pd

from vtlengine.DataTypes import TimePeriod
from vtlengine.duckdb_transpiler.io._validation import (
    build_create_table_sql,
    build_csv_column_types,
    build_select_columns,
    check_missing_identifiers,
    handle_sdmx_columns,
    map_duckdb_error,
    validate_csv_path,
    validate_no_duplicates,
    validate_temporal_columns,
)
from vtlengine.Exceptions import DataLoadError, InputValidationException
from vtlengine.Model import Component, Dataset, Role

# Environment variable to skip post-load validations (for benchmarking)
SKIP_LOAD_VALIDATION = os.environ.get("VTL_SKIP_LOAD_VALIDATION", "").lower() in (
    "1",
    "true",
    "yes",
)


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

    # 1. Create table (NOT NULL only, no PRIMARY KEY)
    conn.execute(build_create_table_sql(dataset_name, components))

    try:
        # 2. Detect CSV format (delimiter, quote, escape) using sniff_csv
        _sniffed_fmt = _detect_csv_format(conn, csv_path)

        # 3. Read CSV header with auto_detect to get column names
        header_rel = conn.sql(
            f"SELECT * FROM read_csv('{csv_path}', header=true, auto_detect=true,"
            f" null_padding=true) LIMIT 0"
        )
        csv_columns = header_rel.columns

        # 4. Handle SDMX-CSV special columns
        keep_columns = handle_sdmx_columns(csv_columns, components)

        # Check required identifier columns exist
        check_missing_identifiers(id_columns, keep_columns, csv_path)

        # 5. Build column type mapping and SELECT expressions
        csv_dtypes = build_csv_column_types(components, keep_columns)
        select_cols = build_select_columns(components, keep_columns, csv_dtypes, dataset_name)

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

        # 9. Normalize TimePeriod columns to canonical internal representation
        _normalize_time_period_columns(conn, dataset_name, components)

    except duckdb.Error as e:
        conn.execute(f'DROP TABLE IF EXISTS "{dataset_name}"')
        raise map_duckdb_error(e, dataset_name, components)

    # 10. Validate constraints (can be skipped via VTL_SKIP_LOAD_VALIDATION for benchmarking)
    if not SKIP_LOAD_VALIDATION:
        try:
            # DWI: no identifiers → max 1 row
            if not id_columns:
                result = conn.execute(f'SELECT COUNT(*) FROM "{dataset_name}"').fetchone()
                if result and result[0] > 1:
                    raise DataLoadError("0-3-1-4", name=dataset_name)

            # Duplicate check (GROUP BY HAVING)
            validate_no_duplicates(conn, dataset_name, id_columns)

            # Temporal type validation
            validate_temporal_columns(conn, dataset_name, components)

        except DataLoadError:
            conn.execute(f'DROP TABLE IF EXISTS "{dataset_name}"')
            raise

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
                # Convert to Path and store
                path_dict[name] = Path(value) if isinstance(value, str) else value
            else:
                raise InputValidationException(
                    f"Invalid datapoint for {name}. Must be DataFrame, Path, or string."
                )
        return path_dict if path_dict else None, df_dict

    # Handle list of paths
    if isinstance(datapoints, list):
        for item in datapoints:
            path = Path(item) if isinstance(item, str) else item
            # Extract dataset name from filename (without extension)
            name = path.stem
            if name in input_datasets:
                path_dict[name] = path
        return path_dict if path_dict else None, df_dict

    # Handle single path
    path = Path(datapoints) if isinstance(datapoints, str) else datapoints
    name = path.stem
    if name in input_datasets:
        path_dict[name] = path
    return path_dict if path_dict else None, df_dict


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

        # Create table with proper schema
        conn.execute(build_create_table_sql(name, components))

        # Register DataFrame and insert data
        temp_view = f"_temp_{name}"
        conn.register(temp_view, df)
        conn.execute(f'INSERT INTO "{name}" SELECT * FROM "{temp_view}"')
        conn.unregister(temp_view)

        # Normalize TimePeriod columns to canonical internal representation
        _normalize_time_period_columns(conn, name, components)
