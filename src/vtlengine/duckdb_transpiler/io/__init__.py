"""
DuckDB-based CSV IO optimized for out-of-core processing.

Public functions:
- load_datapoints_duckdb: Load CSV data into DuckDB table with validation
- save_datapoints_duckdb: Save DuckDB table to CSV file
"""

from pathlib import Path
from typing import Dict, Optional, Union

import duckdb

from vtlengine.Exceptions import DataLoadError
from vtlengine.Model import Component, Role

from ._validation import (
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

# =============================================================================
# CSV Loading Function
# =============================================================================


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
        # 2. Read CSV header
        header_rel = conn.sql(
            f"SELECT * FROM read_csv('{csv_path}', header=true, auto_detect=true) LIMIT 0"
        )
        csv_columns = header_rel.columns

        # 3. Handle SDMX-CSV special columns
        keep_columns = handle_sdmx_columns(csv_columns, components)

        # Check required identifier columns exist
        check_missing_identifiers(id_columns, keep_columns, csv_path)

        # 4. Build column type mapping and SELECT expressions
        csv_dtypes = build_csv_column_types(components, keep_columns)
        select_cols = build_select_columns(components, keep_columns, csv_dtypes, dataset_name)

        # 5. Build type string for read_csv
        type_str = ", ".join(f"'{k}': '{v}'" for k, v in csv_dtypes.items())

        # 6. Build filter for SDMX ACTION column
        action_filter = ""
        if "ACTION" in csv_columns and "ACTION" not in components:
            action_filter = 'WHERE "ACTION" != \'D\' OR "ACTION" IS NULL'

        # 7. Execute INSERT
        insert_sql = f"""
            INSERT INTO "{dataset_name}"
            SELECT {", ".join(select_cols)}
            FROM read_csv(
                '{csv_path}',
                header=true,
                columns={{{type_str}}},
                parallel=true,
                ignore_errors=false
            )
            {action_filter}
        """
        conn.execute(insert_sql)

    except duckdb.Error as e:
        conn.execute(f'DROP TABLE IF EXISTS "{dataset_name}"')
        raise map_duckdb_error(e, dataset_name, components)

    # 8. Validate constraints
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


# =============================================================================
# CSV Saving Function
# =============================================================================


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


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "load_datapoints_duckdb",
    "save_datapoints_duckdb",
]
