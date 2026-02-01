"""
DuckDB-based CSV parser optimized for out-of-core processing.

Validation Strategy (optimized for large datasets):
1. CREATE TABLE with NOT NULL constraints (no PRIMARY KEY)
2. Load CSV with explicit types → DuckDB validates types on load
3. Post-hoc duplicate validation via GROUP BY HAVING COUNT > 1
4. Explicit validation: temporal types (TimePeriod, TimeInterval, Duration)
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import duckdb

from vtlengine.DataTypes import (
    Boolean,
    Date,
    Duration,
    Integer,
    Number,
    TimeInterval,
    TimePeriod,
)
from vtlengine.DataTypes.TimeHandling import PERIOD_IND_MAPPING
from vtlengine.duckdb_transpiler.Config.config import get_decimal_type
from vtlengine.Exceptions import DataLoadError, InputValidationException
from vtlengine.Model import Component, Role

# =============================================================================
# Regex patterns for VTL temporal types (only these need explicit validation)
# =============================================================================

TIME_PERIOD_PATTERN = (
    r"^\d{4}[A]?$|"  # Year - 2024 or 2024A
    r"^\d{4}[S][1-2]$|"  # Semester - 2024S1
    r"^\d{4}[Q][1-4]$|"  # Quarter - 2024Q1
    r"^\d{4}[M](0[1-9]|1[0-2])$|"  # Month - 2024M01
    r"^\d{4}[W](0[1-9]|[1-4][0-9]|5[0-3])$|"  # Week - 2024W01
    r"^\d{4}[D](00[1-9]|0[1-9][0-9]|[1-2][0-9][0-9]|3[0-5][0-9]|36[0-6])$"  # Day
)

TIME_INTERVAL_PATTERN = (
    r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2})?/"
    r"\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2})?$"
)


# =============================================================================
# Error Mapping
# =============================================================================


def _map_duckdb_error(
    error: duckdb.Error,
    dataset_name: str,
    components: Dict[str, Component],
) -> Exception:
    """
    Map DuckDB constraint errors to VTL error codes.

    DuckDB error patterns:
    - PRIMARY KEY violation: "Duplicate key" or "PRIMARY KEY"
    - NOT NULL violation: "NOT NULL constraint failed" or "cannot be null"
    - Type conversion: "Could not convert" or "Conversion Error"
    """
    error_msg = str(error).lower()

    # Duplicate key (PRIMARY KEY violation)
    if "duplicate" in error_msg or "primary key" in error_msg:
        return DataLoadError("0-3-1-7", name=dataset_name, row_index="unknown")

    # NULL in identifier (NOT NULL violation)
    if "null" in error_msg and "constraint" in error_msg:
        # Try to extract column name from error
        for comp_name, comp in components.items():
            if comp.role == Role.IDENTIFIER and comp_name.lower() in error_msg:
                return DataLoadError("0-3-1-3", null_identifier=comp_name, name=dataset_name)
        # Generic null error for identifier
        return DataLoadError("0-3-1-3", null_identifier="unknown", name=dataset_name)

    # Type conversion error
    if "convert" in error_msg or "conversion" in error_msg or "cast" in error_msg:
        # Try to extract column and type info
        for comp_name, comp in components.items():
            if comp_name.lower() in error_msg:
                type_name = (
                    comp.data_type.__name__
                    if hasattr(comp.data_type, "__name__")
                    else str(comp.data_type)
                )
                return DataLoadError(
                    "0-3-1-6",
                    name=dataset_name,
                    column=comp_name,
                    type=type_name,
                    error=str(error),
                )
        return DataLoadError(
            "0-3-1-6",
            name=dataset_name,
            column="unknown",
            type="unknown",
            error=str(error),
        )

    # Generic data load error
    return DataLoadError("0-3-1-6", name=dataset_name, column="", type="", error=str(error))


# =============================================================================
# Column Type Mapping
# =============================================================================


def _get_column_sql_type(comp: Component) -> str:
    """
    Get SQL type for a component with special handling for VTL types.

    - Integer → BIGINT
    - Number → DECIMAL(precision, scale) from config
    - Boolean → BOOLEAN
    - Date → DATE
    - TimePeriod, TimeInterval, Duration, String → VARCHAR
    """
    if comp.data_type == Integer:
        return "BIGINT"
    elif comp.data_type == Number:
        return get_decimal_type()
    elif comp.data_type == Boolean:
        return "BOOLEAN"
    elif comp.data_type == Date:
        return "DATE"
    else:
        # String, TimePeriod, TimeInterval, Duration → VARCHAR
        return "VARCHAR"


def _get_csv_read_type(comp: Component) -> str:
    """
    Get type for CSV reading. DuckDB read_csv needs slightly different types.

    For temporal strings (TimePeriod, etc.) we read as VARCHAR.
    For numerics, we let DuckDB parse directly.
    """
    if comp.data_type == Integer:
        return "BIGINT"
    elif comp.data_type == Number:
        return "DOUBLE"  # Read as DOUBLE, then cast to DECIMAL in table
    elif comp.data_type == Boolean:
        return "BOOLEAN"
    elif comp.data_type == Date:
        return "DATE"
    else:
        return "VARCHAR"


# =============================================================================
# Table Creation
# =============================================================================


def _build_create_table_sql(table_name: str, components: Dict[str, Component]) -> str:
    """
    Build CREATE TABLE statement with NOT NULL constraints only.

    No PRIMARY KEY - duplicate validation is done post-hoc via GROUP BY.
    This is more memory-efficient for large datasets.
    """
    col_defs: List[str] = []

    for comp_name, comp in components.items():
        sql_type = _get_column_sql_type(comp)

        if comp.role == Role.IDENTIFIER or not comp.nullable:
            col_defs.append(f'"{comp_name}" {sql_type} NOT NULL')
        else:
            col_defs.append(f'"{comp_name}" {sql_type}')

    return f'CREATE TABLE "{table_name}" ({", ".join(col_defs)})'


def _validate_no_duplicates(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    id_columns: List[str],
) -> None:
    """
    Validate no duplicate rows exist using GROUP BY HAVING COUNT > 1.

    This is faster than COUNT DISTINCT because it can short-circuit
    as soon as it finds a single duplicate (LIMIT 1).
    """
    if not id_columns:
        return  # DWI check handles this case

    id_list = ", ".join(f'"{c}"' for c in id_columns)
    check_sql = f"""
        SELECT 1 FROM "{table_name}"
        GROUP BY {id_list}
        HAVING COUNT(*) > 1
        LIMIT 1
    """

    result = conn.execute(check_sql).fetchone()
    if result:
        raise DataLoadError("0-3-1-7", name=table_name, row_index="(duplicate keys detected)")


# =============================================================================
# CSV Loading Helpers
# =============================================================================


def _validate_csv_path(csv_path: Path) -> None:
    """Validate CSV file exists."""
    if not csv_path.exists() or not csv_path.is_file():
        raise DataLoadError(code="0-3-1-1", file=csv_path)


def _build_csv_column_types(
    components: Dict[str, Component],
    csv_columns: List[str],
) -> Dict[str, str]:
    """
    Build column type mapping for CSV reading.
    Only include columns that exist in both CSV and components.
    """
    dtypes = {}
    for col in csv_columns:
        if col in components:
            dtypes[col] = _get_csv_read_type(components[col])
    return dtypes


def _handle_sdmx_columns(columns: List[str], components: Dict[str, Component]) -> List[str]:
    """
    Identify SDMX-CSV special columns to exclude.
    Returns list of columns to keep.
    """
    exclude = set()

    # DATAFLOW - drop if first column and not in structure
    if columns and columns[0] == "DATAFLOW" and "DATAFLOW" not in components:
        exclude.add("DATAFLOW")

    # STRUCTURE columns
    if "STRUCTURE" in columns and "STRUCTURE" not in components:
        exclude.add("STRUCTURE")
    if "STRUCTURE_ID" in columns and "STRUCTURE_ID" not in components:
        exclude.add("STRUCTURE_ID")

    # ACTION column (handled specially - need to filter, not just exclude)
    if "ACTION" in columns and "ACTION" not in components:
        exclude.add("ACTION")

    return [c for c in columns if c not in exclude]


# =============================================================================
# Temporal Validation (only explicit validation needed)
# =============================================================================


def _validate_temporal_columns(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    components: Dict[str, Component],
) -> None:
    """
    Validate temporal type columns using SQL regex.

    This is the ONLY explicit validation needed because:
    - Integer/Number: DuckDB validates on CSV read
    - Date: DuckDB validates on CSV read
    - Boolean: DuckDB validates on CSV read
    - Duplicates: PRIMARY KEY constraint validates
    - Nulls in identifiers: NOT NULL constraint validates
    - TimePeriod/TimeInterval/Duration: Stored as VARCHAR, need regex validation
    """
    temporal_checks = []

    for comp_name, comp in components.items():
        if comp.data_type == TimePeriod:
            temporal_checks.append((comp_name, TIME_PERIOD_PATTERN, "Time_Period"))
        elif comp.data_type == TimeInterval:
            temporal_checks.append((comp_name, TIME_INTERVAL_PATTERN, "Time"))
        elif comp.data_type == Duration:
            # Duration must be one of: A, S, Q, M, W, D
            valid_durations = "|".join(PERIOD_IND_MAPPING.keys())
            pattern = f"^({valid_durations})$"
            temporal_checks.append((comp_name, pattern, "Duration"))

    if not temporal_checks:
        return

    # Single query to check all temporal columns at once
    # Returns first invalid value found for any column
    case_expressions = []
    for col_name, pattern, type_name in temporal_checks:
        case_expressions.append(f"""
            CASE WHEN "{col_name}" IS NOT NULL AND "{col_name}" != ''
                 AND NOT regexp_matches(UPPER(TRIM("{col_name}")), '{pattern}')
            THEN '{col_name}|{type_name}|' || "{col_name}"
            ELSE NULL END
        """)

    # Use COALESCE to get first non-null (first invalid)
    coalesce_expr = ", ".join(case_expressions)
    check_query = f"""
        SELECT COALESCE({coalesce_expr}) as invalid
        FROM "{table_name}"
        WHERE COALESCE({coalesce_expr}) IS NOT NULL
        LIMIT 1
    """

    result = conn.execute(check_query).fetchone()
    if result and result[0]:
        # Parse "column|type|value" format
        parts = result[0].split("|", 2)
        col_name, type_name, invalid_value = parts[0], parts[1], parts[2]
        raise DataLoadError(
            "0-3-1-6",
            name=table_name,
            column=col_name,
            type=type_name,
            error=f"Invalid format: '{invalid_value}'",
        )


# =============================================================================
# Main Loading Function
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

    _validate_csv_path(csv_path)

    # Get identifier columns (needed for duplicate validation)
    id_columns = [n for n, c in components.items() if c.role == Role.IDENTIFIER]

    # 1. Create table (NOT NULL only, no PRIMARY KEY)
    conn.execute(_build_create_table_sql(dataset_name, components))

    try:
        # 2. Read CSV header
        header_rel = conn.sql(
            f"SELECT * FROM read_csv('{csv_path}', header=true, auto_detect=true) LIMIT 0"
        )
        csv_columns = header_rel.columns

        # 3. Handle SDMX-CSV special columns
        keep_columns = _handle_sdmx_columns(csv_columns, components)

        # Check required identifier columns exist
        missing_ids = set(id_columns) - set(keep_columns)
        if missing_ids:
            raise InputValidationException(
                code="0-1-1-8",
                ids=", ".join(missing_ids),
                file=str(csv_path.name),
            )

        # 4. Build column type mapping and SELECT expressions
        csv_dtypes = _build_csv_column_types(components, keep_columns)
        select_cols = _build_select_columns(components, keep_columns, csv_dtypes, dataset_name)

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
        raise _map_duckdb_error(e, dataset_name, components)

    # 8. Validate constraints
    try:
        # DWI: no identifiers → max 1 row
        if not id_columns:
            result = conn.execute(f'SELECT COUNT(*) FROM "{dataset_name}"').fetchone()
            if result and result[0] > 1:
                raise DataLoadError("0-3-1-4", name=dataset_name)

        # Duplicate check (GROUP BY HAVING)
        _validate_no_duplicates(conn, dataset_name, id_columns)

        # Temporal type validation
        _validate_temporal_columns(conn, dataset_name, components)

    except DataLoadError:
        conn.execute(f'DROP TABLE IF EXISTS "{dataset_name}"')
        raise

    return conn.table(dataset_name)


def _build_select_columns(
    components: Dict[str, Component],
    keep_columns: List[str],
    csv_dtypes: Dict[str, str],
    dataset_name: str,
) -> List[str]:
    """Build SELECT column expressions with type casting."""
    select_cols = []

    for comp_name, comp in components.items():
        if comp_name in keep_columns:
            csv_type = csv_dtypes.get(comp_name, "VARCHAR")
            table_type = _get_column_sql_type(comp)

            # Cast DOUBLE → DECIMAL for Number type
            if csv_type == "DOUBLE" and "DECIMAL" in table_type:
                select_cols.append(f'CAST("{comp_name}" AS {table_type}) AS "{comp_name}"')
            else:
                select_cols.append(f'"{comp_name}"')
        else:
            # Missing column → NULL (only allowed for nullable)
            if comp.nullable:
                table_type = _get_column_sql_type(comp)
                select_cols.append(f'NULL::{table_type} AS "{comp_name}"')
            else:
                raise DataLoadError("0-3-1-5", name=dataset_name, comp_name=comp_name)

    return select_cols


def _create_empty_table(
    conn: duckdb.DuckDBPyConnection,
    components: Dict[str, Component],
    table_name: str,
) -> duckdb.DuckDBPyRelation:
    """Create empty table with proper schema."""
    conn.execute(_build_create_table_sql(table_name, components))
    return conn.table(table_name)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "load_datapoints_duckdb",
]
