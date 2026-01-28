"""
DuckDB-based CSV parser for VTL Engine.

Provides optimized out-of-core CSV loading with:
- VTL type validation (temporal types, constraints)
- SDMX-CSV format support
- Duplicate key and NULL constraint checking
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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
from vtlengine.Exceptions import DataLoadError, InputValidationException
from vtlengine.Model import Component, Role

# =============================================================================
# Regex patterns for VTL temporal types (only these need explicit validation)
# =============================================================================

# VTL 2.1 Time_Period formats: YYYY, YYYYA, YYYYSn, YYYYQn, YYYYMnn, YYYYWnn, YYYYDnnn
TIME_PERIOD_PATTERN = (
    r"^\d{4}[A]?$|"  # Year
    r"^\d{4}[S][1-2]$|"  # Semester
    r"^\d{4}[Q][1-4]$|"  # Quarter
    r"^\d{4}[M](0[1-9]|1[0-2])$|"  # Month
    r"^\d{4}[W](0[1-9]|[1-4][0-9]|5[0-3])$|"  # Week
    r"^\d{4}[D](00[1-9]|0[1-9][0-9]|[1-2][0-9][0-9]|3[0-5][0-9]|36[0-6])$"  # Day
)

# ISO 8601 interval: start_datetime/end_datetime
TIME_INTERVAL_PATTERN = (
    r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2})?/"
    r"\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2})?$"
)

# SDMX-CSV columns to exclude (not part of VTL data structure)
SDMX_METADATA_COLUMNS = {"DATAFLOW", "STRUCTURE", "STRUCTURE_ID", "ACTION"}


# =============================================================================
# Error Mapping
# =============================================================================


def _map_duckdb_error(
    error: duckdb.Error,
    dataset_name: str,
    components: Dict[str, Component],
) -> DataLoadError:
    """
    Map DuckDB errors to VTL DataLoadError codes.

    Error codes:
        0-3-1-7: Duplicate primary key
        0-3-1-3: NULL in identifier column
        0-3-1-6: Type conversion / generic load error
    """
    msg = str(error).lower()

    if "duplicate" in msg or "primary key" in msg:
        return DataLoadError("0-3-1-7", name=dataset_name, row_index="unknown")

    if "null" in msg and "constraint" in msg:
        # Find which identifier column caused the error
        col = next(
            (n for n, c in components.items() if c.role == Role.IDENTIFIER and n.lower() in msg),
            "unknown",
        )
        return DataLoadError("0-3-1-3", null_identifier=col, name=dataset_name)

    if any(kw in msg for kw in ("convert", "conversion", "cast")):
        for comp_name, comp in components.items():
            if comp_name.lower() in msg:
                type_name = getattr(comp.data_type, "__name__", str(comp.data_type))
                return DataLoadError(
                    "0-3-1-6",
                    name=dataset_name,
                    column=comp_name,
                    type=type_name,
                    error=str(error),
                )

    return DataLoadError("0-3-1-6", name=dataset_name, column="", type="", error=str(error))


# =============================================================================
# VTL → SQL Type Mapping
# =============================================================================

# VTL to DuckDB SQL type mapping
SQL_TYPE_MAPPING = {
    Integer: "BIGINT",
    Number: "DOUBLE",
    Boolean: "BOOLEAN",
    Date: "DATE",
}


def _get_sql_type(comp: Component) -> str:
    return SQL_TYPE_MAPPING.get(comp.data_type, "VARCHAR")


# =============================================================================
# Table Creation & Validation
# =============================================================================


def _build_create_table_sql(table_name: str, components: Dict[str, Component]) -> str:
    """Build CREATE TABLE DDL with NOT NULL for identifiers and non-nullable columns."""
    col_defs = [
        f'"{name}" {_get_sql_type(comp)}{
            " NOT NULL" if comp.role == Role.IDENTIFIER or not comp.nullable else ""
        }'
        for name, comp in components.items()
    ]
    return f'CREATE TABLE "{table_name}" ({", ".join(col_defs)})'


def _validate_duplicates(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    id_columns: List[str],
) -> None:
    """
    Check for duplicate identifier combinations.

    Raises:
        DataLoadError: Code 0-3-1-7 if duplicates found.
    """
    if not id_columns:
        return

    id_list = ", ".join(f'"{c}"' for c in id_columns)
    result = conn.execute(
        f'SELECT 1 FROM "{table_name}" GROUP BY {id_list} HAVING COUNT(*) > 1 LIMIT 1'
    ).fetchone()

    if result:
        raise DataLoadError("0-3-1-7", name=table_name, row_index="(duplicate keys)")


# =============================================================================
# CSV Loading Helpers
# =============================================================================


def _filter_sdmx_columns(
    csv_columns: List[str],
    components: Dict[str, Component],
) -> Tuple[List[str], bool]:
    """
    Filter out SDMX-CSV metadata columns not in the VTL structure.

    Returns:
        Tuple of (filtered columns, has_action_column for row filtering).
    """
    # Only exclude if first column is DATAFLOW (SDMX-CSV indicator)
    if not csv_columns or csv_columns[0] != "DATAFLOW":
        return csv_columns, False

    has_action = "ACTION" in csv_columns and "ACTION" not in components
    filtered = [c for c in csv_columns if c not in SDMX_METADATA_COLUMNS or c in components]
    return filtered, has_action


# =============================================================================
# Temporal Validation
# =============================================================================

# Duration pattern built from valid indicators
DURATION_PATTERN = "^(" + "|".join(PERIOD_IND_MAPPING.keys()) + ")$"


def _get_temporal_checks(
    components: Dict[str, Component],
) -> List[Tuple[str, str, str]]:
    """Build list of (column, regex_pattern, type_name) for temporal validation."""
    checks = []
    for name, comp in components.items():
        if comp.data_type == TimePeriod:
            checks.append((name, TIME_PERIOD_PATTERN, "Time_Period"))
        elif comp.data_type == TimeInterval:
            checks.append((name, TIME_INTERVAL_PATTERN, "Time"))
        elif comp.data_type == Duration:
            checks.append((name, DURATION_PATTERN, "Duration"))
    return checks


def _validate_temporal_columns(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    components: Dict[str, Component],
) -> None:
    """
    Validate temporal columns match VTL format requirements.

    Raises:
        DataLoadError: Code 0-3-1-6 with first invalid value found.
    """
    checks = _get_temporal_checks(components)
    if not checks:
        return

    # Build CASE expressions that return 'col|type|value' on invalid match
    case_exprs = [
        f"""CASE WHEN "{col}" IS NOT NULL AND "{col}" != ''
                  AND NOT regexp_matches(UPPER(TRIM("{col}")), '{pattern}')
             THEN '{col}|{typ}|' || "{col}" END"""
        for col, pattern, typ in checks
    ]

    coalesce = ", ".join(case_exprs)
    result = conn.execute(
        f'SELECT COALESCE({coalesce}) FROM "{table_name}" '
        f"WHERE COALESCE({coalesce}) IS NOT NULL LIMIT 1"
    ).fetchone()

    if result and result[0]:
        col, typ, val = result[0].split("|", 2)
        raise DataLoadError(
            "0-3-1-6", name=table_name, column=col, type=typ, error=f"Invalid format: '{val}'"
        )


# =============================================================================
# Main Loading Method
# =============================================================================


def load_datapoints_duckdb(
    conn: duckdb.DuckDBPyConnection,
    components: Dict[str, Component],
    dataset_name: str,
    csv_path: Optional[Union[Path, str]] = None,
) -> duckdb.DuckDBPyRelation:
    """
    Load and validate CSV data into a DuckDB table.

    Performs full VTL validation:
    - Column presence (identifiers required)
    - Type conversion (VTL types to SQL)
    - NOT NULL constraints (identifiers cannot be null)
    - Duplicate key detection
    - Temporal format validation (TimePeriod, TimeInterval, Duration)

    SDMX-CSV format is auto-detected and metadata columns filtered.

    Args:
        conn: Active DuckDB connection.
        components: VTL component definitions from data structure.
        dataset_name: Table name to create.
        csv_path: CSV file path, or None for empty table.

    Returns:
        DuckDB relation pointing to the created/populated table.

    Raises:
        InputValidationException: Missing required columns (0-1-1-8).
        DataLoadError: Validation failures:
            - 0-3-1-1: File not found
            - 0-3-1-3: NULL in identifier
            - 0-3-1-4: DWI (Data Without Identifiers) has >1 row
            - 0-3-1-5: Missing non-nullable column
            - 0-3-1-6: Type conversion / format error
            - 0-3-1-7: Duplicate primary key
    """
    # Empty dataset case
    csv_path = Path(csv_path) if isinstance(csv_path, str) else csv_path
    if csv_path is None or not csv_path.exists():
        return _create_empty_table(conn, components, dataset_name)

    if not csv_path.is_file():
        raise DataLoadError(code="0-3-1-1", file=csv_path)

    id_columns = [n for n, c in components.items() if c.role == Role.IDENTIFIER]
    conn.execute(_build_create_table_sql(dataset_name, components))

    try:
        _insert_csv_data(conn, dataset_name, components, csv_path, id_columns)
    except duckdb.Error as e:
        conn.execute(f'DROP TABLE IF EXISTS "{dataset_name}"')
        raise _map_duckdb_error(e, dataset_name, components)

    # Post-load validation
    try:
        _validate_loaded_data(conn, dataset_name, components, id_columns)
    except DataLoadError:
        conn.execute(f'DROP TABLE IF EXISTS "{dataset_name}"')
        raise

    return conn.table(dataset_name)


def _insert_csv_data(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    components: Dict[str, Component],
    csv_path: Path,
    id_columns: List[str],
) -> None:
    """Read CSV and insert into table with type conversion."""
    # Get CSV header columns
    csv_columns = conn.sql(
        f"SELECT * FROM read_csv('{csv_path}', header=true, auto_detect=true) LIMIT 0"
    ).columns

    # Filter SDMX columns
    keep_columns, has_action = _filter_sdmx_columns(csv_columns, components)

    # Validate required identifiers present
    missing_ids = set(id_columns) - set(keep_columns)
    if missing_ids:
        raise InputValidationException(
            code="0-1-1-8", ids=", ".join(missing_ids), file=csv_path.name
        )

    # Build type mapping and SELECT expressions
    csv_dtypes = {col: _get_sql_type(components[col]) for col in keep_columns if col in components}
    select_cols = _build_select_columns(components, keep_columns, table_name)
    type_str = ", ".join(f"'{k}': '{v}'" for k, v in csv_dtypes.items())

    # SDMX ACTION='D' rows are deletions, filter them out
    action_filter = 'WHERE "ACTION" != \'D\' OR "ACTION" IS NULL' if has_action else ""

    conn.execute(f"""
        INSERT INTO "{table_name}"
        SELECT {", ".join(select_cols)}
        FROM read_csv('{csv_path}', header=true, columns={{{type_str}}}, parallel=true)
        {action_filter}
    """)


def _validate_loaded_data(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    components: Dict[str, Component],
    id_columns: List[str],
) -> None:
    """Run post-load validations: DWI check, duplicates, temporal formats."""
    # DWI: datasets without identifiers can have at most 1 row
    if not id_columns:
        count = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()
        if count and count[0] > 1:
            raise DataLoadError("0-3-1-4", name=table_name)
    else:
        _validate_duplicates(conn, table_name, id_columns)

    _validate_temporal_columns(conn, table_name, components)


def _build_select_columns(
    components: Dict[str, Component],
    csv_columns: List[str],
    table_name: str,
) -> List[str]:
    """
    Build SELECT expressions for INSERT with proper type casting.

    Handles:
    - DOUBLE → DECIMAL cast for Number type
    - NULL default for missing nullable columns

    Raises:
        DataLoadError: Code 0-3-1-5 if non-nullable column missing from CSV.
    """
    csv_set = set(csv_columns)
    select_cols = []

    for name, comp in components.items():
        if name in csv_set:
            select_cols.append(f'"{name}"')
        elif comp.nullable:
            select_cols.append(f'NULL::{_get_sql_type(comp)} AS "{name}"')
        else:
            raise DataLoadError("0-3-1-5", name=table_name, comp_name=name)

    return select_cols


def _create_empty_table(
    conn: duckdb.DuckDBPyConnection,
    components: Dict[str, Component],
    table_name: str,
) -> duckdb.DuckDBPyRelation:
    """Create empty table with schema from components."""
    conn.execute(_build_create_table_sql(table_name, components))
    return conn.table(table_name)
