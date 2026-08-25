"""
Internal validation helpers for DuckDB CSV and Parquet loading.

This module contains:
- Regex patterns for VTL temporal types
- Error mapping from DuckDB to VTL error codes
- Column type mapping functions
- Table creation and validation helpers
"""

from pathlib import Path
from typing import Dict, List, Optional

import duckdb

from vtlengine.DataTypes import (
    Boolean,
    Date,
    Duration,
    Integer,
    Number,
    String,
    TimeInterval,
    TimePeriod,
)
from vtlengine.Exceptions import DataLoadError, InputValidationException
from vtlengine.Model import Component, Role

# =============================================================================
# Regex patterns for VTL temporal types (only these need explicit validation)
# =============================================================================

TIME_PERIOD_PATTERN = (
    r"^\d{4}$|"  # Year - 2024
    r"^\d{4}A$|"  # Annual - 2024A
    r"^\d{4}[S][1-2]$|"  # Semester - 2024S1
    r"^\d{4}[Q][1-4]$|"  # Quarter - 2024Q1
    r"^\d{4}[M]\d{1,2}$|"  # Month - 2024M01, 2024M1
    r"^\d{4}[W]\d{1,2}$|"  # Week - 2024W01, 2024W1
    r"^\d{4}[D]\d{1,3}$|"  # Day - 2024D001, 2024D01, 2024D1
    # SDMX Gregorian formats (hyphen-separated)
    r"^\d{4}-\d{1,2}$|"  # Month numeric - 2024-01, 2024-1
    r"^\d{4}-A1$|"  # Annual - 2024-A1
    r"^\d{4}-S[1-2]$|"  # Semester - 2024-S1
    r"^\d{4}-Q[1-4]$|"  # Quarter - 2024-Q1
    r"^\d{4}-M\d{1,2}$|"  # Month - 2024-M01, 2024-M1
    r"^\d{4}-W\d{1,2}$|"  # Week - 2024-W01, 2024-W1
    r"^\d{4}-D\d{1,3}$|"  # Day - 2024-D001, 2024-D01, 2024-D1
    r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[1-2][0-9]|3[0-1])$"  # Full date - 2024-01-15
)

TIME_INTERVAL_PATTERN = (
    r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2})?/"
    r"\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2})?$"
)

DURATION_PATTERN = r"^(A|S|Q|M|W|D)$"  # Year, Semester, Quarter, Month, Week, Day

# Canonical Date/datetime input format shared by the CSV and DataFrame loaders and
# kept in sync with the pandas validator (_time_checking._STRICT_DATETIME_RE):
# a bare date, or a date + (T|space) + complete HH:MM:SS + optional fractional seconds
# + optional timezone (+HH:MM or Z). Month/day allow 1-2 digits (DuckDB casts them).
VALID_DATE_REGEX = (
    r"^\d{4}-\d{1,2}-\d{1,2}"
    r"([ T]([01]\d|2[0-3]):[0-5]\d:[0-5]\d(\.\d+)?([+-]\d{2}:\d{2}|Z)?)?$"
)


# =============================================================================
# Error Mapping
# =============================================================================


def map_duckdb_error(
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
    - Corrupt/invalid Parquet: "magic bytes" or "invalid input" in the message.
    """
    error_msg = str(error).lower()

    # Corrupt or invalid Parquet file
    if "magic bytes" in error_msg or "no magic bytes" in error_msg:
        return DataLoadError(
            "0-3-1-16",
            name=dataset_name,
            error=str(error),
        )

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

    # Date/timestamp range error (e.g. 2014-02-31)
    if "timestamp field value out of range" in error_msg:
        import re

        match = re.search(r'"(\d{4}-\d{2}-\d{2})"', str(error))
        date_val = match.group(1) if match else "unknown"
        friendly_msg = f"Date {date_val} is out of range for the month."
        # Find the Date column
        for comp_name, comp in components.items():
            if comp.data_type == Date:
                return DataLoadError(
                    "0-3-1-6",
                    name=dataset_name,
                    column=comp_name,
                    type="Date",
                    error=friendly_msg,
                )
        return DataLoadError(
            "0-3-1-6",
            name=dataset_name,
            column="unknown",
            type="Date",
            error=friendly_msg,
        )

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


def get_column_sql_type(comp: Component) -> str:
    """
    Get SQL type for a component with special handling for VTL types.

    - Integer → BIGINT
    - Number → DOUBLE (IEEE 754 float64, same as the pandas engine)
    - Boolean → BOOLEAN
    - Date → DATE (may be overridden to TIMESTAMP when values contain time)
    - TimePeriod, TimeInterval, Duration, String → VARCHAR
    """
    if comp.data_type == Integer:
        return "BIGINT"
    elif comp.data_type == Number:
        return "DOUBLE"
    elif comp.data_type == Boolean:
        return "BOOLEAN"
    elif comp.data_type == Date:
        return "DATE"
    else:
        # String, TimePeriod, TimeInterval, Duration → VARCHAR
        return "VARCHAR"


def get_csv_read_type(comp: Component) -> str:
    """
    Get type for CSV reading. DuckDB read_csv needs slightly different types.

    For temporal strings (TimePeriod, etc.) we read as VARCHAR.
    For Number, we let DuckDB parse directly.

    Note: Integer columns are read as VARCHAR so integer text casts exactly to
    BIGINT (a DOUBLE hop would corrupt values beyond 2^53, issue #985); the
    strict validation that rejects non-integer values (e.g., 1.5) happens in
    ``build_select_columns``.
    Date columns are read as VARCHAR to preserve original format (date-only vs datetime).
    Boolean columns are read as VARCHAR to handle quoted values (e.g., ``"TRUE"``).
    """
    if comp.data_type == Integer:
        return "VARCHAR"  # Exact BIGINT cast + strict validation in build_select_columns
    elif comp.data_type == Number:
        return "DOUBLE"  # float64, matching the pandas engine's parse
    elif comp.data_type == Boolean:
        return "VARCHAR"  # Read as VARCHAR to handle quoted values; cast during INSERT
    elif comp.data_type == Date:
        return "VARCHAR"  # Read as string; cast to DATE or TIMESTAMP during INSERT
    else:
        return "VARCHAR"


# =============================================================================
# Table Creation
# =============================================================================


def build_create_table_sql(
    table_name: str,
    components: Dict[str, Component],
    type_overrides: Optional[Dict[str, str]] = None,
) -> str:
    """
    Build CREATE TABLE statement with NOT NULL constraints only.

    No PRIMARY KEY - duplicate validation is done post-hoc via GROUP BY.
    This is more memory-efficient for large datasets.

    Args:
        table_name: Name of the table to create.
        components: Mapping of component names to Component definitions.
        type_overrides: Optional dict mapping column names to SQL types,
            used to override the default type (e.g. Date → TIMESTAMP when
            values contain time components).
    """
    col_defs: List[str] = []
    overrides = type_overrides or {}

    for comp_name, comp in components.items():
        sql_type = overrides.get(comp_name, get_column_sql_type(comp))

        if comp.role == Role.IDENTIFIER or not comp.nullable:
            col_defs.append(f'"{comp_name}" {sql_type} NOT NULL')
        else:
            col_defs.append(f'"{comp_name}" {sql_type}')

    return f'CREATE TABLE "{table_name}" ({", ".join(col_defs)})'


def validate_no_duplicates(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    id_columns: List[str],
) -> None:
    """
    Validate no duplicate rows exist using a memory-efficient approach.

    Uses COUNT vs COUNT DISTINCT comparison which is more memory-efficient
    than GROUP BY HAVING for large datasets with many unique values.
    DuckDB can use HyperLogLog approximation for COUNT DISTINCT internally.
    """
    if not id_columns:
        return  # DWI check handles this case

    id_list = ", ".join(f'"{c}"' for c in id_columns)

    # Compare total count with distinct count - memory efficient
    # DuckDB optimizes this better than GROUP BY HAVING for large datasets
    check_sql = f"""
        SELECT
            (SELECT COUNT(*) FROM "{table_name}") AS total,
            (SELECT COUNT(DISTINCT ({id_list})) FROM "{table_name}") AS distinct_count
    """

    result = conn.execute(check_sql).fetchone()
    if result and result[0] != result[1]:
        raise DataLoadError("0-3-1-7", name=table_name, row_index="(duplicate keys detected)")


# =============================================================================
# CSV Loading Helpers
# =============================================================================


def validate_input_path(file_path: Path) -> None:
    """Validate that the input file exists."""
    if not file_path.exists() or not file_path.is_file():
        raise DataLoadError(code="0-3-1-1", file=file_path)


def build_csv_column_types(
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
            dtypes[col] = get_csv_read_type(components[col])
    return dtypes


def handle_sdmx_columns(columns: List[str], components: Dict[str, Component]) -> List[str]:
    """
    Identify SDMX-CSV special columns to exclude.
    Returns list of columns to keep.

    A file is SDMX-CSV when its first column names the structure its rows belong to,
    and only then does it carry the columns around it. A plain CSV that happens to
    hold a column of one of those names holds a column of the Data Set, which the
    DataStructure has to define, as the pandas loader reads it.
    """
    exclude = set()

    # DATAFLOW - drop if first column and not in structure
    if columns and columns[0] == "DATAFLOW" and "DATAFLOW" not in components:
        exclude.add("DATAFLOW")

    # STRUCTURE columns, and the ones an SDMX-CSV file carries beside them
    if columns and columns[0] == "STRUCTURE" and "STRUCTURE" not in components:
        exclude.add("STRUCTURE")
        if "STRUCTURE_ID" in columns and "STRUCTURE_ID" not in components:
            exclude.add("STRUCTURE_ID")
        # ACTION is handled specially - the rows it marks deleted are filtered out
        if "ACTION" in columns and "ACTION" not in components:
            exclude.add("ACTION")

    return [c for c in columns if c not in exclude]


# =============================================================================
# Temporal Validation (only explicit validation needed)
# =============================================================================


def validate_temporal_columns(
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
            temporal_checks.append((comp_name, DURATION_PATTERN, "Duration"))

    if not temporal_checks:
        return

    # Single query to check all temporal columns at once
    # Returns first invalid value found for any column
    case_expressions = []
    for col_name, pattern, type_name in temporal_checks:
        case_expressions.append(f"""
            CASE WHEN "{col_name}" IS NOT NULL AND "{col_name}" != ''
                 AND NOT regexp_matches(TRIM("{col_name}"), '{pattern}')
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


def build_boolean_cast(value_expr: str, column_name: str) -> str:
    """SQL reading a Boolean input value on the documented set: "true"/"false"/"1"/"0"
    whatever the case, a real boolean, or a number, which is compared against zero.

    DuckDB's own VARCHAR->BOOLEAN cast reads "yes", "y" and "t" as well, so a value
    that arrives as text is mapped by hand here and anything outside the set is an
    error, the same answer the pandas loader gives. A column that is
    already a boolean or a number keeps DuckDB's cast, which reads it as pandas does.
    """
    text = f"lower(CAST({value_expr} AS VARCHAR))"
    err = (
        f"'Column {column_name}: value ' || CAST({value_expr} AS VARCHAR) || "
        f"' is not a Boolean, use true, false, 1 or 0'"
    )
    return f"""CASE
        WHEN {value_expr} IS NULL THEN NULL
        WHEN typeof({value_expr}) != 'VARCHAR' THEN CAST({value_expr} AS BOOLEAN)
        WHEN {text} IN ('true', '1') THEN TRUE
        WHEN {text} IN ('false', '0') THEN FALSE
        ELSE error({err})
    END"""


def build_select_columns(
    components: Dict[str, Component],
    keep_columns: List[str],
    csv_dtypes: Dict[str, str],
    dataset_name: str,
    type_overrides: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Build SELECT column expressions with type casting and validation."""
    select_cols = []
    overrides = type_overrides or {}

    for comp_name, comp in components.items():
        if comp_name in keep_columns:
            csv_type = csv_dtypes.get(comp_name, "VARCHAR")
            table_type = overrides.get(comp_name, get_column_sql_type(comp))

            # Strict Integer validation on VARCHAR input. Integer-form text casts
            # straight to BIGINT so values beyond 2^53 stay exact (issue #985);
            # decimal-form text ("1.0", "1e5") goes through DOUBLE and keeps the
            # non-integer rejection (a direct VARCHAR->BIGINT cast would silently
            # round "1.5"); anything else is a conversion error.
            if csv_type == "VARCHAR" and table_type == "BIGINT":
                val = f"NULLIF(\"{comp_name}\", '')" if comp.nullable else f'"{comp_name}"'
                decimal_msg = (
                    f"'Column {comp_name}: value ' || \"{comp_name}\" || "
                    f"' has non-zero decimal component for Integer type'"
                )
                convert_msg = (
                    f"'Column {comp_name}: could not convert value ' || \"{comp_name}\" || "
                    f"' to Integer type'"
                )
                select_cols.append(
                    f"""CASE
                        WHEN {val} IS NULL THEN NULL
                        WHEN regexp_matches(TRIM({val}), '^[+-]?[0-9]+$')
                        THEN CAST({val} AS BIGINT)
                        WHEN TRY_CAST({val} AS DOUBLE) IS NULL
                        THEN error({convert_msg})
                        WHEN CAST({val} AS DOUBLE) <> FLOOR(CAST({val} AS DOUBLE))
                        THEN error({decimal_msg})
                        ELSE CAST(CAST({val} AS DOUBLE) AS BIGINT)
                    END AS "{comp_name}\""""
                )
            # Date columns: read as VARCHAR, validate format, cast to DATE or TIMESTAMP.
            # Accepts a bare date or a full datetime with the T or space separator and an
            # optional timezone (+HH:MM or Z); the same set the pandas loader accepts.
            elif csv_type == "VARCHAR" and comp.data_type == Date:
                date_regex = VALID_DATE_REGEX
                null_check = f'"{comp_name}" IS NOT NULL'
                if comp.nullable:
                    null_check += f""" AND "{comp_name}" != ''"""
                format_err = (
                    f"'Date ' || \"{comp_name}\" || "
                    f"' is not in the correct format. "
                    f"Use YYYY-MM-DD or YYYY-MM-DD HH:MM:SS.'"
                )
                val_expr = f"NULLIF(\"{comp_name}\", '')" if comp.nullable else f'"{comp_name}"'
                select_cols.append(
                    f"""CASE
                        WHEN {null_check}
                             AND NOT regexp_matches("{comp_name}", '{date_regex}')
                        THEN error({format_err})
                        ELSE CAST({val_expr} AS {table_type})
                    END AS "{comp_name}\""""
                )
            elif csv_type == "VARCHAR" and comp.data_type == Boolean:
                # Strip double quotes and read as BOOLEAN (handles """TRUE""" from CSV)
                stripped = f"""REPLACE("{comp_name}", '"', '')"""
                if comp.nullable:
                    stripped = f"NULLIF({stripped}, '')"
                select_cols.append(f'{build_boolean_cast(stripped, comp_name)} AS "{comp_name}"')
            elif csv_type == "VARCHAR" and comp.data_type == String:
                # Strip double quotes from String values (match pandas loader behavior)
                expr = f"""REPLACE("{comp_name}", '"', '')"""
                if comp.nullable:
                    expr = f"NULLIF({expr}, '')"
                select_cols.append(f'{expr} AS "{comp_name}"')
            elif csv_type == "VARCHAR" and comp.nullable:
                # Treat empty strings as NULL for nullable VARCHAR columns
                select_cols.append(f'NULLIF("{comp_name}", \'\') AS "{comp_name}"')
            else:
                select_cols.append(f'"{comp_name}"')
        else:
            # Missing column → NULL (only allowed for nullable)
            if comp.nullable:
                table_type = overrides.get(comp_name, get_column_sql_type(comp))
                select_cols.append(f'NULL::{table_type} AS "{comp_name}"')
            else:
                raise DataLoadError("0-3-1-5", name=dataset_name, comp_name=comp_name)

    return select_cols


def check_extra_columns(
    columns: List[str],
    components: Dict[str, Component],
    dataset_name: str,
) -> None:
    """Reject the columns the DataStructure does not define.

    The SDMX marker columns are taken out by ``handle_sdmx_columns`` before this
    runs, so what is left is what the pandas loader rejects too.
    """
    extra_columns = sorted(set(columns) - set(components))
    if extra_columns:
        raise DataLoadError("0-3-1-15", name=dataset_name, extra_columns=", ".join(extra_columns))


def check_missing_identifiers(
    id_columns: List[str],
    keep_columns: List[str],
    csv_path: Path,
) -> None:
    """Check if required identifier columns are present in CSV."""
    missing_ids = set(id_columns) - set(keep_columns)
    if missing_ids:
        raise InputValidationException(
            code="0-1-1-8",
            ids=", ".join(missing_ids),
            file=str(csv_path.name),
        )
