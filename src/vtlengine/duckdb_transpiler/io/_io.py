"""
Internal IO functions for DuckDB-based CSV and Parquet loading and saving.

This module contains the core load/save implementations to avoid circular imports.
"""

import csv
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import duckdb
import pandas as pd

from vtlengine.DataTypes import Date, String, TimeInterval, TimePeriod
from vtlengine.duckdb_transpiler.io._validation import (
    VALID_DATE_REGEX,
    VALID_DATE_YEAR_REGEX,
    build_boolean_cast,
    build_create_table_sql,
    build_csv_column_types,
    build_select_columns,
    check_extra_columns,
    check_missing_identifiers,
    get_column_sql_type,
    handle_sdmx_columns,
    map_duckdb_error,
    validate_input_path,
    validate_no_duplicates,
    validate_temporal_columns,
)
from vtlengine.Exceptions import DataLoadError, InputValidationException
from vtlengine.files.sdmx_handler import (
    extract_sdmx_dataset_name,
    is_sdmx_csv_file,
    is_sdmx_datapoint_file,
    load_sdmx_datapoints,
)
from vtlengine.Model import Component, Dataset, Role, Scalar
from vtlengine.Utils._number_config import format_scalar_value_for_csv

# A Date value writes its time after the date, with the month and the day allowed to
# be written short, as VALID_DATE_REGEX allows them.
_DATE_WITH_TIME_RE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}[T ]")


def _skip_load_validation() -> bool:
    """Read VTL_SKIP_LOAD_VALIDATION lazily so mutations after import take effect."""
    return os.environ.get("VTL_SKIP_LOAD_VALIDATION", "").lower() in ("1", "true", "yes")


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
    skip_validation = _skip_load_validation()

    if not skip_validation:
        try:
            validate_temporal_columns(conn, table_name, components)
        except DataLoadError:
            conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
            raise

    _normalize_time_period_columns(conn, table_name, components)
    _normalize_time_interval_columns(conn, table_name, components)

    if skip_validation:
        return

    try:
        id_columns = [n for n, c in components.items() if c.role == Role.IDENTIFIER]

        # DWI: no identifiers → max 1 row
        if not id_columns:
            result = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()
            if result and result[0] > 1:
                raise DataLoadError("0-3-1-4", name=table_name)

        # Duplicate check (GROUP BY HAVING), on the normalized values so that two
        # spellings of one period are read as the same Data Point
        validate_no_duplicates(conn, table_name, id_columns)

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
            try:
                conn.execute(
                    f'UPDATE "{table_name}" SET "{comp_name}" = '
                    f'vtl_period_normalize(TRIM("{comp_name}")) '
                    f'WHERE "{comp_name}" IS NOT NULL AND "{comp_name}" != \'\''
                )
            except duckdb.Error as e:
                raise DataLoadError(
                    "0-3-1-6",
                    name=table_name,
                    column=comp_name,
                    type="Time_Period",
                    error=str(e),
                )


def _normalize_time_interval_columns(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    components: Dict[str, Component],
) -> None:
    """Expand a Time interval written as the year or the month it covers, and store
    it without the space around it.

    2020 covers 2020-01-01/2020-12-31 and 2020-05 covers 2020-05-01/2020-05-31, which
    the pandas loader reads them as; a value already written as a pair of dates is
    left as it is (issue #1066). Every form is read from the value without the space
    around it, the way its check reads it and the way the pandas loader strips it
    before reading, so the value stored is the value that was read (issue #1067).
    """
    for comp_name, comp in components.items():
        if comp.data_type != TimeInterval:
            continue
        value = f'TRIM("{comp_name}")'
        year = f"CAST(SUBSTR({value}, 1, 4) AS INTEGER)"
        month = f"CAST(SUBSTR({value}, 6) AS INTEGER)"
        first_day = f"MAKE_DATE({year}, {month}, 1)"
        try:
            conn.execute(
                f'UPDATE "{table_name}" SET "{comp_name}" = CASE '
                f"WHEN regexp_matches({value}, '^\\d{{4}}$') "
                f"THEN {value} || '-01-01/' || {value} || '-12-31' "
                f"WHEN regexp_matches({value}, '^\\d{{4}}-[0-1]?\\d$') "
                f"THEN strftime({first_day}, '%Y-%m-%d') || '/' "
                f"|| strftime(LAST_DAY({first_day}), '%Y-%m-%d') "
                f"ELSE {value} END "
                f'WHERE "{comp_name}" IS NOT NULL AND "{comp_name}" != \'\''
            )
        except duckdb.Error as e:
            raise DataLoadError(
                "0-3-1-6",
                name=table_name,
                column=comp_name,
                type="Time",
                error=str(e),
            )


def _cast_probe(
    conn: duckdb.DuckDBPyConnection,
    read_expr: str,
    components: Dict[str, Component],
    select_cols: List[str],
) -> Optional[Callable[[str], bool]]:
    """Whether reading one column on its own fails, so a failure DuckDB reports
    without naming a column is traced back to the column that caused it.

    One expression was built per component, in the same order, so every column is
    read back through the very expression the load used, over the same file.
    COUNT() reads each value without holding the column in memory.
    """
    if not read_expr or len(select_cols) != len(components):
        return None
    expressions = dict(zip(components, select_cols))

    def fails(comp_name: str) -> bool:
        try:
            conn.execute(
                f'SELECT COUNT("{comp_name}") FROM (SELECT {expressions[comp_name]} {read_expr})'
            )
            return False
        except duckdb.Error:
            return True

    return fails


def _detect_csv_format(
    conn: duckdb.DuckDBPyConnection,
    csv_path: Path,
    expected_columns: Optional[List[str]] = None,
) -> str:
    """Detect CSV delimiter, quote and escape using sniff_csv.

    Returns a string of read_csv format options (e.g. "delim=',', quote='\"', escape='\"'").
    Falls back to defaults if sniffing fails or produces unreliable results.

    Fast path: if every name in ``expected_columns`` appears in the header parsed
    with the default ``,`` delimiter, skip the costly ``sniff_csv`` round-trips.
    """
    if expected_columns:
        try:
            with open(csv_path, newline="", encoding="utf-8-sig") as f:
                reader = csv.reader(f, delimiter=",")
                header = next(reader, [])
            header_set = {h.strip() for h in header}
            if len(header) >= 1 and all(col in header_set for col in expected_columns):
                # Standard RFC 4180 format. Match what sniff_csv returns for
                # well-formed CSVs (double-quote as both quote and escape).
                return "delim=',', quote='\"', escape='\"'"
        except (OSError, UnicodeDecodeError, StopIteration):
            pass

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


def _read_parquet_columns(
    conn: duckdb.DuckDBPyConnection,
    file_path: Path,
) -> Tuple[List[str], Dict[str, str]]:
    """Read the column list and column types of a parquet file via a zero-row scan."""
    rel = conn.sql(f"SELECT * FROM read_parquet('{file_path}') LIMIT 0")
    return list(rel.columns), {name: str(t) for name, t in zip(rel.columns, rel.types)}


# A Date value writes its time after the date, with the month and the day allowed to be
# written short, as VALID_DATE_REGEX allows them. Read by DuckDB, not by Python.
_SQL_DATE_WITH_TIME = r"^\d{4}-\d{1,2}-\d{1,2}[T ]"


def _parquet_date_type_overrides(
    conn: duckdb.DuckDBPyConnection,
    file_path: Path,
    components: Dict[str, Component],
    column_types: Dict[str, str],
) -> Dict[str, str]:
    """The Date columns a parquet file needs stored as TIMESTAMP.

    Every Date column was created as DATE whatever the file held, so a value carrying
    a time was cast down to the bare date, which is what the CSV path and the DataFrame
    path both keep. A column the file types as a timestamp is stored as
    one; a column of text is read the way the CSV path reads it, by looking for a time
    written after the date. ``column_types`` is the name -> DuckDB type mapping the
    caller already read from the file's zero-row scan.
    """
    overrides: Dict[str, str] = {}
    text_columns = []
    for comp_name, comp in components.items():
        column_type = column_types.get(comp_name)
        if comp.data_type != Date or column_type is None:
            continue
        if "TIMESTAMP" in column_type:
            overrides[comp_name] = "TIMESTAMP"
        elif column_type == "VARCHAR":
            text_columns.append(comp_name)

    if text_columns:
        checks = ", ".join(
            f"MAX(CASE WHEN regexp_matches(\"{name}\", '{_SQL_DATE_WITH_TIME}') THEN 1 ELSE 0 END)"
            for name in text_columns
        )
        row = conn.execute(f"SELECT {checks} FROM read_parquet('{file_path}')").fetchone()
        if row is not None:
            overrides.update(
                {name: "TIMESTAMP" for name, has_time in zip(text_columns, row) if has_time}
            )
    return overrides


def load_datapoints_duckdb(
    conn: duckdb.DuckDBPyConnection,
    components: Dict[str, Component],
    dataset_name: str,
    file_path: Optional[Union[Path, str]] = None,
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
        file_path: Path to input file (None for empty table)

    Returns:
        DuckDB relation pointing to the created table

    Raises:
        DataLoadError: If validation fails
    """
    # Handle empty dataset
    if file_path is None:
        return _create_empty_table(conn, components, dataset_name)

    # A path that does not exist names data that never arrived, which validate_input_path
    # reports; only the absence of a path at all is an empty dataset (issue #1061).
    file_path = Path(file_path) if isinstance(file_path, str) else file_path
    validate_input_path(file_path)

    if file_path.suffix.lower() == ".parquet":
        return _load_parquet(conn, components, dataset_name, file_path)

    # Get identifier columns (needed for duplicate validation)
    id_columns = [n for n, c in components.items() if c.role == Role.IDENTIFIER]

    # For CSV, Date columns use TIMESTAMP as safe default (can't inspect values cheaply)
    csv_date_overrides = {n: "TIMESTAMP" for n, c in components.items() if c.data_type == Date}

    # 1. Create table (NOT NULL only, no PRIMARY KEY)
    conn.execute(build_create_table_sql(dataset_name, components, csv_date_overrides))

    # Kept out of the try so a failure before the read can still be reported
    read_expr = ""
    select_cols: List[str] = []

    try:
        # 2. Detect CSV format (delimiter, quote, escape) using sniff_csv.
        # Pass expected component names so the fast-path can skip sniffing
        # when the header already parses cleanly with a comma delimiter.
        _sniffed_fmt = _detect_csv_format(conn, file_path, expected_columns=list(components.keys()))

        sniffed_delim = _sniffed_fmt.split("'")[1] if "delim=" in _sniffed_fmt else ","
        try:
            with open(file_path, newline="", encoding="utf-8-sig") as f:
                reader = csv.reader(f, delimiter=sniffed_delim)
                csv_columns = next(reader, [])
        except UnicodeDecodeError:
            raise InputValidationException(code="0-1-2-5", file=Path(file_path).name) from None

        if not csv_columns:
            raise InputValidationException(code="0-1-1-17", file=file_path.name)

        if len(set(csv_columns)) != len(csv_columns):
            duplicates = list({item for item in csv_columns if csv_columns.count(item) > 1})
            raise InputValidationException(
                code="0-1-2-3",
                element_type="Columns",
                element=f"{', '.join(duplicates)}",
            )

        # 4. Handle SDMX-CSV special columns
        keep_columns = handle_sdmx_columns(csv_columns, components)

        # Check required identifier columns exist, and that no other column is left.
        # An SDMX file does not reach here: extract_datapoint_paths reads it with
        # pysdmx and hands over a DataFrame.
        check_missing_identifiers(id_columns, keep_columns, file_path)
        check_extra_columns(keep_columns, components, dataset_name)

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

        # 7. Build filter for SDMX ACTION column, which only an SDMX-CSV file carries
        action_filter = ""
        if "ACTION" in csv_columns and "ACTION" not in keep_columns:
            action_filter = 'WHERE "ACTION" != \'D\' OR "ACTION" IS NULL'

        # 8. Execute INSERT
        read_expr = f"""
            FROM read_csv(
                '{file_path}',
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
        insert_sql = f"""
            INSERT INTO "{dataset_name}"
            SELECT {", ".join(select_cols)}
            {read_expr}
        """
        conn.execute(insert_sql)

    except duckdb.Error as e:
        conn.execute(f'DROP TABLE IF EXISTS "{dataset_name}"')
        raise map_duckdb_error(
            e, dataset_name, components, _cast_probe(conn, read_expr, components, select_cols)
        )
    except Exception:
        conn.execute(f'DROP TABLE IF EXISTS "{dataset_name}"')
        raise

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


def _load_parquet(
    conn: duckdb.DuckDBPyConnection,
    components: Dict[str, Component],
    dataset_name: str,
    file_path: Path,
) -> duckdb.DuckDBPyRelation:
    """Load a Parquet file into a DuckDB table via read_parquet."""
    id_columns = [n for n, c in components.items() if c.role == Role.IDENTIFIER]

    try:
        parquet_cols, parquet_types = _read_parquet_columns(conn, file_path)

        if len(set(parquet_cols)) != len(parquet_cols):
            duplicates = list({item for item in parquet_cols if parquet_cols.count(item) > 1})
            raise InputValidationException(
                code="0-1-2-3",
                element_type="Columns",
                element=f"{', '.join(duplicates)}",
            )

        keep_columns = handle_sdmx_columns(parquet_cols, components)
        check_missing_identifiers(id_columns, keep_columns, file_path)
        check_extra_columns(keep_columns, components, dataset_name)

        # The table is created once the file has been read, so a Date column holding a
        # time is created as a TIMESTAMP and keeps it (issue #895).
        date_overrides = _parquet_date_type_overrides(conn, file_path, components, parquet_types)
        conn.execute(build_create_table_sql(dataset_name, components, date_overrides))

        select_exprs = _build_dataframe_select_columns(
            components,
            dataset_name,
            df_columns=parquet_cols,
            type_overrides=date_overrides,
            source_types=parquet_types,
        )

        action_filter = ""
        if "ACTION" in parquet_cols and "ACTION" not in keep_columns:
            action_filter = 'WHERE "ACTION" != \'D\' OR "ACTION" IS NULL'

        col_list = ", ".join(f'"{c}"' for c in components)
        insert_sql = (
            f'INSERT INTO "{dataset_name}" ({col_list}) '
            f"SELECT {', '.join(select_exprs)} "
            f"FROM read_parquet('{file_path}') "
            f"{action_filter}"
        )
        conn.execute(insert_sql)

    except duckdb.Error as e:
        conn.execute(f'DROP TABLE IF EXISTS "{dataset_name}"')
        raise map_duckdb_error(e, dataset_name, components)
    except Exception:
        conn.execute(f'DROP TABLE IF EXISTS "{dataset_name}"')
        raise

    _validate_loaded_table(conn, dataset_name, components)
    return conn.table(dataset_name)


def save_datapoints_duckdb(
    conn: duckdb.DuckDBPyConnection,
    dataset_name: str,
    output_path: Union[Path, str],
    delete_after_save: bool = True,
    select_sql: Optional[str] = None,
    output_format: Literal["csv", "parquet"] = "csv",
) -> None:
    """Save dataset to disk using DuckDB's COPY TO.

    Args:
        conn: DuckDB connection.
        dataset_name: Name of the table to save.
        output_path: Directory path where the file will be saved.
        delete_after_save: If True, drop the table after saving.
        select_sql: Optional SELECT query whose rows are saved. When provided,
            COPY runs against ``(select_sql)``; otherwise the raw table is dumped.
        output_format: ``"csv"`` (default) or ``"parquet"``. Determines the
            file extension and the COPY options used.
    """
    if output_format not in ("csv", "parquet"):
        raise InputValidationException(
            code="0-1-1-16",
            value=output_format,
            valid_options="csv, parquet",
        )

    output_path = Path(output_path) if isinstance(output_path, str) else output_path
    output_file = output_path / f"{dataset_name}.{output_format}"

    source = f"({select_sql})" if select_sql else f'"{dataset_name}"'

    if output_format == "parquet":
        copy_options = "(FORMAT PARQUET)"
    else:
        copy_options = "WITH (HEADER true, DELIMITER ',')"

    conn.execute(f"COPY {source} TO '{output_file}' {copy_options}")

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
            if scalar.value is None:
                writer.writerow([name, ""])
            else:
                writer.writerow([name, format_scalar_value_for_csv(scalar.value)])


# The files a directory of datapoints is read from: what the pandas engine reads,
# plus the Parquet files only this engine can.
DATAPOINT_SUFFIXES = (".csv", ".xml", ".parquet")


def _as_datapoint_path(value: Union[str, Path]) -> Path:
    """The Path a datapoint input stands for."""
    return Path(value) if isinstance(value, str) else value


def _datapoint_files(path: Path) -> List[Path]:
    """The files a datapoint path stands for.

    A path that does not exist is rejected rather than read as no data at all, and a
    directory stands for the datapoint files in it, as the pandas engine reads them.
    """
    if not path.exists():
        raise DataLoadError(code="0-3-1-1", file=path)
    if not path.is_dir():
        return [path]
    return sorted(f for f in path.iterdir() if f.suffix.lower() in DATAPOINT_SUFFIXES)


def _sdmx_dataframe(
    path: Path, name: Optional[str], input_datasets: Dict[str, Dataset]
) -> Optional[Tuple[str, pd.DataFrame]]:
    """Read a path with pysdmx where it holds SDMX data, whichever the format.

    SDMX-ML, SDMX-JSON and SDMX-CSV are all read by pysdmx, so this engine loads
    them the way the pandas engine does (issue #1052). The formats that name their
    structure resolve a dataset name from it; an SDMX-CSV file is named after the
    file, as a plain CSV one is.
    """
    if path.suffix.lower() == ".parquet":
        return None
    try:
        if is_sdmx_datapoint_file(path):
            resolved = name or extract_sdmx_dataset_name(path)
        else:
            resolved = name or path.stem
            if resolved not in input_datasets:
                return None
            if not is_sdmx_csv_file(path, input_datasets[resolved].components):
                return None
        if resolved not in input_datasets:
            return None
        components = input_datasets[resolved].components
        data = load_sdmx_datapoints(components, resolved, path)
        # pysdmx reads every cell as text, so an empty one arrives as an empty string
        # rather than as no value. Only a String Component can hold one, which is what
        # the pandas engine reads from the same file.
        for column, comp in components.items():
            if column in data.columns and comp.data_type is not String:
                data[column] = data[column].replace("", None)
        return resolved, data
    except Exception:
        return None


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

    An input that names no dataset, or a path that does not exist, is rejected here
    rather than read as an empty dataset, as the pandas engine rejects it.

    Args:
        datapoints: Dict of DataFrames/paths, list of paths, or single path
        input_datasets: Dict of input dataset structures (for validation)

    Returns:
        Tuple of (path_dict, dataframe_dict):
        - path_dict: Dict mapping dataset names to CSV Paths (None if no paths)
        - dataframe_dict: Dict mapping dataset names to DataFrames (for direct registration)

    Raises:
        InputValidationException: If dataset name not found in structures
        DataLoadError: If a datapoint path does not exist
    """
    if datapoints is None:
        return None, {}

    path_dict: Dict[str, Path] = {}
    df_dict: Dict[str, pd.DataFrame] = {}

    def add_path(path: Path, name: Optional[str]) -> None:
        """Resolve one file to the dataset it holds."""
        loaded = _sdmx_dataframe(path, name, input_datasets)
        if loaded is not None:
            df_dict[loaded[0]] = loaded[1]
            return
        resolved = name or path.stem
        if resolved not in input_datasets:
            raise InputValidationException(f"Not found dataset {resolved} in datastructures.")
        path_dict[resolved] = path

    # Handle dictionary input
    if isinstance(datapoints, dict):
        for name, value in datapoints.items():
            if name not in input_datasets:
                raise InputValidationException(f"Not found dataset {name} in datastructures.")

            if isinstance(value, pd.DataFrame):
                # Store DataFrame for direct DuckDB registration
                df_dict[name] = value
                continue
            if not isinstance(value, (str, Path)):
                # Each value says where one dataset's Data Points are, and either kind
                # may sit beside the other, so what is refused is a value that names no
                # Data Points at all. Both engines say this (issues #1061 and #1072).
                raise InputValidationException(
                    f"Invalid datapoint for {name}. Must be DataFrame, Path, or string."
                )
            # A dictionary names one file per dataset, so a directory is no more a
            # datapoint file here than a missing path is, as the pandas engine reads it.
            path = _as_datapoint_path(value)
            validate_input_path(path)
            add_path(path, name)
        return path_dict if path_dict else None, df_dict

    # Handle list of paths
    if isinstance(datapoints, list):
        for item in datapoints:
            for file_path in _datapoint_files(_as_datapoint_path(item)):
                add_path(file_path, None)
        return path_dict if path_dict else None, df_dict

    # Handle single path
    for file_path in _datapoint_files(_as_datapoint_path(datapoints)):
        add_path(file_path, None)
    return path_dict if path_dict else None, df_dict


def _carries_a_time(value: object) -> bool:
    """Whether a Date input value writes a time beside the date.

    A value is read the way the Date check reads it, rather than by looking for a
    separator at a fixed place in text: a month or a day written short moved it
    along, and a column pandas had already parsed carried no text to look at, so
    either one was stored as a DATE and lost the time it held.
    """
    if isinstance(value, str):
        return bool(_DATE_WITH_TIME_RE.match(value.strip()))
    if isinstance(value, datetime):
        return (value.hour, value.minute, value.second, value.microsecond) != (0, 0, 0, 0)
    return False


def _detect_date_type_overrides(
    df: pd.DataFrame, components: Dict[str, Component]
) -> Dict[str, str]:
    """Determine which Date columns need TIMESTAMP instead of DATE.

    A Date column holding a value that writes a time is stored as TIMESTAMP to keep
    that time; a column of bare dates is stored as DATE.
    """
    overrides: Dict[str, str] = {}
    for comp_name, comp in components.items():
        if comp.data_type != Date or comp_name not in df.columns:
            continue
        if any(_carries_a_time(val) for val in df[comp_name].dropna()):
            overrides[comp_name] = "TIMESTAMP"
    return overrides


def _build_dataframe_select_columns(
    components: Dict[str, Component],
    dataset_name: str,
    df_columns: Optional[List[str]] = None,
    type_overrides: Optional[Dict[str, str]] = None,
    source_types: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Build SELECT expressions with explicit CAST for DataFrame → DuckDB table insertion.

    Ensures type enforcement matches the CSV loading path (load_datapoints_duckdb).
    A missing column is filled with NULL, and, when the component is not nullable,
    reported as missing the way the CSV path and the pandas loader report it: filling
    it left a DataFrame with no rows loading a structure it did not carry.
    ``source_types`` maps source column names to their DuckDB types; a column absent
    from it is assumed to be VARCHAR.
    """
    df_col_set = set(df_columns) if df_columns is not None else None
    overrides = type_overrides or {}
    src_types = source_types or {}
    exprs: List[str] = []
    for comp_name, comp in components.items():
        target_type = overrides.get(comp_name, get_column_sql_type(comp))
        source_type = src_types.get(comp_name, "VARCHAR").upper()
        if df_col_set is not None and comp_name not in df_col_set:
            if not comp.nullable:
                raise DataLoadError("0-3-1-5", name=dataset_name, comp_name=comp_name)
            exprs.append(f'CAST(NULL AS {target_type}) AS "{comp_name}"')
        elif comp.data_type == Date and (
            "VARCHAR" in source_type or source_type.startswith("ENUM")
        ):
            # Accept only a bare date, or a date with a COMPLETE, in-range time
            # (HH:MM:SS, optional fractional seconds / timezone), matching the strict
            # rule on the pandas path. This rejects partial times ("...HH" / "...HH:MM"),
            # a bad separator ("2020-01-01X12:30:45") and out-of-range times
            # ("2020-01-01T25:00:00") here with a clear message, instead of letting the
            # cast silently truncate them or surface a cryptic out-of-range error.
            # The guard only applies to string-like source columns (VARCHAR, or ENUM
            # from pandas categoricals): a temporal source (DATE/TIMESTAMP/TIMESTAMPTZ)
            # cannot hold a malformed string, and its VARCHAR rendering (e.g. a "+01"
            # offset) is not the input format the regex validates, so those keep the
            # plain CAST.
            col_as_varchar = f'TRIM(CAST("{comp_name}" AS VARCHAR))'
            err = (
                f"'Column {comp_name}: Date ' || {col_as_varchar} || "
                f"' has an invalid or incomplete time; expected YYYY-MM-DD HH:MM:SS.'"
            )
            year_err = (
                f"'Date ' || {col_as_varchar} || ' is invalid. Year must be between 1800 and 9999.'"
            )
            exprs.append(
                f'CASE WHEN "{comp_name}" IS NOT NULL '
                f"AND NOT regexp_matches({col_as_varchar}, '{VALID_DATE_REGEX}') "
                f"THEN error({err}) "
                f'WHEN "{comp_name}" IS NOT NULL '
                f"AND NOT regexp_matches({col_as_varchar}, '{VALID_DATE_YEAR_REGEX}') "
                f"THEN error({year_err}) "
                f'ELSE CAST({col_as_varchar} AS {target_type}) END AS "{comp_name}"'
            )
        elif target_type == "BOOLEAN":
            # A Boolean is read on the documented set, not on DuckDB's wider one,
            # so a DataFrame and a CSV are read the same way (issue #1068).
            boolean_cast = build_boolean_cast(f'"{comp_name}"', comp_name)
            exprs.append(f'{boolean_cast} AS "{comp_name}"')
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

        # Normalize the column names the way the pandas loader does (_validate_pandas):
        # labels become str and a leading UTF-8 BOM is dropped (a DataFrame built from a
        # BOM-encoded CSV without utf-8-sig decoding carries it on its first column).
        # A shallow copy keeps the caller's DataFrame untouched without copying its data.
        df = df.copy(deep=False)
        df.columns = pd.Index([str(col).removeprefix("\ufeff") for col in df.columns])

        # A DataFrame carries its columns as they are, so the SDMX markers are taken
        # out before what is left is checked against the DataStructure.
        check_extra_columns(handle_sdmx_columns(list(df.columns), components), components, name)

        # Detect Date columns that contain time values → TIMESTAMP instead of DATE
        type_overrides = _detect_date_type_overrides(df, components)

        # Create table with proper schema
        conn.execute(build_create_table_sql(name, components, type_overrides))

        # Register DataFrame and insert data with explicit type casting
        temp_view = f"_temp_{name}"
        conn.register(temp_view, df)
        try:
            source_types = {
                str(row[0]): str(row[1])
                for row in conn.execute(f'DESCRIBE "{temp_view}"').fetchall()
            }
            select_exprs = _build_dataframe_select_columns(
                components, name, list(df.columns), type_overrides, source_types
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
