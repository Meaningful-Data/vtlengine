from csv import DictReader
from pathlib import Path
from typing import Dict, List, Optional, Union

from duckdb.duckdb import DuckDBPyRelation

from vtlengine.connection import con
from vtlengine.DataTypes import (
    Boolean,
    Date,
    Duration,
    Integer,
    Number,
    TimeInterval,
    TimePeriod,
)
from vtlengine.Exceptions import InputValidationException, SemanticError
from vtlengine.files.parser._rfc_dialect import register_rfc
from vtlengine.files.parser._time_checking import (
    check_time,
    check_time_period,
    load_time_checks,
)
from vtlengine.Model import Component, Dataset, Role

load_time_checks()


def _validate_csv_path(components: Dict[str, Component], csv_path: Path) -> None:
    # GE1 check if the file is empty
    if not csv_path.exists():
        raise Exception(f"Path {csv_path} does not exist.")
    if not csv_path.is_file():
        raise Exception(f"Path {csv_path} is not a file.")
    register_rfc()
    try:
        with open(csv_path, "r", errors="replace", encoding="utf-8") as f:
            reader = DictReader(f, dialect="rfc")
            csv_columns = reader.fieldnames
    except InputValidationException as ie:
        raise InputValidationException("{}".format(str(ie))) from None
    except Exception as e:
        raise InputValidationException(
            f"ERROR: {str(e)}, review file {str(csv_path.as_posix())}"
        ) from None

    if not csv_columns:
        raise InputValidationException(code="0-1-1-6", file=csv_path)

    if len(list(set(csv_columns))) != len(csv_columns):
        duplicates = list(set([item for item in csv_columns if csv_columns.count(item) > 1]))
        raise Exception(f"Duplicated columns {', '.join(duplicates)} found in file.")

    comp_names = set([c.name for c in components.values() if c.role == Role.IDENTIFIER])
    comps_missing: Union[str, List[str]] = (
        [id_m for id_m in comp_names if id_m not in reader.fieldnames] if reader.fieldnames else []
    )
    if comps_missing:
        comps_missing = ", ".join(comps_missing)
        raise InputValidationException(code="0-1-1-8", ids=comps_missing, file=str(csv_path.name))


def _sanitize_duckdb_columns(
    components: Dict[str, Component],
    csv_path: Union[str, Path],
    data: DuckDBPyRelation,
) -> DuckDBPyRelation:
    column_names = data.columns
    modified_data = data

    # Fast loading from SDMX-CSV
    if (
        "DATAFLOW" in column_names
        and column_names[0] == "DATAFLOW"
        and "DATAFLOW" not in components
    ):
        modified_data = modified_data.project(
            ", ".join([f'"{col}"' for col in column_names if col != "DATAFLOW"])
        )
        column_names = modified_data.columns

    if "STRUCTURE" in column_names and column_names[0] == "STRUCTURE":
        cols_to_drop = {"STRUCTURE", "STRUCTURE_ID", "ACTION"} & set(column_names)
        remaining_cols = [col for col in column_names if col not in cols_to_drop]

        if "ACTION" in column_names:
            modified_data = modified_data.filter("ACTION != 'D'")

        modified_data = modified_data.project(
            ", ".join([f'"{col}"' for col in remaining_cols])
        )
        column_names = modified_data.columns

    # Validate identifiers
    comp_names = {c.name for c in components.values() if c.role == Role.IDENTIFIER}
    missing = [name for name in comp_names if name not in column_names]

    if missing:
        file = csv_path if isinstance(csv_path, str) else csv_path.name
        raise InputValidationException(code="0-1-1-7", ids=", ".join(missing), file=file)

    # Add missing nullable columns
    for comp_name, comp in components.items():
        if comp_name not in column_names:
            if not comp.nullable:
                raise InputValidationException(f"Component {comp_name} is missing in the file.")
            modified_data = modified_data.project(
                f"*, NULL::{comp.data_type().sql_type} AS \"{comp_name}\""
            )
            column_names = modified_data.columns

    return modified_data


def _validate_duckdb(
    components: Dict[str, Component],
    data: DuckDBPyRelation,
    dataset_name: str,
) -> DuckDBPyRelation:
    id_names = [name for name, comp in components.items() if comp.role == Role.IDENTIFIER]

    # Check for missing columns
    data_columns = data.columns
    for name in components:
        if name not in data_columns:
            if not components[name].nullable:
                raise SemanticError("0-1-1-10", name=dataset_name, comp_name=name)
            # Add NULL column
            data = data.project(f"*, NULL AS {name}")

    # Null check for identifiers
    for id_name in id_names:
        nulls = data.filter(f"{id_name} IS NULL").limit(1)
        if nulls.count("1").fetchone()[0] > 0:
            raise SemanticError("0-1-1-4", null_identifier=id_name, name=dataset_name)

    # Require at least 1 identifier if more than 1 row
    if not id_names:
        rowcount = data.limit(2).count("1").fetchone()[0]
        if rowcount > 1:
            raise SemanticError("0-1-1-5", name=dataset_name)

    # Type validation and normalization
    for name, comp in components.items():
        col = name
        dtype = comp.data_type

        if dtype in [Integer, Number, Boolean]:
            data = data.project(f"*, TRY_CAST({col} AS {dtype().sql_type}) AS {col}_chk") \
                       .filter(f"{col}_chk IS NOT NULL") \
                       .project(f"* EXCLUDE {col}_chk")

        elif dtype in [Date, TimeInterval, TimePeriod]:
            check_method = f"check_{dtype.__name__}".lower()
            data = data.project(f"*, {check_method}({col}) AS {col}_chk") \
                       .filter(f"{col}_chk IS NOT NULL") \
                       .project(f"* EXCLUDE {col}_chk")

        else:
            # Strip quotes
            data = data.project(f"*, REPLACE({col}::TEXT, '\"', '') AS {col}_clean") \
                       .project(f"* EXCLUDE {col}").project(f"*, {col}_clean AS {col}") \
                       .project(f"* EXCLUDE {col}_clean")

    a = data.df()
    b = 0

    return data


def load_datapoints(
    components: Dict[str, Component],
    dataset_name: str,
    csv_path: Optional[Union[Path, str]] = None,
) -> DuckDBPyRelation:
    if csv_path is None or (isinstance(csv_path, Path) and not csv_path.exists()):
        # Empty dataset as table
        column_defs = ", ".join([f'"{name}" VARCHAR' for name in components])
        rel = con.query(f"SELECT {', '.join(f'NULL::{col.split()[1]}' for col in 
                                            column_defs.split(','))} LIMIT 0")
        return _sanitize_duckdb_columns(components, None, rel)

    elif isinstance(csv_path, (str, Path)):
        path_str = str(csv_path)
        if isinstance(csv_path, Path):
            _validate_csv_path(components, csv_path)

        # Lazy CSV read
        rel = con.from_csv_auto(path_str, header=True)

        # Type validation and normalization
        rel = _validate_duckdb(components, rel, dataset_name)

        return _sanitize_duckdb_columns(components, csv_path, rel)

    else:
        raise Exception("Invalid csv_path type")


def _fill_dataset_empty_data(dataset: Dataset) -> None:
    if not dataset.components:
        dataset.data = con.query("SELECT NULL LIMIT 0")
        return

    column_defs = ", ".join([
        f"NULL::{comp.data_type().sql_type} AS \"{name}\""
        for name, comp in dataset.components.items()
    ])
    dataset.data = con.query(f"SELECT {column_defs} LIMIT 0")

