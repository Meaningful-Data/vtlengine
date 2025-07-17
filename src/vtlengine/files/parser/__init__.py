import csv
from csv import DictReader
from pathlib import Path
from typing import Dict, List, Optional, Union

from duckdb.duckdb import DuckDBPyRelation  # type: ignore[import-untyped]

from vtlengine.connection import con
from vtlengine.DataTypes import Duration, TimeInterval, TimePeriod
from vtlengine.duckdb.duckdb_utils import empty_relation
from vtlengine.Exceptions import InputValidationException, SemanticError
from vtlengine.files.parser._rfc_dialect import register_rfc
from vtlengine.Model import Component, Dataset, Role
from vtlengine.files.parser._time_checking import load_time_checks

load_time_checks(con)


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
        raise InputValidationException(code="0-1-1-7")

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

        modified_data = modified_data.project(", ".join([f'"{col}"' for col in remaining_cols]))
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
                f'*, NULL::{comp.data_type().sql_type} AS "{comp_name}"'
            )
            column_names = modified_data.columns

    return modified_data


def _validate_duckdb(
    components: Dict[str, Component],
    data: DuckDBPyRelation,
    dataset_name: str,
) -> DuckDBPyRelation:
    # Check for missing columns
    data_columns = data.columns
    for col in components:
        if col not in data_columns:
            if not components[col].nullable:
                raise SemanticError(code="0-1-1-10", name=dataset_name, comp_name=col)
            # Add NULL column
            data = data.project(f'*, NULL AS "{col}"')

    # Check dataset integrity
    check_nulls(components, data, dataset_name)
    check_duplicates(components, data, dataset_name)
    check_dwi(components, data, dataset_name)

    exprs = []
    for col, comp in components.items():
        dtype = comp.data_type
        if dtype in [Duration, TimeInterval, TimePeriod]:
            check_method = f'check_{dtype.__name__}'.lower()
            exprs.append(f'{check_method}("{col}") AS "{col}"')
        else:
            exprs.append(f'"{col}"')

    final_query = ', '.join(exprs)
    data = data.project(final_query)

    return data


def check_nulls(
    components: Dict[str, Component], data: DuckDBPyRelation, dataset_name: str
) -> None:
    id_names = [name for name, comp in components.items() if comp.role == Role.IDENTIFIER]
    non_nullable = [
        comp.name
        for comp in components.values()
        if not comp.nullable or comp.role == Role.IDENTIFIER
    ]
    query = (
        'SELECT '
        + ', '.join(
            [
                f'COUNT(CASE WHEN "{col}" IS NULL THEN 1 END) AS "{col}_null_count"'
                for col in non_nullable
            ]
        )
        + ' FROM data'
    )
    null_counts = con.execute(query).fetchone()

    for col, null_count in zip(non_nullable, null_counts):  # type: ignore[arg-type]
        if null_count > 0:
            if col in id_names:
                raise SemanticError(code="0-1-1-4", null_identifier=col, name=dataset_name)
            raise SemanticError(code="0-1-1-15", measure=col, name=dataset_name)


def check_duplicates(
    components: Dict[str, Component],
    data: DuckDBPyRelation,
    dataset_name: str,
) -> None:
    pass
    # id_names = [name for name, comp in components.items() if comp.role == Role.IDENTIFIER]
    #
    # if id_names:
    #     query = f"""
    #             SELECT COUNT(*) > 0 from (
    #                 SELECT COUNT(*) as count
    #                 FROM data
    #                 GROUP BY {', '.join(id_names)}
    #                 HAVING COUNT(*) > 1
    #             ) AS duplicates
    #             """
    #
    #     dup = con.execute(query).fetchone()[0]
    #     if dup:
    #         raise InputValidationException(code="0-1-1-6")


def check_dwi(
    components: Dict[str, Component],
    data: DuckDBPyRelation,
    dataset_name: str,
) -> None:
    id_names = [name for name, comp in components.items() if comp.role == Role.IDENTIFIER]

    if not id_names:
        rowcount = con.execute("SELECT COUNT(*) FROM data LIMIT 2").fetchone()[0]  # type: ignore[index]
        if rowcount > 1:
            raise SemanticError(code="0-1-1-5", name=dataset_name)


def load_datapoints(
    components: Dict[str, Component],
    dataset_name: str,
    csv_path: Optional[Union[Path, str]] = None,
) -> DuckDBPyRelation:
    if csv_path is None or (isinstance(csv_path, Path) and not csv_path.exists()):
        # Empty dataset as table
        return empty_relation(list(components.keys()))

    elif isinstance(csv_path, (str, Path)):
        path_str = str(csv_path)
        if isinstance(csv_path, Path):
            _validate_csv_path(components, csv_path)

        # Lazy CSV read
        with open(path_str, mode="r", encoding="utf-8") as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)  # Extract the header to determine column order

        # Extract data types from components in header
        dtypes = {
            col: comp.data_type().sql_type
            for col in header
            for comp in components.values()
            if comp.name == col
        }

        # Read the CSV file
        rel = con.read_csv(
            path_str,
            header=True,
            columns=dtypes,
            delimiter=",",
            quotechar='"',
            ignore_errors=True,
            date_format="%Y-%m-%d",
        )

        # Type validation and normalization
        rel = _validate_duckdb(components, rel, dataset_name)

        return _sanitize_duckdb_columns(components, csv_path, rel)

    else:
        raise Exception("Invalid csv_path type")


def _fill_dataset_empty_data(dataset: Dataset) -> None:
    if not dataset.components:
        dataset.data = con.query("SELECT NULL LIMIT 0")
        return

    column_defs = ", ".join(
        [
            f'NULL::{comp.data_type().sql_type} AS "{name}"'
            for name, comp in dataset.components.items()
        ]
    )
    dataset.data = con.query(f"SELECT {column_defs} LIMIT 0")
