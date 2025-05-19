from csv import DictReader
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import duckdb
from vtlengine.connection import con

from vtlengine.DataTypes import (
    SCALAR_TYPES_CLASS_REVERSE,
    Boolean,
    Date,
    Duration,
    Integer,
    Number,
    ScalarType,
    TimeInterval,
    TimePeriod, String,
)

from vtlengine.Exceptions import InputValidationException, SemanticError
from vtlengine.files.parser._rfc_dialect import register_rfc
from vtlengine.files.parser._time_checking import (
    check_date,
    check_time,
    check_time_period,
)
from vtlengine.Model import Component, Dataset, Role

TIME_CHECKS_MAPPING: Dict[Type[ScalarType], Any] = {
    Date: check_date,
    TimePeriod: check_time_period,
    TimeInterval: check_time,
}

DUCKDB_TYPES_MAPPING = {
    String: "VARCHAR",
    Integer: "INTEGER",
    Number: "DOUBLE",
    Boolean: "BOOLEAN",
    Date: "DATE",
    Duration: "VARCHAR",
    TimeInterval: "VARCHAR",
    TimePeriod: "VARCHAR",
}


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
    components: Dict[str, Component], csv_path: Union[str, Path], rel: duckdb.DuckDBPyRelation
) -> duckdb.DuckDBPyRelation:
    # Retrieve column names
    cols = rel.columns

    # Handle fast-loading SDMX-CSV extra headers
    if "DATAFLOW" in cols and cols[0] == "DATAFLOW" and "DATAFLOW" not in components:
        rel = rel.project(", ".join([c for c in cols if c != "DATAFLOW"]))
        cols.remove("DATAFLOW")

    if "STRUCTURE" in cols and cols[0] == "STRUCTURE":
        drop_cols = []
        if "STRUCTURE" not in components:
            drop_cols.append("STRUCTURE")
        if "STRUCTURE_ID" in cols:
            drop_cols.append("STRUCTURE_ID")

        if "ACTION" in cols:
            # Filter rows where ACTION != 'D' and remove the ACTION column
            rel = rel.filter("ACTION != 'D'")
            drop_cols.append("ACTION")

        rel = rel.project(", ".join([c for c in cols if c not in drop_cols]))
        cols = [c for c in cols if c not in drop_cols]

    # Check that all required IDENTIFIER components exist
    comp_names = set([c.name for c in components.values() if c.role == Role.IDENTIFIER])
    comps_missing = [id_m for id_m in comp_names if id_m not in cols]
    if comps_missing:
        file = csv_path if isinstance(csv_path, str) else csv_path.name
        raise InputValidationException(code="0-1-1-7", ids=", ".join(comps_missing), file=file)

    # Add missing nullable columns as NULL
    for comp_name, comp in components.items():
        if comp_name not in cols:
            if not comp.nullable:
                raise InputValidationException(f"Component {comp_name} is missing in the file.")
            rel = rel.project(f"*, NULL AS {comp_name}")
            cols.append(comp_name)

    return rel


def _duckdb_load_csv_lazy(components: Dict[str, Component], csv_path: Union[str, Path]) -> duckdb.DuckDBPyRelation:
    csv_path_str = str(csv_path)

    # Build column type mapping from components
    duckdb_types = {name: DUCKDB_TYPES_MAPPING[c.data_type] for name, c in components.items()}

    # Lazily load CSV with type enforcement
    rel = con.read_csv(
        csv_path_str,
        columns=duckdb_types,
    )

    # rel = con.read_csv(
    #     str(csv_path),
    #     header=True,
    #     columns=duckdb_types,
    #     na_values=[""],
    #     union_by_name=True,
    #     all_varchar=True,  # Makes the coltypes definition work consistently
    #     sample_size=-1  # Avoids sampling for auto-detection
    # )

    return _sanitize_duckdb_columns(components, csv_path, rel)


def _parse_boolean(value: str) -> bool:
    if isinstance(value, bool):
        return value
    result = value.lower() == "true" or value == "1"
    return result


def _validate_duckdb(
    components: Dict[str, Component], rel: duckdb.DuckDBPyRelation, dataset_name: str
) -> duckdb.DuckDBPyRelation:
    # Get existing column names
    col_names = rel.columns

    # Add missing nullable columns as NULLs
    for name, comp in components.items():
        if name not in col_names:
            if not comp.nullable:
                raise SemanticError("0-1-1-10", name=dataset_name, comp_name=name)
            rel = rel.project(f"*, NULL AS {name}")
            col_names.append(name)

    # Check for NULLs in identifier columns
    id_names = [name for name, comp in components.items() if comp.role == Role.IDENTIFIER]
    for id_name in id_names:
        null_count = rel.filter(f"{id_name} IS NULL").count("*").fetchone()[0]
        if null_count > 0:
            raise SemanticError("0-1-1-4", null_identifier=id_name, name=dataset_name)

    # If no identifiers and more than one row, raise error
    if not id_names:
        if rel.columns > 1:
            raise SemanticError("0-1-1-5", name=dataset_name)

    # Prepare column transformation expressions
    project_expressions = []

    for comp_name, comp in components.items():
        dtype = comp.data_type

        try:
            if dtype == Integer:
                expr = f"CAST({comp_name} AS BIGINT)"
            elif dtype == Number:
                expr = f"CAST({comp_name} AS DOUBLE)"
            elif dtype == Boolean:
                expr = f"""
                    CASE
                        WHEN LOWER({comp_name}) IN ('true', '1', 'yes') THEN TRUE
                        WHEN LOWER({comp_name}) IN ('false', '0', 'no') THEN FALSE
                        WHEN {comp_name} IS NULL THEN NULL
                        ELSE NULL
                    END
                """
            elif dtype in (Date, TimePeriod, TimeInterval):
                expr = f"CAST({comp_name} AS DATE)"
            elif dtype == Duration:
                expr = f"""
                    CASE
                        WHEN REGEXP_MATCHES(REPLACE({comp_name}, ' ', ''), '^P.*$') THEN {comp_name}
                        ELSE NULL
                    END
                """
            else:
                expr = f"REPLACE(CAST({comp_name} AS VARCHAR), '\"', '')"
        except Exception:
            str_comp = SCALAR_TYPES_CLASS_REVERSE.get(dtype, "Unknown")
            raise SemanticError("0-1-1-12", name=dataset_name, column=comp_name, type=str_comp)

        project_expressions.append(f"{expr} AS {comp_name}")

    # Apply transformations
    rel = rel.project(", ".join(project_expressions))
    return rel


def load_datapoints(
    components: Dict[str, Component],
    dataset_name: str,
    csv_path: Optional[Union[Path, str]] = None,
) -> duckdb.DuckDBPyRelation:
    # If path is missing or file does not exist, return an empty lazy relation
    if csv_path is None or (isinstance(csv_path, Path) and not csv_path.exists()):
        cols = ", ".join(f"NULL::VARCHAR AS {name}" for name in components.keys())
        return duckdb.sql(f"SELECT {cols} WHERE FALSE")

    # Validate path
    if isinstance(csv_path, (str, Path)):
        if isinstance(csv_path, Path):
            _validate_csv_path(components, csv_path)
        data = _duckdb_load_csv_lazy(components, csv_path)
    else:
        raise Exception("Invalid csv_path type")

    # Validate content lazily
    data = _validate_duckdb(components, data, dataset_name)
    return data


def _fill_dataset_empty_data(dataset: Dataset) -> None:
    # Create a DuckDB relation with no rows
    cols = ", ".join(f"NULL::VARCHAR AS {name}" for name in dataset.components.keys())
    dataset.data = duckdb.sql(f"SELECT {cols} WHERE FALSE")
