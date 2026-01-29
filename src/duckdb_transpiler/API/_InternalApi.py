"""
Internal API for loading VTL dataset structures and datapoints into DuckDB.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import duckdb
import jsonschema
import pandas as pd

from duckdb_transpiler.Parser import load_datapoints_duckdb
from duckdb_transpiler.Utils import get_sql_type
from vtlengine.__extras_check import __check_s3_extra
from vtlengine.DataTypes import SCALAR_TYPES
from vtlengine.Exceptions import DataLoadError, InputValidationException, check_key
from vtlengine.files.parser import _validate_pandas
from vtlengine.Model import Component, Dataset, Role, Role_keys, Scalar

_SCALAR_TYPE_KEYS = set(SCALAR_TYPES.keys())

_SCHEMA_PATH = Path(__file__).parent / "data" / "schema"
with open(_SCHEMA_PATH / "json_schema_2.1.json") as f:
    _STRUCTURE_SCHEMA = json.load(f)


# =============================================================================
# Structure Parsing
# =============================================================================


def _build_component(comp_dict: Dict[str, Any]) -> Component:
    """Build VTL Component from dictionary."""
    # Extract type
    type_key = "type" if "type" in comp_dict else "data_type"
    type_value = comp_dict[type_key]
    check_key(type_key, _SCALAR_TYPE_KEYS, type_value)

    # Extract role (ViralAttribute -> Attribute)
    role_str = comp_dict["role"]
    if role_str == "ViralAttribute":
        role_str = "Attribute"
    check_key("role", Role_keys, role_str)
    role = Role(role_str)

    # Nullable defaults: Identifiers=False, others=True
    nullable = comp_dict.get("nullable", role in (Role.MEASURE, Role.ATTRIBUTE))

    return Component(
        name=comp_dict["name"],
        data_type=SCALAR_TYPES[type_value],
        role=role,
        nullable=nullable,
    )


def _parse_structures(
    structures: Dict[str, Any],
) -> Tuple[Dict[str, Dataset], Dict[str, Scalar]]:
    """Parse JSON structure into Dataset and Scalar objects."""
    datasets: Dict[str, Dataset] = {}
    scalars: Dict[str, Scalar] = {}

    # Shared structures lookup
    structure_map = {s["name"]: s for s in structures.get("structures", [])}

    for ds_json in structures.get("datasets", []):
        name = ds_json["name"]
        components: Dict[str, Component] = {}

        # Shared structure reference
        if "structure" in ds_json:
            struct = structure_map.get(ds_json["structure"])
            if struct is None:
                raise InputValidationException(code="0-2-1-2", message="Structure not found.")
            try:
                jsonschema.validate(instance=struct, schema=_STRUCTURE_SCHEMA)
            except jsonschema.exceptions.ValidationError as e:
                raise InputValidationException(code="0-2-1-2", message=e.message)
            for comp in struct["components"]:
                components[comp["name"]] = _build_component(comp)

        # Inline DataStructure
        if "DataStructure" in ds_json:
            for comp in ds_json["DataStructure"]:
                components[comp["name"]] = _build_component(comp)

        datasets[name] = Dataset(name=name, components=components, data=None)

    # Parse scalars
    for sc_json in structures.get("scalars", []):
        check_key("type", _SCALAR_TYPE_KEYS, sc_json["type"])
        scalars[sc_json["name"]] = Scalar(
            name=sc_json["name"],
            data_type=SCALAR_TYPES[sc_json["type"]],
            value=None,
        )

    return datasets, scalars


def load_datasets(
    data_structure: Union[Dict[str, Any], Path, List[Union[Dict[str, Any], Path]]],
) -> Tuple[Dict[str, Dataset], Dict[str, Scalar]]:
    """Load dataset structures from JSON definitions."""
    if isinstance(data_structure, dict):
        return _parse_structures(data_structure)

    if isinstance(data_structure, Path):
        if not data_structure.exists():
            raise DataLoadError(code="0-3-1-1", file=data_structure)

        if data_structure.is_dir():
            datasets, scalars = {}, {}
            for f in data_structure.iterdir():
                if f.suffix == ".json":
                    ds, sc = load_datasets(f)
                    datasets.update(ds)
                    scalars.update(sc)
            return datasets, scalars

        if data_structure.suffix != ".json":
            raise InputValidationException(
                code="0-1-1-3", expected_ext=".json", ext=data_structure.suffix
            )
        with open(data_structure) as f:
            return _parse_structures(json.load(f))

    # List of structures
    datasets, scalars = {}, {}
    for item in data_structure:
        ds, sc = load_datasets(item)
        datasets.update(ds)
        scalars.update(sc)
    return datasets, scalars


# =============================================================================
# Datapoint Resolution
# =============================================================================


def _resolve_datapoints(
    datapoints: Union[Dict[str, Union[str, Path]], List[Union[str, Path]], str, Path],
) -> Dict[str, Union[str, Path]]:
    """Normalize datapoints to {name: path} mapping."""
    if isinstance(datapoints, dict):
        result = {}
        for name, dp in datapoints.items():
            if isinstance(dp, pd.DataFrame):
                result[name] = dp
            else:
                result[name] = _resolve_single_path(dp)
        return result

    if isinstance(datapoints, list):
        result = {}
        for dp in datapoints:
            result.update(_resolve_single_path(dp, return_dict=True))
        return result

    return _resolve_single_path(datapoints, return_dict=True)


def _resolve_single_path(
    datapoint: Union[str, Path], return_dict: bool = False
) -> Union[Path, str, Dict[str, Union[str, Path]]]:
    """Resolve single path/S3 URI."""
    # S3 URI
    if isinstance(datapoint, str) and datapoint.startswith("s3://"):
        __check_s3_extra()
        if return_dict:
            name = datapoint.rsplit("/", 1)[-1].removesuffix(".csv")
            return {name: datapoint}
        return datapoint

    path = Path(datapoint) if isinstance(datapoint, str) else datapoint
    if not path.exists():
        raise DataLoadError(code="0-3-1-1", file=path)

    # Directory: collect all CSVs
    if path.is_dir():
        return {f.stem: f for f in path.iterdir() if f.suffix == ".csv"}

    if return_dict:
        return {path.stem: path}
    return path


# =============================================================================
# Scalar Handling
# =============================================================================


def _apply_scalar_values(
    scalars: Dict[str, Scalar],
    scalar_values: Optional[Dict[str, Optional[Union[int, str, bool, float]]]] = None,
) -> None:
    """Apply provided values to scalar objects with type validation."""
    if not scalar_values:
        return

    for name, value in scalar_values.items():
        if name not in scalars:
            raise InputValidationException(code="0-1-2-6", name=name)

        scalar = scalars[name]
        if not scalar.data_type.check(value):
            raise InputValidationException(
                code="0-1-2-7",
                value=value,
                type_=scalar.data_type.__name__,
                op_type=type(scalar).__name__,
                name=name,
            )
        scalar.value = scalar.data_type.cast(value)


# =============================================================================
# DuckDB Loading
# =============================================================================


def load_datasets_with_data_duckdb(
    data_structures: Any,
    datapoints: Optional[
        Union[Dict[str, Union[pd.DataFrame, Path, str]], List[Union[str, Path]], Path, str]
    ] = None,
    scalar_values: Optional[Dict[str, Optional[Union[int, str, bool, float]]]] = None,
) -> Tuple[duckdb.DuckDBPyConnection, Dict[str, Dataset], Dict[str, Scalar]]:
    """
    Load dataset structures and data into DuckDB.

    Returns:
        Tuple of (connection, datasets, scalars)
    """
    datasets, scalars = load_datasets(data_structures)
    conn = duckdb.connect()

    # Apply scalar values (they get inlined in SQL queries by the transpiler)
    _apply_scalar_values(scalars, scalar_values)

    # No datapoints: create empty tables
    if datapoints is None:
        for name, ds in datasets.items():
            _create_empty_table(conn, name, ds.components)
        return conn, datasets, scalars

    # Handle DataFrame dict
    if isinstance(datapoints, dict) and all(
        isinstance(v, pd.DataFrame) for v in datapoints.values()
    ):
        for name, df in datapoints.items():
            if name not in datasets:
                raise InputValidationException(f"Dataset '{name}' not found in structures.")
            validated = _validate_pandas(datasets[name].components, df, name)  # noqa: F841
            conn.execute(f'CREATE TABLE "{name}" AS SELECT * FROM validated')

        for name, ds in datasets.items():
            if name not in datapoints:
                _create_empty_table(conn, name, ds.components)
        return conn, datasets, scalars

    # Handle path-based datapoints
    resolved = _resolve_datapoints(datapoints)
    for name, path in resolved.items():
        if name not in datasets:
            raise InputValidationException(f"Dataset '{name}' not found in structures.")
        load_datapoints_duckdb(
            conn=conn,
            components=datasets[name].components,
            dataset_name=name,
            csv_path=path,
        )

    for name, ds in datasets.items():
        if name not in resolved:
            _create_empty_table(conn, name, ds.components)

    return conn, datasets, scalars


def _create_empty_table(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    components: Dict[str, Component],
) -> None:
    """Create empty table with schema from VTL components."""
    col_defs = [f'"{name}" {get_sql_type(comp.data_type)}' for name, comp in components.items()]
    conn.execute(f'CREATE TABLE "{table_name}" ({", ".join(col_defs)})')
