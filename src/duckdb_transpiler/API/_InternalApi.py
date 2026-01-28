"""
Internal API for loading VTL dataset structures and datapoints.

Provides functions to:
- Parse JSON data structures into VTL Dataset/Scalar objects
- Load datapoints from CSV files, Pandas DataFrames, or S3 URIs
- Validate data against structure definitions
- Create DuckDB tables for out-of-core processing
"""

import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import duckdb
import jsonschema
import pandas as pd

from duckdb_transpiler.Parser import load_datapoints_duckdb
from vtlengine.__extras_check import __check_s3_extra
from vtlengine.DataTypes import SCALAR_TYPES
from vtlengine.Exceptions import DataLoadError, InputValidationException, check_key
from vtlengine.files.parser import _validate_pandas
from vtlengine.Model import Component as VTL_Component
from vtlengine.Model import Dataset, Role, Role_keys, Scalar

# =============================================================================
# Module Constants
# =============================================================================

_SCALAR_TYPE_KEYS = frozenset(SCALAR_TYPES.keys())

# Load JSON schemas at module import
_SCHEMA_PATH = Path(__file__).parent / "data" / "schema"
with open(_SCHEMA_PATH / "json_schema_2.1.json") as f:
    _STRUCTURE_SCHEMA = json.load(f)
with open(_SCHEMA_PATH / "value_domain_schema.json") as f:
    vd_schema = json.load(f)
with open(_SCHEMA_PATH / "external_routines_schema.json") as f:
    external_routine_schema = json.load(f)


# =============================================================================
# Structure Parsing Helpers
# =============================================================================


def _extract_data_type(component: Dict[str, Any]) -> Tuple[str, type]:
    """
    Extract VTL data type from component dict.

    Supports 'type' (preferred) or 'data_type' (legacy) keys.

    Raises:
        InputValidationException: Unknown type value.
    """
    key = "type" if "type" in component else "data_type"
    value = component[key]
    check_key(key, _SCALAR_TYPE_KEYS, value)
    return key, SCALAR_TYPES[value]


def _build_component(comp_dict: Dict[str, Any]) -> VTL_Component:
    """
    Build VTL Component from dictionary definition.

    Handles nullable defaults: Identifiers=False, Measures/Attributes=True.
    """
    _, scalar_type = _extract_data_type(comp_dict)

    # ViralAttribute is treated as Attribute
    role_str = comp_dict["role"]
    if role_str == "ViralAttribute":
        role_str = "Attribute"
    check_key("role", Role_keys, role_str)
    role = Role(role_str)

    # Default nullable based on role (identifiers=False, measures/attributes=True)
    nullable = comp_dict.get("nullable", role in (Role.MEASURE, Role.ATTRIBUTE))

    return VTL_Component(
        name=comp_dict["name"],
        data_type=scalar_type,
        role=role,
        nullable=nullable,
    )


def _load_dataset_from_structure(
    structures: Dict[str, Any],
) -> Tuple[Dict[str, Dataset], Dict[str, Scalar]]:
    """
    Parse JSON structure definition into Dataset and Scalar objects.

    Supports two formats:
    - Shared structures: datasets reference structures by name
    - Inline DataStructure: components defined directly in dataset

    Raises:
        InputValidationException: Invalid structure format or missing reference.
    """
    datasets: Dict[str, Dataset] = {}
    scalars: Dict[str, Scalar] = {}

    # Build structure lookup for shared definitions
    structure_map = {s["name"]: s for s in structures.get("structures", [])}

    for ds_json in structures.get("datasets", []):
        name = ds_json["name"]
        components: Dict[str, VTL_Component] = {}

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


# =============================================================================
# Datapoint Path Resolution
# =============================================================================


def _path_to_dataset_name(path: Path) -> str:
    """Extract dataset name from CSV filename (removes .csv extension)."""
    return path.name.removesuffix(".csv")


def _load_single_datapoint(datapoint: Union[str, Path]) -> Dict[str, Union[str, Path]]:
    """
    Resolve a single datapoint reference to {dataset_name: path} mapping.

    Handles:
    - S3 URIs (s3://bucket/path/file.csv)
    - Single CSV files
    - Directories containing CSV files

    Raises:
        InputValidationException: Invalid input type.
        DataLoadError: File/directory not found.
    """
    if not isinstance(datapoint, (str, Path)):
        raise InputValidationException(
            code="0-1-1-2", input=datapoint, message="Input must be a Path or S3 URI"
        )

    # S3 URI handling
    if isinstance(datapoint, str) and datapoint.startswith("s3://"):
        __check_s3_extra()
        name = datapoint.rsplit("/", 1)[-1].removesuffix(".csv")
        return {name: datapoint}

    # Convert to Path if string
    try:
        path = Path(datapoint) if isinstance(datapoint, str) else datapoint
    except Exception:
        raise InputValidationException(
            code="0-1-1-2", input=datapoint, message="Invalid path format"
        )

    if not path.exists():
        raise DataLoadError(code="0-3-1-1", file=path)

    # Directory: collect all CSVs
    if path.is_dir():
        return {_path_to_dataset_name(f): f for f in path.iterdir() if f.suffix == ".csv"}

    return {_path_to_dataset_name(path): path}


def _check_duplicate_names(new_names: List[str], existing: List[str]) -> None:
    """Raise if any dataset name already exists."""
    duplicates = set(new_names) & set(existing)
    if duplicates:
        raise InputValidationException(
            f"Duplicate dataset names in datapoints: {', '.join(duplicates)}"
        )


def _load_datapoints_path(
    datapoints: Union[Dict[str, Union[str, Path]], List[Union[str, Path]], str, Path],
) -> Dict[str, Union[str, Path]]:
    """
    Normalize datapoints input to {dataset_name: path} mapping.

    Args:
        datapoints: Dict mapping names to paths, list of paths, or single path.

    Raises:
        InputValidationException: Invalid types or duplicate names.
    """
    result: Dict[str, Union[str, Path]] = {}

    if isinstance(datapoints, dict):
        for name, dp in datapoints.items():
            if not isinstance(name, str):
                raise InputValidationException(
                    code="0-1-1-2", input=name, message="Dict keys must be strings"
                )
            if not isinstance(dp, (str, Path)):
                raise InputValidationException(
                    code="0-1-1-2", input=dp, message="Dict values must be Path or S3 URI"
                )
            resolved = _load_single_datapoint(dp)
            _check_duplicate_names([name], list(result.keys()))
            # Use provided name, not inferred from filename
            result[name] = list(resolved.values())[0]
        return result

    if isinstance(datapoints, list):
        for dp in datapoints:
            resolved = _load_single_datapoint(dp)
            _check_duplicate_names(list(resolved.keys()), list(result.keys()))
            result.update(resolved)
        return result

    return _load_single_datapoint(datapoints)


# =============================================================================
# Structure Loading
# =============================================================================


def _load_datastructure_single(
    data_structure: Union[Dict[str, Any], Path],
) -> Tuple[Dict[str, Dataset], Dict[str, Scalar]]:
    """
    Load a single structure definition from dict or JSON file.

    Recursively processes directories containing .json files.

    Raises:
        InputValidationException: Invalid input type or file extension.
        DataLoadError: File not found.
    """
    if isinstance(data_structure, dict):
        return _load_dataset_from_structure(data_structure)

    if not isinstance(data_structure, Path):
        raise InputValidationException(
            code="0-1-1-2", input=data_structure, message="Must be dict or Path"
        )
    if not data_structure.exists():
        raise DataLoadError(code="0-3-1-1", file=data_structure)

    # Directory: merge all JSON files
    if data_structure.is_dir():
        datasets: Dict[str, Dataset] = {}
        scalars: Dict[str, Scalar] = {}
        for f in data_structure.iterdir():
            if f.suffix == ".json":
                ds, sc = _load_datastructure_single(f)
                datasets.update(ds)
                scalars.update(sc)
        return datasets, scalars

    # Single file
    if data_structure.suffix != ".json":
        raise InputValidationException(
            code="0-1-1-3", expected_ext=".json", ext=data_structure.suffix
        )
    with open(data_structure) as file:
        return _load_dataset_from_structure(json.load(file))


def load_datasets(
    data_structure: Union[Dict[str, Any], Path, List[Union[Dict[str, Any], Path]]],
) -> Tuple[Dict[str, Dataset], Dict[str, Scalar]]:
    """
    Load dataset structures from JSON definitions.

    Args:
        data_structure: Single dict/Path or list of dicts/Paths with structure definitions.

    Returns:
        Tuple of (datasets, scalars) dictionaries keyed by name.

    Raises:
        InputValidationException: Invalid structure format.
        DataLoadError: File not found.
    """
    if isinstance(data_structure, dict):
        return _load_datastructure_single(data_structure)
    if isinstance(data_structure, Path):
        return _load_datastructure_single(data_structure)

    # List of structures: merge all
    datasets: Dict[str, Dataset] = {}
    scalars: Dict[str, Scalar] = {}
    for item in data_structure:
        ds, sc = _load_datastructure_single(item)
        datasets.update(ds)
        scalars.update(sc)
    return datasets, scalars


# =============================================================================
# Scalar and Dataset Helpers
# =============================================================================


def _handle_scalars_values(
    scalars: Dict[str, Scalar],
    scalar_values: Optional[Dict[str, Optional[Union[int, str, bool, float]]]] = None,
) -> None:
    """
    Populate scalar objects with provided values.

    Validates type compatibility and casts values.

    Raises:
        InputValidationException: Unknown scalar name (0-1-2-6) or type mismatch (0-1-2-7).
    """
    if scalar_values is None:
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
# Main Data Loading Functions
# =============================================================================


def _validate_datapoints_dict(datapoints: Dict[str, Any]) -> bool:
    """Check if dict values are all DataFrames (True) or all Path/str (False)."""
    if all(isinstance(v, pd.DataFrame) for v in datapoints.values()):
        return True
    if all(isinstance(v, (str, Path)) for v in datapoints.values()):
        return False
    raise InputValidationException(
        "Invalid datapoints: dict values must be all DataFrames or all Paths/S3 URIs."
    )


def _check_dataset_exists(name: str, datasets: Dict[str, Dataset]) -> None:
    """Raise if dataset not in structures."""
    if name not in datasets:
        raise InputValidationException(f"Dataset '{name}' not found in data structures.")


# =============================================================================
# DuckDB-based Data Loading
# =============================================================================


def load_datasets_with_data_duckdb(
    data_structures: Any,
    datapoints: Optional[
        Union[Dict[str, Union[pd.DataFrame, Path, str]], List[Union[str, Path]], Path, str]
    ] = None,
    scalar_values: Optional[Dict[str, Optional[Union[int, str, bool, float]]]] = None,
) -> Tuple[duckdb.DuckDBPyConnection, Dict[str, Dataset], Dict[str, Scalar]]:
    """
    Load dataset structures and data into DuckDB for out-of-core processing.

    Creates a DuckDB in-memory connection with all datasets as SQL tables.
    Use this for large datasets that don't fit in memory.

    Args:
        data_structures: JSON structure definitions as dict, Path, or list thereof.
        datapoints: Data source (same formats as load_datasets_with_data).
        scalar_values: Values for declared scalars.

    Returns:
        Tuple of:
            - conn: DuckDB connection with tables registered
            - datasets: Dict[str, Dataset] (metadata only, no DataFrame)
            - scalars: Dict[str, Scalar] with values

    Raises:
        InputValidationException: Invalid input or dataset not in structures.
        DataLoadError: File not found or validation failure.
    """
    datasets, scalars = load_datasets(data_structures)
    _handle_scalars_values(scalars, scalar_values)
    conn = duckdb.connect()

    # No datapoints: create empty tables
    if datapoints is None:
        for name, ds in datasets.items():
            _create_empty_table(conn, name, ds.components)
        return conn, datasets, scalars

    # DataFrame dict: validate and create tables
    if isinstance(datapoints, dict) and _validate_datapoints_dict(datapoints):
        for name, data in datapoints.items():
            _check_dataset_exists(name, datasets)
            if isinstance(data, pd.DataFrame):
                validated_data = _validate_pandas(datasets[name].components, data, name)  # noqa: F841
                conn.execute(f'CREATE TABLE "{name}" AS SELECT * FROM validated_data')

        # Empty tables for missing datasets
        for name, ds in datasets.items():
            if name not in datapoints:
                _create_empty_table(conn, name, ds.components)
        return conn, datasets, scalars

    # Path-based: load CSVs directly into DuckDB
    datapoints_path = _load_datapoints_path(datapoints)  # type: ignore[arg-type]
    for name, csv_path in datapoints_path.items():
        _check_dataset_exists(name, datasets)
        load_datapoints_duckdb(
            conn=conn,
            components=datasets[name].components,
            dataset_name=name,
            csv_path=csv_path,
        )

    # Empty tables for missing datasets
    for name, ds in datasets.items():
        if name not in datapoints_path:
            _create_empty_table(conn, name, ds.components)

    gc.collect()
    return conn, datasets, scalars


# =============================================================================
# DuckDB Table Creation Helpers
# =============================================================================


def _create_empty_table(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    components: Dict[str, VTL_Component],
) -> None:
    """Create empty table with schema from VTL components."""
    from duckdb_transpiler.DataTypes import get_duckdb_type

    col_defs = [f'"{name}" {get_duckdb_type(comp.data_type)}' for name, comp in components.items()]
    conn.execute(f'CREATE TABLE "{table_name}" ({", ".join(col_defs)})')
