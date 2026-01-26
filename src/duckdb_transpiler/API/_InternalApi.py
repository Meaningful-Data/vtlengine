import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import duckdb
import jsonschema
import pandas as pd

from duckdb_transpiler.Parser import load_datapoints_duckdb
from vtlengine import AST as AST
from vtlengine.__extras_check import __check_s3_extra
from vtlengine.DataTypes import SCALAR_TYPES
from vtlengine.Exceptions import (
    DataLoadError,
    InputValidationException,
    check_key,
)
from vtlengine.files.parser import (
    _fill_dataset_empty_data,
    _validate_pandas,
    load_datapoints,
)
from vtlengine.Model import (
    Component as VTL_Component,
)
from vtlengine.Model import (
    Dataset,
    Role,
    Role_keys,
    Scalar,
)

# Cache SCALAR_TYPES keys for performance
_SCALAR_TYPE_KEYS = SCALAR_TYPES.keys()

base_path = Path(__file__).parent
schema_path = base_path / "data" / "schema"
sdmx_csv_path = base_path / "data" / "sdmx_csv"
with open(schema_path / "json_schema_2.1.json", "r") as file:
    schema = json.load(file)
with open(schema_path / "value_domain_schema.json", "r") as file:
    vd_schema = json.load(file)
with open(schema_path / "external_routines_schema.json", "r") as file:
    external_routine_schema = json.load(file)


def _extract_data_type(component: Dict[str, Any]) -> Tuple[str, Any]:
    """
    Extract and validate data type from component dictionary.

    Supports both 'type' (preferred) and 'data_type' (backward compatibility) keys.

    Args:
        component: Component dictionary with either 'type' or 'data_type' key

    Returns:
        Tuple of (data_type_key, scalar_type_class)

    Raises:
        InputValidationException: If the data type key or value is invalid
    """
    if "type" in component:
        key = "type"
        value = component["type"]
    else:
        key = "data_type"
        value = component["data_type"]

    check_key(key, _SCALAR_TYPE_KEYS, value)
    return key, SCALAR_TYPES[value]


def _load_dataset_from_structure(
    structures: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Loads a dataset with the structure given.
    """
    datasets = {}
    scalars = {}

    if "datasets" in structures:
        for dataset_json in structures["datasets"]:
            dataset_name = dataset_json["name"]
            components = {}

            if "structure" in dataset_json:
                structure_name = dataset_json["structure"]
                structure_json = None
                for s in structures["structures"]:
                    if s["name"] == structure_name:
                        structure_json = s
                if structure_json is None:
                    raise InputValidationException(code="0-2-1-2", message="Structure not found.")
                try:
                    jsonschema.validate(instance=structure_json, schema=schema)
                except jsonschema.exceptions.ValidationError as e:
                    raise InputValidationException(code="0-2-1-2", message=e.message)

                for component in structure_json["components"]:
                    # Support both 'type' and 'data_type' for backward compatibility
                    _, scalar_type = _extract_data_type(component)
                    if component["role"] == "ViralAttribute":
                        component["role"] = "Attribute"

                    check_key("role", Role_keys, component["role"])

                    if "nullable" not in component:
                        if Role(component["role"]) == Role.IDENTIFIER:
                            component["nullable"] = False
                        elif Role(component["role"]) in (Role.MEASURE, Role.ATTRIBUTE):
                            component["nullable"] = True
                        else:
                            component["nullable"] = False

                    components[component["name"]] = VTL_Component(
                        name=component["name"],
                        data_type=scalar_type,
                        role=Role(component["role"]),
                        nullable=component["nullable"],
                    )

            if "DataStructure" in dataset_json:
                for component in dataset_json["DataStructure"]:
                    # Support both 'type' and 'data_type' for backward compatibility
                    _, scalar_type = _extract_data_type(component)
                    check_key("role", Role_keys, component["role"])
                    components[component["name"]] = VTL_Component(
                        name=component["name"],
                        data_type=scalar_type,
                        role=Role(component["role"]),
                        nullable=component["nullable"],
                    )

            datasets[dataset_name] = Dataset(name=dataset_name, components=components, data=None)
    if "scalars" in structures:
        for scalar_json in structures["scalars"]:
            scalar_name = scalar_json["name"]
            check_key("type", SCALAR_TYPES.keys(), scalar_json["type"])
            scalar = Scalar(
                name=scalar_name,
                data_type=SCALAR_TYPES[scalar_json["type"]],
                value=None,
            )
            scalars[scalar_name] = scalar
    return datasets, scalars


def _generate_single_path_dict(
    datapoint: Path,
) -> Dict[str, Path]:
    """
    Generates a dict with one dataset name and its path. The dataset name is extracted
    from the filename without the .csv extension.
    """
    dataset_name = datapoint.name.removesuffix(".csv")
    dict_paths = {dataset_name: datapoint}
    return dict_paths


def _load_single_datapoint(datapoint: Union[str, Path]) -> Dict[str, Union[str, Path]]:
    """
    Returns a dict with the data given from one dataset.
    """
    if not isinstance(datapoint, (str, Path)):
        raise InputValidationException(
            code="0-1-1-2", input=datapoint, message="Input must be a Path or an S3 URI"
        )
    # Handling of str values
    if isinstance(datapoint, str):
        if "s3://" in datapoint:
            __check_s3_extra()
            dataset_name = datapoint.split("/")[-1].removesuffix(".csv")
            return {dataset_name: datapoint}
        # Converting to Path object if it is not an S3 URI
        try:
            datapoint = Path(datapoint)
        except Exception:
            raise InputValidationException(
                code="0-1-1-2", input=datapoint, message="Input must refer to a Path or an S3 URI"
            )
    # Validation of Path object
    if not datapoint.exists():
        raise DataLoadError(code="0-3-1-1", file=datapoint)

    # Generation of datapoints dictionary with Path objects
    dict_paths: Dict[str, Path] = {}
    if datapoint.is_dir():
        for f in datapoint.iterdir():
            if f.suffix != ".csv":
                continue
            dict_paths.update(_generate_single_path_dict(f))
    else:
        dict_paths = _generate_single_path_dict(datapoint)
    return dict_paths  # type: ignore[return-value]


def _check_unique_datapoints(
    datapoints_to_add: List[str],
    datapoints_present: List[str],
) -> None:
    """
    Checks we don´t add duplicate dataset names in the datapoints.
    """
    for x in datapoints_to_add:
        if x in datapoints_present:
            raise InputValidationException(
                f"Duplicate dataset name found in datapoints: {x}. "
                f"Please check file names and dictionary keys in datapoints."
            )


def _load_datapoints_path(
    datapoints: Union[Dict[str, Union[str, Path]], List[Union[str, Path]], str, Path],
) -> Dict[str, Union[str, Path]]:
    """
    Returns a dict with the data given from a Path.
    """
    dict_datapoints: Dict[str, Union[str, Path]] = {}
    if isinstance(datapoints, dict):
        for dataset_name, datapoint in datapoints.items():
            if not isinstance(dataset_name, str):
                raise InputValidationException(
                    code="0-1-1-2",
                    input=dataset_name,
                    message="Datapoints dictionary keys must be strings.",
                )
            if not isinstance(datapoint, (str, Path)):
                raise InputValidationException(
                    code="0-1-1-2",
                    input=datapoint,
                    message="Datapoints dictionary values must be Paths or S3 URIs.",
                )
            single_datapoint = _load_single_datapoint(datapoint)
            first_datapoint = list(single_datapoint.values())[0]
            _check_unique_datapoints([dataset_name], list(dict_datapoints.keys()))
            dict_datapoints[dataset_name] = first_datapoint
        return dict_datapoints
    if isinstance(datapoints, list):
        for x in datapoints:
            single_datapoint = _load_single_datapoint(x)
            _check_unique_datapoints(list(single_datapoint.keys()), list(dict_datapoints.keys()))
            dict_datapoints.update(single_datapoint)
        return dict_datapoints
    return _load_single_datapoint(datapoints)


def _load_datastructure_single(
    data_structure: Union[Dict[str, Any], Path],
) -> Tuple[Dict[str, Dataset], Dict[str, Scalar]]:
    """
    Loads a single data structure.
    """
    if isinstance(data_structure, dict):
        return _load_dataset_from_structure(data_structure)
    if not isinstance(data_structure, Path):
        raise InputValidationException(
            code="0-1-1-2", input=data_structure, message="Input must be a dict or Path object"
        )
    if not data_structure.exists():
        raise DataLoadError(code="0-3-1-1", file=data_structure)
    if data_structure.is_dir():
        datasets: Dict[str, Dataset] = {}
        scalars: Dict[str, Scalar] = {}
        for f in data_structure.iterdir():
            if f.suffix != ".json":
                continue
            ds, sc = _load_datastructure_single(f)
            datasets = {**datasets, **ds}
            scalars = {**scalars, **sc}
        return datasets, scalars
    else:
        if data_structure.suffix != ".json":
            raise InputValidationException(
                code="0-1-1-3", expected_ext=".json", ext=data_structure.suffix
            )
        with open(data_structure, "r") as file:
            structures = json.load(file)
    return _load_dataset_from_structure(structures)


def load_datasets(
    data_structure: Union[Dict[str, Any], Path, List[Dict[str, Any]], List[Path]],
) -> Tuple[Dict[str, Dataset], Dict[str, Scalar]]:
    """
    Loads multiple datasets.

    Args:
        data_structure: Dict, Path or a List of dicts or Paths.

    Returns:
        The datastructure as a dict or a list of datastructures as dicts. \
        These dicts will have as keys the name, role, \
        type and nullable of the data contained in the dataset.

    Raises:
        Exception: If the Path is invalid or datastructure has a wrong format.
    """
    if isinstance(data_structure, dict):
        return _load_datastructure_single(data_structure)
    if isinstance(data_structure, list):
        ds_structures: Dict[str, Dataset] = {}
        scalar_structures: Dict[str, Scalar] = {}
        for x in data_structure:
            ds, sc = _load_datastructure_single(x)
            ds_structures = {**ds_structures, **ds}  # Overwrite ds_structures dict.
            scalar_structures = {**scalar_structures, **sc}  # Overwrite scalar_structures dict.
        return ds_structures, scalar_structures
    return _load_datastructure_single(data_structure)


def _handle_scalars_values(
    scalars: Dict[str, Scalar],
    scalar_values: Optional[Dict[str, Optional[Union[int, str, bool, float]]]] = None,
) -> None:
    if scalar_values is None:
        return
    # Handling scalar values with the scalar dict
    for name, value in scalar_values.items():
        if name not in scalars:
            raise InputValidationException(code="0-1-2-6", name=name)
        # Casting value to scalar data type
        if not scalars[name].data_type.check(value):
            raise InputValidationException(
                code="0-1-2-7",
                value=value,
                type_=scalars[name].data_type.__name__,
                op_type=type(scalars[name]).__name__,
                name=name,
            )
        scalars[name].value = scalars[name].data_type.cast(value)


def _handle_empty_datasets(datasets: Dict[str, Dataset]) -> None:
    for dataset in datasets.values():
        if dataset.data is None:
            _fill_dataset_empty_data(dataset)


def load_datasets_with_data(
    data_structures: Any,
    datapoints: Optional[
        Union[Dict[str, Union[pd.DataFrame, Path, str]], List[Union[str, Path]], Path, str]
    ] = None,
    scalar_values: Optional[Dict[str, Optional[Union[int, str, bool, float]]]] = None,
) -> Any:
    """
    Loads the dataset structures and fills them with the data contained in the datapoints.

    Args:
        data_structures: Dict, Path or a List of dicts or Paths.
        datapoints: Dict, Path or a List of Paths.
        scalar_values: Dict with the scalar values.

    Returns:
        A dict with the structure and a pandas dataframe with the data.

    Raises:
        Exception: If the Path is wrong or the file is invalid.
    """
    # Load the datasets without data
    datasets, scalars = load_datasets(data_structures)
    # Handle empty datasets and scalar values if no datapoints are given
    if datapoints is None:
        _handle_empty_datasets(datasets)
        _handle_scalars_values(scalars, scalar_values)
        return datasets, scalars, None

    # Handling dictionary of Pandas Dataframes
    if isinstance(datapoints, dict) and all(
        isinstance(v, pd.DataFrame) for v in datapoints.values()
    ):
        for dataset_name, data in datapoints.items():
            if dataset_name not in datasets:
                raise InputValidationException(
                    f"Not found dataset {dataset_name} in datastructures."
                )
            # This exception is not needed due to the all() check above, but it is left for safety
            if not isinstance(data, pd.DataFrame):
                raise InputValidationException(
                    f"Invalid datapoint for dataset {dataset_name}. Must be a Pandas Dataframe."
                )
            datasets[dataset_name].data = _validate_pandas(
                datasets[dataset_name].components, data, dataset_name
            )
        # Handle empty datasets and scalar values for remaining datasets
        _handle_empty_datasets(datasets)
        _handle_scalars_values(scalars, scalar_values)
        return datasets, scalars, None

    # Checking mixed types in the dictionary
    if isinstance(datapoints, dict) and any(
        not isinstance(v, (str, Path)) for v in datapoints.values()
    ):
        raise InputValidationException(
            "Invalid datapoints. All values in the dictionary must be Paths or S3 URIs, "
            "or all values must be Pandas Dataframes."
        )

    # Handling Individual, List or Dict of Paths or S3 URIs
    # NOTE: Adding type: ignore[arg-type] due to mypy issue with Union types
    datapoints_path = _load_datapoints_path(datapoints)  # type: ignore[arg-type]
    for dataset_name, csv_pointer in datapoints_path.items():
        # Check if dataset exists in datastructures
        if dataset_name not in datasets:
            raise InputValidationException(f"Not found dataset {dataset_name} in datastructures.")
        # Validate csv path for this dataset
        components = datasets[dataset_name].components
        _ = load_datapoints(components=components, dataset_name=dataset_name, csv_path=csv_pointer)
    gc.collect()  # Garbage collector to free memory after we loaded everything and discarded them

    _handle_empty_datasets(datasets)
    _handle_scalars_values(scalars, scalar_values)

    return datasets, scalars, datapoints_path


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
    Load dataset structures and data into a DuckDB connection.

    This function creates a DuckDB connection with all datasets loaded as tables,
    ready for SQL execution. Tables are registered with their dataset names.

    Args:
        data_structures: Dict, Path or a List of dicts or Paths with structure definitions.
        datapoints: Dict, Path or a List of Paths with CSV data.
        scalar_values: Dict with scalar values.

    Returns:
        Tuple of (DuckDB connection, datasets dict, scalars dict).
        The connection has tables registered for each dataset.

    Raises:
        InputValidationException: If input validation fails.
        DataLoadError: If data loading fails.
    """
    # Load the dataset structures without data
    datasets, scalars = load_datasets(data_structures)

    # Handle scalar values
    _handle_scalars_values(scalars, scalar_values)

    # Create DuckDB connection
    conn = duckdb.connect()

    # Handle case with no datapoints - create empty tables
    if datapoints is None:
        for dataset_name, dataset in datasets.items():
            rel = _create_empty_duckdb_table(conn, dataset.components)
            conn.register(dataset_name, rel)
        return conn, datasets, scalars

    # Handling dictionary of Pandas Dataframes
    if isinstance(datapoints, dict) and all(
        isinstance(v, pd.DataFrame) for v in datapoints.values()
    ):
        for dataset_name, data in datapoints.items():
            if dataset_name not in datasets:
                raise InputValidationException(
                    f"Not found dataset {dataset_name} in datastructures."
                )
            # Validate and register DataFrame (data is guaranteed to be DataFrame here)
            if not isinstance(data, pd.DataFrame):
                raise InputValidationException(
                    f"Expected DataFrame for dataset {dataset_name}, got {type(data).__name__}"
                )
            validated_data = _validate_pandas(datasets[dataset_name].components, data, dataset_name)
            conn.register(dataset_name, validated_data)

        # Create empty tables for datasets without datapoints
        for dataset_name, dataset in datasets.items():
            if dataset_name not in datapoints:
                rel = _create_empty_duckdb_table(conn, dataset.components)
                conn.register(dataset_name, rel)

        return conn, datasets, scalars

    # Checking mixed types in the dictionary
    if isinstance(datapoints, dict) and any(
        not isinstance(v, (str, Path)) for v in datapoints.values()
    ):
        raise InputValidationException(
            "Invalid datapoints. All values in the dictionary must be Paths or S3 URIs, "
            "or all values must be Pandas Dataframes."
        )

    # Handling Individual, List or Dict of Paths or S3 URIs
    datapoints_path = _load_datapoints_path(datapoints)  # type: ignore[arg-type]

    # Load each dataset into DuckDB
    for dataset_name, csv_path in datapoints_path.items():
        if dataset_name not in datasets:
            raise InputValidationException(f"Not found dataset {dataset_name} in datastructures.")

        # Load CSV directly into DuckDB with validation
        rel = load_datapoints_duckdb(
            conn=conn,
            components=datasets[dataset_name].components,
            dataset_name=dataset_name,
            csv_path=csv_path,
        )
        # Register as table with dataset name
        conn.register(dataset_name, rel)

    # Create empty tables for datasets without datapoints
    for dataset_name, dataset in datasets.items():
        if dataset_name not in datapoints_path:
            rel = _create_empty_duckdb_table(conn, dataset.components)
            conn.register(dataset_name, rel)

    gc.collect()

    return conn, datasets, scalars


def _create_empty_duckdb_table(
    conn: duckdb.DuckDBPyConnection,
    components: Dict[str, VTL_Component],
) -> duckdb.DuckDBPyRelation:
    """
    Create an empty relation with proper column types for DuckDB.

    Args:
        conn: DuckDB connection
        components: Component definitions

    Returns:
        Empty DuckDB relation with typed columns
    """
    from duckdb_transpiler.DataTypes import get_duckdb_type

    col_defs = []
    for col_name, comp in components.items():
        duckdb_type = get_duckdb_type(comp.data_type)
        col_defs.append(f'NULL::{duckdb_type} AS "{col_name}"')

    select_clause = ", ".join(col_defs)
    return conn.sql(f"SELECT {select_clause} WHERE FALSE")
