import gc
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast

import jsonschema
import pandas as pd
from pysdmx.io import get_datasets
from pysdmx.model.dataflow import Dataflow, DataStructureDefinition, Schema
from pysdmx.model.vtl import (
    Ruleset,
    RulesetScheme,
    Transformation,
    TransformationScheme,
    UserDefinedOperator,
    UserDefinedOperatorScheme,
)

from vtlengine import AST as AST
from vtlengine.__extras_check import __check_s3_extra
from vtlengine.AST import Assignment, DPRuleset, HRuleset, Operator, PersistentAssignment, Start
from vtlengine.AST.ASTString import ASTString
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
from vtlengine.files.sdmx_handler import (
    extract_sdmx_dataset_name,
    load_sdmx_structure,
    to_vtl_json,
)
from vtlengine.Model import (
    Component as VTL_Component,
)
from vtlengine.Model import (
    Dataset,
    ExternalRoutine,
    Role,
    Role_keys,
    Scalar,
    ValueDomain,
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
    sdmx_mappings: Optional[Dict[str, str]] = None,
) -> Dict[str, Path]:
    """
    Generates a dict with dataset name(s) and path for lazy loading.

    For SDMX-ML files (.xml): extracts dataset name from structure, returns path.
    For CSV files (plain CSV or SDMX-CSV): uses filename as dataset name, returns path.

    Args:
        datapoint: Path to the datapoint file.
        sdmx_mappings: Optional mapping from SDMX URNs to VTL dataset names.

    Returns:
        Dict mapping dataset name to file path for lazy loading.
    """
    suffix = datapoint.suffix.lower()

    # For SDMX-ML files, extract the dataset name from the file structure
    if suffix == ".xml":
        dataset_name = extract_sdmx_dataset_name(datapoint, sdmx_mappings=sdmx_mappings)
        return {dataset_name: datapoint}

    # For CSV files (plain CSV or SDMX-CSV), use filename as dataset name
    dataset_name = datapoint.name.removesuffix(".csv")
    return {dataset_name: datapoint}


def _load_single_datapoint(
    datapoint: Union[str, Path],
    sdmx_mappings: Optional[Dict[str, str]] = None,
) -> Dict[str, Union[str, Path]]:
    """
    Returns a dict with paths for lazy loading.

    All file types (plain CSV, SDMX-CSV, SDMX-ML) return paths for lazy loading.
    The actual data loading happens in load_datapoints() which supports
    plain CSV, SDMX-CSV, and SDMX-ML file formats.

    Args:
        datapoint: Path or S3 URI to the datapoint file.
        sdmx_mappings: Optional mapping from SDMX URNs to VTL dataset names.
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

    # Generation of datapoints dictionary - all paths for lazy loading
    dict_results: Dict[str, Union[str, Path]] = {}
    if datapoint.is_dir():
        for f in datapoint.iterdir():
            # Handle SDMX files (.xml) and CSV files
            if f.suffix.lower() in (".xml", ".csv"):
                dict_results.update(_generate_single_path_dict(f, sdmx_mappings=sdmx_mappings))
            # Skip other files
    else:
        dict_results.update(_generate_single_path_dict(datapoint, sdmx_mappings=sdmx_mappings))
    return dict_results


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
    sdmx_mappings: Optional[Dict[str, str]] = None,
) -> Dict[str, Union[str, Path]]:
    """
    Returns dict with paths for lazy loading.

    All file types (CSV, SDMX-ML) are returned as paths. The actual data loading
    happens in load_datapoints() which supports both formats.

    Args:
        datapoints: Dict, List, or single Path/S3 URI with datapoints.
        sdmx_mappings: Optional mapping from SDMX URNs to VTL dataset names.

    Returns:
        Dict mapping dataset names to file paths for lazy loading.
    """
    all_paths: Dict[str, Union[str, Path]] = {}

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

            # Convert string to Path if not S3
            if isinstance(datapoint, str) and "s3://" not in datapoint:
                datapoint = Path(datapoint)

            # Validate file exists
            if isinstance(datapoint, Path) and not datapoint.exists():
                raise DataLoadError(code="0-3-1-1", file=datapoint)

            # Use explicit dataset_name from dict key
            _check_unique_datapoints([dataset_name], list(all_paths.keys()))
            all_paths[dataset_name] = datapoint
        return all_paths

    if isinstance(datapoints, list):
        for x in datapoints:
            single_result = _load_single_datapoint(x, sdmx_mappings=sdmx_mappings)
            _check_unique_datapoints(list(single_result.keys()), list(all_paths.keys()))
            all_paths.update(single_result)
        return all_paths

    # Single datapoint
    single_result = _load_single_datapoint(datapoints, sdmx_mappings=sdmx_mappings)
    all_paths.update(single_result)
    return all_paths


def _load_datastructure_single(
    data_structure: Union[Dict[str, Any], Path, Schema, DataStructureDefinition, Dataflow],
    sdmx_mappings: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Dataset], Dict[str, Scalar]]:
    """
    Loads a single data structure.

    Args:
        data_structure: Dict, Path, or pysdmx object (Schema, DSD, Dataflow).
        sdmx_mappings: Optional mapping from SDMX URNs to VTL dataset names.
    """
    # Handle pysdmx objects
    if isinstance(data_structure, (Schema, DataStructureDefinition, Dataflow)):
        # Apply mapping if available
        dataset_name = None
        if (
            sdmx_mappings
            and hasattr(data_structure, "short_urn")
            and data_structure.short_urn in sdmx_mappings
        ):
            dataset_name = sdmx_mappings[data_structure.short_urn]
        vtl_json = to_vtl_json(data_structure, dataset_name=dataset_name)
        return _load_dataset_from_structure(vtl_json)
    if isinstance(data_structure, dict):
        return _load_dataset_from_structure(data_structure)
    if not isinstance(data_structure, Path):
        raise InputValidationException(
            code="0-1-1-2",
            input=data_structure,
            message="Input must be a dict, Path, or pysdmx object",
        )
    if not data_structure.exists():
        raise DataLoadError(code="0-3-1-1", file=data_structure)
    if data_structure.is_dir():
        datasets: Dict[str, Dataset] = {}
        scalars: Dict[str, Scalar] = {}
        for f in data_structure.iterdir():
            if f.suffix not in (".json", ".xml"):
                continue
            ds, sc = _load_datastructure_single(f, sdmx_mappings=sdmx_mappings)
            datasets = {**datasets, **ds}
            scalars = {**scalars, **sc}
        return datasets, scalars
    else:
        suffix = data_structure.suffix.lower()
        # Handle SDMX-ML structure files (.xml) - strict, must be SDMX
        if suffix == ".xml":
            vtl_json = load_sdmx_structure(data_structure, sdmx_mappings=sdmx_mappings)
            return _load_dataset_from_structure(vtl_json)
        # Handle .json files - try SDMX-JSON first, fall back to VTL JSON
        if suffix == ".json":
            try:
                vtl_json = load_sdmx_structure(data_structure, sdmx_mappings=sdmx_mappings)
                return _load_dataset_from_structure(vtl_json)
            except DataLoadError:
                # Not SDMX-JSON, try as VTL JSON
                pass
            with open(data_structure, "r") as file:
                structures = json.load(file)
            return _load_dataset_from_structure(structures)
        # Unsupported extension
        raise InputValidationException(code="0-1-1-3", expected_ext=".json or .xml", ext=suffix)


def load_datasets(
    data_structure: Union[
        Dict[str, Any],
        Path,
        Schema,
        DataStructureDefinition,
        Dataflow,
        List[Union[Dict[str, Any], Path, Schema, DataStructureDefinition, Dataflow]],
    ],
    sdmx_mappings: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Dataset], Dict[str, Scalar]]:
    """
    Loads multiple datasets.

    Args:
        data_structure: Dict, Path or a List of dicts or Paths.
        sdmx_mappings: Optional mapping from SDMX URNs to VTL dataset names.

    Returns:
        The datastructure as a dict or a list of datastructures as dicts. \
        These dicts will have as keys the name, role, \
        type and nullable of the data contained in the dataset.

    Raises:
        Exception: If the Path is invalid or datastructure has a wrong format.
    """
    if isinstance(data_structure, dict):
        return _load_datastructure_single(data_structure, sdmx_mappings=sdmx_mappings)
    if isinstance(data_structure, list):
        ds_structures: Dict[str, Dataset] = {}
        scalar_structures: Dict[str, Scalar] = {}
        for x in data_structure:
            ds, sc = _load_datastructure_single(x, sdmx_mappings=sdmx_mappings)
            ds_structures = {**ds_structures, **ds}  # Overwrite ds_structures dict.
            scalar_structures = {**scalar_structures, **sc}  # Overwrite scalar_structures dict.
        return ds_structures, scalar_structures
    return _load_datastructure_single(data_structure, sdmx_mappings=sdmx_mappings)


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
    sdmx_mappings: Optional[Dict[str, str]] = None,
    validate: bool = False,
) -> Any:
    """
    Loads the dataset structures and fills them with the data contained in the datapoints.

    Args:
        data_structures: Dict, Path or a List of dicts or Paths.
        datapoints: Dict, Path or a List of Paths.
        scalar_values: Dict with the scalar values.
        sdmx_mappings: Optional mapping from SDMX URNs to VTL dataset names.
        validate: If True, load and validate datapoints immediately (for validate_dataset API).
                  If False, defer validation to interpretation time (for run API).

    Returns:
        A dict with the structure and a pandas dataframe with the data.

    Raises:
        Exception: If the Path is wrong or the file is invalid.
    """
    # Load the datasets without data
    datasets, scalars = load_datasets(data_structures, sdmx_mappings=sdmx_mappings)
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
    # At this point, datapoints is narrowed to exclude None and Dict[str, DataFrame]
    # All file types (CSV, SDMX) are returned as paths for lazy loading
    datapoints_paths = _load_datapoints_path(
        cast(Union[Dict[str, Union[str, Path]], List[Union[str, Path]], str, Path], datapoints),
        sdmx_mappings=sdmx_mappings,
    )

    # Validate that all datapoint dataset names exist in structures
    for dataset_name in datapoints_paths:
        if dataset_name not in datasets:
            raise InputValidationException(f"Not found dataset {dataset_name} in datastructures.")

    # If validate=True, load and validate data immediately but don't store it
    # (used by validate_dataset API in memory-constrained scenarios).
    # gc.collect() ensures memory is reclaimed after each large DataFrame is validated.
    if validate:
        for dataset_name, file_path in datapoints_paths.items():
            components = datasets[dataset_name].components
            _ = load_datapoints(
                components=components, dataset_name=dataset_name, csv_path=file_path
            )
            gc.collect()

    _handle_empty_datasets(datasets)
    _handle_scalars_values(scalars, scalar_values)

    return datasets, scalars, datapoints_paths if datapoints_paths else None


def load_vtl(input: Union[str, Path]) -> str:
    """
    Reads the vtl expression.

    Args:
        input: String or Path of the vtl expression.

    Returns:
        If it is a string, it will return the input as a string. \
        If it is a Path, it will return the expression contained in the file as a string.

    Raises:
        Exception: If the vtl does not exist, if the Path is wrong, or if it is not a vtl file.
    """
    if isinstance(input, str):
        if os.path.exists(input):
            input = Path(input)
        else:
            return input
    if not isinstance(input, Path):
        raise InputValidationException(
            code="0-1-1-2", input=input, message="Input is not a Path object"
        )
    if not input.exists():
        raise DataLoadError(code="0-3-1-1", file=input)
    if input.suffix != ".vtl":
        raise InputValidationException(code="0-1-1-3", expected_ext=".vtl", ext=input.suffix)
    with open(input, "r") as f:
        return f.read()


def _validate_json(data: Dict[str, Any], schema: Dict[str, Any]) -> None:
    try:
        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.ValidationError as e:
        raise InputValidationException(code="0-2-1-1", message=f"{e}")


def _load_single_value_domain(input: Path) -> Dict[str, ValueDomain]:
    if input.suffix != ".json":
        raise InputValidationException(code="0-1-1-3", expected_ext=".json", ext=input.suffix)
    with open(input, "r") as f:
        data = json.load(f)
    _validate_json(data, vd_schema)
    vd = ValueDomain.from_dict(data)
    return {vd.name: vd}


def load_value_domains(
    input: Union[Dict[str, Any], Path, List[Union[Dict[str, Any], Path]]],
) -> Dict[str, ValueDomain]:
    """
    Loads the value domains.

    Args:
        input: Dict or Path, or a list of them \
        of the json file that contains the value domains data.

    Returns:
        A dictionary with the value domains data, or a list of dictionaries with them.

    Raises:
        Exception: If the value domains file is wrong, the Path is invalid, \
        or the value domains file does not exist.
    """
    if isinstance(input, dict):
        _validate_json(input, vd_schema)
        vd = ValueDomain.from_dict(input)
        return {vd.name: vd}
    if isinstance(input, list):
        value_domains: Dict[str, Any] = {}
        for item in input:
            value_domains.update(load_value_domains(item))
        return value_domains
    if not isinstance(input, Path):
        raise InputValidationException(
            code="0-1-1-2", input=input, message="Input is not a Path object"
        )
    if not input.exists():
        raise DataLoadError(code="0-3-1-1", file=input)
    if input.is_dir():
        value_domains = {}
        for f in input.iterdir():
            vd = _load_single_value_domain(f)
            value_domains = {**value_domains, **vd}
        return value_domains
    if input.suffix != ".json":
        raise InputValidationException(code="0-1-1-3", expected_ext=".json", ext=input.suffix)
    return _load_single_value_domain(input)


def load_external_routines(
    input: Union[Dict[str, Any], Path, str, List[Union[Dict[str, Any], Path]]],
) -> Any:
    """
    Load the external routines.

    Args:
        input: Dict or Path, or a list of them, \
        of the JSON file that contains the external routine data.

    Returns:
        A dictionary with the external routine data, or a list with \
        the dictionaries from the Path given.

    Raises:
        Exception: If the JSON file does not exist, the Path is wrong, or the file is not a \
        JSON one.
    """
    external_routines = {}
    if isinstance(input, dict):
        _validate_json(input, external_routine_schema)
        ext_routine = ExternalRoutine.from_sql_query(input["name"], input["query"])
        external_routines[ext_routine.name] = ext_routine
        return external_routines
    if isinstance(input, list):
        ext_routines = {}
        for item in input:
            ext_routines.update(load_external_routines(item))
        return ext_routines
    if not isinstance(input, Path):
        raise InputValidationException(
            code="0-1-1-2", input=input, message="Input must be a json file."
        )
    if not input.exists():
        raise DataLoadError(code="0-3-1-1", file=input)
    if input.is_dir():
        for f in input.iterdir():
            if f.suffix != ".sql":
                continue
            ext_rout = _load_single_external_routine_from_file(f)
            external_routines[ext_rout.name] = ext_rout
        return external_routines
    ext_rout = _load_single_external_routine_from_file(input)
    external_routines[ext_rout.name] = ext_rout
    return external_routines


def _return_only_persistent_datasets(
    datasets: Dict[str, Union[Dataset, Scalar]], ast: Start
) -> Dict[str, Union[Dataset, Scalar]]:
    """
    Returns only the datasets with a persistent assignment.
    """
    return {dataset.name: dataset for dataset in datasets.values() if dataset.persistent}


def _load_single_external_routine_from_file(input: Path) -> Any:
    if not isinstance(input, Path):
        raise InputValidationException(code="0-1-1-2", input=input)
    if not input.exists():
        raise DataLoadError(code="0-3-1-1", file=input)
    if input.suffix != ".json":
        raise InputValidationException(code="0-1-1-3", expected_ext=".json", ext=input.suffix)
    routine_name = input.stem
    with open(input, "r") as f:
        data = json.load(f)
    _validate_json(data, external_routine_schema)
    ext_rout = ExternalRoutine.from_sql_query(routine_name, data["query"])
    return ext_rout


def _check_output_folder(output_folder: Union[str, Path]) -> None:
    """
    Check if the output folder exists. If not, it will create it.
    """
    if isinstance(output_folder, str):
        if "s3://" in output_folder:
            __check_s3_extra()
            if not output_folder.endswith("/"):
                raise DataLoadError("0-3-1-2", folder=str(output_folder))
            return
        try:
            output_folder = Path(output_folder)
        except Exception:
            raise DataLoadError("0-3-1-2", folder=str(output_folder))

    if not isinstance(output_folder, Path):
        raise DataLoadError("0-3-1-2", folder=str(output_folder))
    if not output_folder.exists():
        if output_folder.suffix != "":
            raise DataLoadError("0-3-1-2", folder=str(output_folder))
        os.mkdir(output_folder)


def __generate_transformation(
    child: Union[Assignment, PersistentAssignment], is_persistent: bool, count: int
) -> Transformation:
    expression = ASTString().render(ast=child.right)
    result = child.left.value  # type: ignore[attr-defined]
    return Transformation(
        id=f"T{count}",
        expression=expression,
        is_persistent=is_persistent,
        result=result,
        name=f"Transformation {result}",
    )


def __generate_udo(child: Operator, count: int) -> UserDefinedOperator:
    operator_definition = ASTString().render(ast=child)
    return UserDefinedOperator(
        id=f"UDO{count}",
        operator_definition=operator_definition,
        name=f"UserDefinedOperator {child.op}",
    )


def __generate_ruleset(child: Union[DPRuleset, HRuleset], count: int) -> Ruleset:
    ruleset_definition = ASTString().render(ast=child)
    ruleset_type: Literal["datapoint", "hierarchical"] = (
        "datapoint" if isinstance(child, DPRuleset) else "hierarchical"
    )
    ruleset_scope: Literal["variable", "valuedomain"] = (
        "variable" if child.signature_type == "variable" else "valuedomain"
    )
    return Ruleset(
        id=f"R{count}",
        ruleset_definition=ruleset_definition,
        ruleset_type=ruleset_type,
        ruleset_scope=ruleset_scope,
        name=f"{ruleset_type.capitalize()} ruleset {child.name}",
    )


def ast_to_sdmx(ast: AST.Start, agency_id: str, id: str, version: str) -> TransformationScheme:
    """
    Converts a vtl AST into an SDMX compatible `TransformationScheme` object, following
    the pysdmx model.

    This function iterates over the child nodes of the given AST and categorizes each into one of
    the following types:
    - `PersistentAssignment`: Represents a persistent transformation. These are added to the
    transformation list with a persistence flag.
    - `Assignment`: Represents a temporary (non-persistent) transformation. These are added to the
    transformation list without the persistence flag
    - `DPRuleset` or `HRuleset`: Represent validation rule sets.
    These are collected and wrapped into a `RulesetScheme` object.
    - `Operator`: Defines user-defined operators. These are collected
    into a `UserDefinedOperatorScheme` object.

    After parsing all AST elements:
    - If any rulesets were found, a `RulesetScheme` is created and added to the references.
    - If any user-defined operators were found, a `UserDefinedOperatorScheme` is created and added
    to the references.
    - A `TransformationScheme` object is constructed with all collected transformations and any
    additional references.

    Args:
        ast: The root node of the vtl ast representing the set of
        vtl expressions.
        agency_id: The identifier of the agency defining the SDMX structure as a string.
        id: The identifier of the transformation scheme as a string.
        version: The version of the transformation scheme given as a string.

    Returns:
        TransformationScheme: A fully constructed transformation scheme that includes
        transformations, and optionally rule sets and user-defined operator schemes,
        suitable for SDMX.

    """
    list_transformation = []
    list_udos = []
    list_rulesets = []
    count_transformation = 0
    count_udo = 0
    count_ruleset = 0

    for child in ast.children:
        if isinstance(child, PersistentAssignment):
            count_transformation += 1
            list_transformation.append(
                __generate_transformation(
                    child=child, is_persistent=True, count=count_transformation
                )
            )
        elif isinstance(child, Assignment):
            count_transformation += 1
            list_transformation.append(
                __generate_transformation(
                    child=child, is_persistent=False, count=count_transformation
                )
            )
        elif isinstance(child, (DPRuleset, HRuleset)):
            count_ruleset += 1
            list_rulesets.append(__generate_ruleset(child=child, count=count_ruleset))
        elif isinstance(child, Operator):
            count_udo += 1
            list_udos.append(__generate_udo(child=child, count=count_udo))

    references: Any = {}
    if list_rulesets:
        references["ruleset_schemes"] = [
            RulesetScheme(
                items=list_rulesets,
                agency=agency_id,
                id="RS1",
                vtl_version="2.1",
                version=version,
                name=f"RulesetScheme {id}-RS",
            )
        ]
    if list_udos:
        references["user_defined_operator_schemes"] = [
            UserDefinedOperatorScheme(
                items=list_udos,
                agency=agency_id,
                id="UDS1",
                vtl_version="2.1",
                version=version,
                name=f"UserDefinedOperatorScheme {id}-UDS",
            )
        ]

    transformation_scheme = TransformationScheme(
        items=list_transformation,
        agency=agency_id,
        id="TS1",
        vtl_version="2.1",
        version=version,
        name=f"TransformationScheme {id}",
        **references,
    )

    return transformation_scheme


def _is_url(value: Any) -> bool:
    """
    Check if a value is an HTTP/HTTPS URL.

    Args:
        value: Any value to check.

    Returns:
        True if the value is a string starting with http:// or https://.
    """
    return isinstance(value, str) and (value.startswith("http://") or value.startswith("https://"))


def _handle_url_structure(
    url: str,
    sdmx_mappings: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Dataset], Dict[str, Scalar]]:
    """
    Fetch SDMX structure from URL using pysdmx and return datasets.

    Uses pysdmx's read_sdmx to fetch structure messages from HTTP/HTTPS URLs.

    Args:
        url: HTTP/HTTPS URL to an SDMX structure file.
        sdmx_mappings: Optional mapping from SDMX URNs to VTL dataset names.

    Returns:
        Tuple of (datasets, scalars) from the fetched structure.

    Raises:
        DataLoadError: If fetching from URL fails.
    """
    vtl_json = load_sdmx_structure(url, sdmx_mappings=sdmx_mappings)  # type: ignore[arg-type]
    return _load_dataset_from_structure(vtl_json)


def _handle_url_datapoints(
    url_datapoints: Dict[str, str],
    sdmx_structure: Union[str, Path],
    sdmx_mappings: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Dataset], Dict[str, Scalar], Dict[str, pd.DataFrame]]:
    """
    Fetch SDMX data from URLs using pysdmx and return datasets.

    Args:
        url_datapoints: Dict mapping dataset names to HTTP/HTTPS URLs.
        sdmx_structure: Path to SDMX structure file (required for URL fetching).
        sdmx_mappings: Optional mapping from SDMX URNs to VTL dataset names.

    Returns:
        Tuple of (datasets, scalars, dataframes) for merging with other datapoints.

    Raises:
        DataLoadError: If fetching from URL fails.
    """

    datasets: Dict[str, Dataset] = {}
    dataframes: Dict[str, pd.DataFrame] = {}

    for dataset_name, url in url_datapoints.items():
        try:
            sdmx_datasets = get_datasets(data=url, structure=sdmx_structure)
        except Exception as e:
            raise DataLoadError(code="0-3-1-13", url=url, error=str(e))

        if not sdmx_datasets:
            raise DataLoadError(code="0-3-1-13", url=url, error="No data returned")

        sdmx_dataset = sdmx_datasets[0]
        schema = sdmx_dataset.structure

        if isinstance(schema, Schema):
            vtl_json = to_vtl_json(schema, dataset_name=dataset_name)
            ds_dict, _ = _load_dataset_from_structure(vtl_json)
            datasets.update(ds_dict)
        else:
            raise DataLoadError(
                code="0-3-1-13",
                url=url,
                error=f"Expected Schema object, got {type(schema).__name__}",
            )

        dataframes[dataset_name] = sdmx_dataset.data  # type: ignore[attr-defined]

    return datasets, {}, dataframes


def _check_script(script: Union[str, TransformationScheme, Path]) -> str:
    """
    Check if the TransformationScheme object is valid to generate a vtl script.
    """
    if not isinstance(script, (str, TransformationScheme, Path)):
        raise InputValidationException(code="0-1-1-1", format_=type(script).__name__)
    if isinstance(script, TransformationScheme):
        from pysdmx.toolkit.vtl import (
            generate_vtl_script,
        )

        vtl_script = generate_vtl_script(script, model_validation=True)
        return vtl_script
    else:
        return str(script)
