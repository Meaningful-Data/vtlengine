import gc
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union, cast

import jsonschema
import pandas as pd
from pysdmx.io import get_datasets as sdmx_get_datasets
from pysdmx.io.pd import PandasDataset
from pysdmx.model.dataflow import Component as SDMXComponent
from pysdmx.model.dataflow import Dataflow, DataStructureDefinition, Schema
from pysdmx.model.dataflow import Role as SDMX_Role
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
from vtlengine.Utils import VTL_DTYPES_MAPPING, VTL_ROLE_MAPPING

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

# File extensions that trigger SDMX parsing attempt when loading datapoints.
# .xml → SDMX-ML (strict: raises error if parsing fails)
# .json → SDMX-JSON (permissive: falls back to plain file if parsing fails)
# Note: .csv files are handled separately with SDMX-CSV detection and fallback.
SDMX_DATAPOINT_EXTENSIONS = {".xml", ".json"}

# File extensions that indicate SDMX structure files for data_structures parameter.
# .xml → SDMX-ML structure (strict: raises error if parsing fails)
# .json → SDMX-JSON structure (permissive: falls back to VTL JSON if parsing fails)
SDMX_STRUCTURE_EXTENSIONS = {".xml", ".json"}


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


def _is_sdmx_datapoint_file(file_path: Path) -> bool:
    """Check if a file should be treated as SDMX when loading datapoints."""
    return file_path.suffix.lower() in SDMX_DATAPOINT_EXTENSIONS


def _is_sdmx_structure_file(file_path: Path) -> bool:
    """Check if a file should be treated as SDMX structure file."""
    return file_path.suffix.lower() in SDMX_STRUCTURE_EXTENSIONS


def _load_sdmx_structure_file(
    file_path: Path,
    sdmx_mappings: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Load SDMX structure file and convert to VTL JSON format.

    Args:
        file_path: Path to SDMX structure file (.xml)
        sdmx_mappings: Optional mapping from SDMX URNs to VTL dataset names.

    Returns:
        VTL JSON data structure dict with 'datasets' key.

    Raises:
        DataLoadError: If file cannot be parsed or contains no structures.
    """
    from pysdmx.io import read_sdmx

    try:
        msg = read_sdmx(file_path)
    except Exception as e:
        raise DataLoadError(code="0-3-1-11", file=str(file_path), error=str(e))

    # Extract DataStructureDefinitions from the message
    # In pysdmx, msg.structures returns a list of DataStructureDefinition objects directly
    structures = msg.structures if hasattr(msg, "structures") else None
    if structures is None or not structures:
        raise DataLoadError(code="0-3-1-12", file=str(file_path))

    # Filter to only include DataStructureDefinition objects
    dsds = [s for s in structures if isinstance(s, DataStructureDefinition)]
    if not dsds:
        raise DataLoadError(code="0-3-1-12", file=str(file_path))

    # Convert each DSD to VTL JSON and merge
    all_datasets: List[Dict[str, Any]] = []
    for dsd in dsds:
        # Determine dataset name: use mapping if available, otherwise use DSD ID
        dataset_name = dsd.id
        if sdmx_mappings and hasattr(dsd, "short_urn") and dsd.short_urn in sdmx_mappings:
            dataset_name = sdmx_mappings[dsd.short_urn]
        vtl_structure = to_vtl_json(dsd, dataset_name=dataset_name)
        all_datasets.extend(vtl_structure["datasets"])

    return {"datasets": all_datasets}


def _load_sdmx_file(
    file_path: Path,
    explicit_name: Optional[str] = None,
    sdmx_mappings: Optional[Dict[str, str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load SDMX file and return dict of DataFrames.

    The user must provide matching data_structures via run(). The structure
    from the SDMX file is used only for dataset naming (unless explicit_name is given).

    Args:
        file_path: Path to the SDMX file (.xml, .json, or .csv with SDMX structure)
        explicit_name: If provided, use this name instead of URN-derived name.
                      Only valid when file contains exactly one dataset.
        sdmx_mappings: Optional mapping from SDMX URNs to VTL dataset names.

    Returns:
        Dict mapping dataset names to pandas DataFrames with the data.

    Raises:
        DataLoadError: If the file cannot be parsed as SDMX or contains no datasets.
    """
    try:
        # sdmx_get_datasets returns List[Dataset] but actual runtime type is List[PandasDataset]
        pandas_datasets = cast(Sequence[PandasDataset], sdmx_get_datasets(data=file_path))
    except Exception as e:
        raise DataLoadError(
            code="0-3-1-8",
            file=str(file_path),
            error=str(e),
        )

    if not pandas_datasets:
        raise DataLoadError(
            code="0-3-1-9",
            file=str(file_path),
        )

    result: Dict[str, pd.DataFrame] = {}

    # If explicit name provided, only valid for single dataset files
    if explicit_name is not None and len(pandas_datasets) > 1:
        raise InputValidationException(
            f"Cannot use explicit name '{explicit_name}' for SDMX file '{file_path}' "
            f"containing {len(pandas_datasets)} datasets. "
            "Use run_sdmx() with VtlDataflowMapping for multi-dataset files with explicit names."
        )

    for pd_dataset in pandas_datasets:
        # Get dataset name from structure URN or use explicit name
        if explicit_name is not None:
            vtl_name = explicit_name
        else:
            structure = pd_dataset.structure
            # Structure can be a string URN or a Schema object
            if isinstance(structure, str):
                # Check if mapping exists for this URN
                if sdmx_mappings and structure in sdmx_mappings:
                    vtl_name = sdmx_mappings[structure]
                # Extract short name from URN like "DataStructure=BIS:BIS_DER(1.0)" -> "BIS_DER"
                elif "=" in structure and ":" in structure:
                    # Format: DataStructure=AGENCY:ID(VERSION)
                    parts = structure.split(":")
                    vtl_name = parts[-1].split("(")[0] if len(parts) >= 2 else structure
                else:
                    vtl_name = structure
            else:
                # Schema object - check mapping by short_urn first
                if sdmx_mappings and hasattr(structure, "short_urn"):
                    if structure.short_urn in sdmx_mappings:
                        vtl_name = sdmx_mappings[structure.short_urn]
                    else:
                        vtl_name = structure.id
                else:
                    vtl_name = structure.id

        result[vtl_name] = pd_dataset.data

    return result


def _generate_single_path_dict(
    datapoint: Path,
    sdmx_mappings: Optional[Dict[str, str]] = None,
) -> Union[Dict[str, Path], Dict[str, pd.DataFrame]]:
    """
    Generates a dict with dataset name(s) and path or DataFrame.

    For SDMX files (.xml, .json, .csv): attempts SDMX parsing first.
    Falls back to plain file handling if SDMX parsing fails.

    Args:
        datapoint: Path to the datapoint file.
        sdmx_mappings: Optional mapping from SDMX URNs to VTL dataset names.
    """
    suffix = datapoint.suffix.lower()

    # Try SDMX parsing for known SDMX extensions and CSV files
    if suffix in SDMX_DATAPOINT_EXTENSIONS or suffix == ".csv":
        try:
            return _load_sdmx_file(datapoint, sdmx_mappings=sdmx_mappings)
        except DataLoadError:
            # Not a valid SDMX file - fall through to plain file handling
            # For .xml files, re-raise since we can't fall back
            if suffix == ".xml":
                raise
            pass

    # Plain file: return path for lazy loading (works for CSV, fails later for others)
    # Use removesuffix to preserve backward compatibility (e.g., DS_2.json stays DS_2.json)
    dataset_name = datapoint.name.removesuffix(".csv")
    return {dataset_name: datapoint}


def _load_single_datapoint(
    datapoint: Union[str, Path],
    sdmx_mappings: Optional[Dict[str, str]] = None,
) -> Dict[str, Union[str, Path, pd.DataFrame]]:
    """
    Returns a dict with the data given from one dataset.

    For SDMX files (.xml, .json), returns DataFrames with data loaded.
    For CSV files, returns paths for lazy loading.

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

    # Generation of datapoints dictionary
    dict_results: Dict[str, Union[Path, pd.DataFrame]] = {}
    if datapoint.is_dir():
        for f in datapoint.iterdir():
            # Handle SDMX files (.xml, .json)
            if _is_sdmx_datapoint_file(f) or f.suffix.lower() == ".csv":
                dict_results.update(_generate_single_path_dict(f, sdmx_mappings=sdmx_mappings))
            # Skip other files
    else:
        dict_results = _generate_single_path_dict(datapoint, sdmx_mappings=sdmx_mappings)  # type: ignore[assignment]
    return dict_results  # type: ignore[return-value]


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


def _add_loaded_datapoint(
    loaded: Mapping[str, Union[str, Path, pd.DataFrame]],
    csv_paths: Dict[str, Union[str, Path]],
    sdmx_dfs: Dict[str, pd.DataFrame],
    explicit_name: Optional[str] = None,
) -> None:
    """
    Add loaded datapoint results to the appropriate dictionary.

    Args:
        loaded: Result from _load_single_datapoint or _load_sdmx_file
        csv_paths: Dict to accumulate CSV paths
        sdmx_dfs: Dict to accumulate SDMX DataFrames
        explicit_name: If provided, use this name for CSV paths (from dict key)
    """
    existing_names = list(csv_paths.keys()) + list(sdmx_dfs.keys())
    for name, value in loaded.items():
        if isinstance(value, pd.DataFrame):
            _check_unique_datapoints([name], existing_names)
            sdmx_dfs[name] = value
        else:
            final_name = explicit_name if explicit_name is not None else name
            _check_unique_datapoints([final_name], existing_names)
            csv_paths[final_name] = value
        existing_names.append(name if isinstance(value, pd.DataFrame) else final_name)


def _load_datapoints_path(
    datapoints: Union[Dict[str, Union[str, Path]], List[Union[str, Path]], str, Path],
    sdmx_mappings: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Union[str, Path]], Dict[str, pd.DataFrame]]:
    """
    Returns a tuple of:
    - dict with CSV paths for lazy loading
    - dict with pre-loaded SDMX DataFrames

    Args:
        datapoints: Dict, List, or single Path/S3 URI with datapoints.
        sdmx_mappings: Optional mapping from SDMX URNs to VTL dataset names.
    """
    csv_paths: Dict[str, Union[str, Path]] = {}
    sdmx_dfs: Dict[str, pd.DataFrame] = {}

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

            # SDMX files with explicit name
            if isinstance(datapoint, Path) and _is_sdmx_datapoint_file(datapoint):
                if not datapoint.exists():
                    raise DataLoadError(code="0-3-1-1", file=datapoint)
                sdmx_result = _load_sdmx_file(
                    datapoint, explicit_name=dataset_name, sdmx_mappings=sdmx_mappings
                )
                _add_loaded_datapoint(sdmx_result, csv_paths, sdmx_dfs)
            else:
                # CSV or S3 path
                single_result = _load_single_datapoint(datapoint, sdmx_mappings=sdmx_mappings)
                _add_loaded_datapoint(
                    single_result, csv_paths, sdmx_dfs, explicit_name=dataset_name
                )
        return csv_paths, sdmx_dfs

    if isinstance(datapoints, list):
        for x in datapoints:
            single_result = _load_single_datapoint(x, sdmx_mappings=sdmx_mappings)
            _add_loaded_datapoint(single_result, csv_paths, sdmx_dfs)
        return csv_paths, sdmx_dfs

    # Single datapoint
    single_result = _load_single_datapoint(datapoints, sdmx_mappings=sdmx_mappings)
    _add_loaded_datapoint(single_result, csv_paths, sdmx_dfs)
    return csv_paths, sdmx_dfs


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
            vtl_json = _load_sdmx_structure_file(data_structure, sdmx_mappings=sdmx_mappings)
            return _load_dataset_from_structure(vtl_json)
        # Handle .json files - try SDMX-JSON first, fall back to VTL JSON
        if suffix == ".json":
            try:
                vtl_json = _load_sdmx_structure_file(data_structure, sdmx_mappings=sdmx_mappings)
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
) -> Any:
    """
    Loads the dataset structures and fills them with the data contained in the datapoints.

    Args:
        data_structures: Dict, Path or a List of dicts or Paths.
        datapoints: Dict, Path or a List of Paths.
        scalar_values: Dict with the scalar values.
        sdmx_mappings: Optional mapping from SDMX URNs to VTL dataset names.

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
    csv_paths, sdmx_dataframes = _load_datapoints_path(
        cast(Union[Dict[str, Union[str, Path]], List[Union[str, Path]], str, Path], datapoints),
        sdmx_mappings=sdmx_mappings,
    )

    # Merge pre-loaded SDMX DataFrames
    for dataset_name, sdmx_df in sdmx_dataframes.items():
        if dataset_name not in datasets:
            raise InputValidationException(f"Not found dataset {dataset_name} in datastructures.")
        # Validate and assign SDMX DataFrame to the structure-defined dataset
        datasets[dataset_name].data = _validate_pandas(
            datasets[dataset_name].components, sdmx_df, dataset_name
        )

    # Validate CSV paths
    for dataset_name, csv_pointer in csv_paths.items():
        # Check if dataset exists in datastructures
        if dataset_name not in datasets:
            raise InputValidationException(f"Not found dataset {dataset_name} in datastructures.")
        # Validate csv path for this dataset
        components = datasets[dataset_name].components
        _ = load_datapoints(components=components, dataset_name=dataset_name, csv_path=csv_pointer)
    gc.collect()  # Garbage collector to free memory

    _handle_empty_datasets(datasets)
    _handle_scalars_values(scalars, scalar_values)

    return datasets, scalars, csv_paths if csv_paths else None


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


def to_vtl_json(
    structure: Union[DataStructureDefinition, Schema, Dataflow],
    dataset_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Converts a pysdmx `DataStructureDefinition`, `Schema`, or `Dataflow` into a VTL-compatible
    JSON representation.

    This function extracts and transforms the components (dimensions, measures, and attributes)
    from the given SDMX data structure and maps them into a dictionary format that conforms
    to the expected VTL data structure json schema.

    Args:
        structure: An instance of `DataStructureDefinition`, `Schema`, or `Dataflow` from pysdmx.
        dataset_name: The name of the resulting VTL dataset. If not provided, uses the
            structure's ID (or Dataflow's ID for Dataflow objects).

    Returns:
        A dictionary representing the dataset in VTL format, with keys for dataset name and its
        components, including their name, role, data type, and nullability.

    Raises:
        InputValidationException: If a Dataflow has no associated DataStructureDefinition
            or if its structure is an unresolved reference.
    """
    # Handle Dataflow by extracting its DataStructureDefinition
    if isinstance(structure, Dataflow):
        if structure.structure is None:
            raise InputValidationException(
                f"Dataflow '{structure.id}' has no associated DataStructureDefinition."
            )
        if not isinstance(structure.structure, DataStructureDefinition):
            raise InputValidationException(
                f"Dataflow '{structure.id}' structure is a reference, not resolved. "
                "Please provide a resolved Dataflow with embedded DataStructureDefinition."
            )
        # Use Dataflow ID as dataset name if not provided
        if dataset_name is None:
            dataset_name = structure.id
        structure = structure.structure

    # Use structure ID if dataset_name not provided
    if dataset_name is None:
        dataset_name = structure.id

    components = []
    NAME = "name"
    ROLE = "role"
    TYPE = "type"
    NULLABLE = "nullable"

    _components: List[SDMXComponent] = []
    _components.extend(structure.components.dimensions)
    _components.extend(structure.components.measures)
    _components.extend(structure.components.attributes)

    for c in _components:
        _type = VTL_DTYPES_MAPPING[c.dtype]
        _nullability = c.role != SDMX_Role.DIMENSION
        _role = VTL_ROLE_MAPPING[c.role]

        component = {
            NAME: c.id,
            ROLE: _role,
            TYPE: _type,
            NULLABLE: _nullability,
        }

        components.append(component)

    result = {"datasets": [{"name": dataset_name, "DataStructure": components}]}

    return result


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
