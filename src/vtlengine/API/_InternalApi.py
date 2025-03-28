import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import jsonschema
from pysdmx.model.dataflow import Component as SDMXComponent
from pysdmx.model.dataflow import DataStructureDefinition, Schema
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
from vtlengine.Exceptions import InputValidationException, check_key
from vtlengine.files.parser import _fill_dataset_empty_data, _validate_pandas
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
from vtlengine.DataFrame import DataFrame
from vtlengine.Utils import VTL_DTYPES_MAPPING, VTL_ROLE_MAPPING

base_path = Path(__file__).parent
schema_path = base_path / "data" / "schema"
sdmx_csv_path = base_path / "data" / "sdmx_csv"
with open(schema_path / "json_schema_2.1.json", "r") as file:
    schema = json.load(file)


def _load_dataset_from_structure(structures: Dict[str, Any]) -> Dict[str, Any]:
    """
    Loads a dataset with the structure given.
    """
    datasets = {}

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
                    raise InputValidationException(code="0-3-1-1", message="Structure not found.")
                try:
                    jsonschema.validate(instance=structure_json, schema=schema)
                except jsonschema.exceptions.ValidationError as e:
                    raise InputValidationException(code="0-3-1-1", message=e.message)

                for component in structure_json["components"]:
                    check_key("data_type", SCALAR_TYPES.keys(), component["data_type"])
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
                        data_type=SCALAR_TYPES[component["data_type"]],
                        role=Role(component["role"]),
                        nullable=component["nullable"],
                    )

            if "DataStructure" in dataset_json:
                for component in dataset_json["DataStructure"]:
                    check_key("data_type", SCALAR_TYPES.keys(), component["type"])
                    check_key("role", Role_keys, component["role"])
                    components[component["name"]] = VTL_Component(
                        name=component["name"],
                        data_type=SCALAR_TYPES[component["type"]],
                        role=Role(component["role"]),
                        nullable=component["nullable"],
                    )

            datasets[dataset_name] = Dataset(name=dataset_name, components=components, data=None)
    if "scalars" in structures:
        for scalar_json in structures["scalars"]:
            scalar_name = scalar_json["name"]
            scalar = Scalar(
                name=scalar_name,
                data_type=SCALAR_TYPES[scalar_json["type"]],
                value=None,
            )
            datasets[scalar_name] = scalar  # type: ignore[assignment]
    return datasets


def _load_single_datapoint(datapoint: Union[str, Path]) -> Dict[str, Any]:
    """
    Returns a dict with the data given from one dataset.
    """
    if not isinstance(datapoint, (Path, str)):
        raise Exception("Invalid datapoint. Input must be a Path or an S3 URI")
    if isinstance(datapoint, str):
        if "s3://" in datapoint:
            __check_s3_extra()
            dataset_name = datapoint.split("/")[-1].removesuffix(".csv")
            dict_data = {dataset_name: datapoint}
            return dict_data
        try:
            datapoint = Path(datapoint)
        except Exception:
            raise Exception("Invalid datapoint. Input must refer to a Path or an S3 URI")
    if datapoint.is_dir():
        datapoints: Dict[str, Any] = {}
        for f in datapoint.iterdir():
            if f.suffix != ".csv":
                continue
            dp = _load_single_datapoint(f)
            datapoints = {**datapoints, **dp}
        dict_data = datapoints
    else:
        dataset_name = datapoint.name.removesuffix(".csv")
        dict_data = {dataset_name: datapoint}  # type: ignore[dict-item]
    return dict_data


def _load_datapoints_path(
    datapoints: Union[Path, str, List[Union[str, Path]]],
) -> Dict[str, Dataset]:
    """
    Returns a dict with the data given from a Path.
    """
    if isinstance(datapoints, list):
        dict_datapoints: Dict[str, Any] = {}
        for x in datapoints:
            result = _load_single_datapoint(x)
            dict_datapoints = {**dict_datapoints, **result}
        return dict_datapoints
    return _load_single_datapoint(datapoints)


def _load_datastructure_single(data_structure: Union[Dict[str, Any], Path]) -> Dict[str, Dataset]:
    """
    Loads a single data structure.
    """
    if isinstance(data_structure, dict):
        return _load_dataset_from_structure(data_structure)
    if not isinstance(data_structure, Path):
        raise Exception("Invalid datastructure. Input must be a dict or Path object")
    if not data_structure.exists():
        raise Exception("Invalid datastructure. Input does not exist")
    if data_structure.is_dir():
        datasets: Dict[str, Any] = {}
        for f in data_structure.iterdir():
            if f.suffix != ".json":
                continue
            dataset = _load_datastructure_single(f)
            datasets = {**datasets, **dataset}
        return datasets
    else:
        if data_structure.suffix != ".json":
            raise Exception("Invalid datastructure. Must have .json extension")
        with open(data_structure, "r") as file:
            structures = json.load(file)
    return _load_dataset_from_structure(structures)


def load_datasets(
    data_structure: Union[Dict[str, Any], Path, List[Dict[str, Any]], List[Path]],
) -> Dict[str, Dataset]:
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
        ds_structures: Dict[str, Any] = {}
        for x in data_structure:
            result = _load_datastructure_single(x)
            ds_structures = {**ds_structures, **result}  # Overwrite ds_structures dict.
        return ds_structures
    return _load_datastructure_single(data_structure)


def load_datasets_with_data(data_structures: Any, datapoints: Optional[Any] = None) -> Any:
    """
    Loads the dataset structures and fills them with the data contained in the datapoints.

    Args:
        data_structures: Dict, Path or a List of dicts or Paths.
        datapoints: Dict, Path or a List of Paths.

    Returns:
        A dict with the structure and a pandas dataframe with the data.

    Raises:
        Exception: If the Path is wrong or the file is invalid.
    """
    datasets = load_datasets(data_structures)
    if datapoints is None:
        for dataset in datasets.values():
            if isinstance(dataset, Dataset):
                _fill_dataset_empty_data(dataset)
        return datasets, None
    if isinstance(datapoints, dict):
        # Handling dictionary of Pandas Dataframes
        for dataset_name, data in datapoints.items():
            if dataset_name not in datasets:
                raise Exception(f"Not found dataset {dataset_name}")
            datasets[dataset_name].data = _validate_pandas(
                datasets[dataset_name].components, data, dataset_name
            )
        for dataset_name in datasets:
            if datasets[dataset_name].data is None:
                datasets[dataset_name].data = DataFrame(
                    columns=list(datasets[dataset_name].components.keys())
                )
        return datasets, None
    # Handling dictionary of paths
    dict_datapoints = _load_datapoints_path(datapoints)
    for dataset_name, _ in dict_datapoints.items():
        if dataset_name not in datasets:
            raise Exception(f"Not found dataset {dataset_name}")

    return datasets, dict_datapoints


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
        raise Exception("Invalid vtl file. Input is not a Path object")
    if not input.exists():
        raise Exception("Invalid vtl file. Input does not exist")
    if input.suffix != ".vtl":
        raise Exception("Invalid vtl file. Must have .vtl extension")
    with open(input, "r") as f:
        return f.read()


def _load_single_value_domain(input: Path) -> Dict[str, ValueDomain]:
    if input.suffix != ".json":
        raise Exception("Invalid Value Domain file. Must have .json extension")
    with open(input, "r") as f:
        vd = ValueDomain.from_dict(json.load(f))
    return {vd.name: vd}


def load_value_domains(input: Union[Dict[str, Any], Path]) -> Dict[str, ValueDomain]:
    """
    Loads the value domains.

    Args:
        input: Dict or Path of the json file that contains the value domains data.

    Returns:
        A dictionary with the value domains data, or a list of dictionaries with them.

    Raises:
        Exception: If the value domains file is wrong, the Path is invalid, \
        or the value domains file does not exist.
    """
    if isinstance(input, dict):
        vd = ValueDomain.from_dict(input)
        return {vd.name: vd}
    if not isinstance(input, Path):
        raise Exception("Invalid vd file. Input is not a Path object")
    if not input.exists():
        raise Exception("Invalid vd file. Input does not exist")
    if input.is_dir():
        value_domains: Dict[str, Any] = {}
        for f in input.iterdir():
            vd = _load_single_value_domain(f)
            value_domains = {**value_domains, **vd}
        return value_domains
    if input.suffix != ".json":
        raise Exception("Invalid vd file. Must have .json extension")
    return _load_single_value_domain(input)


def load_external_routines(input: Union[Dict[str, Any], Path, str]) -> Any:
    """
    Load the external routines.

    Args:
        input: Dict or Path of the sql file that contains the external routine data.

    Returns:
        A dictionary with the external routine data, or a list with \
        the dictionaries from the Path given.

    Raises:
        Exception: If the sql file does not exist, the Path is wrong, or the file is not a sql one.
    """
    external_routines = {}
    if isinstance(input, dict):
        for name, query in input.items():
            ext_routine = ExternalRoutine.from_sql_query(name, query)
            external_routines[ext_routine.name] = ext_routine
        return external_routines
    if not isinstance(input, Path):
        raise Exception("Input invalid. Input must be a sql file.")
    if not input.exists():
        raise Exception("Input invalid. Input does not exist")
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
    datasets: Dict[str, Dataset], ast: Start
) -> Dict[str, Dataset]:
    """
    Returns only the datasets with a persistent assignment.
    """
    persistent = []
    for child in ast.children:
        if isinstance(child, PersistentAssignment) and hasattr(child.left, "value"):
            persistent.append(child.left.value)
    return {dataset.name: dataset for dataset in datasets.values() if dataset.name in persistent}


def _load_single_external_routine_from_file(input: Path) -> Any:
    """
    Returns a single external routine.
    """
    if not isinstance(input, Path):
        raise Exception("Input invalid")
    if not input.exists():
        raise Exception("Input does not exist")
    if input.suffix != ".sql":
        raise Exception("Input must be a sql file")
    with open(input, "r") as f:
        ext_rout = ExternalRoutine.from_sql_query(input.name.removesuffix(".sql"), f.read())
    return ext_rout


def _check_output_folder(output_folder: Union[str, Path]) -> None:
    """
    Check if the output folder exists. If not, it will create it.
    """
    if isinstance(output_folder, str):
        if "s3://" in output_folder:
            __check_s3_extra()
            if not output_folder.endswith("/"):
                raise ValueError("Output folder must be a Path or S3 URI to a directory")
            return
        try:
            output_folder = Path(output_folder)
        except Exception:
            raise ValueError("Output folder must be a Path or S3 URI to a directory")

    if not isinstance(output_folder, Path):
        raise ValueError("Output folder must be a Path or S3 URI to a directory")
    if not output_folder.exists():
        if output_folder.suffix != "":
            raise ValueError("Output folder must be a Path or S3 URI to a directory")
        os.mkdir(output_folder)


def to_vtl_json(dsd: Union[DataStructureDefinition, Schema]) -> Dict[str, Any]:
    """Formats the DataStructureDefinition as a VTL DataStructure."""
    dataset_name = dsd.id
    components = []
    NAME = "name"
    ROLE = "role"
    TYPE = "type"
    NULLABLE = "nullable"

    _components: List[SDMXComponent] = []
    _components.extend(dsd.components.dimensions)
    _components.extend(dsd.components.measures)
    _components.extend(dsd.components.attributes)

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
        id=f"T{count}", expression=expression, is_persistent=is_persistent, result=result
    )


def __generate_udo(child: Operator, count: int) -> UserDefinedOperator:
    operator_definition = ASTString().render(ast=child)
    return UserDefinedOperator(id=f"UDO{count}", operator_definition=operator_definition)


def __generate_ruleset(child: Union[DPRuleset, HRuleset], count: int) -> Ruleset:
    ruleset_type = ASTString().render(ast=child)
    if isinstance(child, DPRuleset):
        ruleset_type = "datapoint"
    elif isinstance(child, HRuleset):
        ruleset_type = "hierarchical"
    return Ruleset(id=f"RS{count}", ruleset_definition=str(child), ruleset_type=ruleset_type)


def ast_to_sdmx(ast: AST.Start, agency_id: str, version: str) -> TransformationScheme:
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

    transformation_scheme = TransformationScheme(
        items=list_transformation,
        agency=agency_id,
        id="TS1",
        vtl_version="2.1",
        version=version,
        ruleset_schemes=[
            RulesetScheme(
                items=list_rulesets, agency=agency_id, id="RS1", vtl_version="2.1", version=version
            )
        ],
        user_defined_operator_schemes=[
            UserDefinedOperatorScheme(
                items=list_udos, agency=agency_id, id="UDS1", vtl_version="2.1", version=version
            )
        ],
    )

    return transformation_scheme
