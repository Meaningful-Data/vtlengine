import json
from pathlib import Path
from typing import Union, Optional, Dict, List

import pandas as pd

from API import create_ast, load_external_routines
from AST import PersistentAssignment, Start
from AST.DAG import DAGAnalyzer
from DataTypes import SCALAR_TYPES
from Interpreter import InterpreterAnalyzer
from Model import ValueDomain, Dataset, Scalar, Component, Role
from files.output import format_time_period_external_representation, \
    TimePeriodRepresentation
from files.parser import _validate_pandas, _fill_dataset_empty_data

base_path = Path(__file__).parent
filepath_VTL = base_path / "data" / "vtl"
filepath_ValueDomains = base_path / "data" / "ValueDomain"
filepath_sql = base_path / "data" / "sql"
filepath_json = base_path / "data" / "DataStructure" / "input"
filepath_csv = base_path / "data" / "DataSet" / "input"
filepath_out_json = base_path / "data" / "DataStructure" / "output"
filepath_out_csv = base_path / "data" / "DataSet" / "output"


def _load_dataset_from_structure(structures: dict):
    datasets = {}

    if 'datasets' in structures:
        for dataset_json in structures['datasets']:
            dataset_name = dataset_json['name']
            components = {
                component['name']: Component(name=component['name'],
                                             data_type=SCALAR_TYPES[component['type']],
                                             role=Role(component['role']),
                                             nullable=component['nullable'])
                for component in dataset_json['DataStructure']}

            datasets[dataset_name] = Dataset(name=dataset_name,
                                             components=components,
                                             data=None)
    if 'scalars' in structures:
        for scalar_json in structures['scalars']:
            scalar_name = scalar_json['name']
            scalar = Scalar(name=scalar_name,
                            data_type=SCALAR_TYPES[scalar_json['type']],
                            value=None)
            datasets[scalar_name] = scalar
    return datasets


def _load_single_datapoint(datapoint: Path):
    if datapoint.is_dir():
        datapoints = {}
        for f in datapoint.iterdir():
            dp = _load_single_datapoint(f)
            datapoints = {**datapoints, **dp}
        dict_data = datapoints
    else:
        dataset_name = datapoint.name.removesuffix('.csv')
        dict_data = {dataset_name: datapoint}
    return dict_data


def _load_datapoints_path(datapoints: Union[Path, List[Path]]):
    if isinstance(datapoints, list):
        dict_datapoints = {}
        for x in datapoints:
            result = _load_single_datapoint(x)
            dict_datapoints = {**dict_datapoints, **result}
        return dict_datapoints
    return _load_single_datapoint(datapoints)


def _load_datastructure_single(data_structure: Union[dict, Path]):
    if isinstance(data_structure, dict):
        structures = data_structure
    elif data_structure.is_dir():
        ds_structures = {}
        for f in data_structure.iterdir():
            ds_r = _load_datastructure_single(f)
            ds_structures = {**ds_structures, **ds_r}
        structures = ds_structures
    else:
        with open(data_structure, 'r') as file:
            structures = json.load(file)
    return _load_dataset_from_structure(structures)


def load_datasets(data_structure: Union[dict, Path, List[Union[dict, Path]]]):
    if not data_structure.exists():
        raise Exception('Invalid datastructure. Input does not exist')
    if data_structure.suffix != '.json':
        raise Exception('Invalid datastructure. Must have .json extension')
    if isinstance(data_structure, list):
        ds_structures = {}
        for x in data_structure:
            result = _load_datastructure_single(x)
            ds_structures = {**ds_structures, **result}  # Overwrite ds_structures dict.
        return ds_structures
    return _load_datastructure_single(data_structure)


def load_datasets_with_data(data_structures: Union[dict, Path, List[Union[dict, Path]]],
                            datapoints: Optional[Union[dict, Path, List[Path]]] = None):
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
            datasets[dataset_name].data = _validate_pandas(datasets[dataset_name].components, data)
        for dataset_name in datasets:
            if datasets[dataset_name].data is None:
                datasets[dataset_name].data = pd.DataFrame(
                    columns=list(datasets[dataset_name].components.keys()))
        return datasets, None
    # Handling dictionary of paths
    dict_datapoints = _load_datapoints_path(datapoints)
    for dataset_name, file_path in dict_datapoints.items():
        if dataset_name not in datasets:
            raise Exception(f"Not found dataset {dataset_name}")

    return datasets, dict_datapoints


def load_vtl(input: Union[str, Path]):
    if isinstance(input, str):
        return input
    if not isinstance(input, Path):
        raise Exception('Invalid vtl file. Input is not a Path object')
    if not input.exists():
        raise Exception('Invalid vtl file. Input does not exist')
    if input.suffix != '.vtl':
        raise Exception('Invalid vtl file. Must have .vtl extension')
    with open(input, 'r') as f:
        return f.read()


def load_value_domains(input: Union[dict, Path]):
    if isinstance(input, dict):
        vd = ValueDomain.from_dict(input)
        return {vd.name: vd}
    if not isinstance(input, Path):
        raise Exception('Invalid vd file. Input is not a Path object')
    if not input.exists():
        raise Exception('Invalid vd file. Input does not exist')
    if input.suffix != '.json':
        raise Exception('Invalid vd file. Must have .json extension')
    value_domains = {}
    with open(input, 'r') as file:
        vd = ValueDomain.from_json(file.read())
        value_domains[vd.name] = vd
    return value_domains


def _return_only_persistent_datasets(datasets: Dict[str, Dataset], ast: Start):
    persistent = []
    for child in ast.children:
        if isinstance(child, PersistentAssignment):
            persistent.append(child.left.value)
    return {dataset.name: dataset for dataset in datasets.values() if
            isinstance(dataset, Dataset) and dataset.name in persistent}


def semantic_analysis(script: Union[str, Path],
                      data_structures: Union[dict, Path, List[Union[dict, Path]]],
                      value_domains: Union[dict, Path] = None,
                      external_routines: Union[str, Path] = None):
    # AST generation
    vtl = load_vtl(script)
    ast = create_ast(vtl)

    # Loading datasets
    structures = load_datasets(data_structures)

    # Handling of library items
    vd = None
    if value_domains is not None:
        vd = load_value_domains(value_domains)
    ext_routines = None
    if external_routines is not None:
        ext_routines = load_external_routines(external_routines)

    # Running the interpreter
    interpreter = InterpreterAnalyzer(datasets=structures, value_domains=vd,
                                      external_routines=ext_routines,
                                      only_semantic=True)
    result = interpreter.visit(ast)
    return result


def run(script: Union[str, Path], data_structures: Union[dict, Path, List[Union[dict, Path]]],
        datapoints: Union[dict, Path, List[Path]],
        value_domains: Union[dict, Path] = None, external_routines: Union[str, Path] = None,
        time_period_output_format: str = "vtl",
        return_only_persistent=False, output_path: Optional[Path] = None):
    # AST generation
    vtl = load_vtl(script)
    ast = create_ast(vtl)


    # Loading datasets and datapoints
    datasets, path_dict = load_datasets_with_data(data_structures, datapoints)

    # Handling of library items
    vd = None
    if value_domains is not None:
        vd = load_value_domains(value_domains)
    ext_routines = None
    if external_routines is not None:
        ext_routines = load_external_routines(external_routines)

    # Checking time period output format value
    time_period_representation = TimePeriodRepresentation.check_value(time_period_output_format)

    # VTL Efficient analysis
    ds_analysis = DAGAnalyzer.ds_structure(ast)
    if output_path and not isinstance(output_path, Path):
        raise Exception('Output path must be a Path object')
    # Running the interpreter
    interpreter = InterpreterAnalyzer(datasets=datasets, value_domains=vd,
                                      external_routines=ext_routines,
                                      ds_analysis=ds_analysis,
                                      datapoints_paths=path_dict,
                                      output_path=output_path,
                                      time_period_representation=time_period_representation)
    result = interpreter.visit(ast)

    # Applying time period output format
    if output_path is None:
        for dataset in result.values():
            format_time_period_external_representation(dataset, time_period_representation)

    # Returning only persistent datasets
    if return_only_persistent:
        return _return_only_persistent_datasets(result, ast)
    return result


if __name__ == '__main__':
    # print(run(script=(filepath_VTL / 'test.vtl'),
    #           data_structures=[filepath_json / 'DS_1.json', filepath_json / 'DS_2.json'],
    #           datapoints=[filepath_csv / 'DS_1.csv', filepath_csv / 'DS_2.csv'],
    #           value_domains=None, external_routines=None,
    #           return_only_persistent=False))
    print(load_datasets(filepath_csv / 'DS_1.csv'))