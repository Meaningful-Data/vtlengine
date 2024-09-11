import json
from pathlib import Path
from typing import Union, Optional, Dict, List

import pandas as pd

from API import create_ast, load_external_routines
from DataTypes import SCALAR_TYPES
from Interpreter import InterpreterAnalyzer
from Model import ValueDomain, Dataset, Scalar, Component, Role

base_path = Path(__file__).parent
filepath_VTL = base_path / "data" / "vtl"
filepath_ValueDomains = base_path / "data" / "ValueDomain"
filepath_sql = base_path / "data" / "sql"
filepath_json = base_path / "data" / "DataStructure" / "input"
filepath_csv = base_path / "data" / "DataSet" / "input"


def _check_columns(component_names: List[str], columns: List[str]):
    pass


def _add_data_to_dataset(data: pd.DataFrame, dataset: Dataset):
    _check_columns(dataset.get_components_names(), data.columns())
    dataset.data = data


def _fill_datasets_empty_data(datasets: dict):
    for dataset in datasets.values():
        dataset.data = pd.DataFrame(columns=list(dataset.components.keys()))
    return datasets


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


def load_datastructures(data_structure: Union[dict, Path, List[Union[dict, Path]]]):
    if isinstance(data_structure, list):
        ds_structures = {}
        for x in data_structure:
            result = _load_datastructure_single(x)
            ds_structures = {**ds_structures, **result}  # Overwrite ds_structures dict.
        return ds_structures
    return _load_datastructure_single(data_structure)


def load_dataset(data_structures: Union[dict, Path, List[Union[dict, Path]]],
                 datapoints: Optional[Union[dict, Path, List[Path]]] = None) -> Dict[
    str, Union[Dataset, Scalar]]:
    datasets = load_datastructures(data_structures)
    if datapoints == None:
        return _fill_datasets_empty_data(datasets)
    if isinstance(datapoints, dict):
        for dataset_name, data in datapoints.items():
            if dataset_name not in datasets:
                raise Exception('Not found dataset name')
            dataset = datasets[dataset_name]
            _add_data_to_dataset(data, dataset)
        return datasets
    if isinstance(datapoints, Path):
        dataset_name = datapoints.name
        if dataset_name not in datasets:
            raise Exception('Not found dataset name')

        dataset = datasets[dataset_name]
        _add_data_to_dataset(data, dataset)


def load_vtl(input: Union[str, Path]):
    if isinstance(input, str):
        return input
    if isinstance(input, Path):
        input = Path(input)
    if not isinstance(input, Path):
        raise Exception('Input invalid')
    if not input.exists():
        raise Exception('Input does not exist')
    with open(input, 'r') as f:
        return f.read()


def load_value_domains(input: Union[dict, Path]):
    if isinstance(input, dict):
        return input
    if not isinstance(input, Path):
        raise Exception('Invalid input')
    value_domains = {}
    with open(input, 'r') as file:
        vd = ValueDomain.from_json(file.read())
        value_domains[vd.name] = vd
    return value_domains


def semantic_analysis(script: Union[str, Path], data_structures: Union[dict, Path, List[Union[dict, Path]]],
                      value_domains: Union[dict, Path] = None, external_routines: Union[str, Path] = None):
    vtl = load_vtl(script)
    ast = create_ast(vtl)
    structures = load_datastructures(data_structures)
    vd = None
    if value_domains is not None:
        vd = load_value_domains(value_domains)
    ext_routines = None
    if external_routines is not None:
        ext_routines = load_external_routines(external_routines)

    interpreter = InterpreterAnalyzer(datasets=structures, value_domains=vd, external_routines=ext_routines,
                                      only_semantic=True)
    result = interpreter.visit(ast)
    return result


def run(script: Union[str, Path], data_structures: Union[dict, Path, List[Union[dict, Path]]],
        datapoints: Union[dict, Path, List[Path]],
        value_domains: Union[dict, Path] = None, external_routines: Union[str, Path] = None):
    vtl = load_vtl(script)
    ast = create_ast(vtl)
    datasets = load_dataset(data_structures, datapoints)
    vd = None
    if value_domains is not None:
        vd = load_value_domains(value_domains)
    ext_routines = None
    if external_routines is not None:
        ext_routines = load_external_routines(external_routines)

    interpreter = InterpreterAnalyzer(datasets=datasets, value_domains=vd, external_routines=ext_routines)
    result = interpreter.visit(ast)
    return result


if __name__ == '__main__':
    print(run(script=(filepath_VTL / '1-1-1-1.vtl'),
              data_structures=[filepath_json / '2-1-DS_1.json', filepath_json / '2-1-DS_2.json'],
              datapoints=[filepath_csv / '2-1-DS_1.csv', filepath_csv / '2-1-DS_2.csv'],
              value_domains=None, external_routines=None))
    # print(load_dataset(data_structures=(filepath_json / '1-2-DS_1.json'), datapoints=(filepath_csv / '1-2-DS_1.csv')))
    # print(load_datastructures(filepath_json))
