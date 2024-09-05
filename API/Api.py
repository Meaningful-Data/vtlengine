import json
from pathlib import Path
from typing import Union, Optional, Dict

from DataTypes import SCALAR_TYPES
from Model import ValueDomain, Dataset, Scalar, Component, Role
from files.parser import load_datapoints

base_path = Path(__file__).parent
filepath_VTL = base_path / "data" / "vtl"
filepath_ValueDomains = base_path / "data" / "ValueDomain"
filepath_sql = base_path / "data" / "sql"
filepath_json = base_path / "data" / "DataStructure" / "input"


def load_dataset(data_structures: Union[dict, Path], datapoints: Optional[Union[dict, Path]] = None) -> Dict[
    str, Union[Dataset, Scalar]]:
    with open(data_structures, 'r') as file:
        structures = json.load(file)

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
            data = load_datapoints(components, Path(datapoints))

            datasets[dataset_name] = Dataset(name=dataset_name,
                                             components=components,
                                             data=data)
    if 'scalars' in structures:
        for scalar_json in structures['scalars']:
            scalar_name = scalar_json['name']
            scalar = Scalar(name=scalar_name,
                            data_type=SCALAR_TYPES[scalar_json['type']],
                            value=None)
            datasets[scalar_name] = scalar
    return datasets


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
