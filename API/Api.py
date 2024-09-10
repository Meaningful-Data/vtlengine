import json
from pathlib import Path
from typing import Union, Optional, Dict

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


def load_datastructures(data_structures: Union[dict, Path, list[dict, Path]]):
    with open(data_structures, 'r') as file:
        structures = json.load(file)
        return structures


def load_dataset(data_structures: Union[dict, Path], datapoints: Optional[Union[dict, Path]] = None) -> Dict[
    str, Union[Dataset, Scalar]]:
    structures = load_datastructures(data_structures)
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

            # data = load_datapoints(components, Path(datapoints))

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


def semantic_analysis(script: Union[str, Path], data_structures: Union[dict, Path],
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

    interpreter = InterpreterAnalyzer(datasets=structures, value_domains=vd, external_routines=ext_routines)
    result = interpreter.visit(ast)
    return result


if __name__ == '__main__':
    print(semantic_analysis(script=(filepath_VTL / '1-1-1-1.vtl'), data_structures=(filepath_json / '1-1-1-1-1.json'),
                            value_domains=None, external_routines=None))
