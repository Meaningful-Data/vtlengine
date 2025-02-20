import json
from pathlib import Path

import pandas as pd

from vtlengine import API, DataTypes, run
from vtlengine.DataTypes import Null
from vtlengine.Model import Dataset, Scalar, ValueDomain

base_path = Path(__file__).parent / "data"
dataset_input_path = base_path / "DataSet" / "input"
dataset_output_path = base_path / "DataSet" / "output"
datastructure_input_path = base_path / "DataStructure" / "input"
datastructure_output_path = base_path / "DataStructure" / "output"
sql_path = base_path / "SQL"
vd_path = base_path / "ValueDomain"
vtl_path = base_path / "vtl"

def test_grammar():
    script_name = "test_grammar.vtl"
    with open(vtl_path / script_name, "r") as file:
        script = file.read()

    sql_name = "SQL1.sql"
    external_routines = sql_path / sql_name

    vd_name = "countries"
    value_domains = vd_path / f"{vd_name}.json"

    data_structures, datapoints = load_input_data("DS_1")
    reference_datasets, reference_scalars = load_reference_data()

    run_result = run(script=script,
                     data_structures=data_structures,
                     datapoints=datapoints,
                     external_routines=external_routines,
                     value_domains=value_domains)

    check_results(run_result, reference_datasets, reference_scalars)

def check_results(run_result, reference_datasets, reference_scalars):
    for result in run_result.values():
        if isinstance(result, Dataset):
            assert result.name in reference_datasets
            reference = reference_datasets[result.name]

            assert len(result.components) == len(reference.components)
            assert result.components == reference.components

            dataset_data = result.data.reset_index(drop=True)
            reference_data = reference.data.reset_index(drop=True)
            assert all(dataset_data == reference_data)

        else:
            assert result.name in reference_scalars
            reference = reference_scalars[result.name]

            assert result.data_type == reference.data_type
            assert result.value == reference.value


def load_input_data(dataset_name):
    with open(datastructure_input_path / f"{dataset_name}.json", "r") as file:
        data_structures = json.load(file)

    data = pd.read_csv(dataset_input_path / f"{dataset_name}.csv")

    datapoints = {dataset_name: data}

    return data_structures, datapoints

def load_reference_data():
    with open(datastructure_output_path / "reference.json", "r") as file:
        structures = json.load(file)

    datasets = {}
    for file in dataset_output_path.iterdir():
        if file.suffix == ".csv":
            data = pd.read_csv(file)
            dataset_name = file.stem
            datasets[dataset_name] = data

    scalars = {}
    for scalar in structures['scalars']:
        d_type = Null
        if scalar['data_type'] == 'TimePeriod':
            scalar['data_type'] = 'Time_Period'
        if scalar['data_type'] != 'Null':
            d_type = DataTypes.SCALAR_TYPES[scalar['data_type']]

        scalars[scalar['name']] = Scalar(name=scalar['name'],
                                         data_type=d_type,
                                         value=scalar['value'])

    datasets = API.load_datasets_with_data({'datasets': structures['datasets']}, datasets)[0]

    return datasets, scalars
