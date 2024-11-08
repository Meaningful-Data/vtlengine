import json
import os

import pandas as pd
import pytest

from vtlengine.API._InternalApi import load_datasets_with_data
from vtlengine.Exceptions import SemanticError


def load_datasets(base_path, code, folder_type):
    datapoints_path = base_path / 'DataSet' / folder_type
    input_path = base_path / 'DataStructure' / folder_type

    num_inputs = len([f for f in os.listdir(input_path) if f.startswith(f"{code}-")])
    datasets = {}

    for i in range(1, num_inputs + 1):
        with open(input_path / f"{code}-{i}.json", 'r') as file:
            datastructure = json.load(file)
        if 'datasets' in datastructure:
            ds_name = datastructure['datasets'][0]['name']
            datapoint = {ds_name: pd.read_csv(datapoints_path / f"{code}-{i}.csv")}
        else:
            datapoint = None
        datasets.update(load_datasets_with_data(datastructure, datapoint)[0])

    return datasets


@pytest.fixture
def load_input(request, code):
    base_path = request.node.get_closest_marker('input_path').args[0]
    return load_datasets(base_path, code, folder_type="input")


@pytest.fixture
def load_reference(request, code):
    base_path = request.node.get_closest_marker('input_path').args[0]
    return load_datasets(base_path, code, folder_type="output")

@pytest.fixture
def load_error(request, code):
    with pytest.raises(SemanticError) as context:
        base_path = request.node.get_closest_marker('input_path').args[0]
        return load_datasets(base_path, code, folder_type="input")
    return context.value.args[1]