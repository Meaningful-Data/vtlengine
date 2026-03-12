import json
import os

import pandas as pd
import pytest

from vtlengine.API._InternalApi import load_datasets_with_data

VTL_ENGINE_BACKEND = os.environ.get("VTL_ENGINE_BACKEND", "duckdb").lower()
use_duckdb = VTL_ENGINE_BACKEND == "duckdb"


def _build_run_inputs(code: str, base_path):
    """Build data_structures and datapoints arguments for run() from test code."""
    ds_input_path = base_path / "DataStructure" / "input"
    dp_input_path = base_path / "DataSet" / "input"

    num_inputs = len([f for f in os.listdir(ds_input_path) if f.startswith(f"{code}-")])
    data_structures = []
    datapoints = {}
    for i in range(1, num_inputs + 1):
        ds_path = ds_input_path / f"{code}-{i}.json"
        data_structures.append(ds_path)
        with open(ds_path, "r") as file:
            structure = json.load(file)
        ds_name = structure["datasets"][0]["name"]
        datapoints[ds_name] = dp_input_path / f"{code}-{i}.csv"
    return data_structures, datapoints


def load_datasets(base_path, code, folder_type):
    datapoints_path = base_path / "DataSet" / folder_type
    input_path = base_path / "DataStructure" / folder_type

    num_inputs = len([f for f in os.listdir(input_path) if f.startswith(f"{code}-")])
    datasets = {}

    for i in range(1, num_inputs + 1):
        with open(input_path / f"{code}-{i}.json", "r") as file:
            datastructure = json.load(file)

        ds_name = datastructure["datasets"][0]["name"]
        datapoint = {ds_name: pd.read_csv(datapoints_path / f"{code}-{i}.csv")}
        datasets.update(load_datasets_with_data(datastructure, datapoint)[0])

    return datasets


@pytest.fixture
def load_input(request, code):
    base_path = request.node.get_closest_marker("input_path").args[0]
    return load_datasets(base_path, code, folder_type="input")


@pytest.fixture
def load_reference(request, code):
    base_path = request.node.get_closest_marker("input_path").args[0]
    return load_datasets(base_path, code, folder_type="output")
