import json
import os

import pandas as pd
import pytest

from tests.Helper import _use_duckdb_backend
from vtlengine.API import run
from vtlengine.API._InternalApi import load_datasets_with_data


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


def _load_duckdb_paths(base_path, code, folder_type):
    """Load data structure files and datapoint paths for DuckDB backend."""
    input_path = base_path / "DataStructure" / folder_type
    datapoints_path = base_path / "DataSet" / folder_type

    num_inputs = len([f for f in os.listdir(input_path) if f.startswith(f"{code}-")])
    data_structures = []
    datapoints = {}

    for i in range(1, num_inputs + 1):
        json_file = input_path / f"{code}-{i}.json"
        csv_file = datapoints_path / f"{code}-{i}.csv"
        data_structures.append(json_file)
        with open(json_file, "r") as f:
            structure = json.load(f)
        if "datasets" in structure:
            for ds in structure["datasets"]:
                datapoints[ds["name"]] = csv_file

    return data_structures, datapoints


@pytest.fixture
def load_input(request, code):
    base_path = request.node.get_closest_marker("input_path").args[0]
    return load_datasets(base_path, code, folder_type="input")


@pytest.fixture
def load_reference(request, code):
    base_path = request.node.get_closest_marker("input_path").args[0]
    return load_datasets(base_path, code, folder_type="output")


@pytest.fixture
def duckdb_input(request, code):
    """Provide data_structures and datapoints paths for DuckDB backend."""
    base_path = request.node.get_closest_marker("input_path").args[0]
    return _load_duckdb_paths(base_path, code, folder_type="input")


def run_expression(expression, load_input, duckdb_input):
    """Run a VTL expression using the configured backend."""
    if _use_duckdb_backend():
        data_structures, datapoints = duckdb_input
        return run(
            script=expression,
            data_structures=data_structures,
            datapoints=datapoints,
            return_only_persistent=False,
            use_duckdb=True,
        )
    else:
        from vtlengine.API import create_ast
        from vtlengine.Interpreter import InterpreterAnalyzer

        ast = create_ast(expression)
        interpreter = InterpreterAnalyzer(load_input)
        return interpreter.visit(ast)


def run_scalar_expression(expression):
    """Run a scalar VTL expression using the configured backend."""
    if _use_duckdb_backend():
        return run(
            script=expression,
            data_structures={"datasets": []},
            datapoints={},
            return_only_persistent=False,
            use_duckdb=True,
        )
    else:
        from vtlengine.API import create_ast
        from vtlengine.Interpreter import InterpreterAnalyzer

        ast = create_ast(expression)
        interpreter = InterpreterAnalyzer({})
        return interpreter.visit(ast)
