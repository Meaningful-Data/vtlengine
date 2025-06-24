import itertools
import json
import os
import warnings
from pathlib import Path
import pandas as pd
import pytest

from vtlengine.API import create_ast
from vtlengine.DataTypes import SCALAR_TYPES
from vtlengine.files.parser import load_datapoints
from vtlengine.Interpreter import InterpreterAnalyzer
from vtlengine.Model import Component, Dataset, Role, ValueDomain

base_path = Path(__file__).parent
input_dp_dir = base_path / "data/DataSet/input"
reference_dp_dir = base_path / "data/DataSet/output"
input_ds_dir = base_path / "data/DataStructure/input"
reference_ds_dir = base_path / "data/DataStructure/output"
vtl_dir = base_path / "data/vtl"
vtl_def_operators_dir = base_path / "data/vtl_defined_operators"
value_domain_dir = base_path / "data/ValueDomain"

general_operators = list(range(1, 6))
join_operators = list(range(6, 13))
string_operators = list(range(13, 32))
numeric_operators = list(range(32, 73))
comparison_operators = list(range(73, 91))
boolean_operators = list(range(91, 99))
time_operators = list(range(99, 126))
set_operators = list(range(126, 132))
hierarchy_operators = list(range(132, 135))
aggregation_operators = list(range(135, 151))
analytic_operators = list(range(151, 157))
validation_operators = list(range(157, 161))
conditional_operators = list(range(161, 163))
clause_operators = list(range(163, 177))
new_operators = list(range(177, 186))

# Remove tests because Reference Manual is wrong (Pivot)
clause_operators.remove(172)

# TODO: check if test 107 is correct
# remove test until re-evaluation
time_operators.remove(107)

# Remove tests because Time Interval with Periods
time_operators.remove(101)
time_operators.remove(102)
time_operators.remove(109)
time_operators.remove(113)
time_operators.remove(117)

# Remove tests because missing test files
time_operators.remove(121)
time_operators.remove(122)
time_operators.remove(123)
time_operators.remove(124)
time_operators.remove(125)

# Remove tests due to wrong implicit cast (Duration to String)
time_operators.remove(100)

# Multimeasures on specific operators that must raise errors
exceptions_tests = [27, 31]

params = itertools.chain(
    general_operators,
    join_operators,
    string_operators,
    numeric_operators,
    comparison_operators,
    boolean_operators,
    time_operators,
    set_operators,
    hierarchy_operators,
    aggregation_operators,
    analytic_operators,
    validation_operators,
    conditional_operators,
    clause_operators,
    new_operators,
)

params = [x for x in list(params) if x not in exceptions_tests]


@pytest.fixture
def ast(input_datasets, param):
    with open(os.path.join(vtl_dir, f"RM{param:03d}.vtl"), "r") as f:
        vtl = f.read()
    return create_ast(vtl)


@pytest.fixture
def ast_defined_operators(input_datasets, param):
    with open(os.path.join(vtl_def_operators_dir, f"RM{param:03d}.vtl"), "r") as f:
        vtl = f.read()
    return create_ast(vtl)


@pytest.fixture
def value_domains():
    vds = {}
    for f in os.listdir(value_domain_dir):
        with open(os.path.join(value_domain_dir, f), "r") as file:
            value_domain = ValueDomain.from_json(file.read())
            vds[value_domain.name] = value_domain
    return vds


@pytest.fixture
def input_datasets(param):
    prefix = f"{param}-"
    suffix_csv = ".csv"
    datapoints = [
        f.removeprefix(prefix).removesuffix(suffix_csv)
        for f in os.listdir(input_dp_dir)
        if f.lower().startswith(prefix)
    ]
    datastructures = [
        f"{input_ds_dir}/{f}" for f in os.listdir(input_ds_dir) if f.lower().startswith(prefix)
    ]
    return datapoints, datastructures


@pytest.fixture
def reference_datasets(param):
    prefix = f"{param}-"
    suffix_csv = ".csv"
    datapoints = [
        f.removeprefix(prefix).removesuffix(suffix_csv)
        for f in os.listdir(reference_dp_dir)
        if f.lower().startswith(prefix)
    ]
    datastructures = [
        f"{reference_ds_dir}/{f}"
        for f in os.listdir(reference_ds_dir)
        if f.lower().startswith(prefix)
    ]
    return datapoints, datastructures


def load_dataset(dataPoints, dataStructures, dp_dir, param):
    datasets = {}
    for f in dataStructures:
        with open(f, "r") as file:
            structures = json.load(file)

        for dataset_json in structures["datasets"]:
            dataset_name = dataset_json["name"]
            components = {
                component["name"]: Component(
                    name=component["name"],
                    data_type=SCALAR_TYPES[component["type"]],
                    role=Role(component["role"]),
                    nullable=component["nullable"],
                )
                for component in dataset_json["DataStructure"]
            }
            if dataset_name not in dataPoints:
                data = pd.DataFrame(columns=components.keys())
            else:
                data = load_datapoints(
                    components=components,
                    dataset_name=dataset_name,
                    csv_path=Path(f"{dp_dir}/{param}-{dataset_name}.csv"),
                )

            datasets[dataset_name] = Dataset(name=dataset_name, components=components, data=data)
    if len(datasets) == 0:
        raise FileNotFoundError("No datasets found")
    return datasets


@pytest.mark.parametrize("param", params)
def test_reference(input_datasets, reference_datasets, ast, param, value_domains):
    # try:
    warnings.filterwarnings("ignore", category=FutureWarning)
    input_datasets = load_dataset(*input_datasets, dp_dir=input_dp_dir, param=param)
    reference_datasets = load_dataset(*reference_datasets, dp_dir=reference_dp_dir, param=param)
    interpreter = InterpreterAnalyzer(input_datasets, value_domains=value_domains)
    result = interpreter.visit(ast)
    assert result == reference_datasets
    # except NotImplementedError:
    #     pass


@pytest.mark.parametrize("param", params)
def test_reference_defined_operators(
    input_datasets, reference_datasets, ast_defined_operators, param, value_domains
):
    warnings.filterwarnings("ignore", category=FutureWarning)
    input_datasets = load_dataset(*input_datasets, dp_dir=input_dp_dir, param=param)
    reference_datasets = load_dataset(*reference_datasets, dp_dir=reference_dp_dir, param=param)
    interpreter = InterpreterAnalyzer(input_datasets, value_domains=value_domains)
    result = interpreter.visit(ast_defined_operators)
    assert result == reference_datasets


@pytest.mark.parametrize("param", exceptions_tests)
def test_reference_exceptions(input_datasets, reference_datasets, ast, param):
    # try:
    warnings.filterwarnings("ignore", category=FutureWarning)
    input_datasets = load_dataset(*input_datasets, dp_dir=input_dp_dir, param=param)
    interpreter = InterpreterAnalyzer(input_datasets)
    with pytest.raises(Exception, match="Operation not allowed for multimeasure datasets"):
        # result = interpreter.visit(ast) # to match with F841
        interpreter.visit(ast)
