import itertools
import json
import os

from Interpreter import InterpreterAnalyzer

if os.environ.get("SPARK", False):
    import sys

    virtualenv_path = sys.prefix
    sys.path.append(virtualenv_path)
    # os.environ['PYTHONPATH'] = f'{virtualenv_path}'
    os.environ['PYSPARK_PYTHON'] = f'{virtualenv_path}/bin/python'
    # os.environ['PYSPARK_PYTHON'] = f'{virtualenv_path}\\Scripts\\python'
    # os.environ['VIRTUAL_ENV'] = os.environ.get('PYTHONPATH', f'{virtualenv_path}')

    from pyspark import SparkConf, SparkContext

    conf = SparkConf()
    conf.set('spark.driver.cores', '2')
    conf.set('spark.executor.cores', '2')
    conf.set('spark.driver.memory', '2g')
    conf.set('spark.executor.memory', '2g')
    # conf.set('spark.sql.execution.arrow.pyspark.enabled', 'true')
    conf.set('spark.pyspark.virtualenv.enabled', 'true')
    conf.set('spark.pyspark.virtualenv.type', 'native')
    conf.set('spark.pyspark.virtualenv.requirements', 'requirements.txt')
    # conf.set('spark.pyspark.virtualenv.bin.path', f'{virtualenv_path}/Scripts/python')
    # Pandas API on Spark automatically uses this Spark context with the configurations set.
    SparkContext(conf=conf)

    import pyspark.pandas as pd

    pd.set_option('compute.ops_on_diff_frames', True)
    os.environ["PYSPARK_SUBMIT_ARGS"] = "--conf spark.network.timeout=600s pyspark-shell"
else:
    import pandas as pd

import pytest

from API import create_ast
from DataTypes import SCALAR_TYPES
from Model import Component, Role, Dataset

input_dp_dir = 'data/DataSet/input'
reference_dp_dir = 'data/DataSet/output'
input_ds_dir = 'data/DataStructure/input'
reference_ds_dir = 'data/DataStructure/output'
vtl_dir = 'data/vtl'

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

# Remove tests not implemented (Value Domains)
comparison_operators.remove(84)

# Remove tests because Reference Manual is wrong (Pivot)
clause_operators.remove(172)

comparison_operators.remove(85)

analytic_operators.remove(155)

# TODO: Median test 144 inconsistent result on odd number of elements on pyspark
aggregation_operators.remove(144)

# Multimeasures on specific operators that must raise errors
exceptions_tests = [27, 31]

params = itertools.chain(
    # general_operators,
    # join_operators,
    # string_operators,
    # numeric_operators,
    # comparison_operators,
    # boolean_operators,
    time_operators,
    # set_operators,
    # hierarchy_operators,
    # aggregation_operators,
    # analytic_operators,
    # validation_operators,
    # conditional_operators,
    # clause_operators
)

params = [x for x in list(params) if x not in exceptions_tests]

@pytest.fixture
def ast(input_datasets, param):
    with open(os.path.join(vtl_dir, f'RM{param:03d}.vtl'), 'r') as f:
        vtl = f.read()
    return create_ast(vtl)


@pytest.fixture
def input_datasets(param):
    prefix = f'{param}-'
    suffix_csv = '.csv'
    datapoints = [f.removeprefix(prefix).removesuffix(suffix_csv) for f in os.listdir(input_dp_dir)
                  if f.lower().startswith(prefix)]
    datastructures = [f'{input_ds_dir}/{f}' for f in os.listdir(input_ds_dir)
                      if f.lower().startswith(prefix)]
    return datapoints, datastructures


@pytest.fixture
def reference_datasets(param):
    prefix = f'{param}-'
    suffix_csv = '.csv'
    datapoints = [f.removeprefix(prefix).removesuffix(suffix_csv) for f in
                  os.listdir(reference_dp_dir)
                  if f.lower().startswith(prefix)]
    datastructures = [f'{reference_ds_dir}/{f}' for f in os.listdir(reference_ds_dir)
                      if f.lower().startswith(prefix)]
    return datapoints, datastructures


def load_dataset(dataPoints, dataStructures, dp_dir, param):
    datasets = {}
    for f in dataStructures:
        with open(f, 'r') as file:
            structures = json.load(file)

        for dataset_json in structures['datasets']:
            dataset_name = dataset_json['name']
            components = {
                component['name']: Component(name=component['name'],
                                             data_type=SCALAR_TYPES[component['type']],
                                             role=Role(component['role']),
                                             nullable=component['nullable'])
                for component in dataset_json['DataStructure']}
            if dataset_name not in dataPoints:
                data = pd.DataFrame(columns=components.keys())
            else:
                data = pd.read_csv(os.path.join(dp_dir, f'{param}-{dataset_name}.csv'), sep=',')

            datasets[dataset_name] = Dataset(name=dataset_name, components=components, data=data)
    if len(datasets) == 0:
        raise FileNotFoundError("No datasets found")
    return datasets

# params = [99]
# params = [109, 110, 111, 112]
params = [113, 114, 115, 116]

@pytest.mark.parametrize('param', params)
def test_reference(input_datasets, reference_datasets, ast, param):
    # try:
    input_datasets = load_dataset(*input_datasets, dp_dir=input_dp_dir, param=param)
    reference_datasets = load_dataset(*reference_datasets, dp_dir=reference_dp_dir, param=param)
    interpreter = InterpreterAnalyzer(input_datasets)
    result = interpreter.visit(ast)
    assert result == reference_datasets
    # except NotImplementedError:
    #     pass

@pytest.mark.parametrize('param', exceptions_tests)
def test_reference_exceptions(input_datasets, reference_datasets, ast, param):
    # try:
    input_datasets = load_dataset(*input_datasets, dp_dir=input_dp_dir, param=param)
    interpreter = InterpreterAnalyzer(input_datasets)
    with pytest.raises(Exception, match="Operation not allowed for multimeasure datasets"):
        result = interpreter.visit(ast)
