import json

from API import create_ast, load_datasets
from Interpreter import InterpreterAnalyzer
from Model import Dataset

datapoints_path = '../data/dataPoints'
datastructures_path = '../data/dataStructures'
results_path = 'results/binary_comparison'

datasets = load_datasets(datapoints_path, datastructures_path)

def test_equal():
    ast = create_ast("DS_r := DS_1 = DS_2;")
    interpreter = InterpreterAnalyzer(datasets)
    result = interpreter.visit(ast)
    with open(f'{results_path}/Equal.json', 'r') as f:
        dataset = Dataset.from_json(json.load(f))
        assert result['DS_r'] == dataset

def test_not_equal():
    ast = create_ast("DS_r := DS_1 <> DS_2;")
    interpreter = InterpreterAnalyzer(datasets)
    result = interpreter.visit(ast)
    with open(f'{results_path}/NotEqual.json', 'r') as f:
        dataset = Dataset.from_json(json.load(f))
        assert result['DS_r'] == dataset

def test_greater():
    ast = create_ast("DS_r := DS_1 > DS_2;")
    interpreter = InterpreterAnalyzer(datasets)
    result = interpreter.visit(ast)
    with open(f'{results_path}/Greater.json', 'r') as f:
        dataset = Dataset.from_json(json.load(f))
        assert result['DS_r'] == dataset

def test_greater_equal():
    ast = create_ast("DS_r := DS_1 >= DS_2;")
    interpreter = InterpreterAnalyzer(datasets)
    result = interpreter.visit(ast)
    with open(f'{results_path}/GreaterEqual.json', 'r') as f:
        dataset = Dataset.from_json(json.load(f))
        assert result['DS_r'] == dataset

def test_less():
    ast = create_ast("DS_r := DS_1 < DS_2;")
    interpreter = InterpreterAnalyzer(datasets)
    result = interpreter.visit(ast)
    with open(f'{results_path}/Less.json', 'r') as f:
        dataset = Dataset.from_json(json.load(f))
        assert result['DS_r'] == dataset

def test_less_equal():
    ast = create_ast("DS_r := DS_1 <= DS_2;")
    interpreter = InterpreterAnalyzer(datasets)
    result = interpreter.visit(ast)
    with open(f'{results_path}/LessEqual.json', 'r') as f:
        dataset = Dataset.from_json(json.load(f))
        assert result['DS_r'] == dataset