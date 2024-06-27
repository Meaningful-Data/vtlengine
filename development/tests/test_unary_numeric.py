import json

from API import create_ast, load_datasets
from Interpreter import InterpreterAnalyzer
from Model import Dataset

datapoints_path = '../data/dataPoints'
datastructures_path = '../data/dataStructures'
results_path = 'results/unary_numeric'

datasets = load_datasets(datapoints_path, datastructures_path)

def test_unplus():
    ast = create_ast("DS_r := + DS_1;")
    interpreter = InterpreterAnalyzer(datasets)
    result = interpreter.visit(ast)
    with open(f'{results_path}/UnPlus.json', 'r') as f:
        dataset = Dataset.from_json(json.load(f))
        assert result['DS_r'] == dataset

def test_unminus():
    ast = create_ast("DS_r := - DS_1;")
    interpreter = InterpreterAnalyzer(datasets)
    result = interpreter.visit(ast)
    with open(f'{results_path}/UnMinus.json', 'r') as f:
        dataset = Dataset.from_json(json.load(f))
        assert result['DS_r'] == dataset

def test_absolute():
    ast = create_ast("DS_r := abs(- DS_1);")
    interpreter = InterpreterAnalyzer(datasets)
    result = interpreter.visit(ast)
    with open(f'{results_path}/Absolute.json', 'r') as f:
        dataset = Dataset.from_json(json.load(f))
        assert result['DS_r'] == dataset