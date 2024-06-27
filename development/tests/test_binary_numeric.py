import json

from API import create_ast, load_datasets
from Interpreter import InterpreterAnalyzer
from Model import Dataset

datapoints_path = '../data/dataPoints'
datastructures_path = '../data/dataStructures'
results_path = 'results/binary_numeric'

datasets = load_datasets(datapoints_path, datastructures_path)

def test_binplus():
    ast = create_ast("DS_r := DS_1 + DS_2;")
    interpreter = InterpreterAnalyzer(datasets)
    result = interpreter.visit(ast)
    with open(f'{results_path}/BinPlus.json', 'r') as f:
        dataset = Dataset.from_json(json.load(f))
        assert result['DS_r'] == dataset

def test_binminus():
    ast = create_ast("DS_r := DS_1 - DS_2;")
    interpreter = InterpreterAnalyzer(datasets)
    result = interpreter.visit(ast)
    with open(f'{results_path}/BinMinus.json', 'r') as f:
        dataset = Dataset.from_json(json.load(f))
        assert result['DS_r'] == dataset

def test_mult():
    ast = create_ast("DS_r := DS_1 * DS_2;")
    interpreter = InterpreterAnalyzer(datasets)
    result = interpreter.visit(ast)
    with open(f'{results_path}/Mult.json', 'r') as f:
        dataset = Dataset.from_json(json.load(f))
        assert result['DS_r'] == dataset

def test_div():
    ast = create_ast("DS_r := DS_1 / DS_2;")
    interpreter = InterpreterAnalyzer(datasets)
    result = interpreter.visit(ast)
    with open(f'{results_path}/Div.json', 'r') as f:
        dataset = Dataset.from_json(json.load(f))
        assert result['DS_r'] == dataset