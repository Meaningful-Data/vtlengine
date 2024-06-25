import json

import pandas as pd

from API import create_ast, load_datasets
from Interpreter import InterpreterAnalyzer
from Model import Dataset, Component

if __name__ == '__main__':
    ast = create_ast("DS_r := sqrt(DS_1);")
    datasets = load_datasets("development/data/dataPoints", "development/data/dataStructures")
    interpreter = InterpreterAnalyzer(datasets)
    result = interpreter.visit(ast)

    with open('development/tests/results/unary_numeric/SquareRoot.json', 'w') as f:
        f.write(result['DS_r'].to_json())
    print(result['DS_r'].name)
    print(result['DS_r'].components)
    print(result['DS_r'].data)
