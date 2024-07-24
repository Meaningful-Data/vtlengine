import json

from API import create_ast, load_datasets
from Interpreter import InterpreterAnalyzer

if __name__ == '__main__':
    ast = create_ast('DS_r := inner_join ( DS_1 as d1, DS_2 as d2, DS_3 as d3 keep Me_1, d2#Me_2, d3#Me_1B); ')
    datasets = load_datasets("development/data/dataPoints", "development/data/dataStructures")
    interpreter = InterpreterAnalyzer(datasets)
    result = interpreter.visit(ast)

    # with open('development/tests/results/binary_boolean/Xor.json', 'w') as f:
    #     f.write(result['DS_r'].to_json())
    print(result['DS_r'].name)
    print(result['DS_r'].components)
    print(result['DS_r'].data)
