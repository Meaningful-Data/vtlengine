import json

from API import create_ast, load_datasets
from Interpreter import InterpreterAnalyzer

if __name__ == '__main__':
    ast = create_ast('DS_r := inner_join (DS_1 as d1, DS_2 as d2 filter Me_1 = "A" calc Me_4 := Me_1 || Me_1A drop d1#Me_2);')
    # ast = create_ast('DS_r := DS_1 [ filter Id_1 = 1 and Me_1 < 10 ];')
    datasets = load_datasets("development/data/dataPoints", "development/data/dataStructures")
    interpreter = InterpreterAnalyzer(datasets)
    result = interpreter.visit(ast)

    # with open('development/tests/results/binary_boolean/Xor.json', 'w') as f:
    #     f.write(result['DS_r'].to_json())
    print(result['DS_r'].name)
    print(result['DS_r'].components)
    print(result['DS_r'].data)
