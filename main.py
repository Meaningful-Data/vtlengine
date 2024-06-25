import json

from API import create_ast, load_datasets
from Interpreter import InterpreterAnalyzer

if __name__ == '__main__':
    # ast = create_ast("DS_r := DS_1 + DS_2;")
    # ast = create_ast("DS_r := DS_1[calc Me_2 := abs(Me_1)];")
    # ast = create_ast("DS_r := DS_1#Me_1 = DS_2#Me_1;")
    ast = create_ast("DS_r := exists_in(DS_1, DS_2);")
    datasets = load_datasets("development/data/dataPoints",
                             "development/data/dataStructures")
    # print(datasets)
    interpreter = InterpreterAnalyzer(datasets)
    result = interpreter.visit(ast)
    print(result['DS_r'].name)
    print(result['DS_r'].components)
    print(result['DS_r'].data)
