import json

from API import create_ast, load_datasets
from Interpreter import InterpreterAnalyzer

if __name__ == '__main__':
    # ast = create_ast("DS_r := DS_1 + DS_2;")
    # ast = create_ast("DS_r := DS_1[calc Me_2 := -Me_1][keep Me_2][drop Me_2][calc Me_1 := 1];")
    # ast = create_ast("DS_r := DS_1[rename Me_1 to Me_2];")
    ast = create_ast('DS_r := DS_1[sub Id_1 = "A"][sub Id_2 = "B"];')
    datasets = load_datasets("development/data/dataPoints", "development/data/dataStructures")
    # print(datasets)
    interpreter = InterpreterAnalyzer(datasets)
    result = interpreter.visit(ast)
    print(result['DS_r'].name)
    print(result['DS_r'].components)
    print(result['DS_r'].data)
