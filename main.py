import json

from API import create_ast, load_datasets
from Interpreter import InterpreterAnalyzer

if __name__ == '__main__':
    # ast = create_ast(' DS_r := if 1 then DS_1 else DS_2; ')
    # ast = create_ast(' DS_r := DS_1[calc Me_10:=if Me_1 > 10 then "ok" else "false"]; ')
    # ast = create_ast(' DS_r := if ( DS_3#Id_4 = "F" ) then 1 else 0; ')
    # ast = create_ast(' DS_r := DS_1[calc Me_10:=if Me_1 = 0 then 0 else 5/Me_1]; ')
    # ast = create_ast(' DS_r := DS_1[calc Me_10:=if Me_1 > 5 then if Me_1 = 0 then 0 else 5/Me_1 else -155]; ')
    # ast = create_ast(' DS_r := DS_1[calc Me_10:=if Me_1 > 5 then if Me_1 = 0 then Me_1 else 5/Me_1 else if Me_1 = 0 then 0 else if Me_1 = 2 then -Me_1 else Me_1]; ')
    ast = create_ast(
        ' DS_r := DS_1[calc Me_10:=if Me_1 > 5 then if Me_1 = 0 then 0 else if Me_1 < 10 then if Me_1 = 0 then 0 else 10/Me_1 else if Me_1 = 0 then 0 else 100/Me_1 else if Me_1 = 0 then 0 else if Me_1 = 2 then -Me_1 else Me_1]; ')
    # ast = create_ast(' DS_r := DS_2#Me_1; ')
    datasets = load_datasets("development/data/dataPoints", "development/data/dataStructures")
    interpreter = InterpreterAnalyzer(datasets)
    result = interpreter.visit(ast)

    # with open('development/tests/results/binary_boolean/Xor.json', 'w') as f:
    #     f.write(result['DS_r'].to_json())
    print(result['DS_r'].name)
    print(result['DS_r'].components)
    print(result['DS_r'].data)
