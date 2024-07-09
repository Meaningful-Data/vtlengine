import json

from API import create_ast, load_datasets
from AST.DLModifier import DLModifier
from Interpreter import InterpreterAnalyzer

if __name__ == '__main__':
    ast = create_ast("""define operator drop_identifier (ds dataset, comp component)
  returns dataset is
    max(ds group except comp)
end operator;

DS_r := drop_identifier (DS_1, Id_1);""")
    datasets = load_datasets("development/data/dataPoints", "development/data/dataStructures")
    ast = DLModifier(datasets).visit(ast)
    interpreter = InterpreterAnalyzer(datasets)

    result = interpreter.visit(ast)

    # with open('development/tests/results/binary_boolean/Xor.json', 'w') as f:
    #     f.write(result['DS_r'].to_json())
    print(result['DS_r'].name)
    print(result['DS_r'].components)
    print(result['DS_r'].data)
