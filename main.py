from API import create_ast
from Interpreter import InterpreterAnalyzer

if __name__ == '__main__':
    ast = create_ast("DS_r := DS_1 + DS_2;")
    interpreter = InterpreterAnalyzer()
    interpreter.visit(ast)
