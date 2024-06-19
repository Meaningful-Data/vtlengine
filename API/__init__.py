from pathlib import Path
from typing import Any, Union

from antlr4 import CommonTokenStream, InputStream
from antlr4.error.ErrorListener import ErrorListener

from AST import AST
from AST.ASTConstructor import ASTVisitor
from Grammar.lexer import Lexer
from Grammar.parser import Parser
from Model import Dataset


class __VTLSingleErrorListener(ErrorListener):
    """

    """

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        raise Exception(f"Not valid VTL Syntax \n "
                        f"offendingSymbol: {offendingSymbol} \n "
                        f"msg: {msg} \n "
                        f"line: {line}")


def _lexer(text: str) -> CommonTokenStream:
    """
    Lexing
    """
    lexer_ = Lexer(InputStream(text))
    lexer_._listeners = []
    stream = CommonTokenStream(lexer_)

    return stream


def _parser(stream: CommonTokenStream) -> Any:
    """
    Parse the expresion
    """
    vtl_parser = Parser(stream)
    vtl_parser._listeners = [__VTLSingleErrorListener()]
    return vtl_parser.start()


def create_ast(text: str) -> AST:
    """
    Generates the AST
    """
    stream = _lexer(text)
    cst = _parser(stream)
    visitor = ASTVisitor()
    return visitor.visit(cst)

def load_datasets(file_path: Union[str, Path]) -> Dataset:
    """
    Load the datasets
    """
    # TODO: Method to load CSV files into PySpark and Pandas DataFrames
    # TODO: Use Data Types and Lazy Loading on PySpark DataFrames
    pass
