import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from AST.DAG import DAGAnalyzer

if os.environ.get("SPARK", False):
    pass
else:
    pass

from antlr4 import CommonTokenStream, InputStream
from antlr4.error.ErrorListener import ErrorListener

from AST import Start
from AST.ASTConstructor import ASTVisitor
from AST.Grammar.lexer import Lexer
from AST.Grammar.parser import Parser
from Model import ExternalRoutine


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
    lexer_._listeners = [__VTLSingleErrorListener()]
    stream = CommonTokenStream(lexer_)

    return stream


def _parser(stream: CommonTokenStream) -> Any:
    """
    Parse the expression
    """
    vtl_parser = Parser(stream)
    vtl_parser._listeners = [__VTLSingleErrorListener()]
    return vtl_parser.start()


def create_ast(text: str) -> Start:
    """
    Generates the AST
    """
    stream = _lexer(text)
    cst = _parser(stream)
    visitor = ASTVisitor()
    ast = visitor.visit(cst)
    DAGAnalyzer.createDAG(ast)
    return ast


def _load_single_external_routine_from_file(input: Path):
    if not isinstance(input, Path):
        raise Exception('Input invalid')
    if not input.exists():
        raise Exception('Input does not exist')
    if not '.sql' in input.name:
        raise Exception('Input must be a sql file')
    with open(input, 'r') as f:
        ext_rout = ExternalRoutine.from_sql_query(input.name.removesuffix('.sql'), f.read())
    return ext_rout


def load_external_routines(input: Union[dict, Path]) -> Optional[
    Dict[str, ExternalRoutine]]:
    """
    Load the external routines
    """
    external_routines = {}
    if isinstance(input, dict):
        for name, query in input.items():
            ext_routine = ExternalRoutine.from_sql_query(name, query)
            external_routines[ext_routine.name] = ext_routine
        return external_routines
    if not isinstance(input, Path):
        raise Exception('Input invalid')
    if not input.exists():
        raise Exception('Input does not exist')
    if input.is_dir():
        for f in input.iterdir():
            ext_rout = _load_single_external_routine_from_file(f)
            external_routines[ext_rout.name] = ext_rout
        return external_routines
    ext_rout = _load_single_external_routine_from_file(input)
    external_routines[ext_rout.name] = ext_rout
    return external_routines
