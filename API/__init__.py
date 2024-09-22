import os
from pathlib import Path
from typing import Any, Union, List, Optional

from API._InternalApi import load_vtl, load_datasets, load_value_domains, load_external_routines, \
    load_datasets_with_data, _return_only_persistent_datasets
from AST.DAG import DAGAnalyzer
from Interpreter import InterpreterAnalyzer
from files.output import TimePeriodRepresentation, format_time_period_external_representation

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


def semantic_analysis(script: Union[str, Path],
                      data_structures: Union[dict, Path, List[Union[dict, Path]]],
                      value_domains: Union[dict, Path] = None,
                      external_routines: Union[str, Path] = None):
    # AST generation
    vtl = load_vtl(script)
    ast = create_ast(vtl)

    # Loading datasets
    structures = load_datasets(data_structures)

    # Handling of library items
    vd = None
    if value_domains is not None:
        vd = load_value_domains(value_domains)
    ext_routines = None
    if external_routines is not None:
        ext_routines = load_external_routines(external_routines)

    # Running the interpreter
    interpreter = InterpreterAnalyzer(datasets=structures, value_domains=vd,
                                      external_routines=ext_routines,
                                      only_semantic=True)
    result = interpreter.visit(ast)
    return result


def run(script: Union[str, Path], data_structures: Union[dict, Path, List[Union[dict, Path]]],
        datapoints: Union[dict, Path, List[Path]],
        value_domains: Union[dict, Path] = None, external_routines: Union[str, Path] = None,
        time_period_output_format: str = "vtl",
        return_only_persistent=False, output_path: Optional[Path] = None):
    # AST generation
    vtl = load_vtl(script)
    ast = create_ast(vtl)

    # Loading datasets and datapoints
    datasets, path_dict = load_datasets_with_data(data_structures, datapoints)

    # Handling of library items
    vd = None
    if value_domains is not None:
        vd = load_value_domains(value_domains)
    ext_routines = None
    if external_routines is not None:
        ext_routines = load_external_routines(external_routines)

    # Checking time period output format value
    time_period_representation = TimePeriodRepresentation.check_value(time_period_output_format)

    # VTL Efficient analysis
    ds_analysis = DAGAnalyzer.ds_structure(ast)
    if output_path and not isinstance(output_path, Path):
        raise Exception('Output path must be a Path object')
    # Running the interpreter
    interpreter = InterpreterAnalyzer(datasets=datasets, value_domains=vd,
                                      external_routines=ext_routines,
                                      ds_analysis=ds_analysis,
                                      datapoints_paths=path_dict,
                                      output_path=output_path,
                                      time_period_representation=time_period_representation)
    result = interpreter.visit(ast)

    # Applying time period output format
    if output_path is None:
        for dataset in result.values():
            format_time_period_external_representation(dataset, time_period_representation)

    # Returning only persistent datasets
    if return_only_persistent:
        return _return_only_persistent_datasets(result, ast)
    return result
