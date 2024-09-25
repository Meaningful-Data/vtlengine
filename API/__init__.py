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
    Generates the AST object.
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
    """
    Checks if the vtl operation can be done.To do that, it generates the AST with the vtl script given and also reviews
    if the data structure given can fit with it.If there are any value domains or external routines, this data is taken
    into account. Finally, the :obj:`Interpreter <vtl-engine-spark.Interpreter.InterpreterAnalyzer>`
    class takes all of this information and checks it with the ast generated to return the semantic analysis result.

    Concepts you may know:
    - Vtl script: The expression that informs of the operation to be done.
    - Data Structure: Json file that contains the structure and the name for the dataset(s) (and/or scalar)
    about the datatype (String, integer or number) and the role
    (Measure or Identifier) each data has.
    - Value domains: Collection of unique values of the same datatype.
    - External routines: SQL query used to transform a dataset.

    This function has the following params:

    :param script: String or Path of the vtl expression.

    :param data_structures: Dict or Path(file or folder), or List of Dicts or Paths with the data_structures json files.

    :param value_domains: Dict or Path of the value_domains json files. (default: None)

    :param external_routines: String or Path of the external routines sql files. (default: None)

    :return: The analysis.
    """
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
    """
    Run is the main function of the API, which mission is to ensure the vtl operation is ready to be performed. When the vtl expression is given,
    an AST object is created. At the same time, data structures are loaded with its datapoints.
    This data structure information is contained in the json file given, and establish the datatype (string, integer or number),
    and the role that each component is going to have (Identifier or Measure). Moreover, a csv file with the data to operate with is going to be loaded.
    Also, the DAG analysis reviews if this data has direct acyclic graphs.

    This information is taken by the Interpreter class, to analyze if the operation correlates with the AST object.
    Also, if value domain data or external routines are required, the function loads this information and integrates
    them into the :obj:`Interpreter <vtl-engine-spark.Interpreter.InterpreterAnalyzer>` class. Moreover,
    if it is a vtl time operation, the operator is integrated in this Interpreter class.

    Finally, run function returns the vtl operation ready to perform it.

    :param script: String or Path with the vtl expression.

    :param data_structures: Dict, Path or a List of Dicts or Paths with the data structures.

    :param datapoints: Dict, Path or List of Paths with data.

    :param value_domains: (default:None)

    :param external_routines: (default: None)

    :param time_period_output_format: String with the possible values () for the representation of the Time Period components

    :param return_only_persistent: If it is True, run function will only return the expression with an only persistent argument. (default: False)

    :param output_path: Path with the output folder. (default: None)

    :return: The operation to be performed.

    """
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
