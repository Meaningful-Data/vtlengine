import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from AST.DAG import DAGAnalyzer
from DataTypes import SCALAR_TYPES

if os.environ.get("SPARK", False):
    import pyspark.pandas as pd
else:
    import pandas as pd

from antlr4 import CommonTokenStream, InputStream
from antlr4.error.ErrorListener import ErrorListener

from AST import AST
from AST.ASTConstructor import ASTVisitor
from AST.Grammar.lexer import Lexer
from AST.Grammar.parser import Parser
from Model import Dataset, Component, ExternalRoutine, Role


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


def create_ast(text: str) -> AST:
    """
    Generates the AST
    """
    stream = _lexer(text)
    cst = _parser(stream)
    visitor = ASTVisitor()
    ast = visitor.visit(cst)
    DAGAnalyzer.createDAG(ast)
    return ast


def load_datasets(dataPoints_path: Union[str, Path], dataStructures_path: Union[str, Path]) -> Dict[
    str, Dataset]:
    """
    Load the datasets
    """

    if isinstance(dataPoints_path, str):
        dataPoints_path = Path(dataPoints_path)

    if isinstance(dataStructures_path, str):
        dataStructures_path = Path(dataStructures_path)

    datasets = {}
    dataStructures = [dataStructures_path / f for f in os.listdir(dataStructures_path)
                      if f.lower().endswith('.json')]

    for f in dataStructures:
        with open(f, 'r') as file:
            structures = json.load(file)

        for dataset_json in structures['datasets']:
            dataset_name = dataset_json['name']
            components = {component['name']: Component(name=component['name'],
                                                       data_type=SCALAR_TYPES[component['type']],
                                                       role=Role(component['role']),
                                                       nullable=component['nullable'])
                          for component in dataset_json['DataStructure']}
            dataPoint = dataPoints_path / f"{dataset_name}.csv"
            if not os.path.exists(dataPoint):
                data = pd.DataFrame(columns=components.keys())
            else:
                data = pd.read_csv(str(dataPoint), sep=',')

            datasets[dataset_name] = Dataset(name=dataset_name, components=components, data=data)
    if len(datasets) == 0:
        raise FileNotFoundError("No datasets found")
    return datasets


def load_external_routines(external_routines_path: Union[str, Path]) -> Optional[
    Dict[str, ExternalRoutine]]:
    """
    Load the external routines
    """
    if isinstance(external_routines_path, str):
        external_routines_path = Path(external_routines_path)

    if len(list(external_routines_path.iterdir())) == 0:
        return

    external_routines = {}
    for f in external_routines_path.iterdir():
        with open(f, 'r') as file:
            sql_query = file.read()
        external_routines[f.stem] = ExternalRoutine.from_sql_query(f.stem, sql_query)
    return external_routines
