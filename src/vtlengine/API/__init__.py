from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from antlr4 import CommonTokenStream, InputStream  # type: ignore[import-untyped]
from antlr4.error.ErrorListener import ErrorListener  # type: ignore[import-untyped]
from pysdmx.io import read_sdmx
from pysdmx.io.pd import PandasDataset

from vtlengine.API._InternalApi import (
    _check_output_folder,
    _return_only_persistent_datasets,
    load_datasets,
    load_datasets_with_data,
    load_external_routines,
    load_value_domains,
    load_vtl, to_vtl_json,
)
from vtlengine.AST import Start
from vtlengine.AST.ASTConstructor import ASTVisitor
from vtlengine.AST.DAG import DAGAnalyzer
from vtlengine.AST.Grammar.lexer import Lexer
from vtlengine.AST.Grammar.parser import Parser
from vtlengine.Interpreter import InterpreterAnalyzer
from vtlengine.files.output._time_period_representation import (
    TimePeriodRepresentation,
    format_time_period_external_representation,
)

pd.options.mode.chained_assignment = None

base_path = Path(__file__).parent
schema_path = base_path / "data" / "schema"
sdmx_csv_path = base_path / "data" / "sdmx_csv"
sdmx_xml_path = base_path / "data" / "sdmx_csv"


class __VTLSingleErrorListener(ErrorListener):  # type: ignore[misc]
    """ """

    def syntaxError(
            self,
            recognizer: Any,
            offendingSymbol: str,
            line: str,
            column: str,
            msg: str,
            e: Any,
    ) -> None:
        raise Exception(
            f"Not valid VTL Syntax \n "
            f"offendingSymbol: {offendingSymbol} \n "
            f"msg: {msg} \n "
            f"line: {line}"
        )


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
    Function that creates the AST object.

    Args:
        text: Vtl string expression that will be used to create the AST object.

    Returns:
        The ast object.

    Raises:
        Exception: When the vtl syntax expression is wrong.
    """
    stream = _lexer(text)
    cst = _parser(stream)
    visitor = ASTVisitor()
    ast = visitor.visitStart(cst)
    DAGAnalyzer.createDAG(ast)
    return ast


def semantic_analysis(
        script: Union[str, Path],
        data_structures: Union[Dict[str, Any], Path, List[Union[Dict[str, Any], Path]]],
        value_domains: Optional[Union[Dict[str, Any], Path]] = None,
        external_routines: Optional[Union[Dict[str, Any], Path]] = None,
) -> Any:
    """
    Checks if the vtl operation can be done.To do that, it generates the AST with the vtl script
    given and also reviews if the data structure given can fit with it.

    This vtl script can be a string with the actual expression or a filepath to the folder
    that contains the vtl file.

    Moreover, the data structure can be a dictionary or a filepath to the folder that contains it.

    If there are any value domains or external routines, this data is taken into account.
    Both can be loaded the same way as data structures or vtl scripts are.

    Finally, the :obj:`Interpreter <vtl-engine-spark.Interpreter.InterpreterAnalyzer>`
    class takes all of this information and checks it with the ast generated to
    return the semantic analysis result.

    Concepts you may know:

    - Vtl script: The expression that shows the operation to be done.

    - Data Structure: JSON file that contains the structure and the name for the dataset(s) \
    (and/or scalar) about the datatype (String, integer or number), \
    the role (Identifier, Attribute or Measure) and the nullability each component has.

    - Value domains: Collection of unique values on the same datatype.

    - External routines: SQL query used to transform a dataset.

    This function has the following params:

    Args:
        script: String or Path of the vtl expression.
        data_structures: Dict or Path (file or folder), \
        or List of Dicts or Paths with the data structures JSON files.
        value_domains: Dict or Path of the value domains JSON files. (default: None)
        external_routines: String or Path of the external routines SQL files. (default: None)

    Returns:
        The computed datasets.

    Raises:
        Exception: If the files have the wrong format, or they do not exist, \
        or their Paths are invalid.
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
    interpreter = InterpreterAnalyzer(
        datasets=structures,
        value_domains=vd,
        external_routines=ext_routines,
        only_semantic=True,
    )
    with pd.option_context("future.no_silent_downcasting", True):
        result = interpreter.visit(ast)
    return result


def run(
        script: Union[str, Path],
        data_structures: Union[Dict[str, Any], Path, List[Union[Dict[str, Any], Path]]],
        datapoints: Union[Dict[str, Any], str, Path, List[Union[str, Path]]],
        value_domains: Optional[Union[Dict[str, Any], Path]] = None,
        external_routines: Optional[Union[str, Path]] = None,
        time_period_output_format: str = "vtl",
        return_only_persistent: bool = False,
        output_folder: Optional[Union[str, Path]] = None,
) -> Any:
    """
    Run is the main function of the ``API``, which mission is to ensure the vtl operation is ready
    to be performed.
    When the vtl expression is given, an AST object is created.
    This vtl script can be given as a string or a path with the folder or file that contains it.
    At the same time, data structures are loaded with its datapoints.

    The data structure information is contained in the JSON file given,
    and establish the datatype (string, integer or number),
    and the role that each component is going to have (Identifier, Attribute or Measure).
    It can be a dictionary or a path to the JSON file or folder that contains it.

    Moreover, a csv file with the data to operate with is going to be loaded.
    It can be given with a dictionary (dataset name : pandas Dataframe),
    a path or S3 URI to the folder, path or S3 to the csv file that contains the data.

    .. important::
        The data structure and the data points must have the same dataset
        name to be loaded correctly.

    .. important::
        If pointing to a Path or an S3 URI, dataset_name will be taken from the file name.
        Example: If the path is 'path/to/data.csv', the dataset name will be 'data'.

    .. important::
        If using an S3 URI, the path must be in the format:

        s3://bucket-name/path/to/data.csv

        The following environment variables must be set (from the AWS account):

        - AWS_ACCESS_KEY_ID
        - AWS_SECRET_ACCESS_KEY

        For more details, see
        `s3fs documentation <https://s3fs.readthedocs.io/en/latest/index.html#credentials>`_.

    Before the execution, the DAG analysis reviews if the VTL script is a direct acyclic graphs.


    If value domain data or external routines are required, the function loads this information
    and integrates them into the
    :obj:`Interpreter <vtl-engine-spark.Interpreter.InterpreterAnalyzer>` class.

    Moreover, if any component has a Time Period component, the external representation
    is passed to the Interpreter class.

    Concepts you may need to know:

    - Vtl script: The expression that shows the operation to be done.

    - Data Structure: JSON file that contains the structure and the name for the dataset(s) \
    (and/or scalar) about the datatype (String, integer or number), \
    the role (Identifier, Attribute or Measure) and the nullability each component has.

    - Data point: Pointer to the data. It will be loaded as a `Pandas Dataframe \
    <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.

    - Value domains: Collection of unique values that have the same datatype.

    - External routines: SQL query used to transform a dataset.

    This function has the following params:

    Args:
        script: String or Path with the vtl expression.

        data_structures: Dict, Path or a List of Dicts or Paths with the data structures.

        datapoints: Dict, Path, S3 URI or List of S3 URIs or Paths with data.

        value_domains: Dict or Path of the value domains JSON files. (default:None)

        external_routines: String or Path of the external routines SQL files. (default: None)

        time_period_output_format: String with the possible values \
        ("sdmx_gregorian", "sdmx_reporting", "vtl") for the representation of the \
        Time Period components.

        return_only_persistent: If True, run function will only return the results of \
        Persistent Assignments. (default: False)

        output_folder: Path or S3 URI to the output folder. (default: None)


    Returns:
       The datasets are produced without data if the output folder is defined.

    Raises:
        Exception: If the files have the wrong format, or they do not exist, \
        or their Paths are invalid.

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

    # Checking the output path to be a Path object to a directory
    if output_folder is not None:
        _check_output_folder(output_folder)

    # Running the interpreter
    interpreter = InterpreterAnalyzer(
        datasets=datasets,
        value_domains=vd,
        external_routines=ext_routines,
        ds_analysis=ds_analysis,
        datapoints_paths=path_dict,
        output_path=output_folder,
        time_period_representation=time_period_representation,
    )
    with pd.option_context("future.no_silent_downcasting", True):
        result = interpreter.visit(ast)

    # Applying time period output format
    if output_folder is None:
        for dataset in result.values():
            format_time_period_external_representation(dataset, time_period_representation)

    # Returning only persistent datasets
    if return_only_persistent:
        return _return_only_persistent_datasets(result, ast)
    return result


def run_sdmx(script: str, data_file, structure_file, definition: str):
    # Reading sdmx file
    data_msg = read_sdmx(data_file)

    # Getting the structure
    str_msg = read_sdmx(structure_file)

    data_structure = str_msg.get_data_structure_definition(definition)
    # Load the dataset
    datasets = PandasDataset(data=data_msg.get_datasets(), structure=data_structure)

    # Generate data_structure from VTL
    vtl_structure = to_vtl_json(data_structure)

    # Run the VTL script
    result = run(script, data_structures=vtl_structure, datapoints={definition: datasets.data})
    return result


# if __name__ == '__main__':
#     print(run_sdmx("DS_r := DS_1 * 2;", sdmx_csv_path / "data_v1.csv", sdmx_xml_path / "metadata.xml",
#                    "DataStructure=BIS:BIS_DER(1.0)"))
#     # leer = read_sdmx(sdmx_xml_path / "str_all.xml")
#     # print(leer.get_dataset("DataStructure=BIS:BIS_DER(1.0)").structure)
