from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd
from antlr4 import CommonTokenStream, InputStream  # type: ignore[import-untyped]
from antlr4.error.ErrorListener import ErrorListener  # type: ignore[import-untyped]
from pysdmx.io.pd import PandasDataset
from pysdmx.model import TransformationScheme
from pysdmx.model.dataflow import Schema

from vtlengine.API._InternalApi import (
    _check_output_folder,
    _check_script,
    _return_only_persistent_datasets,
    ast_to_sdmx,
    load_datasets,
    load_datasets_with_data,
    load_external_routines,
    load_value_domains,
    load_vtl,
    to_vtl_json,
)
from vtlengine.AST import Start
from vtlengine.AST.ASTConstructor import ASTVisitor
from vtlengine.AST.ASTString import ASTString
from vtlengine.AST.DAG import DAGAnalyzer
from vtlengine.AST.Grammar.lexer import Lexer
from vtlengine.AST.Grammar.parser import Parser
from vtlengine.Exceptions import SemanticError
from vtlengine.files.output._time_period_representation import (
    TimePeriodRepresentation,
    format_time_period_external_representation,
)
from vtlengine.Interpreter import InterpreterAnalyzer
from vtlengine.Model import Dataset

pd.options.mode.chained_assignment = None


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


def prettify(script: Union[str, TransformationScheme, Path]) -> str:
    """
    Function that prettifies the VTL script given.

    Args:
        script: VTL script as a string, a Transformation Scheme object or Path with the VTL script.

    Returns:
        A str with the prettified VTL script.
    """
    from vtlengine.AST.ASTComment import create_ast_with_comments

    checking = _check_script(script)
    vtl = load_vtl(checking)
    ast = create_ast_with_comments(vtl)
    return ASTString(pretty=True).render(ast)


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
    script: Union[str, TransformationScheme, Path],
    data_structures: Union[Dict[str, Any], Path, List[Dict[str, Any]], List[Path]],
    value_domains: Optional[Union[Dict[str, Any], Path]] = None,
    external_routines: Optional[Union[Dict[str, Any], Path]] = None,
) -> Dict[str, Dataset]:
    """
    Checks if the vtl scripts and its related datastructures are valid. As part of the compatibility
    with pysdmx library, the vtl script can be a Transformation Scheme object, which availability as
    input is going to be serialized as a string VTL script.

    Concepts you may need to know:

    - Vtl script: The script that shows the set of operations to be executed.

    - Data Structure: JSON file that contains the structure and the name for the dataset(s) \
    (and/or scalar) about the datatype (String, integer or number), \
    the role (Identifier, Attribute or Measure) and the nullability each component has.

    - Value domains: Collection of unique values on the same datatype.

    - External routines: SQL query used to transform a dataset.

    This function has the following params:

    Args:
        script: Vtl script as a string, Transformation Scheme object or Path to the folder \
        that holds the vtl script.
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
    checking = _check_script(script)
    vtl = load_vtl(checking)
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
    result = interpreter.visit(ast)
    return result


def run(
    script: Union[str, TransformationScheme, Path],
    data_structures: Union[Dict[str, Any], Path, List[Dict[str, Any]], List[Path]],
    datapoints: Union[Dict[str, pd.DataFrame], str, Path, List[Dict[str, Any]], List[Path]],
    value_domains: Optional[Union[Dict[str, Any], Path]] = None,
    external_routines: Optional[Union[str, Path]] = None,
    time_period_output_format: str = "vtl",
    return_only_persistent: bool = False,
    output_folder: Optional[Union[str, Path]] = None,
) -> Dict[str, Dataset]:
    """
    Run is the main function of the ``API``, which mission is to execute
    the vtl operation over the data.

    Concepts you may need to know:

    - Vtl script: The script that shows the set of operations to be executed.

    - Data Structure: JSON file that contains the structure and the name for the dataset(s) \
    (and/or scalar) about the datatype (String, integer or number), \
    the role (Identifier, Attribute or Measure) and the nullability each component has.

    - Data point: `Pandas Dataframe \
    <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_ \
    that holds the data related to the Dataset.

    - Value domains: Collection of unique values on the same datatype.

    - External routines: SQL query used to transform a dataset.

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

    Before the execution, the DAG analysis reviews if the VTL script is a direct acyclic graph.

    This function has the following params:

    Args:
        script: VTL script as a string, a Transformation Scheme object or Path with the VTL script.

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
    script = _check_script(script)
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
    result = interpreter.visit(ast)

    # Applying time period output format
    if output_folder is None:
        for dataset in result.values():
            format_time_period_external_representation(dataset, time_period_representation)

    # Returning only persistent datasets
    if return_only_persistent:
        return _return_only_persistent_datasets(result, ast)
    return result


def run_sdmx(
    script: Union[str, TransformationScheme, Path],
    datasets: Sequence[PandasDataset],
    value_domains: Optional[Union[Dict[str, Any], Path]] = None,
    external_routines: Optional[Union[str, Path]] = None,
    time_period_output_format: str = "vtl",
    return_only_persistent: bool = False,
    output_folder: Optional[Union[str, Path]] = None,
) -> Dict[str, Dataset]:
    """
    Executes a VTL script using a list of pysdmx `PandasDataset` objects.

    This function prepares the required VTL data structures and datapoints from
    the given list of pysdmx `PandasDataset` objects. It validates each
    `PandasDataset` uses a valid `Schema` instance as its structure. Each `Schema` is converted
    to the appropriate VTL JSON data structure, and the Pandas Dataframe is extracted.

    .. important::
        We recommend to use this function in combination with the
        `get_datasets <https://py.sdmx.io/howto/data_rw.html#pysdmx.io.get_datasets>`_
        pysdmx method.

    .. important::
        The mapping between pysdmx `PandasDataset
        <https://py.sdmx.io/howto/data_rw.html#pysdmx.io.pd.PandasDataset>`_ \
        and VTL datasets is done using the `Schema` instance of the `PandasDataset`.
        The Schema ID is used as the dataset name.

        DataStructure=MD:TEST_DS(1.0) -> TEST_DS

    The function then calls the :obj:`run <vtlengine.API>` function with the provided VTL
    script and prepared inputs.

    Before the execution, the DAG analysis reviews if the generated VTL script is a direct acyclic
    graph.

    Args:
        script: VTL script as a string, a Transformation Scheme object or Path with the VTL script.

        datasets: A list of PandasDataset.

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
        SemanticError: If any dataset does not contain a valid `Schema` instance as its structure.

    """
    datapoints = {}
    data_structures = []
    for dataset in datasets:
        schema = dataset.structure
        if not isinstance(schema, Schema):
            raise SemanticError("0-3-1-2", schema=schema)
        vtl_structure = to_vtl_json(schema)
        data_structures.append(vtl_structure)
        datapoints[schema.id] = dataset.data

    result = run(
        script=script,
        data_structures=data_structures,
        datapoints=datapoints,
        value_domains=value_domains,
        external_routines=external_routines,
        time_period_output_format=time_period_output_format,
        return_only_persistent=return_only_persistent,
        output_folder=output_folder,
    )
    return result


def generate_sdmx(
    script: Union[str, Path], agency_id: str, id: str, version: str = "1.0"
) -> TransformationScheme:
    """
    Function that generates a TransformationScheme object from a VTL script.

    The TransformationScheme object is the SDMX representation of the VTL script. \
    For more details please check the `SDMX IM VTL objects \
    <https://sdmx.org/wp-content/uploads/SDMX_3-0-0_SECTION_2_FINAL-1_0.pdf#page=146>`_, line 2266.

    Args:
        script: A string with the VTL script.
        agency_id: The Agency ID used in the generated `TransformationScheme` object.
        id: The given id of the generated `TransformationScheme` object.
        version: The Version used in the generated `TransformationScheme` object. (default: "1.0")

    Returns:
        The generated Transformation Scheme object.
    """
    vtl = load_vtl(script)
    ast = create_ast(vtl)
    result = ast_to_sdmx(ast, agency_id, id, version)
    return result
