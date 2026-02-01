from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union, cast

import pandas as pd
from antlr4 import CommonTokenStream, InputStream  # type: ignore[import-untyped]
from antlr4.error.ErrorListener import ErrorListener  # type: ignore[import-untyped]
from pysdmx.io.pd import PandasDataset
from pysdmx.model import TransformationScheme
from pysdmx.model.dataflow import Dataflow, DataStructureDefinition, Schema
from pysdmx.model.vtl import VtlDataflowMapping

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
from vtlengine.API._sdmx_utils import _build_mapping_dict, _convert_sdmx_mappings
from vtlengine.AST import Start
from vtlengine.AST.ASTConstructor import ASTVisitor
from vtlengine.AST.ASTString import ASTString
from vtlengine.AST.DAG import DAGAnalyzer
from vtlengine.AST.Grammar.lexer import Lexer
from vtlengine.AST.Grammar.parser import Parser
from vtlengine.Exceptions import InputValidationException
from vtlengine.files.output._time_period_representation import (
    TimePeriodRepresentation,
    format_time_period_external_representation,
)
from vtlengine.Interpreter import InterpreterAnalyzer
from vtlengine.Model import Dataset, Scalar

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


def _extract_input_datasets(script: Union[str, TransformationScheme, Path]) -> List[str]:
    if isinstance(script, TransformationScheme):
        vtl_script = _check_script(script)
    elif isinstance(script, (str, Path)):
        vtl_script = load_vtl(script)
    else:
        raise TypeError("Unsupported script type.")

    ast = create_ast(vtl_script)
    dag_inputs = DAGAnalyzer.ds_structure(ast)["global_inputs"]

    return dag_inputs


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
    text = text + "\n"
    stream = _lexer(text)
    cst = _parser(stream)
    visitor = ASTVisitor()
    ast = visitor.visitStart(cst)
    DAGAnalyzer.createDAG(ast)
    return ast


def validate_dataset(
    data_structures: Union[Dict[str, Any], Path, List[Union[Dict[str, Any], Path]]],
    datapoints: Optional[
        Union[Dict[str, Union[pd.DataFrame, Path, str]], List[Union[str, Path]], Path, str]
    ] = None,
    scalar_values: Optional[Dict[str, Optional[Union[int, str, bool, float]]]] = None,
) -> None:
    """
    Validate that datasets can be loaded from the given data_structures and optional datapoints.

    Args:
        data_structures: Dict, Path, or List of Dict/Path objects representing data structures.
        datapoints: Optional Dict, Path, or List of Dict/Path objects representing datapoints.
        scalar_values: Optional Dict with scalar values to be used in the datasets.

    Raises:
        Exception: If the data structures or datapoints are invalid or cannot be loaded.
    """
    load_datasets_with_data(data_structures, datapoints, scalar_values)


def validate_value_domain(
    input: Union[Dict[str, Any], Path, List[Union[Dict[str, Any], Path]]],
) -> None:
    """
    Validate ValueDomain(s) using JSON Schema.

    Args:
        input: Dict, Path, or List of Dict/Path objects representing value domain definitions.

    Raises:
        Exception: If the input file is invalid, does not exist,
                   or the JSON content does not follow the schema.
    """
    load_value_domains(input)


def validate_external_routine(
    input: Union[Dict[str, Any], Path, List[Union[Dict[str, Any], Path]]],
) -> None:
    """
    Validate External Routine(s) using JSON Schema and SQLGlot.

    Args:
        input: Dict, Path, or List of Dict/Path objects representing external routines.

    Raises:
        Exception: If JSON schema validation fails,
                   SQL syntax is invalid, or file type is wrong.
    """
    load_external_routines(input)


def semantic_analysis(
    script: Union[str, TransformationScheme, Path],
    data_structures: Union[
        Dict[str, Any],
        Path,
        Schema,
        DataStructureDefinition,
        Dataflow,
        List[Union[Dict[str, Any], Path, Schema, DataStructureDefinition, Dataflow]],
    ],
    value_domains: Optional[Union[Dict[str, Any], Path, List[Union[Dict[str, Any], Path]]]] = None,
    external_routines: Optional[
        Union[Dict[str, Any], Path, List[Union[Dict[str, Any], Path]]]
    ] = None,
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
        value_domains: Dict or Path, or List of Dicts or Paths of the \
        value domains JSON files. (default:None) It is passed as an object, that can be read from \
        a Path or from a dictionary. Furthermore, a list of those objects can be passed. \
        Check the following example: \
        :ref:`Example 5 <example_5_run_with_multiple_value_domains_and_external_routines>`.

        external_routines: String or Path, or List of Strings or Paths of the \
        external routines SQL files. (default: None) It is passed as an object, that can be read \
        from a Path or from a dictionary. Furthermore, a list of those objects can be passed. \
        Check the following example: \
        :ref:`Example 5 <example_5_run_with_multiple_value_domains_and_external_routines>`.

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
    datasets, scalars = load_datasets(data_structures)

    # Handling of library items
    vd = None
    if value_domains is not None:
        vd = load_value_domains(value_domains)
    ext_routines = None
    if external_routines is not None:
        ext_routines = load_external_routines(external_routines)

    # Running the interpreter
    interpreter = InterpreterAnalyzer(
        datasets=datasets,
        value_domains=vd,
        external_routines=ext_routines,
        scalars=scalars,
        only_semantic=True,
    )
    result = interpreter.visit(ast)
    return result


def _run_with_duckdb(
    script: Union[str, TransformationScheme, Path],
    data_structures: Union[
        Dict[str, Any],
        Path,
        Schema,
        DataStructureDefinition,
        Dataflow,
        List[Union[Dict[str, Any], Path, Schema, DataStructureDefinition, Dataflow]],
    ],
    datapoints: Union[Dict[str, Union[pd.DataFrame, str, Path]], List[Union[str, Path]], str, Path],
    value_domains: Optional[Union[Dict[str, Any], Path, List[Union[Dict[str, Any], Path]]]] = None,
    external_routines: Optional[
        Union[Dict[str, Any], Path, List[Union[Dict[str, Any], Path]]]
    ] = None,
    return_only_persistent: bool = True,
    scalar_values: Optional[Dict[str, Optional[Union[int, str, bool, float]]]] = None,
    output_folder: Optional[Union[str, Path]] = None,
) -> Dict[str, Union[Dataset, Scalar]]:
    """
    Run VTL script using DuckDB as the execution engine.

    This function transpiles VTL to SQL and executes it using DuckDB.
    When output_folder is provided, uses efficient CSV IO with DuckDB's
    native read_csv and COPY TO for memory-efficient processing.
    """
    import duckdb

    from vtlengine.AST.DAG._words import DELETE, GLOBAL, INSERT, PERSISTENT
    from vtlengine.duckdb_transpiler import SQLTranspiler
    from vtlengine.duckdb_transpiler.io import load_datapoints_duckdb, save_datapoints_duckdb

    # AST generation
    script = _check_script(script)
    vtl = load_vtl(script)
    ast = create_ast(vtl)

    # Load datasets structure (without data)
    input_datasets, input_scalars = load_datasets(data_structures)

    # Apply scalar values if provided
    if scalar_values:
        for name, value in scalar_values.items():
            if name in input_scalars:
                input_scalars[name].value = value

    # Run semantic analysis to get output structures
    interpreter = InterpreterAnalyzer(
        datasets=input_datasets,
        value_domains=load_value_domains(value_domains) if value_domains else None,
        external_routines=load_external_routines(external_routines) if external_routines else None,
        scalars=input_scalars,
        only_semantic=True,
    )
    semantic_results = interpreter.visit(ast)

    # Separate output datasets and scalars
    output_datasets: Dict[str, Dataset] = {}
    output_scalars: Dict[str, Scalar] = {}
    for name, result in semantic_results.items():
        if isinstance(result, Dataset):
            output_datasets[name] = result
        elif isinstance(result, Scalar):
            output_scalars[name] = result

    # Get DAG analysis for efficient load/save scheduling
    ds_analysis = DAGAnalyzer.ds_structure(ast)

    # Create DuckDB connection
    conn = duckdb.connect()

    # Normalize output folder path
    output_folder_path = Path(output_folder) if output_folder else None

    # Load datapoints - get path mappings
    _, _, path_dict = load_datasets_with_data(data_structures, datapoints, scalar_values)

    # If output_folder is provided, use efficient CSV IO with DAG scheduling
    if output_folder_path:
        # Ensure output folder exists
        output_folder_path.mkdir(parents=True, exist_ok=True)
    else:
        # Without output_folder, load all data upfront via pandas (original behavior)
        datasets_with_data, _, _ = load_datasets_with_data(
            data_structures, datapoints, scalar_values
        )
        for ds_name, ds in datasets_with_data.items():
            # Prioritize loading from CSV path if available
            if path_dict and ds_name in path_dict:
                df = pd.read_csv(path_dict[ds_name])
                conn.register(ds_name, df)
            elif ds.data is not None and len(ds.data) > 0:
                conn.register(ds_name, ds.data)

    # Create transpiler and generate SQL
    transpiler = SQLTranspiler(
        input_datasets=input_datasets,
        output_datasets=output_datasets,
        input_scalars=input_scalars,
        output_scalars=output_scalars,
        value_domains=load_value_domains(value_domains) if value_domains else {},
        external_routines=load_external_routines(external_routines) if external_routines else {},
    )
    queries = transpiler.transpile(ast)

    # Get persistent and global datasets from DAG analysis
    persistent_datasets = ds_analysis.get(PERSISTENT, [])
    global_inputs = ds_analysis.get(GLOBAL, [])

    # Execute queries with efficient IO
    results: Dict[str, Union[Dataset, Scalar]] = {}

    for statement_num, (result_name, sql_query, _) in enumerate(queries, start=1):
        # Load datasets scheduled for this statement (efficient mode with output_folder)
        if output_folder_path and statement_num in ds_analysis.get(INSERT, {}):
            for ds_name in ds_analysis[INSERT][statement_num]:
                if path_dict and ds_name in path_dict and ds_name in input_datasets:
                    load_datapoints_duckdb(
                        conn=conn,
                        components=input_datasets[ds_name].components,
                        dataset_name=ds_name,
                        csv_path=path_dict[ds_name],
                    )

        # Execute query and create table
        conn.execute(f'CREATE TABLE "{result_name}" AS {sql_query}')

        # Save/delete datasets scheduled for deletion (efficient mode)
        if output_folder_path and statement_num in ds_analysis.get(DELETE, {}):
            for ds_name in ds_analysis[DELETE][statement_num]:
                if ds_name in global_inputs:
                    # Drop global inputs without saving
                    conn.execute(f'DROP TABLE IF EXISTS "{ds_name}"')
                elif not return_only_persistent or ds_name in persistent_datasets:
                    # Save to CSV and drop table
                    save_datapoints_duckdb(conn, ds_name, output_folder_path)
                    ds = output_datasets.get(
                        ds_name, Dataset(name=ds_name, components={}, data=None)
                    )
                    results[ds_name] = ds
                else:
                    # Drop non-persistent intermediate results
                    conn.execute(f'DROP TABLE IF EXISTS "{ds_name}"')

    # Handle final results
    for result_name, _, is_persistent in queries:
        if result_name in results:
            continue

        should_include = not return_only_persistent or is_persistent
        if not should_include:
            continue

        if output_folder_path:
            # Save to CSV
            save_datapoints_duckdb(conn, result_name, output_folder_path)
            ds = output_datasets.get(
                result_name, Dataset(name=result_name, components={}, data=None)
            )
            results[result_name] = ds
        else:
            # Return as DataFrame
            result_df = conn.execute(f'SELECT * FROM "{result_name}"').fetchdf()

            if result_name in output_scalars:
                if len(result_df) == 1 and len(result_df.columns) == 1:
                    scalar = output_scalars[result_name]
                    scalar.value = result_df.iloc[0, 0]
                    results[result_name] = scalar
                else:
                    results[result_name] = Dataset(name=result_name, components={}, data=result_df)
            else:
                ds = output_datasets.get(
                    result_name, Dataset(name=result_name, components={}, data=None)
                )
                ds.data = result_df
                results[result_name] = ds

    conn.close()
    return results


def run(
    script: Union[str, TransformationScheme, Path],
    data_structures: Union[
        Dict[str, Any],
        Path,
        Schema,
        DataStructureDefinition,
        Dataflow,
        List[Union[Dict[str, Any], Path, Schema, DataStructureDefinition, Dataflow]],
    ],
    datapoints: Union[Dict[str, Union[pd.DataFrame, str, Path]], List[Union[str, Path]], str, Path],
    value_domains: Optional[Union[Dict[str, Any], Path, List[Union[Dict[str, Any], Path]]]] = None,
    external_routines: Optional[
        Union[Dict[str, Any], Path, List[Union[Dict[str, Any], Path]]]
    ] = None,
    time_period_output_format: str = "vtl",
    return_only_persistent: bool = True,
    output_folder: Optional[Union[str, Path]] = None,
    scalar_values: Optional[Dict[str, Optional[Union[int, str, bool, float]]]] = None,
    sdmx_mappings: Optional[Union[VtlDataflowMapping, Dict[str, str]]] = None,
    use_duckdb: bool = False,
) -> Dict[str, Union[Dataset, Scalar]]:
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

        data_structures: Dict, Path, pysdmx object, or a List of these with the data structures. \
        Supports VTL JSON format (dict or .json file), SDMX structure files (.xml or SDMX-JSON), \
        or pysdmx objects (Schema, DataStructureDefinition, Dataflow).

        datapoints: Dict, Path, S3 URI or List of S3 URIs or Paths with data. \
        Supports plain CSV files and SDMX files (.xml for SDMX-ML, .json for SDMX-JSON, \
        and .csv for SDMX-CSV with embedded structure). SDMX files are automatically \
        detected by extension and loaded using pysdmx. For SDMX files requiring \
        external structure files, use the :obj:`run_sdmx` function instead. \
        You can also use a custom name for the dataset by passing a dictionary with \
        the dataset name as key and the Path, S3 URI or DataFrame as value. \
        Check the following example: \
        :ref:`Example 6 <example_6_run_using_paths>`.

        value_domains: Dict or Path, or List of Dicts or Paths of the \
        value domains JSON files. (default:None) It is passed as an object, that can be read from \
        a Path or from a dictionary. Furthermore, a list of those objects can be passed. \
        Check the following example: \
        :ref:`Example 5 <example_5_run_with_multiple_value_domains_and_external_routines>`.

        external_routines: String or Path, or List of Strings or Paths of the \
        external routines JSON files. (default: None) It is passed as an object, that can be read \
        from a Path or from a dictionary. Furthermore, a list of those objects can be passed. \
        Check the following example: \
        :ref:`Example 5 <example_5_run_with_multiple_value_domains_and_external_routines>`.

        time_period_output_format: String with the possible values \
        ("sdmx_gregorian", "sdmx_reporting", "vtl") for the representation of the \
        Time Period components.

        return_only_persistent: If True, run function will only return the results of \
        Persistent Assignments. (default: True)

        output_folder: Path or S3 URI to the output folder. (default: None)

        scalar_values: Dict with the scalar values to be used in the VTL script.

        sdmx_mappings: A dictionary or VtlDataflowMapping object that maps SDMX URNs \
        (e.g., "Dataflow=MD:TEST_DF(1.0)") to VTL dataset names. This parameter is \
        primarily used when calling run() from run_sdmx() to pass mapping configuration.

        use_duckdb: If True, use DuckDB as the execution engine instead of pandas. \
        This transpiles VTL to SQL and executes it using DuckDB, which can be more \
        efficient for large datasets. (default: False)

    Returns:
       The datasets are produced without data if the output folder is defined.

    Raises:
        Exception: If the files have the wrong format, or they do not exist, \
        or their Paths are invalid.

    """
    # Use DuckDB execution engine if requested (check early to avoid unnecessary processing)
    if use_duckdb:
        return _run_with_duckdb(
            script=script,
            data_structures=data_structures,
            datapoints=datapoints,
            value_domains=value_domains,
            external_routines=external_routines,
            return_only_persistent=return_only_persistent,
            scalar_values=scalar_values,
            output_folder=output_folder,
        )

    # Convert sdmx_mappings to dict format for internal use
    mapping_dict = _convert_sdmx_mappings(sdmx_mappings)

    # AST generation
    script = _check_script(script)
    vtl = load_vtl(script)
    ast = create_ast(vtl)

    # Loading datasets and datapoints
    datasets, scalars, path_dict = load_datasets_with_data(
        data_structures, datapoints, scalar_values, sdmx_mappings=mapping_dict
    )

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
        return_only_persistent=return_only_persistent,
        scalars=scalars,
    )
    result = interpreter.visit(ast)

    # Applying time period output format
    if output_folder is None:
        for obj in result.values():
            if isinstance(obj, (Dataset, Scalar)):
                format_time_period_external_representation(obj, time_period_representation)

    # Returning only persistent datasets
    if return_only_persistent:
        return _return_only_persistent_datasets(result, ast)
    return result


def run_sdmx(
    script: Union[str, TransformationScheme, Path],
    datasets: Sequence[PandasDataset],
    mappings: Optional[Union[VtlDataflowMapping, Dict[str, str]]] = None,
    value_domains: Optional[Union[Dict[str, Any], Path, List[Union[Dict[str, Any], Path]]]] = None,
    external_routines: Optional[
        Union[Dict[str, Any], Path, List[Union[Dict[str, Any], Path]]]
    ] = None,
    time_period_output_format: str = "vtl",
    return_only_persistent: bool = True,
    output_folder: Optional[Union[str, Path]] = None,
) -> Dict[str, Union[Dataset, Scalar]]:
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

        mappings: A dictionary or VtlDataflowMapping object that maps the dataset names.

        value_domains: Dict or Path, or List of Dicts or Paths of the \
        value domains JSON files. (default:None) It is passed as an object, that can be read from \
        a Path or from a dictionary. Furthermore, a list of those objects can be passed. \
        Check the following example: \
        :ref:`Example 5 <example_5_run_with_multiple_value_domains_and_external_routines>`.

        external_routines: String or Path, or List of Strings or Paths of the \
        external routines JSON files. (default: None) It is passed as an object, that can be read \
        from a Path or from a dictionary. Furthermore, a list of those objects can be passed. \
        Check the following example: \
        :ref:`Example 5 <example_5_run_with_multiple_value_domains_and_external_routines>`.

        time_period_output_format: String with the possible values \
        ("sdmx_gregorian", "sdmx_reporting", "vtl") for the representation of the \
        Time Period components.

        return_only_persistent: If True, run function will only return the results of \
        Persistent Assignments. (default: True)

        output_folder: Path or S3 URI to the output folder. (default: None)

    Returns:
       The datasets are produced without data if the output folder is defined.

    Raises:
        SemanticError: If any dataset does not contain a valid `Schema` instance as its structure.

    """
    # Validate datasets input type
    if not isinstance(datasets, (list, tuple)) or any(
        not isinstance(ds, PandasDataset) for ds in datasets
    ):
        type_ = type(datasets).__name__
        if isinstance(datasets, (list, tuple)):
            object_typing = {type(o).__name__ for o in datasets}
            type_ = f"{type_}[{', '.join(object_typing)}]"
        raise InputValidationException(code="0-1-3-7", type_=type_)

    # Build mapping from SDMX URNs to VTL dataset names
    input_names = _extract_input_datasets(script)
    mapping_dict = _build_mapping_dict(datasets, mappings, input_names)

    # Validate all mapped names exist in the script
    for vtl_name in mapping_dict.values():
        if vtl_name not in input_names:
            raise InputValidationException(code="0-1-3-5", dataset_name=vtl_name)

    # Convert PandasDatasets to VTL data structures and datapoints
    datapoints_dict: Dict[str, pd.DataFrame] = {}
    data_structures_list: List[Dict[str, Any]] = []
    for dataset in datasets:
        schema = dataset.structure
        if not isinstance(schema, Schema):
            raise InputValidationException(code="0-1-3-2", schema=schema)
        if schema.short_urn not in mapping_dict:
            raise InputValidationException(code="0-1-3-4", short_urn=schema.short_urn)
        dataset_name = mapping_dict[schema.short_urn]
        vtl_structure = to_vtl_json(schema, dataset_name)
        data_structures_list.append(vtl_structure)
        datapoints_dict[dataset_name] = dataset.data

    # Validate all script inputs are mapped
    missing = [name for name in input_names if name not in mapping_dict.values()]
    if missing:
        raise InputValidationException(code="0-1-3-6", missing=missing)

    return run(
        script=script,
        data_structures=cast(
            List[Union[Dict[str, Any], Path, Schema, DataStructureDefinition, Dataflow]],
            data_structures_list,
        ),
        datapoints=cast(Dict[str, Union[pd.DataFrame, str, Path]], datapoints_dict),
        value_domains=value_domains,
        external_routines=external_routines,
        time_period_output_format=time_period_output_format,
        return_only_persistent=return_only_persistent,
        output_folder=output_folder,
        sdmx_mappings=mappings,
    )


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
