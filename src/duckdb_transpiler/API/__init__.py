from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pysdmx.model import TransformationScheme

from duckdb_transpiler.Config.config import create_configured_connection
from duckdb_transpiler.IO._execution import execute_queries
from duckdb_transpiler.IO._model import Query
from duckdb_transpiler.Transpiler import SQLTranspiler
from vtlengine.API import create_ast, semantic_analysis
from vtlengine.API._InternalApi import _check_script, load_datasets, load_vtl
from vtlengine.AST.DAG import DAGAnalyzer
from vtlengine.Model import Dataset, Scalar


def transpile(
    script: Union[str, TransformationScheme, Path],
    data_structures: Union[Dict[str, Any], Path, List[Dict[str, Any]], List[Path]],
    value_domains: Optional[Union[Dict[str, Any], Path, List[Union[Dict[str, Any], Path]]]] = None,
    external_routines: Optional[
        Union[Dict[str, Any], Path, List[Union[Dict[str, Any], Path]]]
    ] = None,
    scalar_values: Optional[Dict[str, Optional[Union[int, str, bool, float]]]] = None,
) -> List[Query]:
    """
    Transpile a VTL script to SQL queries.

    Args:
        script: VTL script as string, TransformationScheme object, or Path.
        data_structures: Dict or Path with data structure definitions.
        value_domains: Optional value domains.
        external_routines: Optional external routines.
        scalar_values: Optional dict of scalar values to inline in queries.

    Returns:
        List of Query objects with name, inputs, structure and SQL.
    """
    # 1. Parse script and create AST
    script = _check_script(script)
    vtl = load_vtl(script)
    ast = create_ast(vtl)
    dag = DAGAnalyzer().createDAG(ast)

    # 2. Load input datasets and scalars from data structures
    input_datasets, input_scalars = load_datasets(data_structures)

    # 3. Apply scalar values if provided
    if scalar_values:
        for name, value in scalar_values.items():
            if name in input_scalars:
                input_scalars[name].value = value

    # 4. Run semantic analysis to get output structures and validate script
    semantic_results = semantic_analysis(
        script=vtl,
        data_structures=data_structures,
        value_domains=value_domains,
        external_routines=external_routines,
    )

    # 5. Separate output datasets and scalars from semantic results
    output_datasets: Dict[str, Dataset] = {}
    output_scalars: Dict[str, Scalar] = {}

    for name, result in semantic_results.items():
        if isinstance(result, Dataset):
            output_datasets[name] = result
        elif isinstance(result, Scalar):
            output_scalars[name] = result

    # 6. Create the SQL transpiler with all known datasets and scalars
    transpiler = SQLTranspiler(
        input_datasets=input_datasets,
        input_scalars=input_scalars,
        output_datasets=output_datasets,
        output_scalars=output_scalars,
    )

    # 7. Transpile AST to SQL queries
    queries = transpiler.transpile(ast)

    # 6. Format queries structures
    fomatted_queries = []
    output_structures = {**output_datasets, **output_scalars}
    for dependencies in dag.dependencies.values():
        name = (dependencies.get("persistent") or dependencies.get("outputs"))[0]
        inputs = dependencies.get("inputs", [])
        structure = output_structures[name]
        sql = next((query[1] for query in queries if query[0] == name))
        fomatted_queries.append(Query(name, sql, inputs, structure))

    return fomatted_queries


def run(
    script: Union[str, TransformationScheme, Path],
    data_structures: Union[Dict[str, Any], Path, List[Dict[str, Any]], List[Path]],
    datapoints: Union[Dict[str, Union[pd.DataFrame, str, Path]], List[Union[str, Path]], str, Path],
    value_domains: Optional[Union[Dict[str, Any], Path, List[Union[Dict[str, Any], Path]]]] = None,
    external_routines: Optional[
        Union[Dict[str, Any], Path, List[Union[Dict[str, Any], Path]]]
    ] = None,
    time_period_output_format: str = "vtl",
    return_only_persistent: bool = True,
    output_folder: Optional[Union[str, Path]] = None,
    scalar_values: Optional[Dict[str, Optional[Union[int, str, bool, float]]]] = None,
) -> Dict[str, Union[Dataset, Scalar]]:
    """
    Run a VTL script using DuckDB as the execution engine.

    Args:
        script: VTL script as string, TransformationScheme object, or Path.
        data_structures: Dict or Path with data structure definitions.
        datapoints: Dict mapping dataset names to DataFrames or CSV paths.
        value_domains: Optional value domains.
        external_routines: Optional external routines.
        time_period_output_format: Output format for time periods.
        return_only_persistent: If True, only return persistent assignments.
        output_folder: Optional folder to write output files.
        scalar_values: Optional dict of scalar values.
        output_format: Output format ("pandas" or "duckdb").

    Returns:
        Dict mapping result names to Dataset, Scalar, or DataFrame objects.
    """
    # 1. Transpile script to SQL queries (with scalar values inlined)
    queries = transpile(
        script=script,
        data_structures=data_structures,
        value_domains=value_domains,
        external_routines=external_routines,
        scalar_values=scalar_values,
    )

    # 2. DAG analysis for execution scheduling
    script = _check_script(script)
    vtl = load_vtl(script)
    ast = create_ast(vtl)
    dag = DAGAnalyzer().createDAG(ast)

    # 3. Create DuckDB connection
    conn = create_configured_connection()

    # 4. Execute queries in DuckDB
    results = execute_queries(
        conn=conn,
        queries=queries,
        ds_analysis=dag.dependencies,
        path_dict=None,
        dataframe_dict={},
        input_datasets={},
        output_datasets={},
        output_scalars={},
        output_folder=output_folder,
        return_only_persistent=return_only_persistent,
        insert_key="insertion",
        delete_key="deletion",
        global_key="global_inputs",
        persistent_key="persistent_outputs",
    )

    conn.close()
    return results
