from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pysdmx.model import TransformationScheme

from duckdb_transpiler.API._InternalApi import load_datasets_with_data_duckdb
from duckdb_transpiler.Transpiler import SQLTranspiler
from duckdb_transpiler.Utils import get_sql_type
from vtlengine.API import create_ast, semantic_analysis
from vtlengine.API._InternalApi import _check_script, load_datasets, load_vtl
from vtlengine.AST.DAG import DAGAnalyzer
from vtlengine.Model import Dataset, Scalar


class Query:
    def __init__(self, name: str, sql: str, inputs: List[str], structure: Union[Dataset, Scalar]):
        self.name = name
        self.sql = sql
        self.inputs = inputs
        self.structure = structure


def transpile(
    script: Union[str, TransformationScheme, Path],
    data_structures: Union[Dict[str, Any], Path, List[Dict[str, Any]], List[Path]],
    value_domains: Optional[Union[Dict[str, Any], Path, List[Union[Dict[str, Any], Path]]]] = None,
    external_routines: Optional[
        Union[Dict[str, Any], Path, List[Union[Dict[str, Any], Path]]]
    ] = None,
) -> List[Query]:
    """
    Transpile a VTL script to SQL queries.

    Args:
        script: VTL script as string, TransformationScheme object, or Path.
        data_structures: Dict or Path with data structure definitions.
        value_domains: Optional value domains.
        external_routines: Optional external routines.

    Returns:
        List of tuples: (result_name, sql_query, is_persistent)
        Each tuple represents one top-level assignment.
    """
    # 1. Parse script and create AST
    script = _check_script(script)
    vtl = load_vtl(script)
    ast = create_ast(vtl)
    dag = DAGAnalyzer().createDAG(ast)

    # 2. Load input datasets and scalars from data structures
    input_datasets, input_scalars = load_datasets(data_structures)

    # 3. Run semantic analysis to get output structures and validate script
    semantic_results = semantic_analysis(
        script=vtl,
        data_structures=data_structures,
        value_domains=value_domains,
        external_routines=external_routines,
    )

    # 4. Separate output datasets and scalars from semantic results
    output_datasets: Dict[str, Dataset] = {}
    output_scalars: Dict[str, Scalar] = {}

    for name, result in semantic_results.items():
        if isinstance(result, Dataset):
            output_datasets[name] = result
        elif isinstance(result, Scalar):
            output_scalars[name] = result

    # 5. Create the SQL transpiler
    transpiler = SQLTranspiler(
        input_datasets=input_datasets,
        output_datasets=output_datasets,
        input_scalars=input_scalars,
        output_scalars=output_scalars,
    )

    # 6. Transpile AST to SQL queries
    queries = transpiler.transpile(ast)

    # 7. Format queries
    formatted_queries = []
    output_structures = {**output_datasets, **output_scalars}
    for query, dependencies in zip(queries.items(), dag.dependencies.values()):
        name, sql = query
        structure = output_structures[name]
        inputs = dependencies.get("inputs", [])
        formatted_queries.append(Query(name, sql, inputs, structure))

    return formatted_queries


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
    # 1. Parse script and create AST
    queries = transpile(
        script=script,
        data_structures=data_structures,
        value_domains=value_domains,
        external_routines=external_routines,
    )

    # 2. Load data into DuckDB connection
    conn, input_datasets, input_scalars = load_datasets_with_data_duckdb(
        data_structures=data_structures,
        datapoints=datapoints,
        scalar_values=scalar_values,
    )

    # 3. Execute queries in DuckDB
    results = {}
    for query in queries:
        # TODO: Remove pandas dependency here (Used for fast debugging and testing)
        # Execute query
        result_df = conn.execute(query.sql).fetchdf()

        # Cast result columns to component types if needed
        structure = query.structure
        if isinstance(structure, Dataset):
            _types = {comp.name: get_sql_type(comp.type) for comp in structure.data_structure}
        else:
            _types = {structure.name: get_sql_type(structure.type)}
        result_df = result_df.astype(_types)

        # Register result for subsequent queries
        conn.register(query.name, result_df)

        # Store result if needed
        if not return_only_persistent or structure.persistent:
            structure.data = result_df.iloc[0, 0] if isinstance(structure, Scalar) else result_df
            results[query.name] = structure

    conn.close()
    return results
