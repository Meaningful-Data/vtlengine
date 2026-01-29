from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pysdmx.model import TransformationScheme

from duckdb_transpiler.API._InternalApi import load_datasets_with_data_duckdb
from duckdb_transpiler.Transpiler import SQLTranspiler
from duckdb_transpiler.Utils import get_pandas_type
from vtlengine.API import create_ast, semantic_analysis
from vtlengine.API._InternalApi import _check_script, load_datasets, load_vtl
from vtlengine.AST.DAG import DAGAnalyzer
from vtlengine.Model import Dataset, Scalar

OUTPUT_DTYPES = Union[Dataset, Scalar]


class Query:
    def __init__(self, name: str, inputs: List[str], structure: OUTPUT_DTYPES, sql: str = ""):
        self.name = name
        self.inputs = inputs
        self.structure = structure
        self.sql = sql


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

    # 6. Format queries structures
    queries = []
    output_structures = {**output_datasets, **output_scalars}
    for dependencies in dag.dependencies.values():
        name = (dependencies.get("persistent") or dependencies.get("outputs"))[0]
        inputs = dependencies.get("inputs", [])
        structure = output_structures[name]
        queries.append(Query(name, inputs, structure))

    # 7. Create the SQL transpiler with all known datasets and scalars
    all_datasets = {**input_datasets, **output_datasets}
    all_scalars = {**input_scalars, **output_scalars}

    transpiler = SQLTranspiler(
        datasets=all_datasets,
        scalars=all_scalars,
    )

    # 8. Transpile AST to SQL queries
    queries = transpiler.transpile(ast, queries)

    return queries


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
) -> Dict[str, OUTPUT_DTYPES]:
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

    # 2. Load data into DuckDB connection
    conn, input_datasets, input_scalars = load_datasets_with_data_duckdb(
        data_structures=data_structures,
        datapoints=datapoints,
        scalar_values=scalar_values,
    )

    # 3. Execute queries in DuckDB
    results = {}
    for query in queries:
        # Execute query
        result_df = conn.execute(query.sql).fetchdf()

        # Cast result columns to component types if needed
        structure = query.structure
        if isinstance(structure, Dataset):
            _types = {
                comp.name: get_pandas_type(comp.data_type) for comp in structure.get_components()
            }
            result_df = result_df.astype(_types)

        # Register result for subsequent queries
        conn.register(query.name, result_df)

        # Store result
        if isinstance(structure, Scalar):
            # Convert numpy types to Python native types for Scalar validation
            raw_value = result_df.iloc[0, 0]
            structure.value = raw_value.item() if hasattr(raw_value, "item") else raw_value
        else:
            structure.data = result_df

        # Add to results if persistent or return_only_persistent is False
        if not return_only_persistent or structure.persistent:
            results[query.name] = structure

    conn.close()
    return results
