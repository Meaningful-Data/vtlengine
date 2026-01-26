from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import duckdb
import pandas as pd
from pysdmx.model import TransformationScheme

from duckdb_transpiler.API._InternalApi import load_datasets_with_data_duckdb
from duckdb_transpiler.Transpiler import SQLTranspiler
from vtlengine.API import create_ast, semantic_analysis
from vtlengine.API._InternalApi import _check_script, load_datasets, load_vtl
from vtlengine.Model import Dataset, Scalar


def transpile(
    script: Union[str, TransformationScheme, Path],
    data_structures: Union[Dict[str, Any], Path, List[Dict[str, Any]], List[Path]],
    value_domains: Optional[Union[Dict[str, Any], Path, List[Union[Dict[str, Any], Path]]]] = None,
    external_routines: Optional[
        Union[Dict[str, Any], Path, List[Union[Dict[str, Any], Path]]]
    ] = None,
) -> List[Tuple[str, str, bool]]:
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

    # 5. Create the SQL transpiler with:
    #    - input_datasets: Tables available for querying (inputs)
    #    - output_datasets: Expected output structures (for validation)
    #    - scalars: Both input and output scalars
    transpiler = SQLTranspiler(
        input_datasets=input_datasets,
        output_datasets=output_datasets,
        input_scalars=input_scalars,
        output_scalars=output_scalars,
    )

    # 6. Transpile AST to SQL queries
    queries = transpiler.transpile(ast)

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
    output_format: str = "pandas",
) -> Dict[str, Union[Dataset, Scalar, pd.DataFrame, duckdb.DuckDBPyRelation]]:
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
    script_checked = _check_script(script)
    vtl = load_vtl(script_checked)
    ast = create_ast(vtl)

    # 2. Load data into DuckDB connection
    conn, input_datasets, input_scalars = load_datasets_with_data_duckdb(
        data_structures=data_structures,
        datapoints=datapoints,
        scalar_values=scalar_values,
    )

    # 3. Run semantic analysis to get output structures
    semantic_results = semantic_analysis(
        script=vtl,
        data_structures=data_structures,
        value_domains=value_domains,
        external_routines=external_routines,
    )

    # 4. Separate output datasets and scalars
    output_datasets: Dict[str, Dataset] = {}
    output_scalars: Dict[str, Scalar] = {}

    for name, result in semantic_results.items():
        if isinstance(result, Dataset):
            output_datasets[name] = result
        elif isinstance(result, Scalar):
            output_scalars[name] = result

    # 5. Create transpiler and generate SQL queries
    transpiler = SQLTranspiler(
        input_datasets=input_datasets,
        output_datasets=output_datasets,
        input_scalars=input_scalars,
        output_scalars=output_scalars,
    )
    queries = transpiler.transpile(ast)

    # 6. Execute queries in DuckDB
    results: Dict[str, Union[Dataset, Scalar, pd.DataFrame, duckdb.DuckDBPyRelation]] = {}

    for result_name, sql_query, is_persistent in queries:
        # Skip non-persistent results if requested
        if return_only_persistent and not is_persistent:
            # Still execute to register intermediate results
            result_df = conn.execute(sql_query).fetchdf()
            conn.register(result_name, result_df)
            continue

        # Execute query
        result_df = conn.execute(sql_query).fetchdf()

        # Register result for subsequent queries
        conn.register(result_name, result_df)

        # Check if this is a scalar result (single value, no identifiers)
        if result_name in output_scalars:
            # Extract scalar value
            if len(result_df) == 1 and len(result_df.columns) == 1:
                scalar = output_scalars[result_name]
                scalar.value = result_df.iloc[0, 0]
                results[result_name] = scalar
            else:
                results[result_name] = result_df
        else:
            # Dataset result
            if output_format == "pandas":
                results[result_name] = result_df
            else:
                # Return as DuckDB relation
                results[result_name] = conn.from_df(result_df)

    # Close connection if returning pandas
    if output_format == "pandas":
        conn.close()

    return results
