"""
DuckDB Transpiler for VTL.

This module provides SQL transpilation capabilities for VTL scripts,
converting VTL AST to DuckDB-compatible SQL queries.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pysdmx.model import TransformationScheme
from pysdmx.model.dataflow import Dataflow, DataStructureDefinition, Schema

from vtlengine.API import create_ast, semantic_analysis
from vtlengine.API._InternalApi import (
    _check_script,
    load_datasets,
    load_external_routines,
    load_value_domains,
    load_vtl,
)
from vtlengine.duckdb_transpiler.Transpiler import SQLTranspiler
from vtlengine.Model import Dataset, Scalar

__all__ = ["SQLTranspiler", "transpile"]


def transpile(
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

    # 5. Load value domains and external routines
    loaded_vds = load_value_domains(value_domains) if value_domains else {}
    loaded_routines = load_external_routines(external_routines) if external_routines else {}

    # 6. Create the SQL transpiler with:
    #    - input_datasets: Tables available for querying (inputs)
    #    - output_datasets: Expected output structures (for validation)
    #    - scalars: Both input and output scalars
    #    - value_domains: Loaded value domains
    #    - external_routines: Loaded external routines
    transpiler = SQLTranspiler(
        input_datasets=input_datasets,
        output_datasets=output_datasets,
        input_scalars=input_scalars,
        output_scalars=output_scalars,
        value_domains=loaded_vds,
        external_routines=loaded_routines,
    )

    # 7. Transpile AST to SQL queries
    queries = transpiler.transpile(ast)

    return queries
