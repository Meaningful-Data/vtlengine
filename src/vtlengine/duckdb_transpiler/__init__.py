"""DuckDB transpiler for VTL scripts."""

from typing import Any, Dict, List, Optional, Tuple

from vtlengine.duckdb_transpiler.Transpiler import SQLTranspiler

__all__ = ["SQLTranspiler", "transpile"]


def transpile(
    vtl_script: str,
    data_structures: Optional[Dict[str, Any]] = None,
    value_domains: Any = None,
    external_routines: Any = None,
) -> List[Tuple[str, str, bool]]:
    """
    Transpile a VTL script to a list of (name, SQL, is_persistent) tuples.

    This is a convenience function that runs the full pipeline:
    1. Parses the VTL script into an AST
    2. Runs semantic analysis to determine output structures
    3. Transpiles the AST to SQL queries

    Args:
        vtl_script: The VTL script to transpile.
        data_structures: Input dataset structures (raw dict format as used by the API).
        value_domains: Value domain definitions.
        external_routines: External routine definitions.

    Returns:
        List of (dataset_name, sql_query, is_persistent) tuples.
    """
    from vtlengine.API import create_ast, load_datasets, load_external_routines, load_value_domains
    from vtlengine.AST.DAG import DAGAnalyzer
    from vtlengine.Interpreter import InterpreterAnalyzer
    from vtlengine.Model import Dataset, Scalar

    if data_structures is None:
        data_structures = {}

    # Parse VTL to AST
    ast = create_ast(vtl_script)
    dag = DAGAnalyzer.create_dag(ast)

    # Load datasets structure (without data) from raw dict format
    input_datasets, input_scalars = load_datasets(data_structures)

    # Load value domains and external routines
    loaded_vds = load_value_domains(value_domains) if value_domains else None
    loaded_routines = load_external_routines(external_routines) if external_routines else None

    # Run semantic analysis to get output structures
    interpreter = InterpreterAnalyzer(
        datasets=input_datasets,
        value_domains=loaded_vds,
        external_routines=loaded_routines,
        scalars=input_scalars,
        only_semantic=True,
        return_only_persistent=False,
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

    # Create transpiler and generate SQL
    transpiler = SQLTranspiler(
        input_datasets=input_datasets,
        output_datasets=output_datasets,
        input_scalars=input_scalars,
        output_scalars=output_scalars,
        value_domains=loaded_vds or {},
        external_routines=loaded_routines or {},
        dag=dag,
    )

    return transpiler.transpile(ast)
