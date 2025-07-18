from duckdb.duckdb import DuckDBPyConnection  # type: ignore[import-untyped]

from vtlengine.duckdb.custom_methods.clause import load_clause_methods


def load_custom_methods(con: DuckDBPyConnection) -> None:
    load_clause_methods(con)
