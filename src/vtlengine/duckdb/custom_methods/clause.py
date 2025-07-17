from typing import Any
from duckdb.duckdb import DuckDBPyConnection
from vtlengine.AST.Grammar.tokens import ISNULL


def load_clause_methods(con: DuckDBPyConnection) -> None:
    con.create_function(ISNULL, _isnull)


def _isnull(x: Any) -> bool:
    return x is None

