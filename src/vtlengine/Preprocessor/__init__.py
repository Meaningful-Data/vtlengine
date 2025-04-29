import os

DUCKDB_TOKEN = "duckdb"

BACKENDS = {
    # eager execution
    "pandas": "pd",
    "pd": "pd",
    "eager": "pd",
    # lazy execution
    "duckdb": DUCKDB_TOKEN,
    "db": DUCKDB_TOKEN,
    "lazy": DUCKDB_TOKEN,
    "streaming": DUCKDB_TOKEN,
}

backend_df = BACKENDS.get(os.getenv("BACKEND_DF", "").lower(), "pd")

con = None
sql_promotion = None
if backend_df == DUCKDB_TOKEN:
    from .utils import con as duckdb_con
    from .utils import sql_column_type_promotion as sql_promotion
    con = duckdb_con

__all__ = ["con", "sql_promotion"]