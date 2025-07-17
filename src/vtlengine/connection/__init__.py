from .connection import ConnectionManager
from ..duckdb.custom_methods import load_custom_methods

con = ConnectionManager.get_connection()

load_custom_methods(con)
