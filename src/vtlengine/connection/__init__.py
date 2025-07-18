from ..duckdb.custom_methods import load_custom_methods
from .connection import ConnectionManager

con = ConnectionManager.get_connection()

load_custom_methods(con)
