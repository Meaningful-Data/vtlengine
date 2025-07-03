from duckdb.duckdb import DuckDBPyRelation


def duckdb_concat(left: DuckDBPyRelation, right: DuckDBPyRelation) -> DuckDBPyRelation:
    cols = set(left.columns) | set(right.columns)
    left = left.project("*, ROW_NUMBER() OVER () AS __row_id__").set_alias("base")
    right = right.project("*, ROW_NUMBER() OVER () AS __row_id__").set_alias("other")
    condition = "base.__row_id__ = other.__row_id__"
    return left.join(right, condition=condition, how="inner").project(", ".join(cols))