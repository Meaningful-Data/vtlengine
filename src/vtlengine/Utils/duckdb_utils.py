from duckdb.duckdb import DuckDBPyRelation


def duckdb_concat(left: DuckDBPyRelation, right: DuckDBPyRelation) -> DuckDBPyRelation:
    cols = set(left.columns) | set(right.columns)
    common_cols = set(left.columns).intersection(set(right.columns))
    cols_left = "*"
    cols_right = "*"
    if common_cols:
        if len(left.columns) > len(right.columns):
            cols_left += f" EXCLUDE ({', '.join(common_cols)})"
        else:
            cols_right += f" EXCLUDE ({', '.join(common_cols)})"

    print(left)
    print(right)

    left = left.project(f"{cols_left}, ROW_NUMBER() OVER () AS __row_id__").set_alias("base")
    right = right.project(f"{cols_right}, ROW_NUMBER() OVER () AS __row_id__").set_alias("other")
    condition = "base.__row_id__ = other.__row_id__"
    return left.join(right, condition=condition, how="inner").project(", ".join(cols))