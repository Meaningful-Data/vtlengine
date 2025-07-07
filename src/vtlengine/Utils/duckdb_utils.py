from typing import Any, Dict, List, Optional, Union

from duckdb.duckdb import DuckDBPyRelation

from vtlengine.connection import con


def empty_relation() -> DuckDBPyRelation:
    """Returns an empty DuckDB relation with no columns."""
    return con.sql("SELECT 1 LIMIT 0")


# TODO: implement other merge types: left, outer...
def duckdb_merge(
    base_relation: Optional[DuckDBPyRelation],
    other_relation: Optional[DuckDBPyRelation],
    join_keys: list[str],
    how: str = "inner",
) -> DuckDBPyRelation:
    base_relation = base_relation or empty_relation()
    other_relation = other_relation or empty_relation()

    suffixes = ["_x", "_y"]
    base_cols = set(base_relation.columns)
    other_cols = set(other_relation.columns)
    common_cols = (base_cols & other_cols) - set(join_keys)

    base_proj_cols = []
    for c in base_relation.columns:
        if c in common_cols:
            base_proj_cols.append(f"{c} AS {c}{suffixes[0]}")
        else:
            base_proj_cols.append(c)
    base_relation = base_relation.project(", ".join(base_proj_cols))

    other_proj_cols = []
    for c in other_relation.columns:
        if c in common_cols:
            other_proj_cols.append(f"{c} AS {c}{suffixes[1]}")
        else:
            other_proj_cols.append(c)
    other_relation = other_relation.project(", ".join(other_proj_cols))

    base_alias = "base"
    other_alias = "other"
    base_relation = base_relation.set_alias(base_alias)
    other_relation = other_relation.set_alias(other_alias)

    join_condition = " AND ".join([f"{base_alias}.{k} = {other_alias}.{k}" for k in join_keys])
    joined = base_relation.join(
        other_relation,
        condition=join_condition,
        how=how,
    )

    keep_cols = []
    for c in joined.columns:
        if c in join_keys:
            c = f"{base_alias}.{c}"
        keep_cols.append(c)

    return joined.project(", ".join(set(keep_cols)))


def duckdb_concat(left: DuckDBPyRelation, right: DuckDBPyRelation) -> DuckDBPyRelation:
    if left is None or right is None:
        return empty_relation()

    cols = set(left.columns) | set(right.columns)
    common_cols = set(left.columns).intersection(set(right.columns))
    cols_left = "*"
    cols_right = "*"
    if common_cols:
        if len(left.columns) > len(right.columns):
            cols_left += f" EXCLUDE ({', '.join(common_cols)})"
        else:
            cols_right += f" EXCLUDE ({', '.join(common_cols)})"

    left = left.project(f"{cols_left}, ROW_NUMBER() OVER () AS __row_id__").set_alias("base")
    right = right.project(f"{cols_right}, ROW_NUMBER() OVER () AS __row_id__").set_alias("other")
    condition = "base.__row_id__ = other.__row_id__"
    return left.join(right, condition=condition, how="inner").project(", ".join(cols))


def duckdb_rename(data: DuckDBPyRelation, name_dict: Dict[str, str]) -> DuckDBPyRelation:
    """Renames a column in a DuckDB relation."""
    cols = set(data.columns)
    for old_name, new_name in name_dict.items():
        if old_name not in cols:
            raise ValueError(f"Column '{old_name}' not found in relation.")
        cols.remove(old_name)
        cols.add(f'{old_name} AS "{new_name}"')
    return data.project(", ".join(cols))


def duckdb_fill(data: DuckDBPyRelation, col_name: str, value: Any) -> DuckDBPyRelation:
    """
    Fills a column in a DuckDB relation with a specified value.

    If the column does not exist, it will be created.
    """
    if col_name not in data.columns:
        data = data.project(f"*, {value} AS {col_name}")
    else:
        data = data.project(f"{col_name} COALESCE({value}) AS {col_name}")
    return data


def duckdb_drop(data: DuckDBPyRelation, cols_to_drop: Union[str, List[str]]) -> DuckDBPyRelation:
    """
    Drops a column from a DuckDB relation.

    If the column does not exist, it will be ignored.
    """
    cols = set(data.columns) - set(cols_to_drop)
    if not cols:
        return empty_relation()
    return data.project(", ".join(cols))


def duckdb_select(data: DuckDBPyRelation, cols: Union[str, List[str]]) -> DuckDBPyRelation:
    """
    Selects specific columns from a DuckDB relation.

    If the column does not exist, it will be ignored.
    """
    cols = set(cols)
    if not cols:
        return empty_relation()
    return data.project(", ".join(cols))
