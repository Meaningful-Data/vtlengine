from typing import Any, Dict, List, Optional, Union

import pandas as pd
from duckdb import duckdb
from duckdb.duckdb import DuckDBPyRelation
from duckdb.duckdb.typing import DuckDBPyType

from vtlengine.connection import con


def duckdb_concat(left: DuckDBPyRelation, right: DuckDBPyRelation) -> DuckDBPyRelation:
    """
    Concatenates two DuckDB relations by row, ensuring that columns are aligned.

    If either relation is None, returns an empty relation.

    Its behavior is similar to pandas dataframe-series assignment.
    """

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


def duckdb_drop(
    data: DuckDBPyRelation, cols_to_drop: Union[str, List[str]], as_query: bool = False
) -> DuckDBPyRelation:
    """
    Drops a column from a DuckDB relation.

    If no columns are specified, returns an empty relation.
    """
    cols = set(data.columns) - set(cols_to_drop)
    if not cols:
        return empty_relation(as_query=as_query)
    query = ", ".join(cols)
    return query if as_query else data.project(query)


def duckdb_fill(
    data: DuckDBPyRelation, value: Any, col_name: str, as_query: bool = False
) -> DuckDBPyRelation:
    """
    Fills a column in a DuckDB relation with a specified value.

    If the column does not exist, it will be created.
    """
    if col_name not in data.columns:
        query = f'*, {value} AS "{col_name}"'
    else:
        query = f'{col_name} COALESCE({value}) AS "{col_name}"'
    return query if as_query else data.project(query)


def duckdb_fillna(
    data: DuckDBPyRelation,
    value: Any,
    cols: Optional[Union[str, List[str]]] = None,
    as_query: bool = False,
) -> DuckDBPyRelation:
    """
    Fills NaN values in specified columns of a DuckDB relation with a specified value.

    If no columns are specified, all columns will be filled.
    """
    cols = set(cols) if cols else data.columns
    fill_exprs = [f'COALESCE({col}, {value}) AS "{col}"' for col in cols]
    query = ", ".join(fill_exprs) if fill_exprs else ""

    if as_query:
        return query
    return data.project(", ".join(fill_exprs)) if query else data


# TODO: implement other merge types: left, outer...
def duckdb_merge(
    base_relation: Optional[DuckDBPyRelation],
    other_relation: Optional[DuckDBPyRelation],
    join_keys: list[str],
    how: str = "inner",
) -> DuckDBPyRelation:
    """
    Merges two DuckDB relations on specified join keys and mode.
    """
    base_relation = base_relation if base_relation is not None else empty_relation()
    other_relation = other_relation if other_relation is not None else empty_relation()

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


def duckdb_rename(
    data: DuckDBPyRelation, name_dict: Dict[str, str], as_query: bool = False
) -> DuckDBPyRelation:
    """Renames a column in a DuckDB relation."""
    cols = set(data.columns)
    for old_name, new_name in name_dict.items():
        if old_name not in cols:
            raise ValueError(f"Column '{old_name}' not found in relation.")
        cols.remove(old_name)
        cols.add(f'{old_name} AS "{new_name}"')
    query = ", ".join(cols)
    return query if as_query else data.project(", ".join(cols))


def duckdb_select(
    data: DuckDBPyRelation, cols: Union[str, List[str]] = "*", as_query: bool = False
) -> DuckDBPyRelation:
    """
    Selects specific columns from a DuckDB relation.

    If no columns are specified, returns an empty relation.

    If `as_query` is True, returns the SQL query string instead of the relation.
    """
    cols = set(cols)
    query = ", ".join(cols)
    return query if as_query else data.project(query)


def empty_relation(
    cols: Optional[Union[str, List[str]]] = None, as_query: bool = False
) -> DuckDBPyRelation:
    """
    Returns an empty DuckDB relation.

    If `cols` is provided, it will create an empty relation with those columns.
    """
    if cols:
        return con.from_df(pd.DataFrame(columns=list(cols)))
    query = "SELECT 1 LIMIT 0"
    return query if as_query else con.sql(query)


def normalize_data(data: DuckDBPyRelation, as_query: bool = False) -> DuckDBPyRelation:
    """
    Normalizes the data by launching a remove_null_str and round_doubles operations.
    """
    if as_query:
        return remove_null_str(data, as_query=True) + f", {round_doubles(data, as_query=True)}"
    return remove_null_str(round_doubles(data))


def remove_null_str(data: DuckDBPyRelation, cols: Optional[Union[str, List[str]]] = None, as_query: bool = False) -> DuckDBPyRelation:
    """
    Removes rows where specified columns contain null or empty string values.

    If no columns are specified, it checks all str columns.
    """
    cols = data.columns if cols is None else set(cols)
    str_columns = [
        col for col, dtype in zip(data.columns, data.dtypes)
        if col in cols and isinstance(dtype, DuckDBPyType)
           and dtype in [duckdb.type("VARCHAR"), duckdb.type("STRING")]
    ]
    return duckdb_fillna(data, "''", str_columns, as_query=as_query) if str_columns else data


def round_doubles(data: DuckDBPyRelation, num_dec: int = 6, as_query: bool = False) -> DuckDBPyRelation:
    """
    Rounds double values in the dataset to avoid precision issues.
    """
    exprs = []
    double_columns = [
        col
        for col, dtype in zip(data.columns, data.dtypes)
        if isinstance(dtype, DuckDBPyType)
        and dtype in [duckdb.type("DOUBLE"), duckdb.type("FLOAT"), duckdb.type("REAL")]
    ]
    for col in data.columns:
        if col in double_columns:
            exprs.append(f'ROUND({col}, {num_dec}) AS "{col}"')
        else:
            exprs.append(f'"{col}"')

    query = ", ".join(exprs)
    return query if as_query else data.project(query)
