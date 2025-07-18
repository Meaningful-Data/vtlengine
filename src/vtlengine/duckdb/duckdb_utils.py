from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd
from duckdb import duckdb
from duckdb.duckdb import DuckDBPyRelation  # type: ignore[import-untyped]
from duckdb.duckdb.typing import DuckDBPyType  # type: ignore[import-untyped]

from vtlengine.connection import con
from vtlengine.DataTypes.TimeHandling import PERIOD_IND_MAPPING, PERIOD_IND_MAPPING_REVERSE

TYPES_DICT = {
    "STRING": [duckdb.type("VARCHAR"), duckdb.type("STRING")],
    "INTEGER": [duckdb.type("INTEGER"), duckdb.type("BIGINT")],
    "DOUBLE": [duckdb.type("DOUBLE"), duckdb.type("FLOAT"), duckdb.type("REAL")],
    "NUMBER": [duckdb.type("DOUBLE"), duckdb.type("FLOAT"), duckdb.type("REAL")],
    "BOOLEAN": [duckdb.type("BOOLEAN")],
    "DATE": [duckdb.type("DATE")],
}


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
    if common_cols:
        cols_left += f" EXCLUDE ({', '.join(quote_cols(common_cols))})"

    left = left.project(f"{cols_left}, ROW_NUMBER() OVER () AS __row_id__").set_alias("base")
    right = right.project("*, ROW_NUMBER() OVER () AS __row_id__").set_alias("other")

    condition = "base.__row_id__ = other.__row_id__"
    return left.join(right, condition=condition, how="inner").project(", ".join(quote_cols(cols)))


def duckdb_drop(
    data: DuckDBPyRelation, cols_to_drop: Union[str, List[str]], as_query: bool = False
) -> DuckDBPyRelation:
    """
    Drops (remove) a column from a DuckDB relation.

    If no columns are specified, returns an empty relation.

    Its behavior is similar to pandas dataframe drop method.
    """
    cols = set(data.columns) - set(cols_to_drop)
    if not cols:
        return empty_relation(as_query=as_query)
    query = ", ".join(quote_cols(cols))
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
    types: Optional[Union[str, List[str], Dict[str, str]]] = None,
    as_query: bool = False,
) -> DuckDBPyRelation:
    """
    Fills NaN values in specified columns of a DuckDB relation with a specified value.

    If no columns are specified, all columns will be filled.
    """

    exprs = ["*"]
    cols_set: Set[str] = set(cols) if cols is not None else data.columns
    for idx, col in enumerate(cols_set):
        type_ = (
            (
                types
                if isinstance(types, str)
                else types[0]
                if isinstance(types, list) and len(types) == 1
                else types[idx]
                if isinstance(types, list) and len(types) == len(cols_set)
                else types.get(col)
                if isinstance(types, dict)
                else None
            )
            if types
            else None
        )

        if isinstance(types, list) and len(types) not in (1, len(cols_set)):
            raise ValueError("Length of types must match length of columns.")

        value = f"CAST({value} AS {type_})" if type_ else value
        exprs.append(f'COALESCE("{col}", {value}) AS "{col}"')

    query = ", ".join(exprs)
    return query if as_query else data.project(query)


def duckdb_merge(
    base_relation: Optional[DuckDBPyRelation],
    other_relation: Optional[DuckDBPyRelation],
    join_keys: Optional[List[str]],
    how: str = "inner",
) -> DuckDBPyRelation:
    """
    Merges two DuckDB relations on specified join keys and mode.

    Supports: inner, left, right, full (outer) and cross joins.
    """
    base_relation = base_relation if base_relation is not None else empty_relation()
    other_relation = other_relation if other_relation is not None else empty_relation()
    join_keys = join_keys if join_keys is not None else []

    suffixes = ["_x", "_y"]
    base_cols = set(base_relation.columns)
    other_cols = set(other_relation.columns)
    common_cols = (base_cols & other_cols) - set(join_keys)

    base_proj_cols = []
    for c in base_relation.columns:
        if c in common_cols:
            base_proj_cols.append(f'"{c}" AS "{c}{suffixes[0]}"')
        else:
            base_proj_cols.append(f'"{c}"')
    base_relation = base_relation.project(", ".join(base_proj_cols))

    other_proj_cols = []
    for c in other_relation.columns:
        if c in common_cols:
            other_proj_cols.append(f'"{c}" AS "{c}{suffixes[1]}"')
        else:
            other_proj_cols.append(f'"{c}"')
    other_relation = other_relation.project(", ".join(other_proj_cols))

    if how == "cross":
        return base_relation.cross(other_relation)

    base_alias = "base"
    other_alias = "other"
    base_relation = base_relation.set_alias(base_alias)
    other_relation = other_relation.set_alias(other_alias)

    join_condition = " AND ".join([f'{base_alias}."{k}" = {other_alias}."{k}"' for k in join_keys])

    joined = base_relation.join(
        other_relation,
        condition=join_condition,
        how=how,
    )

    coalesced_cols = [
        f'COALESCE({base_alias}."{k}", {other_alias}."{k}") AS "{k}"' for k in join_keys
    ]

    other_proj = []
    for c in base_relation.columns:
        if c not in join_keys:
            other_proj.append(f'{base_alias}."{c}"')
    for c in other_relation.columns:
        if c not in join_keys:
            other_proj.append(f'{other_alias}."{c}"')

    final_cols = coalesced_cols + other_proj

    return joined.project(", ".join(final_cols))


def duckdb_rename(
    data: DuckDBPyRelation, name_dict: Dict[str, str], as_query: bool = False
) -> DuckDBPyRelation:
    """Renames columns in a DuckDB relation."""
    cols_set = set()
    cols = set(data.columns)
    for old_name, new_name in name_dict.items():
        if old_name not in cols:
            raise ValueError(f"Column '{old_name}' not found in relation.")
        cols.remove(old_name)
        cols_set.add(f'"{old_name}" AS "{new_name}"')
    query = ", ".join(quote_cols(cols) | cols_set)
    return query if as_query else data.project(query)


def duckdb_select(
    data: DuckDBPyRelation, cols: Union[str, List[str], Any] = "*", as_query: bool = False
) -> DuckDBPyRelation:
    """
    Selects specific columns from a DuckDB relation.

    If no columns are specified, returns an empty relation.

    If `as_query` is True, returns the SQL query string instead of the relation.
    """
    cols = set(cols)
    query = ", ".join(quote_cols(cols))
    return query if as_query else data.project(query)


def duration_handler(col: str, reverse: bool = False) -> str:
    """
    Returns a CASE expression to handle duration columns in DuckDB.

    It allows for converting between string representations and integer indices.
    """

    expr = "CASE"
    mapping = PERIOD_IND_MAPPING_REVERSE if reverse else PERIOD_IND_MAPPING
    type_to_cast = "VARCHAR" if reverse else "INTEGER"
    for k, v in mapping.items():
        k, v = (k, f"'{v}'") if reverse else (f"'{k}'", v)
        expr += f" WHEN {col} = {k} THEN {v}"
    expr += f" ELSE CAST({col} AS {type_to_cast}) END"
    return expr


def empty_relation(
    cols: Optional[Union[str, List[str]]] = None, as_query: bool = False
) -> DuckDBPyRelation:
    """
    Returns an empty DuckDB relation.

    If `cols` is provided, it will create an empty relation with those columns.
    """
    if cols:
        df = pd.DataFrame(columns=list(cols) if isinstance(cols, (list, set)) else [cols])
        return con.from_df(pd.DataFrame(df))
    query = "SELECT 1 LIMIT 0"
    return query if as_query else con.sql(query)


def get_col_type(rel: DuckDBPyRelation, col_name: str) -> DuckDBPyType:
    """
    Returns the specified column type from the DuckDB relation.
    """
    return rel.types[rel.columns.index(col_name)]


def normalize_data(data: DuckDBPyRelation, as_query: bool = False) -> DuckDBPyRelation:
    """
    Normalizes the data by launching a remove_null_str and round_doubles operations.
    """
    query_set = {
        f'"{c}"' for c in data.columns if c not in get_cols_by_types(data, ["DOUBLE", "STRING"])
    }
    query_set.add(remove_null_str(data, as_query=True))
    query_set.add(round_doubles(data, as_query=True))
    query = ", ".join(query_set).replace("*, ", "").replace(", *", "")
    return query if as_query else data.project(query)


def null_counter(data: DuckDBPyRelation, name: str, as_query: bool = False) -> Any:
    query = f"COUNT(*) FILTER (WHERE {name} IS NULL) AS null_count"
    return query if as_query else data.aggregate(query).fetchone()[0]


def quote_cols(cols: Union[str, List[str], Set[str]]) -> Set[str]:
    """
    Quotes column names for use in SQL queries.
    """
    cols_set = set(cols)
    return {f'"{c}"' for c in cols_set}


def remove_null_str(
    data: DuckDBPyRelation, cols: Optional[Union[str, List[str]]] = None, as_query: bool = False
) -> DuckDBPyRelation:
    """
    Removes rows where specified columns contain null or empty string values.

    If no columns are specified, it checks all str columns.
    """
    str_columns = get_cols_by_types(data, "STRING")
    if not str_columns:
        return data if not as_query else "*"
    query = duckdb_fillna(data, "''", str_columns, as_query=True)
    return query if as_query else data.project(query)


def round_doubles(
    data: DuckDBPyRelation, num_dec: int = 6, as_query: bool = False
) -> DuckDBPyRelation:
    """
    Rounds double values in the dataset to avoid precision issues.
    """
    exprs = ["*"]
    double_columns = get_cols_by_types(data, "DOUBLE")
    for col in double_columns:
        exprs.append(f'ROUND("{col}", {num_dec}) AS "{col}"')
    query = ", ".join(exprs)
    return query if as_query else data.project(query)


def get_cols_by_types(rel: DuckDBPyRelation, types: Union[str, List[str]]) -> Set[str]:
    cols = set()
    types = {types} if isinstance(types, str) else set(types)
    types = {t.upper() for t in types}

    for type_ in types:
        type_columns = set(
            [
                col
                for col, dtype in zip(rel.columns, rel.dtypes)
                if isinstance(dtype, DuckDBPyType) and dtype in TYPES_DICT.get(type_, [])
            ]
        )
        cols.update(type_columns)

    return cols


def clean_execution_graph(rel: DuckDBPyRelation):
    df = rel.df()
    return con.from_df(df)
