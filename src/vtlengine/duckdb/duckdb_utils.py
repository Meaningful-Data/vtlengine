from typing import Any, Dict, List, Optional, Set, Tuple, Union

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

INDEX_COL = "__index__"


def duckdb_concat(
    left: DuckDBPyRelation, right: DuckDBPyRelation, on: Optional[Union[str, List[str]]] = None
) -> DuckDBPyRelation:
    """
    Horizontal concatenation (axis=1) of two relations.
    - If `on` is provided, align rows using those key columns (OUTER JOIN).
    - If `on` is None and both have INDEX_COL, align by INDEX_COL.
    - Otherwise, align by position using a temporary row id (not exposed).
    - For shared columns, prefer right-hand values when the right row exists.
    """
    if left is None or right is None:
        return empty_relation()

    # Normalize key list
    if on is not None:
        on_cols = [on] if isinstance(on, str) else list(on)
    else:
        on_cols = [INDEX_COL] if (INDEX_COL in left.columns and INDEX_COL in right.columns) else None

    l = left.set_alias("l")
    r = right.set_alias("r")

    used_rowid = False
    if on_cols is None:
        # Positional alignment when no usable key is available
        l = l.project("row_number() OVER () - 1 AS __row_id__, *").set_alias("l")
        r = r.project("row_number() OVER () - 1 AS __row_id__, *").set_alias("r")
        join_cond = "l.__row_id__ = r.__row_id__"
        presence_col = "r.__row_id__"
        used_rowid = True
    else:
        join_cond = " AND ".join(f'l."{c}" = r."{c}"' for c in on_cols)
        presence_col = f'r."{on_cols[0]}"'

    # Use 'outer' (DuckDBPyRelation.join supports 'outer' but not 'full')
    joined = l.join(r, join_cond, how="outer")

    # Union of columns preserving order: left first, then right extras
    left_cols = list(left.columns)
    right_cols = list(right.columns)
    union_cols: List[str] = []
    seen: Set[str] = set()
    for c in left_cols + right_cols:
        if c not in seen:
            seen.add(c)
            union_cols.append(c)

    # Build projection; prefer right values when there is a matching right row
    select_exprs: List[str] = []
    for c in union_cols:
        if used_rowid and c == "__row_id__":
            continue  # hide helper column
        if c in left_cols and c in right_cols:
            select_exprs.append(f'CASE WHEN {presence_col} IS NOT NULL THEN r."{c}" ELSE l."{c}" END AS "{c}"')
        elif c in left_cols:
            select_exprs.append(f'l."{c}" AS "{c}"')
        else:
            select_exprs.append(f'r."{c}" AS "{c}"')

    # If index exists in either side but was not included (positional mode without index), keep a coalesced index
    if INDEX_COL in (set(left_cols) | set(right_cols)) and INDEX_COL not in union_cols:
        select_exprs.append(f'COALESCE(r."{INDEX_COL}", l."{INDEX_COL}") AS "{INDEX_COL}"')

    return joined.project(", ".join(select_exprs))


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
    cols: Optional[Union[str, List[str], Set[str]]] = None,
    types: Optional[Union[str, List[str], Set[str], Dict[str, str]]] = None,
    as_query: bool = False,
) -> DuckDBPyRelation:
    """
    Fills NaN values in specified columns of a DuckDB relation with a specified value.

    If no columns are specified, all columns will be filled.
    """

    exprs = []
    cols_set = set(data.columns) if cols is None else {cols} if isinstance(cols, str) else set(cols)
    for idx, col in enumerate(cols_set):
        col = col.replace('"', "")
        col_type = data.dtypes[col]
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

        cast_type = type_ if type_ else col_type
        exprs.append(f'COALESCE("{col}", CAST({value} AS {cast_type})) AS "{col}"')

    exprs.extend([f'"{c}"' for c in data.columns if c not in cols_set])
    query = ", ".join(exprs)
    return query if as_query else data.project(query)


def duckdb_merge(
    base_relation: Optional[DuckDBPyRelation],
    other_relation: Optional[DuckDBPyRelation],
    join_keys: Optional[List[str]],
    how: str = "inner",
) -> DuckDBPyRelation:
    """
    Merges two DuckDB relations using SQL syntax and temporary views.

    Supports: inner, left, full, and cross joins.
    """
    base_relation = base_relation if base_relation is not None else empty_relation()
    other_relation = other_relation if other_relation is not None else empty_relation()
    join_keys = join_keys if join_keys is not None else []

    from vtlengine.Utils.__Virtual_Assets import VirtualCounter

    base_name = VirtualCounter._new_temp_view_name()
    other_name = VirtualCounter._new_temp_view_name()
    con.register(base_name, base_relation)
    con.register(other_name, other_relation)

    if how == "cross":
        return con.sql(f"SELECT * FROM {base_name} CROSS JOIN {other_name}")
    elif how == "outer":
        how = "FULL OUTER"

    if not join_keys:
        raise ValueError("Join keys required for non-cross joins")

    using_clause = ", ".join(f'"{k}"' for k in join_keys)

    base_cols = set(base_relation.columns)
    other_cols = set(other_relation.columns)
    common_cols = (base_cols & other_cols) - set(join_keys)

    select_cols = [f'COALESCE({base_name}."{k}", {other_name}."{k}") AS "{k}"' for k in join_keys]

    for col in base_relation.columns:
        if col not in join_keys:
            suffix = "_x" if col in common_cols and how != "left" else ""
            select_cols.append(f'{base_name}."{col}" AS "{col}{suffix}"')

    for col in other_relation.columns:
        if col not in join_keys:
            suffix = "_y" if col in common_cols and how != "left" else ""
            select_cols.append(f'{other_name}."{col}" AS "{col}{suffix}"')

    query = f"""
        SELECT {", ".join(select_cols)}
        FROM {base_name}
        {how.upper()} JOIN {other_name}
        USING ({using_clause})
    """
    return con.sql(query)


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
    data = clean_execution_graph(data)
    cols = {cols} if isinstance(cols, str) else set(cols)
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


def normalize_data(
    self_data: DuckDBPyRelation, other_data: DuckDBPyRelation
) -> Tuple[DuckDBPyRelation, DuckDBPyRelation]:
    """
    Normalizes the data by launching a remove_null_str and round_doubles operations.
    """
    double_cols = set(get_cols_by_types(self_data, "DOUBLE"))
    if not len(double_cols):
        return self_data, other_data

    round_exprs = []
    base = empty_relation()
    for col in double_cols:
        base = duckdb_concat(
            base,
            duckdb_concat(
                self_data.project(f'"{col}" AS "self_{col}"'),
                other_data.project(f'"{col}" AS "other_{col}"'),
            ),
        )
        round_exprs.append(f'round_to_ref("self_{col}", "other_{col}") AS "self_{col}"')
        round_exprs.append(f'round_to_ref("other_{col}", "self_{col}") AS "other_{col}"')

    base = base.project(", ".join(round_exprs))
    self_data = duckdb_concat(
        self_data, base.project(", ".join(f"self_{col} AS {col}" for col in double_cols))
    )
    other_data = duckdb_concat(
        other_data, base.project(", ".join(f"other_{col} AS {col}" for col in double_cols))
    )
    return self_data, other_data


def null_counter(data: DuckDBPyRelation, name: str, as_query: bool = False) -> Any:
    query = f"COUNT(*) FILTER (WHERE {name} IS NULL) AS null_count"
    return query if as_query else data.aggregate(query).fetchone()[0]


def quote_cols(cols: Union[str, List[str], Set[str]]) -> Set[str]:
    """
    Quotes column names for use in SQL queries.
    """
    cols_set = set(cols)
    return {f'"{c}"' for c in cols_set}


def get_cols_by_types(rel: DuckDBPyRelation, types: Union[str, List[str], Set[str]]) -> Set[str]:
    cols = set()
    types = {types} if isinstance(types, str) else set(types)
    types = {t.upper() for t in types}

    for type_ in types:
        type_columns = set(
            [
                col
                for col, dtype in rel.dtypes.items()
                if isinstance(dtype, DuckDBPyType) and dtype in TYPES_DICT.get(type_, [])
            ]
        )
        cols.update(type_columns)

    return cols


def clean_execution_graph(rel: DuckDBPyRelation) -> DuckDBPyRelation:
    df = rel.df()
    return con.from_df(df)
