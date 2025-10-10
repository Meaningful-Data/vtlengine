from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
from duckdb import duckdb
from duckdb.duckdb import DuckDBPyRelation  # type: ignore[import-untyped]
from duckdb.duckdb.typing import DuckDBPyType  # type: ignore[import-untyped]

from vtlengine.connection import con
from vtlengine.DataTypes.TimeHandling import PERIOD_IND_MAPPING, PERIOD_IND_MAPPING_REVERSE
from vtlengine.Model.relation_proxy import RelationProxy

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

    if on is not None:
        on_cols = [on] if isinstance(on, str) else list(on)
    else:
        on_cols = (
            [INDEX_COL] if (INDEX_COL in left.columns and INDEX_COL in right.columns) else None
        )

    left_rel = left.set_alias("l")
    right_rel = right.set_alias("r")

    if not on_cols and INDEX_COL in left.columns and INDEX_COL in right.columns:
        on_cols = [INDEX_COL]

    used_rowid = False
    if on_cols is None:
        # Positional alignment when no usable key is available
        left_rel = left_rel.project("row_number() OVER () - 1 AS __row_id__, *").set_alias("l")
        right_rel = right_rel.project("row_number() OVER () - 1 AS __row_id__, *").set_alias("r")
        join_cond = "l.__row_id__ = r.__row_id__"
        presence_col = "r.__row_id__"
        used_rowid = True
    else:
        join_cond = " AND ".join(f'l."{c}" = r."{c}"' for c in on_cols)
        presence_col = f'r."{on_cols[0]}"'

    joined = left_rel.join(right_rel, join_cond, how="outer")

    left_cols = list(left.columns)
    right_cols = list(right.columns)
    union_cols: List[str] = []
    seen: Set[str] = set()
    for c in left_cols + right_cols:
        if c not in seen:
            seen.add(c)
            union_cols.append(c)

    select_exprs: List[str] = []
    for c in union_cols:
        if used_rowid and c == "__row_id__":
            continue
        if c in left_cols and c in right_cols:
            select_exprs.append(
                f'CASE WHEN {presence_col} IS NOT NULL THEN r."{c}" ELSE l."{c}" END AS "{c}"'
            )
        elif c in left_cols:
            select_exprs.append(f'l."{c}" AS "{c}"')
        else:
            select_exprs.append(f'r."{c}" AS "{c}"')

    # If index exists in either side but was not included (positional mode without index),
    # keep a coalesced index
    if INDEX_COL in (set(left_cols) | set(right_cols)) and INDEX_COL not in union_cols:
        select_exprs.append(f'COALESCE(r."{INDEX_COL}", l."{INDEX_COL}") AS "{INDEX_COL}"')

    return RelationProxy(joined.project(", ".join(select_exprs)))


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
    return query if as_query else RelationProxy(data.project(query))


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
    return query if as_query else RelationProxy(data.project(query))


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
        col_type = get_col_type(data, col)
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
        # problematic default value
        if value == "default":
            value = "default"
        if value == "":
            value = "''"
        exprs.append(f'COALESCE("{col}", CAST({value} AS {cast_type})) AS "{col}"')

    exprs.extend([f'"{c}"' for c in data.columns if c not in cols_set])
    query = ", ".join(exprs)
    return query if as_query else RelationProxy(data.project(query))


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
        query = f"SELECT {base_name}.*, {other_name}.* FROM {base_name} CROSS JOIN {other_name} ORDER BY {base_name}.\"{INDEX_COL}\""
        return RelationProxy(con.sql(query))

    join_keyword = "FULL OUTER" if how.lower() == "outer" else how.upper()
    if join_keyword not in ("INNER", "LEFT", "RIGHT", "FULL OUTER"):
        raise ValueError(f"Unsupported join type: {how}")

    if not join_keys:
        raise ValueError("Join keys required for non-cross joins")

    using_clause = ", ".join(f'"{k}"' for k in join_keys)

    base_cols = set(base_relation.columns)
    other_cols = set(other_relation.columns)
    common_cols = (base_cols & other_cols) - set(join_keys)

    if join_keyword == "RIGHT":
        index_expr = f'{other_name}."{INDEX_COL}" AS "{INDEX_COL}"'
        order_by = f'ORDER BY {other_name}."{INDEX_COL}"'
    elif join_keyword == "LEFT" or join_keyword == "INNER":
        index_expr = f'{base_name}."{INDEX_COL}" AS "{INDEX_COL}"'
        order_by = f'ORDER BY {base_name}."{INDEX_COL}"'
    else:
        index_expr = (
            f'COALESCE({base_name}."{INDEX_COL}", {other_name}."{INDEX_COL}") AS "{INDEX_COL}"'
        )
        order_by = (
            f'ORDER BY COALESCE({base_name}."{INDEX_COL}", {other_name}."{INDEX_COL}")'
        )

    select_cols = [index_expr] + [
        f'COALESCE({base_name}."{k}", {other_name}."{k}") AS "{k}"' for k in join_keys
    ]

    for col in base_relation.columns:
        if col in (INDEX_COL, *join_keys):
            continue
        suffix = "_x" if col in common_cols and join_keyword != "LEFT" else ""
        select_cols.append(f'{base_name}."{col}" AS "{col}{suffix}"')

    for col in other_relation.columns:
        if col in (INDEX_COL, *join_keys):
            continue
        suffix = "_y" if col in common_cols and join_keyword != "LEFT" else ""
        select_cols.append(f'{other_name}."{col}" AS "{col}{suffix}"')

    query = f"""
        SELECT {", ".join(select_cols)}
        FROM {base_name}
        {join_keyword} JOIN {other_name}
        USING ({using_clause})
        {order_by}
    """
    return RelationProxy(con.sql(query))


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
    return query if as_query else RelationProxy(data.project(query))


def duckdb_select(
    data: DuckDBPyRelation, cols: Union[str, List[str], Any] = "*", as_query: bool = False
) -> DuckDBPyRelation:
    """
    Selects specific columns from a DuckDB relation.

    If no columns are specified, returns an empty relation.

    If `as_query` is True, returns the SQL query string instead of the relation.
    """
    data.clean_exec_graph()
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


def get_col_type(rel: RelationProxy, col_name: str) -> DuckDBPyType:
    """
    Returns the specified column type from the DuckDB relation.
    """
    empty_row = rel.project(f'"{col_name}"', include_index=False).limit(0)
    return empty_row.types[0]


def normalize_data(
    self_data: DuckDBPyRelation, other_data: DuckDBPyRelation
) -> Tuple[DuckDBPyRelation, DuckDBPyRelation]:
    """
    Normalizes the data by launching a remove_null_str and round_doubles operations.
    """
    # Target columns: intersection of DOUBLEs on both sides (exclude index)
    self_double = get_cols_by_types(self_data, "DOUBLE")
    other_double = get_cols_by_types(other_data, "DOUBLE")
    target_cols = sorted(self_double & other_double)
    if not target_cols:
        return self_data, other_data

    s = self_data.set_alias("s")
    o = other_data.set_alias("o")

    # Single OUTER join by index to compute rounded values once
    j = s.join(o, f"s.{INDEX_COL} = o.{INDEX_COL}", how="outer")

    # Build rounded projections for both sides
    self_exprs = [f"s.{INDEX_COL} AS {INDEX_COL}"] + [
        f'round_to_ref(s."{c}", o."{c}") AS "{c}"' for c in target_cols
    ]
    other_exprs = [f"o.{INDEX_COL} AS {INDEX_COL}"] + [
        f'round_to_ref(o."{c}", s."{c}") AS "{c}"' for c in target_cols
    ]

    new_self_vals = j.project(", ".join(self_exprs)).set_alias("ns")
    new_other_vals = j.project(", ".join(other_exprs)).set_alias("no")

    # Assemble final self side: replace only target columns
    s_cols = list(self_data.columns)
    js = self_data.set_alias("s").join(new_self_vals, f"s.{INDEX_COL} = ns.{INDEX_COL}", how="left")
    s_select = []
    for c in s_cols:
        if c == INDEX_COL:
            s_select.append(f"s.{INDEX_COL} AS {INDEX_COL}")
        elif c in target_cols:
            s_select.append(f'ns."{c}" AS "{c}"')
        else:
            s_select.append(f's."{c}" AS "{c}"')
    self_out = js.project(", ".join(s_select))

    # Assemble final other side: replace only target columns
    o_cols = list(other_data.columns)
    jo = other_data.set_alias("o").join(
        new_other_vals, f"o.{INDEX_COL} = no.{INDEX_COL}", how="left"
    )
    o_select = []
    for c in o_cols:
        if c == INDEX_COL:
            o_select.append(f"o.{INDEX_COL} AS {INDEX_COL}")
        elif c in target_cols:
            o_select.append(f'no."{c}" AS "{c}"')
        else:
            o_select.append(f'o."{c}" AS "{c}"')
    other_out = jo.project(", ".join(o_select))

    return self_out, other_out


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
                if col != INDEX_COL
                and isinstance(dtype, DuckDBPyType)
                and dtype in TYPES_DICT.get(type_, [])
            ]
        )
        cols.update(type_columns)

    return cols


def clean_execution_graph(rel: DuckDBPyRelation) -> DuckDBPyRelation:
    df = rel.df()
    return con.from_df(df)
