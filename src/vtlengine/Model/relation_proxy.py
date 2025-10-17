from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import pandas as pd
from duckdb import DuckDBPyRelation

from vtlengine.connection import con

INDEX_COL = "__index__"


@dataclass
class RelationProxy:
    _relation: DuckDBPyRelation
    __slots__ = "_relation"

    def __init__(
        self,
        relation: Union[DuckDBPyRelation, pd.DataFrame],
        index: Optional[DuckDBPyRelation] = None,
    ):
        if isinstance(relation, pd.DataFrame):
            relation = con.from_df(relation)
        if index is not None and INDEX_COL in index.columns:
            # setting index explicitly
            idx = index.project(INDEX_COL).set_alias("idx")
            rel = relation.set_alias("rel")
            joined = rel.join(idx, f"rel.{INDEX_COL} = idx.{INDEX_COL}", how="right")
            self.relation = joined.project(f"idx.{INDEX_COL} AS {INDEX_COL}, * EXCLUDE {INDEX_COL}")
        self.relation = relation

    # Dynamic dispatch
    def __getattribute__(self, name: str) -> Any:
        reserved = set(RelationProxy.__dict__.keys())
        if name in reserved:
            return object.__getattribute__(self, name)

        rel = object.__getattribute__(self, "_relation")
        if hasattr(rel, name):
            attr = getattr(rel, name)
            if callable(attr):

                def rel_wrapper(*args: Any, **kwargs: Any) -> Any:
                    return attr(*args, **kwargs)

                self._wrap_relation(rel_wrapper)
            return attr

        raise AttributeError(
            f"Attribute or method '{name}' not found in DuckDBPyRelation nor RelationProxy"
        ) from None

    def __getitem__(self, key: Any) -> "RelationProxy":
        if isinstance(key, str):
            if key == INDEX_COL:
                return RelationProxy(self.relation.project(INDEX_COL))
            return RelationProxy(self.relation.project(f'{INDEX_COL}, "{key}"'))

        if isinstance(key, int):
            if key < 0:
                raise IndexError("Negative indexing is not supported")
            data = self.relation.filter(f"{INDEX_COL} = {key}")
            if len(self.columns) == 1:
                data = data.project(f"* EXCLUDE {INDEX_COL}").execute().fetchone()[0]
            return RelationProxy(data)

        if isinstance(key, (RelationProxy, DuckDBPyRelation)):
            mask_rel = key.relation if isinstance(key, RelationProxy) else key
            left_rel = self.relation.set_alias("l")
            mask = mask_rel.set_alias("m")
            data_cols = [c for c in mask.columns if c != INDEX_COL]

            if len(data_cols) == 0:
                mpos = mask.project(f"row_number() OVER () - 1 AS __pos__, {INDEX_COL}").set_alias(
                    "m"
                )
                joined = left_rel.join(mpos, f"l.{INDEX_COL} = m.{INDEX_COL}", how="inner")
                return RelationProxy(joined.order("m.__pos__").project("l.*"))

            mask_col = data_cols[0]
            mask_true = mask.filter(f'coalesce(m."{mask_col}", true)')
            joined = left_rel.join(mask_true, f"l.{INDEX_COL} = m.{INDEX_COL}", how="semi")
            return RelationProxy(joined)

        if isinstance(key, Sequence) and not isinstance(key, (str, bytes)):
            if all(isinstance(x, str) for x in key):
                cols = [c for c in key if c in self.relation.columns and c != INDEX_COL]
                if not cols:
                    return RelationProxy(self.relation.project(INDEX_COL))
                proj = ", ".join([INDEX_COL] + [f'"{c}"' for c in cols])
                return RelationProxy(self.relation.project(proj))

            if all(isinstance(x, int) and not isinstance(x, bool) for x in key):
                idx_list = [int(x) for x in key]
                if not idx_list:
                    return RelationProxy(self.relation.limit(0))

                data = list(enumerate(idx_list))
                idx = con.values(data).project("col0 AS __pos__, col1 AS idx").set_alias("idx")
                left_rel = self.relation.set_alias("l")
                joined = left_rel.join(idx, f"l.{INDEX_COL} = idx.idx", how="inner")
                return RelationProxy(joined.order("idx.__pos__").project("l.*"))

        raise TypeError(f"Unsupported key type for __getitem__: {type(key)!r}")

    # pandas-like assignment
    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, str):
            if key == INDEX_COL:
                raise ValueError("Cannot assign to the index column")
            self._assign_column(key, value)
            return

        if isinstance(key, (RelationProxy, DuckDBPyRelation)):
            self._assign_rows(key, value)
            return

        if (
            isinstance(key, Sequence)
            and not isinstance(key, (str, bytes))
            and (
                (len(key) > 0 and all(type(x) is bool for x in key))
                or (all(isinstance(x, int) and not isinstance(x, bool) for x in key))
            )
        ):
            self._assign_rows(key, value)
            return

        if isinstance(key, tuple) and len(key) == 2:
            row_key, col_key = key

            if isinstance(col_key, str):
                cols = [col_key]
            elif isinstance(col_key, Sequence) and not isinstance(col_key, (str, bytes)):
                if not all(isinstance(c, str) for c in col_key):
                    raise TypeError("Column selector in tuple must be str or sequence of str")
                cols = list(col_key)
            else:
                raise TypeError("Unsupported column selector type in tuple assignment")

            if any(c == INDEX_COL for c in cols):
                raise ValueError("Cannot assign to the index column")

            seen = set()
            cols = [c for c in cols if not (c in seen or seen.add(c))]

            self._assign_rows_to_columns(row_key, cols, value)
            return

        raise TypeError(f"Unsupported key type for __setitem__: {type(key)!r}")

    def __contains__(self, item: Any) -> bool:
        cols = [c for c in self.relation.columns if c != INDEX_COL]
        if len(cols) != 1:
            return False
        col = cols[0]
        proj = self.relation.project(f'{INDEX_COL}, "{col}"')
        cond = f'"{col}" IS NULL' if item is None else f'"{col}" = {self._to_sql_literal(item)}'
        cur = proj.filter(cond).limit(1).execute()
        return cur.fetchone() is not None

    def __eq__(self, other: Any) -> "RelationProxy":
        return self._binary_compare(other, "=")

    def __ne__(self, other: Any) -> "RelationProxy":
        return self._binary_compare(other, "!=")

    def __lt__(self, other: Any) -> "RelationProxy":
        return self._binary_compare(other, "<")

    def __le__(self, other: Any) -> "RelationProxy":
        return self._binary_compare(other, "<=")

    def __gt__(self, other: Any) -> "RelationProxy":
        return self._binary_compare(other, ">")

    def __ge__(self, other: Any) -> "RelationProxy":
        return self._binary_compare(other, ">=")

    def __len__(self) -> int:
        return int(self.relation.aggregate("count(*) AS cnt").execute().fetchone()[0])

    def __repr__(self) -> str:
        sorted_cols = sorted(self.relation.columns)
        try:
            data = self.relation.project(", ".join(sorted_cols)).limit(30)
        except Exception:
            data = self.relation
        return f"RelationProxy(\ncolumns={sorted_cols},\ndata=\n{data}\n)"

    @property
    def all_columns(self) -> list[str]:
        return self.relation.columns

    @property
    def columns(self) -> list[str]:
        return [c for c in self.relation.columns if c != INDEX_COL]

    @property
    def dtypes(self) -> dict[str, str]:
        return dict(zip(self.relation.columns, self.relation.types))

    @property
    def index(self) -> DuckDBPyRelation:
        return self.relation.project(INDEX_COL)

    @property
    def relation(self) -> DuckDBPyRelation:
        return self._relation

    @relation.setter
    def relation(self, value: DuckDBPyRelation) -> None:
        value = value.relation if isinstance(value, RelationProxy) else value
        if INDEX_COL not in value.columns:
            value = value.project(f"*, row_number() OVER () - 1 AS {INDEX_COL}")
        self._relation = value

    def _enumerate_sequence(self, seq: Sequence[Any], value_col: str, alias: str = "m"):
        if not seq:
            return con.values([]).project(
                f"CAST(NULL AS BIGINT) AS __pos__, CAST(NULL AS ANY) AS {value_col}"
            )
        data = list(enumerate(seq))
        return (
            con.values(data).project(f"column0 AS __pos__, column1 AS {value_col}").set_alias(alias)
        )

    def _prepare_positional_join(self, rel: DuckDBPyRelation, alias: str = "l"):
        return rel.project(f"row_number() OVER (ORDER BY {INDEX_COL}) - 1 AS __pos__, *").set_alias(
            alias
        )

    def _assign_column(self, col_name: str, value: Any) -> None:
        left_rel = self.relation.set_alias("l")
        has_col = col_name in self.columns
        base_proj = f'l.* EXCLUDE "{col_name}"' if has_col else "l.*"

        if isinstance(value, (RelationProxy, DuckDBPyRelation)):
            right_rel = (
                value.relation
                if isinstance(value, RelationProxy)
                else RelationProxy(value).relation
            )
            if len(right_rel.columns) == 0:
                self.relation = left_rel.project(f'{base_proj}, NULL AS "{col_name}"')
                return
            r_data_cols = [c for c in right_rel.columns if c != INDEX_COL]
            if not r_data_cols:
                raise ValueError("Right-hand side relation has no data columns to assign")
            rcol = r_data_cols[0]
            right_rel = right_rel.project(f'{INDEX_COL}, "{rcol}"').set_alias("r")
            joined = left_rel.join(right_rel, f"l.{INDEX_COL} = r.{INDEX_COL}", how="left")
            self.relation = joined.project(f'{base_proj}, r."{rcol}" AS "{col_name}"')
            return

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            expected = len(self)
            if len(value) != expected:
                raise ValueError(
                    f"Sequence length {len(value)} does not match relation length {expected}"
                )
            mask = self._enumerate_sequence(value, "__val__", alias="m")
            lpos = self._prepare_positional_join(left_rel, alias="l")
            joined = lpos.join(mask, "l.__pos__ = m.__pos__", how="left")
            self.relation = joined.project(f'{base_proj}, m.__val__ AS "{col_name}"')
            return

        scalar = self._to_sql_literal(value)
        self.relation = left_rel.project(f'{base_proj}, {scalar} AS "{col_name}"')

    def _assign_rows(self, key: Any, value: Any) -> None:
        if isinstance(value, (RelationProxy, DuckDBPyRelation, Sequence)) and not isinstance(
            value, (str, bytes)
        ):
            raise NotImplementedError("Row-wise assignment currently supports only scalar values")

        left_rel = self.relation.set_alias("l")
        cond: Optional[str] = None
        joined: Optional[DuckDBPyRelation] = None

        if isinstance(key, (RelationProxy, DuckDBPyRelation)):
            m_rel = key.relation if isinstance(key, RelationProxy) else RelationProxy(key).relation
            mask = m_rel.set_alias("m")
            m_cols = [c for c in mask.columns if c != INDEX_COL]

            if not m_cols:
                joined = left_rel.join(
                    mask.project(INDEX_COL).set_alias("m"),
                    f"l.{INDEX_COL} = m.{INDEX_COL}",
                    how="left",
                )
                cond = f"m.{INDEX_COL} IS NOT NULL"
            else:
                mask_col = m_cols[0]
                mask = mask.project(f'{INDEX_COL}, "{mask_col}"').set_alias("m")
                joined = left_rel.join(mask, f"l.{INDEX_COL} = m.{INDEX_COL}", how="left")
                cond = f'coalesce(m."{mask_col}", false)'

        elif isinstance(key, Sequence) and not isinstance(key, (str, bytes)):
            if len(key) > 0 and all(type(x) is bool for x in key):
                true_pos = [i for i, f in enumerate(key) if f]
                if not true_pos:
                    return
                joined = left_rel
                cond = f"l.{INDEX_COL} IN ({', '.join(map(str, true_pos))})"

            elif all(isinstance(x, int) and not isinstance(x, bool) for x in key):
                if not key:
                    return
                idx_list = [int(i) for i in key]
                joined = left_rel
                cond = f"l.{INDEX_COL} IN ({', '.join(map(str, idx_list))})"
            else:
                raise TypeError("Unsupported sequence type for row-wise assignment")
        else:
            raise TypeError(f"Unsupported key type for row-wise assignment: {type(key)!r}")

        if joined is None or cond is None:
            raise RuntimeError("Internal error constructing row-wise assignment")

        lit = self._to_sql_literal(value)
        replace_parts = [
            f'CASE WHEN {cond} THEN {lit} ELSE l."{c}" END AS "{c}"' for c in self.columns
        ]
        projection = f"l.* REPLACE ({', '.join(replace_parts)})"
        self.relation = (joined or left_rel).project(projection)

    def _assign_rows_to_columns(self, key: Any, columns: Sequence[str], value: Any) -> None:
        if isinstance(value, (RelationProxy, DuckDBPyRelation, Sequence)) and not isinstance(
            value, (str, bytes)
        ):
            raise NotImplementedError("Row-wise assignment currently supports only scalar values")

        left_rel = self.relation.set_alias("l")
        joined, cond = self._build_row_join_and_condition(left_rel, key)
        if joined is None or cond is None:
            return

        value = self._to_sql_literal(value)
        columns = set(columns)

        parts: list[str] = []
        replace_parts = [
            f'CASE WHEN {cond} THEN {value} ELSE l."{c}" END AS "{c}"'
            for c in self.columns
            if c in columns
        ]
        parts.append(f"l.* REPLACE ({', '.join(replace_parts)})" if replace_parts else "l.*")

        for c in columns:
            if c not in self.columns:
                parts.append(f'CASE WHEN {cond} THEN {value} ELSE NULL END AS "{c}"')

        self.relation = joined.project(", ".join(parts))

    def _build_row_join_and_condition(
        self, left_rel: DuckDBPyRelation, key: Any
    ) -> tuple[Optional[DuckDBPyRelation], Optional[str]]:
        cond: Optional[str] = None
        joined: Optional[DuckDBPyRelation] = None

        if isinstance(key, (RelationProxy, DuckDBPyRelation)):
            m_rel = key.relation if isinstance(key, RelationProxy) else RelationProxy(key).relation
            mask = m_rel.set_alias("m")
            m_cols = [c for c in mask.columns if c != INDEX_COL]

            if not m_cols:
                joined = left_rel.join(
                    mask.project(INDEX_COL).set_alias("m"),
                    f"l.{INDEX_COL} = m.{INDEX_COL}",
                    how="left",
                )
                cond = f"m.{INDEX_COL} IS NOT NULL"
            else:
                mask_col = m_cols[0]
                mask = mask.project(f'{INDEX_COL}, "{mask_col}"').set_alias("m")
                joined = left_rel.join(mask, f"l.{INDEX_COL} = m.{INDEX_COL}", how="left")
                cond = f'coalesce(m."{mask_col}", false)'

        elif isinstance(key, Sequence) and not isinstance(key, (str, bytes)):
            if len(key) > 0 and all(type(x) is bool for x in key):
                true_pos = [i for i, f in enumerate(key) if f]
                if not true_pos:
                    return None, None
                joined = left_rel
                cond = f"l.{INDEX_COL} IN ({', '.join(map(str, true_pos))})"

            elif all(isinstance(x, int) and not isinstance(x, bool) for x in key):
                if not key:
                    return None, None
                idx_list = [int(i) for i in key]
                joined = left_rel
                cond = f"l.{INDEX_COL} IN ({', '.join(map(str, idx_list))})"
            else:
                raise TypeError("Unsupported sequence type for row-wise assignment")
        else:
            raise TypeError(f"Unsupported key type for row-wise assignment: {type(key)!r}")

        return joined, cond

    def _binary_compare(self, other: Any, op: str) -> "RelationProxy":
        left = self.relation
        lcol = self._get_single_data_column(left)

        if isinstance(other, (RelationProxy, DuckDBPyRelation)):
            right = other.relation if isinstance(other, RelationProxy) else other
            rcol = self._get_single_data_column(right)

            left_rel = left.project(f'{INDEX_COL}, "{lcol}"').set_alias("l")
            right_rel = right.project(f'{INDEX_COL}, "{rcol}"').set_alias("r")
            expr = f'(l."{lcol}" {op} r."{rcol}") AS __mask__'
            joined = left_rel.join(right_rel, f"l.{INDEX_COL} = r.{INDEX_COL}", how="left")
            return RelationProxy(joined.project(f"l.{INDEX_COL} AS {INDEX_COL}, {expr}"))

        value = self._to_sql_literal(other)
        proj_left = left.project(f'{INDEX_COL}, "{lcol}"')
        expr = f'("{lcol}" {op} {value}) AS __mask__'
        return RelationProxy(proj_left.project(f"{INDEX_COL}, {expr}"))

    def _get_single_data_column(self, rel: DuckDBPyRelation) -> str:
        cols = [c for c in rel.columns if c != INDEX_COL]
        if len(cols) != 1:
            raise ValueError("Element-wise comparison requires exactly one data column per side")
        return cols[0]

    def _ensure_index(self, projection: str) -> str:
        cols_lower = [c.strip().lower() for c in projection.split(",")]
        if INDEX_COL not in cols_lower:
            return f"{INDEX_COL}, " + projection
        return projection

    def _materialize(self, data: DuckDBPyRelation) -> DuckDBPyRelation:
        tmp = f"__mat_{uuid.uuid4().hex}"
        con.execute(f"CREATE TEMP TABLE {tmp} AS SELECT * FROM data")
        return con.table(tmp)

    def _to_sql_literal(self, v: Any) -> str:
        if v is None:
            return "NULL"
        if isinstance(v, bool):
            return "TRUE" if v else "FALSE"
        if isinstance(v, (int, float)):
            return repr(v)
        if isinstance(v, str):
            return "'" + v.replace("'", "''") + "'"
        return "'" + str(v).replace("'", "''") + "'"

    def _resolve_duckdb_type(self, dtype: Any) -> str:
        if dtype is str:
            return "VARCHAR"
        if dtype is int:
            return "BIGINT"
        if dtype is float:
            return "DOUBLE"
        if dtype is bool:
            return "BOOLEAN"
        return "VARCHAR"

    def _wrap_relation(self, value: Any) -> Any:
        return RelationProxy(value) if isinstance(value, DuckDBPyRelation) else value

    def clean_exec_graph(self, verbose: bool = False) -> None:
        if verbose:
            print("Pre-clean plan:")
            print(self.explain())
        self.relation = self._materialize(self.relation)
        if verbose:
            print("Post-clean plan:")
            print(self.explain())

    def distinct(self) -> "RelationProxy":
        return RelationProxy(self.project(include_index=False).distinct())

    def df(self, limit: int | None = None) -> Any:
        rel = self.relation if limit is None else self.relation.limit(limit)
        df = rel.df()
        if INDEX_COL in df.columns:
            df = df.set_index(INDEX_COL)
            df.index.name = None
        return df

    def drop(self, columns: Any) -> "RelationProxy":
        drop_set = {columns} if isinstance(columns, str) else set(columns)
        drop_set.discard(INDEX_COL)

        keep_cols = [INDEX_COL] + [c for c in self.columns if c not in drop_set]
        proj = ", ".join([f'"{c}"' for c in keep_cols])
        return RelationProxy(self.relation.project(proj))

    def explain(self, type_: Optional[str] = None) -> str:
        if type_ in ["analyze", "logical"]:
            return self.relation.explain(type=type_)
        return self.relation.explain()

    def isnull(self) -> "RelationProxy":
        if len(self.columns) == 0:
            raise ValueError("No data columns to check for nulls")
        expr = f'("{self.columns[0]}" IS NULL) AS __mask__'
        return RelationProxy(self.relation.project(f"{INDEX_COL}, {expr}"))

    def notnull(self) -> "RelationProxy":
        if len(self.columns) == 0:
            raise ValueError("No data columns to check for not nulls")
        expr = f'("{self.columns[0]}" IS NOT NULL) AS __mask__'
        return RelationProxy(self.relation.project(f"{INDEX_COL}, {expr}"))

    def any(self) -> bool:
        col = self._get_single_data_column(self.relation)
        proj = self.relation.project(f'{INDEX_COL}, "{col}"')
        agg = proj.aggregate(f'coalesce(bool_or("{col}"), false) AS __any__')
        return bool(agg.execute().fetchone()[0])

    def all(self) -> bool:
        col = self._get_single_data_column(self.relation)
        proj = self.relation.project(f'{INDEX_COL}, "{col}"')
        agg = proj.aggregate(f'coalesce(bool_and(coalesce("{col}", true)), true) AS __all__')
        return bool(agg.execute().fetchone()[0])

    def is_empty(self):
        pass

    def order_by_index(self) -> "RelationProxy":
        return RelationProxy(self.relation.order(INDEX_COL))

    def project(
        self, projection: str = f"* EXCLUDE {INDEX_COL}", include_index: bool = True
    ) -> "RelationProxy":
        self.clean_exec_graph()
        if include_index or len(self.columns) == 0:
            projection = self._ensure_index(projection)
        return self.relation.project(projection)

    def reset_index(self, **kwargs: Any) -> "RelationProxy":
        new_rel = self.relation.project(
            f"row_number() OVER () - 1 AS {INDEX_COL}, * EXCLUDE {INDEX_COL}"
        )
        return RelationProxy(new_rel)

    def astype(self, dtype: Any) -> "RelationProxy":
        col = self._get_single_data_column(self.relation)
        duck_type = self._resolve_duckdb_type(dtype)
        proj = self.relation.project(f'{INDEX_COL}, CAST("{col}" AS {duck_type}) AS "{col}"')
        return RelationProxy(proj)

    def cast(self, dtype: Any) -> "RelationProxy":
        return self.astype(dtype)
