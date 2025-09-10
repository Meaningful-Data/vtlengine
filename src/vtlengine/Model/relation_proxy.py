from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Sequence

from duckdb import DuckDBPyRelation  # type: ignore
import duckdb

INDEX_COL = "__index__"


@dataclass
class RelationProxy:
    _relation: DuckDBPyRelation
    __slots__ = ("_relation")

    def __init__(self, relation: DuckDBPyRelation):
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
        # str: select column
        if isinstance(key, str):
            proj = ", ".join([INDEX_COL, key])
            return RelationProxy(self.relation.project(proj))

        # RelationProxy or DuckDBPyRelation: boolean mask
        if isinstance(key, RelationProxy):
            mask_rel = key.relation
        elif isinstance(key, DuckDBPyRelation):
            mask_rel = key
        else:
            mask_rel = None

        if mask_rel is not None:
            l = self.relation.set_alias("l")
            m = mask_rel.set_alias("m")
            m_cols = [c for c in m.columns if c != INDEX_COL]
            if not m_cols:
                raise ValueError("The mask does not contain data column(s)")
            mask_col = m_cols[0]
            joined = l.join(m, f"l.{INDEX_COL} = m.{INDEX_COL}", how="left")
            filtered = joined.filter(f"coalesce(m.{mask_col}, false)")
            out_cols = [f"l.{INDEX_COL} AS {INDEX_COL}"] + [f"l.{c} AS {c}" for c in self.columns]
            return RelationProxy(filtered.project(", ".join(out_cols)))

        if isinstance(key, Sequence) and not isinstance(key, str):
            if len(key) > 0 and all(type(x) is bool for x in key):
                data = list(enumerate(key))
                m = duckdb.values(data).project("column0 AS __pos__, column1 AS __flag__").set_alias("m")
                l = self.relation.project(f"row_number() OVER (ORDER BY {INDEX_COL}) - 1 AS __pos__, *").set_alias("l")
                joined = l.join(m, "l.__pos__ = m.__pos__", how="left")
                filtered = joined.filter("coalesce(m.__flag__, false)")
                out_cols = [f"l.{INDEX_COL} AS {INDEX_COL}"] + [f"l.{c} AS {c}" for c in self.columns]
                return RelationProxy(filtered.project(", ".join(out_cols)))

            if all(isinstance(x, str) for x in key):
                cols = [c for c in key if c in self.relation.columns and c != INDEX_COL]
                if not cols:
                    return RelationProxy(self.relation.project(INDEX_COL))
                proj = ", ".join([INDEX_COL] + cols)
                return RelationProxy(self.relation.project(proj))

            if all(isinstance(x, int) and not isinstance(x, bool) for x in key):
                pos_vals = list(enumerate(key))
                idx = duckdb.values(pos_vals).project("column0 AS __pos__, column1 AS idx").set_alias("idx")
                l = self.relation.set_alias("l")
                joined = l.join(idx, f"l.{INDEX_COL} = idx.idx", how="inner")
                proj = [f"l.{INDEX_COL} AS {INDEX_COL}"] + [f"l.{c} AS {c}" for c in self.columns]
                result = joined.order("__pos__").project(", ".join(proj))
                return RelationProxy(result)

        raise TypeError(f"Unsupported key type for __getitem__: {type(key)!r}")

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
        return (f"RelationProxy(\n"
                f"columns={self.relation.columns},\n"
                f"data=\n{self.relation}\n"
                f")")

    @property
    def columns(self) -> list[str]:
        return [c for c in self.relation.columns if c != INDEX_COL]

    @property
    def all_columns(self) -> list[str]:
        return self.relation.columns

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

    def _wrap_relation(self, value: Any) -> Any:
        return RelationProxy(value) if isinstance(value, DuckDBPyRelation) else value

    def _ensure_index(self, projection: str) -> str:
        cols_lower = [c.strip().lower() for c in projection.split(",")]
        if INDEX_COL not in cols_lower:
            return f"{INDEX_COL}, " + projection
        return projection

    def _to_sql_literal(self, v: Any) -> str:
        # Convert a Python value into a safe SQL literal for DuckDB
        if v is None:
            return "NULL"
        if isinstance(v, bool):
            return "TRUE" if v else "FALSE"
        if isinstance(v, (int, float)):
            return repr(v)
        if isinstance(v, str):
            return "'" + v.replace("'", "''") + "'"
        # Fallback: use DuckDB VALUES to build a literal and select it via join if needed
        # For simplicity, convert to string literal
        return "'" + str(v).replace("'", "''") + "'"

    def _binary_compare(self, other: Any, op: str) -> "RelationProxy":
        # Compare a single data column with a scalar or another single-column relation, aligned by index
        left = self.relation
        left_cols = [c for c in left.columns if c != INDEX_COL]
        if len(left_cols) != 1:
            raise ValueError("Element-wise comparison requires a single data column on the left side")

        # Other is a RelationProxy or DuckDBPyRelation: align by index and compare columns
        if isinstance(other, RelationProxy):
            right = other.relation
        elif isinstance(other, DuckDBPyRelation):
            right = other
        else:
            right = None

        if right is not None:
            r_cols = [c for c in right.columns if c != INDEX_COL]
            if len(r_cols) != 1:
                raise ValueError("Element-wise comparison requires a single data column on the right side")
            l = left.set_alias("l")
            r = right.set_alias("r")
            expr = f"(l.{left_cols[0]} {op} r.{r_cols[0]}) AS __mask__"
            proj = f"l.{INDEX_COL} AS {INDEX_COL}, {expr}"
            joined = l.join(r, f"l.{INDEX_COL} = r.{INDEX_COL}", how="left")
            return RelationProxy(joined.project(proj))

        # Other is a scalar: compare left column with literal
        lit = self._to_sql_literal(other)
        expr = f"({left_cols[0]} {op} {lit}) AS __mask__"
        proj = f"{INDEX_COL}, {expr}"
        return RelationProxy(left.project(proj))

    def df(self, limit: int | None = None) -> Any:
        rel = self.relation if limit is None else self.relation.limit(limit)
        df = rel.df()
        if INDEX_COL in df.columns:
            df = df.set_index(INDEX_COL)
        return df

    @property
    def values(self):
        # DataFrame/Series-like: return 1D for single data column, otherwise 2D
        df = self.df()
        cols = self.columns
        if len(cols) == 1:
            return df[cols[0]].values
        return df.values

    def project(self, projection: str = f"* EXCLUDE {INDEX_COL}", include_index: bool = True) -> "RelationProxy":
        if include_index:
            projection = self._ensure_index(projection)
        return self.relation.project(projection)

    def reindex(self, **kwargs: Any) -> "RelationProxy":
        new_rel = self.relation.project(f"row_number() OVER () - 1 AS {INDEX_COL}, * EXCLUDE {INDEX_COL}")
        return RelationProxy(new_rel)
