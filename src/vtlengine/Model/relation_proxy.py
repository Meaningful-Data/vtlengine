from __future__ import annotations

import uuid
from typing import Any, Sequence

from duckdb import DuckDBPyRelation  # type: ignore

from vtlengine.connection import con


class RelationProxy:
    __slots__ = ("_relation", "_id")

    def __init__(self, relation: DuckDBPyRelation):
        self._id = f"_relproxy_{uuid.uuid4().hex}"
        if "__index__" not in relation.columns:
            base_sql = relation.to_sql()
            relation = con.sql(
                f"SELECT row_number() OVER() - 1 AS __index__, t.* FROM ({base_sql}) t"
            )
        self._relation: DuckDBPyRelation = relation

    def _wrap_relation(self, value: Any) -> Any:
        if isinstance(value, DuckDBPyRelation):
            return RelationProxy(value)
        return value

    def _ensure_index(self, projection: str) -> str:
        cols_lower = [c.strip().lower() for c in projection.split(",")]
        if "__index__" not in cols_lower:
            return "__index__, " + projection
        return projection

    @property
    def relation(self) -> DuckDBPyRelation:
        return self._relation

    @property
    def index(self) -> DuckDBPyRelation:
        return self._relation.project("__index__")

    def df(self, limit: int | None = None) -> Any:
        rel = self._relation if limit is None else self._relation.limit(limit)
        df = rel.df()
        if "__index__" in df.columns:
            df = df.set_index("__index__")
        return df

    def reindex(self) -> "RelationProxy":
        cols = [f"'{c}'" for c in self._relation.columns if c != "__index__"]
        new_rel = self._relation.project(", ".join(cols))
        new_rel = new_rel.with_column("__index__", "row_number() OVER() - 1")
        return RelationProxy(new_rel)

    def assign_from(
        self, other: "RelationProxy", columns: Sequence[str] | None = None
    ) -> "RelationProxy":
        left_cols = [c for c in self._relation.columns if c != "__index__"]
        right_cols = [c for c in other._relation.columns if c != "__index__"]
        if columns is None:
            columns = [c for c in left_cols if c in right_cols]
        assign_set = set(columns)
        out_cols: list[str] = ["l.__index__ AS __index__"]
        for c in left_cols:
            if c in assign_set:
                out_cols.append(f"r.{c} AS {c}")
            else:
                out_cols.append(f"l.{c} AS {c}")
        sql = f"""
        SELECT {", ".join(out_cols)}
        FROM ({self._relation.to_sql()}) l
        LEFT JOIN ({other._relation.to_sql()}) r USING (__index__)
        """
        return RelationProxy(con.sql(sql))

    # Dynamic dispatch
    def __getattribute__(self, name: str) -> Any:
        reserved = {
            "__class__",
            "__slots__",
            "__init__",
            "__repr__",
            "__iter__",
            "__eq__",
            "__getattribute__",
            "_wrap_relation",
            "_ensure_index_in_projection",
            "_relation",
            "_id",
            "relation",
            "index",
            "df",
            "reindex",
            "assign_from",
        }
        if name in reserved:
            return object.__getattribute__(self, name)

        if name in ("project", "select"):

            def _proj_wrapper(projection: str, *args: Any, **kwargs: Any) -> Any:
                projection = self._ensure_index(projection)
                attr = getattr(self._relation, name)
                result = attr(projection, *args, **kwargs)
                return self._wrap_relation(result)

            return _proj_wrapper

        if hasattr(self._relation, name):
            attr = getattr(self._relation, name)
            if callable(attr):

                def rel_wrapper(*args: Any, **kwargs: Any) -> Any:
                    result = attr(*args, **kwargs)
                    return self._wrap_relation(result)

                return rel_wrapper
            return attr

        # Fallback
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            raise AttributeError(
                f"Attribute or method '{name}' not found in DuckDBPyRelation nor RelationProxy"
            ) from None

    def __repr__(self) -> str:
        try:
            cnt = con.sql(f"SELECT COUNT(*) FROM ({self._relation.to_sql()})").fetchone()[0]
        except Exception:
            cnt = "?"
        return f"<RelationProxy rows={cnt} sql={self._relation.to_sql()!r}>"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, RelationProxy):
            other = other._relation
        return self._relation == other
