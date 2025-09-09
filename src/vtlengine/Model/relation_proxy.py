from __future__ import annotations

import uuid
from typing import Any, Sequence

from duckdb import DuckDBPyRelation  # type: ignore

from vtlengine.connection import con


INDEX_COL = "__index__"


class RelationProxy:
    __slots__ = ("_relation", "_id")

    def __init__(self, relation: DuckDBPyRelation):
        self._id = f"_relproxy_{uuid.uuid4().hex}"
        if INDEX_COL not in relation.columns:
            relation = relation.project(f"row_number() OVER () - 1 AS {INDEX_COL}, *")
        self.relation = relation

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, RelationProxy):
            other = other.relation
        return self.relation == other

    def __len__(self) -> int:
        count_rel = self.relation.aggregate("count(*) AS cnt")
        count_df = count_rel.df()
        return int(count_df["cnt"][0])

    def __repr__(self) -> str:
        return (f"RelationProxy(\n"
                f"id={self._id},\n"
                f"columns={self.relation.columns},\n"
                f"data=\n{self.relation}\n"
                f")")

    @property
    def index(self) -> DuckDBPyRelation:
        return self.relation.project(INDEX_COL)

    @property
    def relation(self) -> DuckDBPyRelation:
        return self._relation

    @relation.setter
    def relation(self, value: DuckDBPyRelation) -> None:
        self._relation = value.relation if isinstance(value, RelationProxy) else value

    def _wrap_relation(self, value: Any) -> Any:
        return RelationProxy(value) if isinstance(value, DuckDBPyRelation) else value

    def _ensure_index(self, projection: str) -> str:
        cols_lower = [c.strip().lower() for c in projection.split(",")]
        if INDEX_COL not in cols_lower:
            return f"{INDEX_COL}, " + projection
        return projection

    def assign(
        self, other: "RelationProxy", columns: Sequence[str] | None = None
    ) -> "RelationProxy":
        left_cols = [c for c in self.relation.columns if c != INDEX_COL]
        right_cols = [c for c in other.relation.columns if c != INDEX_COL]
        if columns is None:
            columns = [c for c in left_cols if c in right_cols]
        assign_set = set(columns)
        out_cols: list[str] = [f"l.{INDEX_COL} AS {INDEX_COL}"]
        for c in left_cols:
            if c in assign_set:
                out_cols.append(f"r.{c} AS {c}")
            else:
                out_cols.append(f"l.{c} AS {c}")
        sql = f"""
        SELECT {", ".join(out_cols)}
        FROM ({self.relation.to_sql()}) l
        LEFT JOIN ({other.relation.to_sql()}) r USING ({INDEX_COL})
        """
        return RelationProxy(con.sql(sql))

    def df(self, limit: int | None = None) -> Any:
        rel = self.relation if limit is None else self.relation.limit(limit)
        df = rel.df()
        if INDEX_COL in df.columns:
            df = df.set_index(INDEX_COL)
        return df

    def project(self, projection: str = f"* EXCLUDE {INDEX_COL}", include_index: bool = True) -> "RelationProxy":
        if include_index:
            projection = self._ensure_index(projection)
        return self.relation.project(projection)

    def reindex(self) -> "RelationProxy":
        cols = [f"'{c}'" for c in self.relation.columns if c != INDEX_COL]
        new_rel = self.relation.project(", ".join(cols))
        new_rel = new_rel.with_column(INDEX_COL, "row_number() OVER() - 1")
        return RelationProxy(new_rel)

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

        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            raise AttributeError(
                f"Attribute or method '{name}' not found in DuckDBPyRelation nor RelationProxy"
            ) from None
