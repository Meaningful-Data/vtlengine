from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from duckdb import DuckDBPyRelation  # type: ignore

from vtlengine.connection import con

INDEX_COL = "__index__"


@dataclass
class RelationProxy:
    _relation: DuckDBPyRelation
    __slots__ = "_relation"

    def __init__(self, relation: DuckDBPyRelation, index: Optional[DuckDBPyRelation] = None):
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

        if isinstance(key, (RelationProxy, DuckDBPyRelation)):
            m_rel = key.relation if isinstance(key, RelationProxy) else key
            l = self.relation.set_alias("l")
            m = m_rel.set_alias("m")
            data_cols = [c for c in m.columns if c != INDEX_COL]

            if len(data_cols) == 0:
                mpos = m.project(f"row_number() OVER () - 1 AS __pos__, {INDEX_COL}").set_alias("m")
                joined = l.join(mpos, f"l.{INDEX_COL} = m.{INDEX_COL}", how="inner")
                return RelationProxy(joined.order("m.__pos__").project("l.*"))

            mask_col = data_cols[0]
            m_true = m.filter(f'coalesce(m."{mask_col}", false)')
            joined = l.join(m_true, f"l.{INDEX_COL} = m.{INDEX_COL}", how="semi")
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
                idx = con.values(data).project("column0 AS __pos__, column1 AS idx").set_alias("idx")
                l = self.relation.set_alias("l")
                joined = l.join(idx, f"l.{INDEX_COL} = idx.idx", how="inner")
                return RelationProxy(joined.order("idx.__pos__").project("l.*"))

        raise TypeError(f"Unsupported key type for __getitem__: {type(key)!r}")

    # New: pandas-like assignment
    def __setitem__(self, key: Any, value: Any) -> None:
        # Column assignment: df["col"] = ...
        if isinstance(key, str):
            if key == INDEX_COL:
                raise ValueError("Cannot assign to the index column")
            self._assign_column(key, value)
            return

        # Row-wise assignment via mask Relation or DuckDB relation
        if isinstance(key, (RelationProxy, DuckDBPyRelation)):
            self._assign_rows(key, value)
            return

        # Row-wise assignment via Python sequences: bool mask or list of indices
        if isinstance(key, Sequence) and not isinstance(key, str):
            # bool mask list
            if len(key) > 0 and all(type(x) is bool for x in key):
                self._assign_rows(key, value)
                return
            # list of indices (ints)
            if all(isinstance(x, int) and not isinstance(x, bool) for x in key):
                self._assign_rows(key, value)
                return

        raise TypeError(f"Unsupported key type for __setitem__: {type(key)!r}")

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
        # print(f"\nOLD EXEC GRAPH:\n{self.relation.explain()}")
        # self.clean_exec_graph()
        # print(f"\nNEW EXEC GRAPH:\n{self.relation.explain()}")
        return int(self.relation.aggregate("count(*) AS cnt").execute().fetchone()[0])

    def __repr__(self) -> str:
        return (
            f"RelationProxy(\ncolumns={self.relation.columns},\ndata=\n{self.relation.limit(10)}\n)"
        )

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
        if value is None:
            pass
        if INDEX_COL not in value.columns:
            value = value.project(f"*, row_number() OVER () - 1 AS {INDEX_COL}")
        self._relation = value

    def _assign_column(self, col_name: str, value: Any) -> None:
        l = self.relation.set_alias("l")
        other_cols = [c for c in self.columns if c != col_name]

        # Relation-like value: align by index, take single data column
        if isinstance(value, (RelationProxy, DuckDBPyRelation)):
            r_rel = (
                value.relation
                if isinstance(value, RelationProxy)
                else RelationProxy(value).relation
            )
            r = r_rel.set_alias("r")
            r_cols = [c for c in r.columns if c != INDEX_COL]
            if len(r_cols) == 0:
                raise ValueError("Right-hand side relation has no data columns to assign")
            r_col = r_cols[0] if len(r_cols) > 1 else r_cols[0]
            joined = l.join(r, f"l.{INDEX_COL} = r.{INDEX_COL}", how="left")
            proj_cols = (
                [f"l.{INDEX_COL} AS {INDEX_COL}"]
                + [f'l."{c}" AS "{c}"' for c in other_cols]
                + [f'r."{r_col}" AS "{col_name}"']
            )
            self.relation = joined.project(", ".join(proj_cols))
            return

        # Sequence value: align by position
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            expected = len(self)
            if len(value) != expected:
                raise ValueError(
                    f"Sequence length {len(value)} does not match relation length {expected}"
                )
            data = list(enumerate(value))
            m = con.values(data).project("column0 AS __pos__, column1 AS __val__").set_alias("m")
            lpos = l.project(
                f"row_number() OVER (ORDER BY {INDEX_COL}) - 1 AS __pos__, *"
            ).set_alias("l")
            joined = lpos.join(m, "l.__pos__ = m.__pos__", how="left")
            proj_cols = (
                [f"l.{INDEX_COL} AS {INDEX_COL}"]
                + [f'l."{c}" AS "{c}"' for c in other_cols]
                + ["m.__val__ AS " + col_name]
            )
            self.relation = joined.project(", ".join(proj_cols))
            return

        # Scalar value: broadcast
        value = self._to_sql_literal(value)
        # Build projection explicitly to avoid duplicate column names
        proj_expr = ", ".join(
            [f"{INDEX_COL} AS {INDEX_COL}"]
            + [f'"{c}" AS "{c}"' for c in other_cols]
            + [f'{value} AS "{col_name}"']
        )
        self.relation = l.project(proj_expr)

    def _assign_rows(self, key: Any, value: Any) -> None:
        # Only scalar RHS supported for row-wise assignment
        if isinstance(value, (RelationProxy, DuckDBPyRelation, Sequence)) and not isinstance(
            value, (str, bytes)
        ):
            raise NotImplementedError("Row-wise assignment currently supports only scalar values")

        value = self._to_sql_literal(value)
        l = self.relation.set_alias("l")
        cond = None
        joined = None

        # Mask is Relation/DuckDB relation
        if isinstance(key, (RelationProxy, DuckDBPyRelation)):
            m_rel = key.relation if isinstance(key, RelationProxy) else RelationProxy(key).relation
            m = m_rel.set_alias("m")
            m_cols = [c for c in m.columns if c != INDEX_COL]
            if not m_cols:
                # Treat as index-list relation: select rows where index exists in 'm'
                joined = l.join(m, f"l.{INDEX_COL} = m.{INDEX_COL}", how="left")
                cond = f"m.{INDEX_COL} IS NOT NULL"
            else:
                mask_col = m_cols[0]
                joined = l.join(m, f"l.{INDEX_COL} = m.{INDEX_COL}", how="left")
                cond = f'coalesce(m."{mask_col}", false)'

        # Mask is a Python sequence
        elif isinstance(key, Sequence) and not isinstance(key, str):
            # Boolean mask list
            if len(key) > 0 and all(type(x) is bool for x in key):
                data = list(enumerate(key))
                m = (
                    con.values(data)
                    .project("column0 AS __pos__, column1 AS __flag__")
                    .set_alias("m")
                )
                lpos = l.project(
                    f"row_number() OVER (ORDER BY {INDEX_COL}) - 1 AS __pos__, *"
                ).set_alias("l")
                joined = lpos.join(m, "l.__pos__ = m.__pos__", how="left")
                cond = "coalesce(m.__flag__, false)"
            # Index list (ints)
            elif all(isinstance(x, int) and not isinstance(x, bool) for x in key):
                idx_data = [(int(i), True) for i in key]
                m = (
                    con.values(idx_data)
                    .project("column0 AS idx, column1 AS __flag__")
                    .set_alias("m")
                )
                joined = l.join(m, f"l.{INDEX_COL} = m.idx", how="left")
                cond = "coalesce(m.__flag__, false)"
            else:
                raise TypeError("Unsupported sequence type for row-wise assignment")

        else:
            raise TypeError(f"Unsupported key type for row-wise assignment: {type(key)!r}")

        # Build projection with CASE expressions per data column
        proj_cols = [f"l.{INDEX_COL} AS {INDEX_COL}"]
        for c in self.columns:
            proj_cols.append(f'CASE WHEN "{cond}" THEN {value} ELSE l."{c}" END AS "{c}"')
        self.relation = joined.project(", ".join(proj_cols))

    def _binary_compare(self, other: Any, op: str) -> "RelationProxy":
        # Compare a single data column with a scalar or another single-column relation, aligned by index
        left = self.relation
        left_cols = [c for c in left.columns if c != INDEX_COL]
        if len(left_cols) != 1:
            raise ValueError(
                "Element-wise comparison requires a single data column on the left side"
            )

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
                raise ValueError(
                    "Element-wise comparison requires a single data column on the right side"
                )
            l = left.set_alias("l")
            r = right.set_alias("r")
            expr = f"(l.{left_cols[0]} {op} r.{r_cols[0]}) AS __mask__"
            proj = f"l.{INDEX_COL} AS {INDEX_COL}, {expr}"
            joined = l.join(r, f"l.{INDEX_COL} = r.{INDEX_COL}", how="left")
            return RelationProxy(joined.project(proj))

        # Other is a scalar: compare left column with literal
        value = self._to_sql_literal(other)
        expr = f"({left_cols[0]} {op} {value}) AS __mask__"
        proj = f"{INDEX_COL}, {expr}"
        return RelationProxy(left.project(proj))

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

    def _wrap_relation(self, value: Any) -> Any:
        return RelationProxy(value) if isinstance(value, DuckDBPyRelation) else value

    def clean_exec_graph(self):
        self.relation.execute().fetchall()

    def distinct(self) -> "RelationProxy":
        return RelationProxy(self.project(include_index=False).distinct())

    def df(self, limit: int | None = None) -> Any:
        rel = self.relation if limit is None else self.relation.limit(limit)
        df = rel.df()
        if INDEX_COL in df.columns:
            df = df.set_index(INDEX_COL)
            df.index.name = None
        return df

    def isnull(self) -> "RelationProxy":
        if len(self.columns) == 0:
            raise ValueError("No data columns to check for nulls")
        expr = f'("{self.columns[0]}" IS NULL) AS __mask__'
        return RelationProxy(self.relation.project(f"{INDEX_COL}, {expr}"))

    def project(
        self, projection: str = f"* EXCLUDE {INDEX_COL}", include_index: bool = True
    ) -> "RelationProxy":
        if include_index or len(self.columns) == 0:
            projection = self._ensure_index(projection)
        return self.relation.project(projection)

    def reset_index(self, **kwargs: Any) -> "RelationProxy":
        new_rel = self.relation.project(
            f"row_number() OVER () - 1 AS {INDEX_COL}, * EXCLUDE {INDEX_COL}"
        )
        return RelationProxy(new_rel)

    def drop(self, columns: Any) -> "RelationProxy":
        drop_set = {columns} if isinstance(columns, str) else set(columns)
        drop_set.discard(INDEX_COL)

        keep_cols = [INDEX_COL] + [c for c in self.columns if c not in drop_set]
        proj = ", ".join([f'"{c}"' for c in keep_cols])
        return RelationProxy(self.relation.project(proj))
