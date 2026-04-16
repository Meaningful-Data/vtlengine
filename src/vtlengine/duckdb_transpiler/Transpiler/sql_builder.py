"""Fluent SQL builder used by the DuckDB transpiler."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SQLBuilder:
    """Chainable builder for SELECT queries."""

    _select_cols: List[str] = field(default_factory=list)
    _from_clause: str = ""
    _from_alias: str = ""
    _joins: List[str] = field(default_factory=list)
    _where_conditions: List[str] = field(default_factory=list)
    _group_by_cols: List[str] = field(default_factory=list)
    _having_conditions: List[str] = field(default_factory=list)
    _order_by_cols: List[str] = field(default_factory=list)
    _limit_value: Optional[int] = None
    _distinct: bool = False
    _distinct_on: List[str] = field(default_factory=list)

    def select(self, *cols: str) -> "SQLBuilder":
        """Add columns to SELECT."""
        self._select_cols.extend(cols)
        return self

    def select_all(self) -> "SQLBuilder":
        """Select all columns."""
        self._select_cols.append("*")
        return self

    def distinct(self) -> "SQLBuilder":
        """Add DISTINCT."""
        self._distinct = True
        return self

    def distinct_on(self, *cols: str) -> "SQLBuilder":
        """Add DISTINCT ON columns."""
        self._distinct_on.extend(cols)
        return self

    def from_table(self, table: str, alias: str = "") -> "SQLBuilder":
        """Set FROM with a table reference."""
        self._from_clause = table
        self._from_alias = alias
        return self

    def from_subquery(self, subquery: str, alias: str = "t") -> "SQLBuilder":
        """Set FROM with a subquery."""
        self._from_clause = f"({subquery})"
        self._from_alias = alias
        return self

    def join(
        self,
        table: str,
        alias: str,
        on: str = "",
        using: Optional[List[str]] = None,
        join_type: str = "INNER",
    ) -> "SQLBuilder":
        """Add a JOIN clause."""
        op = join_type.replace("_join", "").upper()
        join_sql = f"{op} JOIN {table} AS {alias}"
        if using:
            using_cols = ", ".join([f'"{c}"' for c in using])
            join_sql += f" USING ({using_cols})"
        elif on:
            join_sql += f" ON {on}"
        self._joins.append(join_sql)
        return self

    def inner_join(
        self, table: str, alias: str, on: str = "", using: Optional[List[str]] = None
    ) -> "SQLBuilder":
        """Add INNER JOIN."""
        return self.join(table, alias, on, using, "INNER")

    def left_join(
        self, table: str, alias: str, on: str = "", using: Optional[List[str]] = None
    ) -> "SQLBuilder":
        """Add LEFT JOIN."""
        return self.join(table, alias, on, using, "LEFT")

    def cross_join(self, table: str, alias: str) -> "SQLBuilder":
        """Add CROSS JOIN."""
        self._joins.append(f"CROSS JOIN {table} AS {alias}")
        return self

    def where(self, condition: str) -> "SQLBuilder":
        """Add a WHERE condition."""
        self._where_conditions.append(condition)
        return self

    def where_all(self, conditions: List[str]) -> "SQLBuilder":
        """Add multiple WHERE conditions."""
        self._where_conditions.extend(conditions)
        return self

    def group_by(self, *cols: str) -> "SQLBuilder":
        """Add GROUP BY columns."""
        self._group_by_cols.extend(cols)
        return self

    def having(self, condition: str) -> "SQLBuilder":
        """Add a HAVING condition."""
        self._having_conditions.append(condition)
        return self

    def order_by(self, *cols: str) -> "SQLBuilder":
        """Add ORDER BY columns."""
        self._order_by_cols.extend(cols)
        return self

    def limit(self, n: int) -> "SQLBuilder":
        """Set LIMIT."""
        self._limit_value = n
        return self

    def build(self) -> str:
        """Build the SQL query string."""
        parts: List[str] = []

        # SELECT
        select_prefix = "SELECT"
        if self._distinct_on:
            distinct_cols = ", ".join(self._distinct_on)
            select_prefix = f"SELECT DISTINCT ON ({distinct_cols})"
        elif self._distinct:
            select_prefix = "SELECT DISTINCT"

        if self._select_cols:
            parts.append(f"{select_prefix} {', '.join(self._select_cols)}")
        else:
            parts.append(f"{select_prefix} *")

        # FROM
        if self._from_clause:
            if self._from_alias:
                parts.append(f"FROM {self._from_clause} AS {self._from_alias}")
            else:
                parts.append(f"FROM {self._from_clause}")

        # JOIN
        parts.extend(self._joins)

        # WHERE
        if self._where_conditions:
            parts.append(f"WHERE {' AND '.join(self._where_conditions)}")

        # GROUP BY
        if self._group_by_cols:
            parts.append(f"GROUP BY {', '.join(self._group_by_cols)}")

        # HAVING
        if self._having_conditions:
            parts.append(f"HAVING {' AND '.join(self._having_conditions)}")

        # ORDER BY
        if self._order_by_cols:
            parts.append(f"ORDER BY {', '.join(self._order_by_cols)}")

        # LIMIT
        if self._limit_value is not None:
            parts.append(f"LIMIT {self._limit_value}")

        return " ".join(parts)

    def reset(self) -> "SQLBuilder":
        """Reset the builder state."""
        self._select_cols = []
        self._from_clause = ""
        self._from_alias = ""
        self._joins = []
        self._where_conditions = []
        self._group_by_cols = []
        self._having_conditions = []
        self._order_by_cols = []
        self._limit_value = None
        self._distinct = False
        self._distinct_on = []
        return self


class CTEBuilder:
    """Builder for WITH ... SELECT queries using named CTEs."""

    def __init__(self) -> None:
        self._ctes: List[tuple[str, str, bool]] = []  # (name, sql, recursive)

    def cte(self, name: str, sql: str) -> "CTEBuilder":
        """Add a regular CTE."""
        self._ctes.append((name, sql.strip(), False))
        return self

    def recursive_cte(self, name: str, columns: str, seed: str, step: str) -> "CTEBuilder":
        """Add a RECURSIVE CTE with seed UNION ALL step."""
        sql = f"{seed.strip()}\n    UNION ALL\n    {step.strip()}"
        self._ctes.append((f"{name}({columns})", sql, True))
        return self

    def select(self, final_sql: str) -> str:
        """Build the full WITH ... SELECT statement."""
        if not self._ctes:
            return final_sql.strip()
        has_recursive = any(r for _, _, r in self._ctes)
        keyword = "WITH RECURSIVE" if has_recursive else "WITH"
        parts = ["{} AS (\n    {}\n)".format(name, sql) for name, sql, _ in self._ctes]
        sep = ",\n"
        return "{} {}\n{}".format(keyword, sep.join(parts), final_sql.strip())


def quote_identifier(name: str) -> str:
    """Quote a SQL identifier."""
    return f'"{name}"'


def quote_identifiers(names: List[str]) -> List[str]:
    """Quote multiple SQL identifiers."""
    return [quote_identifier(n) for n in names]


def build_column_expr(col: str, alias: str = "", table_alias: str = "") -> str:
    """Build a column expression with optional alias and table prefix."""
    col_ref = f'{table_alias}."{col}"' if table_alias else f'"{col}"'
    if alias:
        return f'{col_ref} AS "{alias}"'
    return col_ref


def build_function_expr(func: str, col: str, alias: str = "") -> str:
    """Build a function expression."""
    expr = f'{func}("{col}")'
    if alias:
        return f'{expr} AS "{alias}"'
    return expr


def build_binary_expr(left: str, op: str, right: str, alias: str = "") -> str:
    """Build a binary expression."""
    expr = f"({left} {op} {right})"
    if alias:
        return f'{expr} AS "{alias}"'
    return expr
