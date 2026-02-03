"""
SQL Builder for DuckDB Transpiler.

This module provides a fluent SQL query builder for constructing SQL statements
in a readable and maintainable way.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SQLBuilder:
    """
    Fluent SQL query builder.

    Provides a chainable interface for building SQL SELECT statements
    with proper formatting and component management.

    Example:
        >>> builder = SQLBuilder()
        >>> sql = (builder
        ...     .select('"Id_1"', '"Me_1" * 2 AS "Me_1"')
        ...     .from_table('"DS_1"')
        ...     .where('"Me_1" > 10')
        ...     .build())
        >>> print(sql)
        SELECT "Id_1", "Me_1" * 2 AS "Me_1" FROM "DS_1" WHERE "Me_1" > 10
    """

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
        """
        Add columns to SELECT clause.

        Args:
            *cols: Column expressions to select.

        Returns:
            Self for chaining.
        """
        self._select_cols.extend(cols)
        return self

    def select_all(self) -> "SQLBuilder":
        """
        Select all columns (*).

        Returns:
            Self for chaining.
        """
        self._select_cols.append("*")
        return self

    def distinct(self) -> "SQLBuilder":
        """
        Add DISTINCT modifier.

        Returns:
            Self for chaining.
        """
        self._distinct = True
        return self

    def distinct_on(self, *cols: str) -> "SQLBuilder":
        """
        Add DISTINCT ON clause (DuckDB/PostgreSQL specific).

        Args:
            *cols: Columns for DISTINCT ON.

        Returns:
            Self for chaining.
        """
        self._distinct_on.extend(cols)
        return self

    def from_table(self, table: str, alias: str = "") -> "SQLBuilder":
        """
        Set the FROM clause with a table reference.

        Args:
            table: Table name or subquery.
            alias: Optional table alias.

        Returns:
            Self for chaining.
        """
        self._from_clause = table
        self._from_alias = alias
        return self

    def from_subquery(self, subquery: str, alias: str = "t") -> "SQLBuilder":
        """
        Set the FROM clause with a subquery.

        Args:
            subquery: SQL subquery.
            alias: Subquery alias (default: "t").

        Returns:
            Self for chaining.
        """
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
        """
        Add a JOIN clause.

        Args:
            table: Table name or subquery to join.
            alias: Table alias.
            on: ON condition (mutually exclusive with using).
            using: USING columns (mutually exclusive with on).
            join_type: Type of join (INNER, LEFT, RIGHT, FULL, CROSS).

        Returns:
            Self for chaining.
        """
        join_sql = f"{join_type} JOIN {table} AS {alias}"
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
        """
        Add a WHERE condition.

        Multiple conditions are combined with AND.

        Args:
            condition: WHERE condition.

        Returns:
            Self for chaining.
        """
        self._where_conditions.append(condition)
        return self

    def where_all(self, conditions: List[str]) -> "SQLBuilder":
        """
        Add multiple WHERE conditions (AND).

        Args:
            conditions: List of conditions.

        Returns:
            Self for chaining.
        """
        self._where_conditions.extend(conditions)
        return self

    def group_by(self, *cols: str) -> "SQLBuilder":
        """
        Add GROUP BY columns.

        Args:
            *cols: Columns to group by.

        Returns:
            Self for chaining.
        """
        self._group_by_cols.extend(cols)
        return self

    def having(self, condition: str) -> "SQLBuilder":
        """
        Add a HAVING condition.

        Multiple conditions are combined with AND.

        Args:
            condition: HAVING condition.

        Returns:
            Self for chaining.
        """
        self._having_conditions.append(condition)
        return self

    def order_by(self, *cols: str) -> "SQLBuilder":
        """
        Add ORDER BY columns.

        Args:
            *cols: Columns to order by (can include ASC/DESC).

        Returns:
            Self for chaining.
        """
        self._order_by_cols.extend(cols)
        return self

    def limit(self, n: int) -> "SQLBuilder":
        """
        Set LIMIT clause.

        Args:
            n: Maximum number of rows.

        Returns:
            Self for chaining.
        """
        self._limit_value = n
        return self

    def build(self) -> str:
        """
        Build the SQL query string.

        Returns:
            Complete SQL query string.
        """
        parts: List[str] = []

        # SELECT clause
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

        # FROM clause
        if self._from_clause:
            if self._from_alias:
                parts.append(f"FROM {self._from_clause} AS {self._from_alias}")
            else:
                parts.append(f"FROM {self._from_clause}")

        # JOINs
        parts.extend(self._joins)

        # WHERE clause
        if self._where_conditions:
            parts.append(f"WHERE {' AND '.join(self._where_conditions)}")

        # GROUP BY clause
        if self._group_by_cols:
            parts.append(f"GROUP BY {', '.join(self._group_by_cols)}")

        # HAVING clause
        if self._having_conditions:
            parts.append(f"HAVING {' AND '.join(self._having_conditions)}")

        # ORDER BY clause
        if self._order_by_cols:
            parts.append(f"ORDER BY {', '.join(self._order_by_cols)}")

        # LIMIT clause
        if self._limit_value is not None:
            parts.append(f"LIMIT {self._limit_value}")

        return " ".join(parts)

    def reset(self) -> "SQLBuilder":
        """
        Reset the builder to initial state.

        Returns:
            Self for chaining.
        """
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


def quote_identifier(name: str) -> str:
    """
    Quote a SQL identifier.

    Args:
        name: Identifier name.

    Returns:
        Quoted identifier.
    """
    return f'"{name}"'


def quote_identifiers(names: List[str]) -> List[str]:
    """
    Quote multiple SQL identifiers.

    Args:
        names: List of identifier names.

    Returns:
        List of quoted identifiers.
    """
    return [quote_identifier(n) for n in names]


def build_column_expr(col: str, alias: str = "", table_alias: str = "") -> str:
    """
    Build a column expression with optional alias and table prefix.

    Args:
        col: Column name.
        alias: Optional column alias.
        table_alias: Optional table alias prefix.

    Returns:
        Column expression string.
    """
    col_ref = f'{table_alias}."{col}"' if table_alias else f'"{col}"'
    if alias:
        return f'{col_ref} AS "{alias}"'
    return col_ref


def build_function_expr(func: str, col: str, alias: str = "") -> str:
    """
    Build a function expression.

    Args:
        func: SQL function name.
        col: Column to apply function to.
        alias: Optional result alias.

    Returns:
        Function expression string.
    """
    expr = f'{func}("{col}")'
    if alias:
        return f'{expr} AS "{alias}"'
    return expr


def build_binary_expr(left: str, op: str, right: str, alias: str = "") -> str:
    """
    Build a binary expression.

    Args:
        left: Left operand.
        op: Operator.
        right: Right operand.
        alias: Optional result alias.

    Returns:
        Binary expression string.
    """
    expr = f"({left} {op} {right})"
    if alias:
        return f'{expr} AS "{alias}"'
    return expr
