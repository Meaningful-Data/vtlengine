"""
Operator Registry for SQL Transpiler.

This module provides a centralized registry for VTL operators and their SQL mappings.
It decouples operator definitions from the transpiler logic, making it easier to:
- Add new operators
- Customize operator behavior
- Test operator mappings independently

Usage:
    from vtlengine.duckdb_transpiler.Transpiler.operators import (
        registry,
        OperatorCategory,
    )

    # Get SQL for binary operator
    sql = registry.binary.generate("+", "a", "b")  # Returns "(a + b)"

    # Get SQL for unary operator
    sql = registry.unary.generate("ceil", "x")  # Returns "CEIL(x)"

    # Check if operator is registered
    if registry.binary.is_registered("+"):
        ...
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import vtlengine.AST.Grammar.tokens as tokens


class OperatorCategory(Enum):
    """Categories of VTL operators."""

    BINARY = auto()  # Two operands: a + b
    UNARY = auto()  # One operand: ceil(x)
    AGGREGATE = auto()  # Aggregation: sum(x)
    ANALYTIC = auto()  # Window functions: sum(x) over (...)
    PARAMETERIZED = auto()  # With parameters: round(x, 2)
    SET = auto()  # Set operations: union, intersect


@dataclass
class SQLOperator:
    """
    SQL operator definition.

    Attributes:
        sql_template: SQL template string with placeholders.
            - For binary: "{0} + {1}" where {0}=left, {1}=right
            - For unary function: "CEIL({0})"
            - For unary prefix: "{op}{0}" (e.g., "-{0}")
        category: The operator category.
        is_prefix: For unary operators, whether it's prefix (e.g., -x) vs function (e.g., CEIL(x)).
        dataset_handler: Optional callback for dataset-level operations.
        requires_context: Whether the operator needs transpiler context.
        custom_generator: Optional custom SQL generator function.
    """

    sql_template: str
    category: OperatorCategory
    is_prefix: bool = False
    dataset_handler: Optional[Callable[..., Any]] = None
    requires_context: bool = False
    custom_generator: Optional[Callable[..., str]] = None

    def generate(self, *operands: str) -> str:
        """
        Generate SQL from the template with the given operands.

        Args:
            *operands: The SQL expressions for each operand.

        Returns:
            The generated SQL expression.
        """
        if self.custom_generator:
            return self.custom_generator(*operands)

        if self.category == OperatorCategory.BINARY:
            if len(operands) < 2:
                raise ValueError(f"Binary operator requires 2 operands, got {len(operands)}")
            return self.sql_template.format(operands[0], operands[1])

        elif self.category == OperatorCategory.UNARY:
            if len(operands) < 1:
                raise ValueError(f"Unary operator requires 1 operand, got {len(operands)}")
            if self.is_prefix:
                # Template like "{op}{0}" for prefix operators
                return self.sql_template.format(operands[0])
            # Function style: FUNC(operand)
            return self.sql_template.format(operands[0])

        elif self.category in (OperatorCategory.AGGREGATE, OperatorCategory.ANALYTIC):
            if len(operands) < 1:
                raise ValueError(f"Aggregate operator requires 1 operand, got {len(operands)}")
            return self.sql_template.format(operands[0])

        elif self.category == OperatorCategory.PARAMETERIZED:
            # Template uses numbered placeholders: {0}, {1}, {2}, ...
            return self.sql_template.format(*operands)

        elif self.category == OperatorCategory.SET:
            # Set operations join multiple queries
            sql_op = self.sql_template
            return f" {sql_op} ".join([f"({q})" for q in operands])

        # Default: use format with all operands
        return self.sql_template.format(*operands)


@dataclass
class OperatorRegistry:
    """
    Registry for SQL operators of a specific category.

    Provides registration, lookup, and SQL generation for operators.
    """

    category: OperatorCategory
    _operators: Dict[str, SQLOperator] = field(default_factory=dict)

    def register(self, vtl_token: str, operator: SQLOperator) -> "OperatorRegistry":
        """
        Register an operator.

        Args:
            vtl_token: The VTL operator token (from tokens.py).
            operator: The SQLOperator definition.

        Returns:
            Self for chaining.
        """
        self._operators[vtl_token] = operator
        return self

    def register_simple(
        self,
        vtl_token: str,
        sql_template: str,
        is_prefix: bool = False,
    ) -> "OperatorRegistry":
        """
        Register a simple operator with just a template.

        Args:
            vtl_token: The VTL operator token.
            sql_template: The SQL template string.
            is_prefix: For unary operators, whether it's prefix style.

        Returns:
            Self for chaining.
        """
        operator = SQLOperator(
            sql_template=sql_template,
            category=self.category,
            is_prefix=is_prefix,
        )
        self._operators[vtl_token] = operator
        return self

    def get(self, vtl_token: str) -> Optional[SQLOperator]:
        """
        Get an operator by VTL token.

        Args:
            vtl_token: The VTL operator token.

        Returns:
            The SQLOperator or None if not registered.
        """
        return self._operators.get(vtl_token)

    def is_registered(self, vtl_token: str) -> bool:
        """Check if an operator is registered."""
        return vtl_token in self._operators

    def generate(self, vtl_token: str, *operands: str) -> str:
        """
        Generate SQL for an operator.

        Args:
            vtl_token: The VTL operator token.
            *operands: The SQL expressions for operands.

        Returns:
            The generated SQL.

        Raises:
            ValueError: If operator is not registered.
        """
        operator = self.get(vtl_token)
        if not operator:
            raise ValueError(f"Unknown operator: {vtl_token}")
        return operator.generate(*operands)

    def get_sql_symbol(self, vtl_token: str) -> Optional[str]:
        """
        Get the SQL symbol/function name for an operator.

        For simple operators, extracts the SQL part from the template.

        Args:
            vtl_token: The VTL operator token.

        Returns:
            The SQL symbol or None if not registered.
        """
        operator = self.get(vtl_token)
        if not operator:
            return None

        template = operator.sql_template

        # For binary operators like "({0} + {1})", extract "+"
        if operator.category == OperatorCategory.BINARY:
            cleaned = (
                template.replace("{0}", "").replace("{1}", "").replace("(", "").replace(")", "")
            )
            return cleaned.strip()

        # For prefix unary operators like "+{0}", "-{0}", "NOT {0}"
        if operator.is_prefix:
            return template.replace("{0}", "").strip()

        # For function-style like "CEIL({0})", "SUM({0})", extract "CEIL", "SUM"
        if "({" in template:
            return template.split("(")[0]

        # For templates like "RANK()" (no placeholder), extract "RANK"
        if template.endswith("()"):
            return template[:-2]

        return template

    def list_operators(self) -> List[Tuple[str, str]]:
        """
        List all registered operators.

        Returns:
            List of (vtl_token, sql_template) tuples.
        """
        return [(token, op.sql_template) for token, op in self._operators.items()]


@dataclass
class SQLOperatorRegistries:
    """
    Collection of all operator registries.

    Provides centralized access to operators by category.
    """

    binary: OperatorRegistry = field(
        default_factory=lambda: OperatorRegistry(OperatorCategory.BINARY)
    )
    unary: OperatorRegistry = field(
        default_factory=lambda: OperatorRegistry(OperatorCategory.UNARY)
    )
    aggregate: OperatorRegistry = field(
        default_factory=lambda: OperatorRegistry(OperatorCategory.AGGREGATE)
    )
    analytic: OperatorRegistry = field(
        default_factory=lambda: OperatorRegistry(OperatorCategory.ANALYTIC)
    )
    parameterized: OperatorRegistry = field(
        default_factory=lambda: OperatorRegistry(OperatorCategory.PARAMETERIZED)
    )
    set_ops: OperatorRegistry = field(
        default_factory=lambda: OperatorRegistry(OperatorCategory.SET)
    )

    def get_by_category(self, category: OperatorCategory) -> OperatorRegistry:
        """Get registry by category."""
        mapping = {
            OperatorCategory.BINARY: self.binary,
            OperatorCategory.UNARY: self.unary,
            OperatorCategory.AGGREGATE: self.aggregate,
            OperatorCategory.ANALYTIC: self.analytic,
            OperatorCategory.PARAMETERIZED: self.parameterized,
            OperatorCategory.SET: self.set_ops,
        }
        return mapping[category]

    def find_operator(self, vtl_token: str) -> Optional[Tuple[OperatorCategory, SQLOperator]]:
        """
        Find an operator across all registries.

        Args:
            vtl_token: The VTL operator token.

        Returns:
            Tuple of (category, operator) or None if not found.
        """
        for category in OperatorCategory:
            registry = self.get_by_category(category)
            operator = registry.get(vtl_token)
            if operator:
                return (category, operator)
        return None


def _create_default_registries() -> SQLOperatorRegistries:
    """
    Create and populate the default operator registries.

    Returns:
        Fully populated SQLOperatorRegistries instance.
    """
    registries = SQLOperatorRegistries()

    # =========================================================================
    # Binary Operators
    # =========================================================================

    # Arithmetic
    registries.binary.register_simple(tokens.PLUS, "({0} + {1})")
    registries.binary.register_simple(tokens.MINUS, "({0} - {1})")
    registries.binary.register_simple(tokens.MULT, "({0} * {1})")
    registries.binary.register_simple(tokens.DIV, "({0} / {1})")
    registries.binary.register_simple(tokens.MOD, "({0} % {1})")

    # Comparison
    registries.binary.register_simple(tokens.EQ, "({0} = {1})")
    registries.binary.register_simple(tokens.NEQ, "({0} <> {1})")
    registries.binary.register_simple(tokens.GT, "({0} > {1})")
    registries.binary.register_simple(tokens.LT, "({0} < {1})")
    registries.binary.register_simple(tokens.GTE, "({0} >= {1})")
    registries.binary.register_simple(tokens.LTE, "({0} <= {1})")

    # Logical
    registries.binary.register_simple(tokens.AND, "({0} AND {1})")
    registries.binary.register_simple(tokens.OR, "({0} OR {1})")
    registries.binary.register(
        tokens.XOR,
        SQLOperator(
            sql_template="",
            category=OperatorCategory.BINARY,
            custom_generator=lambda a, b: f"(({a} AND NOT {b}) OR (NOT {a} AND {b}))",
        ),
    )
    registries.binary.register_simple(tokens.IN, "({0} IN {1})")
    registries.binary.register_simple(tokens.NOT_IN, "({0} NOT IN {1})")

    # String
    registries.binary.register_simple(tokens.CONCAT, "({0} || {1})")

    # Numeric functions (come through BinOp AST)
    registries.binary.register_simple(tokens.POWER, "POWER({0}, {1})")
    registries.binary.register_simple(tokens.LOG, "LOG({1}, {0})")  # DuckDB: LOG(base, value)

    # Conditional (come through BinOp AST)
    registries.binary.register_simple(tokens.NVL, "COALESCE({0}, {1})")

    # Date/Time
    registries.binary.register_simple(tokens.DATEDIFF, "ABS(DATE_DIFF('day', {0}, {1}))")

    # String matching
    registries.binary.register_simple(tokens.CHARSET_MATCH, "regexp_full_match({0}, {1})")

    # =========================================================================
    # Unary Operators
    # =========================================================================

    # Arithmetic prefix
    registries.unary.register_simple(tokens.PLUS, "+{0}", is_prefix=True)
    registries.unary.register_simple(tokens.MINUS, "-{0}", is_prefix=True)

    # Arithmetic functions
    registries.unary.register_simple(tokens.CEIL, "CEIL({0})")
    registries.unary.register_simple(tokens.FLOOR, "FLOOR({0})")
    registries.unary.register_simple(tokens.ABS, "ABS({0})")
    registries.unary.register_simple(tokens.EXP, "EXP({0})")
    registries.unary.register_simple(tokens.LN, "LN({0})")
    registries.unary.register_simple(tokens.SQRT, "SQRT({0})")

    # Logical
    registries.unary.register_simple(tokens.NOT, "NOT {0}", is_prefix=True)

    # String functions
    registries.unary.register_simple(tokens.LEN, "LENGTH({0})")
    registries.unary.register_simple(tokens.TRIM, "TRIM({0})")
    registries.unary.register_simple(tokens.LTRIM, "LTRIM({0})")
    registries.unary.register_simple(tokens.RTRIM, "RTRIM({0})")
    registries.unary.register_simple(tokens.UCASE, "UPPER({0})")
    registries.unary.register_simple(tokens.LCASE, "LOWER({0})")

    # Null check
    registries.unary.register_simple(tokens.ISNULL, "({0} IS NULL)")

    # Time extraction functions
    registries.unary.register_simple(tokens.YEAR, "YEAR({0})")
    registries.unary.register_simple(tokens.MONTH, "MONTH({0})")
    registries.unary.register_simple(tokens.DAYOFMONTH, "DAY({0})")
    registries.unary.register_simple(tokens.DAYOFYEAR, "DAYOFYEAR({0})")

    # =========================================================================
    # Aggregate Operators
    # =========================================================================

    registries.aggregate.register_simple(tokens.SUM, "SUM({0})")
    registries.aggregate.register_simple(tokens.AVG, "AVG({0})")
    registries.aggregate.register_simple(tokens.COUNT, "COUNT({0})")
    registries.aggregate.register_simple(tokens.MIN, "MIN({0})")
    registries.aggregate.register_simple(tokens.MAX, "MAX({0})")
    registries.aggregate.register_simple(tokens.MEDIAN, "MEDIAN({0})")
    registries.aggregate.register_simple(tokens.STDDEV_POP, "STDDEV_POP({0})")
    registries.aggregate.register_simple(tokens.STDDEV_SAMP, "STDDEV_SAMP({0})")
    registries.aggregate.register_simple(tokens.VAR_POP, "VAR_POP({0})")
    registries.aggregate.register_simple(tokens.VAR_SAMP, "VAR_SAMP({0})")

    # =========================================================================
    # Analytic (Window) Operators
    # =========================================================================

    # Aggregate functions can also be used as analytics
    registries.analytic.register_simple(tokens.SUM, "SUM({0})")
    registries.analytic.register_simple(tokens.AVG, "AVG({0})")
    registries.analytic.register_simple(tokens.COUNT, "COUNT({0})")
    registries.analytic.register_simple(tokens.MIN, "MIN({0})")
    registries.analytic.register_simple(tokens.MAX, "MAX({0})")
    registries.analytic.register_simple(tokens.MEDIAN, "MEDIAN({0})")
    registries.analytic.register_simple(tokens.STDDEV_POP, "STDDEV_POP({0})")
    registries.analytic.register_simple(tokens.STDDEV_SAMP, "STDDEV_SAMP({0})")
    registries.analytic.register_simple(tokens.VAR_POP, "VAR_POP({0})")
    registries.analytic.register_simple(tokens.VAR_SAMP, "VAR_SAMP({0})")

    # Pure analytic functions
    registries.analytic.register_simple(tokens.FIRST_VALUE, "FIRST_VALUE({0})")
    registries.analytic.register_simple(tokens.LAST_VALUE, "LAST_VALUE({0})")
    registries.analytic.register_simple(tokens.LAG, "LAG({0})")
    registries.analytic.register_simple(tokens.LEAD, "LEAD({0})")
    registries.analytic.register_simple(tokens.RANK, "RANK()")  # RANK takes no argument
    registries.analytic.register_simple(tokens.RATIO_TO_REPORT, "RATIO_TO_REPORT({0})")

    # =========================================================================
    # Parameterized Operators
    # =========================================================================

    # Comparison
    registries.parameterized.register_simple(tokens.BETWEEN, "({0} BETWEEN {1} AND {2})")

    # Single parameter operations
    # DuckDB does not support ROUND/TRUNC(DECIMAL, col) with non-constant
    # precision.  Casting the value to DOUBLE avoids this limitation.
    # VTL semantics: null precision defaults to 0.
    def _round_generator(*args: Optional[str]) -> str:
        precision = "0" if (len(args) < 2 or args[1] is None) else str(args[1])
        return f"ROUND(CAST({args[0]} AS DOUBLE), COALESCE(CAST({precision} AS INTEGER), 0))"

    registries.parameterized.register(
        tokens.ROUND,
        SQLOperator(
            sql_template="ROUND({0}, CAST({1} AS INTEGER))",
            category=OperatorCategory.PARAMETERIZED,
            custom_generator=_round_generator,
        ),
    )

    def _trunc_generator(*args: Optional[str]) -> str:
        precision = "0" if (len(args) < 2 or args[1] is None) else str(args[1])
        return f"TRUNC(CAST({args[0]} AS DOUBLE), COALESCE(CAST({precision} AS INTEGER), 0))"

    registries.parameterized.register(
        tokens.TRUNC,
        SQLOperator(
            sql_template="TRUNC({0}, CAST({1} AS INTEGER))",
            category=OperatorCategory.PARAMETERIZED,
            custom_generator=_trunc_generator,
        ),
    )

    def _instr_generator(*args: Optional[str]) -> str:
        """Generate INSTR SQL emulating VTL instr(string, pattern, start, occurrence).

        DuckDB's INSTR only supports 2 args: INSTR(string, pattern).
        VTL's instr supports: instr(string, pattern, start=1, occurrence=1).
        We always use vtl_instr macro for consistency (handles null pattern → 0).
        """
        # Build args with defaults for missing values
        params = []
        params.append(args[0] if len(args) > 0 and args[0] is not None else "NULL")
        params.append(args[1] if len(args) > 1 and args[1] is not None else "NULL")
        params.append(args[2] if len(args) > 2 and args[2] is not None else "1")
        params.append(args[3] if len(args) > 3 and args[3] is not None else "1")

        return f"vtl_instr({', '.join(params)})"

    registries.parameterized.register(
        tokens.INSTR,
        SQLOperator(
            sql_template="INSTR({0}, {1})",
            category=OperatorCategory.PARAMETERIZED,
            custom_generator=_instr_generator,
        ),
    )
    registries.parameterized.register_simple(tokens.LOG, "LOG({1}, {0})")  # LOG(base, value)
    registries.parameterized.register_simple(tokens.POWER, "POWER({0}, {1})")

    # Multi-parameter operations (variable args)
    def _substr_generator(*args: Optional[str]) -> str:
        """Generate SUBSTR SQL handling None args (VTL defaults: start=1)."""
        if len(args) == 1:
            # Just the string, no start/length → return as-is
            return str(args[0])
        filtered = []
        for i, a in enumerate(args):
            if a is None:
                if i == 1:  # start position defaults to 1
                    filtered.append("1")
                # if i == 2 (length) is None, omit it
            else:
                filtered.append(str(a))
        return f"SUBSTR({', '.join(filtered)})"

    registries.parameterized.register(
        tokens.SUBSTR,
        SQLOperator(
            sql_template="SUBSTR({0}, {1}, {2})",
            category=OperatorCategory.PARAMETERIZED,
            custom_generator=_substr_generator,
        ),
    )

    def _replace_generator(*args: Optional[str]) -> str:
        """Generate REPLACE SQL. DuckDB requires 3 args; VTL allows 2.

        VTL replace(op, s1, s2):
        - s1=null → return op unchanged
        - s2=null → replace s1 with empty string
        - only 2 args → replace s1 with empty string
        """
        # args order: string, pattern, replacement
        if len(args) < 2 or args[1] is None:
            # Pattern is null/missing → return original string unchanged
            return str(args[0]) if args else "''"
        string_arg = str(args[0])
        pattern_arg = str(args[1])
        if len(args) < 3 or args[2] is None:
            # Replacement is null/missing → replace with empty string
            return f"REPLACE({string_arg}, {pattern_arg}, '')"
        return f"REPLACE({string_arg}, {pattern_arg}, {args[2]})"

    registries.parameterized.register(
        tokens.REPLACE,
        SQLOperator(
            sql_template="REPLACE({0}, {1}, {2})",
            category=OperatorCategory.PARAMETERIZED,
            custom_generator=_replace_generator,
        ),
    )

    # =========================================================================
    # Set Operations
    # =========================================================================

    registries.set_ops.register_simple(tokens.UNION, "UNION ALL")
    registries.set_ops.register_simple(tokens.INTERSECT, "INTERSECT")
    registries.set_ops.register_simple(tokens.SETDIFF, "EXCEPT")
    # SYMDIFF requires special handling (not a simple SQL operator)
    registries.set_ops.register(
        tokens.SYMDIFF,
        SQLOperator(
            sql_template="SYMDIFF",
            category=OperatorCategory.SET,
            requires_context=True,  # Needs custom handling
        ),
    )

    return registries


# Global registry instance
registry = _create_default_registries()


# =========================================================================
# Convenience Functions
# =========================================================================


def generate_sql(vtl_token: str, *args: str) -> str:
    """
    Generate SQL for a given VTL operator token and operands.

    Searches all registries for the token and delegates to the operator.
    Prefer using registry.<category>.generate() directly from the visitor
    when the category is known (e.g., registry.unary.generate(token, operand)).

    Args:
        vtl_token: The VTL operator token.
        *args: The SQL expressions for operands.

    Returns:
        The generated SQL expression.

    Raises:
        ValueError: If operator is not registered.
    """
    result = registry.find_operator(vtl_token)
    if result is None:
        raise ValueError(f"Unknown operator: {vtl_token}")
    _, op = result
    return op.generate(*args)


def get_sql_operator_symbol(vtl_token: str) -> Optional[str]:
    """
    Get the raw SQL operator symbol for a VTL token.

    This returns just the SQL operator/function name without placeholders.

    Args:
        vtl_token: The VTL operator token.

    Returns:
        The SQL symbol (e.g., "+" for PLUS, "CEIL" for CEIL) or None.
    """
    # Check each registry
    for reg in [
        registry.binary,
        registry.unary,
        registry.aggregate,
        registry.analytic,
        registry.parameterized,
        registry.set_ops,
    ]:
        symbol = reg.get_sql_symbol(vtl_token)
        if symbol:
            return symbol
    return None


def is_operator_registered(vtl_token: str) -> bool:
    """
    Check if an operator is registered in any registry.

    Args:
        vtl_token: The VTL operator token.

    Returns:
        True if operator is registered.
    """
    return registry.find_operator(vtl_token) is not None


def get_binary_sql(vtl_token: str, left: str, right: str) -> str:
    """Convenience: generate SQL for a binary operator."""
    return registry.binary.generate(vtl_token, left, right)


def get_unary_sql(vtl_token: str, operand: str) -> str:
    """Convenience: generate SQL for a unary operator."""
    return registry.unary.generate(vtl_token, operand)


def get_aggregate_sql(vtl_token: str, operand: str) -> str:
    """Convenience: generate SQL for an aggregate operator."""
    return registry.aggregate.generate(vtl_token, operand)


# =========================================================================
# Type Mappings (moved from Transpiler)
# =========================================================================

VTL_TO_DUCKDB_TYPES: Dict[str, str] = {
    "Integer": "BIGINT",
    "Number": "DOUBLE",
    "String": "VARCHAR",
    "Boolean": "BOOLEAN",
    "Date": "DATE",
    "TimePeriod": "VARCHAR",
    "TimeInterval": "VARCHAR",
    "Duration": "VARCHAR",
    "Null": "VARCHAR",
}


def get_duckdb_type(vtl_type: str) -> str:
    """
    Map VTL type name to DuckDB SQL type.

    Args:
        vtl_type: VTL type name (e.g., "Integer", "Number").

    Returns:
        DuckDB SQL type (e.g., "BIGINT", "DOUBLE").
    """
    return VTL_TO_DUCKDB_TYPES.get(vtl_type, "VARCHAR")
