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

from vtlengine.AST.Grammar.tokens import (
    ABS,
    AND,
    AVG,
    CEIL,
    CONCAT,
    COUNT,
    DAYOFMONTH,
    DAYOFYEAR,
    DIV,
    EQ,
    EXP,
    FIRST_VALUE,
    FLOOR,
    GT,
    GTE,
    INSTR,
    INTERSECT,
    LAG,
    LAST_VALUE,
    LCASE,
    LEAD,
    LEN,
    LN,
    LOG,
    LT,
    LTE,
    LTRIM,
    MAX,
    MEDIAN,
    MIN,
    MINUS,
    MOD,
    MONTH,
    MULT,
    NEQ,
    NOT,
    NVL,
    OR,
    PLUS,
    POWER,
    RANK,
    RATIO_TO_REPORT,
    REPLACE,
    ROUND,
    RTRIM,
    SETDIFF,
    SQRT,
    STDDEV_POP,
    STDDEV_SAMP,
    SUBSTR,
    SUM,
    SYMDIFF,
    TRIM,
    TRUNC,
    UCASE,
    UNION,
    VAR_POP,
    VAR_SAMP,
    XOR,
    YEAR,
)


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
    registries.binary.register_simple(PLUS, "({0} + {1})")
    registries.binary.register_simple(MINUS, "({0} - {1})")
    registries.binary.register_simple(MULT, "({0} * {1})")
    registries.binary.register_simple(DIV, "({0} / {1})")
    registries.binary.register_simple(MOD, "({0} % {1})")

    # Comparison
    registries.binary.register_simple(EQ, "({0} = {1})")
    registries.binary.register_simple(NEQ, "({0} <> {1})")
    registries.binary.register_simple(GT, "({0} > {1})")
    registries.binary.register_simple(LT, "({0} < {1})")
    registries.binary.register_simple(GTE, "({0} >= {1})")
    registries.binary.register_simple(LTE, "({0} <= {1})")

    # Logical
    registries.binary.register_simple(AND, "({0} AND {1})")
    registries.binary.register_simple(OR, "({0} OR {1})")
    registries.binary.register_simple(XOR, "({0} XOR {1})")

    # String
    registries.binary.register_simple(CONCAT, "({0} || {1})")

    # =========================================================================
    # Unary Operators
    # =========================================================================

    # Arithmetic prefix
    registries.unary.register_simple(PLUS, "+{0}", is_prefix=True)
    registries.unary.register_simple(MINUS, "-{0}", is_prefix=True)

    # Arithmetic functions
    registries.unary.register_simple(CEIL, "CEIL({0})")
    registries.unary.register_simple(FLOOR, "FLOOR({0})")
    registries.unary.register_simple(ABS, "ABS({0})")
    registries.unary.register_simple(EXP, "EXP({0})")
    registries.unary.register_simple(LN, "LN({0})")
    registries.unary.register_simple(SQRT, "SQRT({0})")

    # Logical
    registries.unary.register_simple(NOT, "NOT {0}", is_prefix=True)

    # String functions
    registries.unary.register_simple(LEN, "LENGTH({0})")
    registries.unary.register_simple(TRIM, "TRIM({0})")
    registries.unary.register_simple(LTRIM, "LTRIM({0})")
    registries.unary.register_simple(RTRIM, "RTRIM({0})")
    registries.unary.register_simple(UCASE, "UPPER({0})")
    registries.unary.register_simple(LCASE, "LOWER({0})")

    # Time extraction functions
    registries.unary.register_simple(YEAR, "YEAR({0})")
    registries.unary.register_simple(MONTH, "MONTH({0})")
    registries.unary.register_simple(DAYOFMONTH, "DAY({0})")
    registries.unary.register_simple(DAYOFYEAR, "DAYOFYEAR({0})")

    # =========================================================================
    # Aggregate Operators
    # =========================================================================

    registries.aggregate.register_simple(SUM, "SUM({0})")
    registries.aggregate.register_simple(AVG, "AVG({0})")
    registries.aggregate.register_simple(COUNT, "COUNT({0})")
    registries.aggregate.register_simple(MIN, "MIN({0})")
    registries.aggregate.register_simple(MAX, "MAX({0})")
    registries.aggregate.register_simple(MEDIAN, "MEDIAN({0})")
    registries.aggregate.register_simple(STDDEV_POP, "STDDEV_POP({0})")
    registries.aggregate.register_simple(STDDEV_SAMP, "STDDEV_SAMP({0})")
    registries.aggregate.register_simple(VAR_POP, "VAR_POP({0})")
    registries.aggregate.register_simple(VAR_SAMP, "VAR_SAMP({0})")

    # =========================================================================
    # Analytic (Window) Operators
    # =========================================================================

    # Aggregate functions can also be used as analytics
    registries.analytic.register_simple(SUM, "SUM({0})")
    registries.analytic.register_simple(AVG, "AVG({0})")
    registries.analytic.register_simple(COUNT, "COUNT({0})")
    registries.analytic.register_simple(MIN, "MIN({0})")
    registries.analytic.register_simple(MAX, "MAX({0})")
    registries.analytic.register_simple(MEDIAN, "MEDIAN({0})")
    registries.analytic.register_simple(STDDEV_POP, "STDDEV_POP({0})")
    registries.analytic.register_simple(STDDEV_SAMP, "STDDEV_SAMP({0})")
    registries.analytic.register_simple(VAR_POP, "VAR_POP({0})")
    registries.analytic.register_simple(VAR_SAMP, "VAR_SAMP({0})")

    # Pure analytic functions
    registries.analytic.register_simple(FIRST_VALUE, "FIRST_VALUE({0})")
    registries.analytic.register_simple(LAST_VALUE, "LAST_VALUE({0})")
    registries.analytic.register_simple(LAG, "LAG({0})")
    registries.analytic.register_simple(LEAD, "LEAD({0})")
    registries.analytic.register_simple(RANK, "RANK()")  # RANK takes no argument
    registries.analytic.register_simple(RATIO_TO_REPORT, "RATIO_TO_REPORT({0})")

    # =========================================================================
    # Parameterized Operators
    # =========================================================================

    # Single parameter operations
    registries.parameterized.register_simple(ROUND, "ROUND({0}, {1})")
    registries.parameterized.register_simple(TRUNC, "TRUNC({0}, {1})")
    registries.parameterized.register_simple(INSTR, "INSTR({0}, {1})")
    registries.parameterized.register_simple(LOG, "LOG({1}, {0})")  # LOG(base, value)
    registries.parameterized.register_simple(POWER, "POWER({0}, {1})")
    registries.parameterized.register_simple(NVL, "COALESCE({0}, {1})")

    # Multi-parameter operations
    registries.parameterized.register_simple(SUBSTR, "SUBSTR({0}, {1}, {2})")
    registries.parameterized.register_simple(REPLACE, "REPLACE({0}, {1}, {2})")

    # =========================================================================
    # Set Operations
    # =========================================================================

    registries.set_ops.register_simple(UNION, "UNION ALL")
    registries.set_ops.register_simple(INTERSECT, "INTERSECT")
    registries.set_ops.register_simple(SETDIFF, "EXCEPT")
    # SYMDIFF requires special handling (not a simple SQL operator)
    registries.set_ops.register(
        SYMDIFF,
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


def get_binary_sql(vtl_token: str, left: str, right: str) -> str:
    """
    Generate SQL for a binary operation.

    Args:
        vtl_token: The VTL operator token.
        left: SQL for left operand.
        right: SQL for right operand.

    Returns:
        Generated SQL expression.
    """
    return registry.binary.generate(vtl_token, left, right)


def get_unary_sql(vtl_token: str, operand: str) -> str:
    """
    Generate SQL for a unary operation.

    Args:
        vtl_token: The VTL operator token.
        operand: SQL for the operand.

    Returns:
        Generated SQL expression.
    """
    return registry.unary.generate(vtl_token, operand)


def get_aggregate_sql(vtl_token: str, operand: str) -> str:
    """
    Generate SQL for an aggregate operation.

    Args:
        vtl_token: The VTL operator token.
        operand: SQL for the operand.

    Returns:
        Generated SQL expression.
    """
    return registry.aggregate.generate(vtl_token, operand)


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
