"""Operator registry used by the DuckDB transpiler."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Set, Tuple

import vtlengine.AST.Grammar.tokens as tokens
from vtlengine.DataTypes import Duration, TimePeriod
from vtlengine.Exceptions import SemanticError

# Ordering-only comparisons (TimeInterval ordering is forbidden).
_ORDERING_OPS: Set[str] = {tokens.GT, tokens.GTE, tokens.LT, tokens.LTE}

# String operators needing VARCHAR input.
_STRING_UNARY_OPS: Set[str] = {
    tokens.UCASE,
    tokens.LCASE,
    tokens.LEN,
    tokens.TRIM,
    tokens.LTRIM,
    tokens.RTRIM,
}

_STRING_PARAM_OPS: Set[str] = {tokens.SUBSTR, tokens.REPLACE, tokens.INSTR}

# Type mappings
VTL_TO_DUCKDB_TYPES: Dict[str, str] = {
    "Integer": "BIGINT",
    "Number": "DOUBLE",
    "String": "VARCHAR",
    "Boolean": "BOOLEAN",
    "Date": "TIMESTAMP",
    "TimePeriod": "VARCHAR",
    "TimeInterval": "VARCHAR",
    "Duration": "VARCHAR",
    "Null": "VARCHAR",
}


def get_duckdb_type(vtl_type: str) -> str:
    """Map a VTL type name to a DuckDB SQL type."""
    return VTL_TO_DUCKDB_TYPES.get(vtl_type, "VARCHAR")


@dataclass
class SQLOperator:
    """Definition of a SQL operator mapping."""

    sql_template: str
    is_prefix: bool = False
    dataset_handler: Optional[Callable[..., Any]] = None
    requires_context: bool = False
    custom_generator: Optional[Callable[..., str]] = None

    def sql(self, *operands: str) -> str:
        if self.custom_generator:
            return self.custom_generator(*operands)
        return self.sql_template.format(*operands)


@dataclass
class OperatorRegistry:
    """Unified registry for SQL operators. ``(vtl_token, arity/dtype)``"""

    _operators: Dict[Tuple[str, int], SQLOperator] = field(default_factory=dict)
    _typed_overrides: Dict[Tuple[str, type], SQLOperator] = field(default_factory=dict)

    def register(
        self, vtl_token: str, sql_template: str, *, arity: int = 0, is_prefix: bool = False
    ) -> "OperatorRegistry":
        """Register a simple operator with just a template.

        Args:
            vtl_token: The VTL operator token (from tokens.py).
            sql_template: SQL template with ``{0}``, ``{1}`` placeholders.
            arity: Number of operands (1=unary, 2=binary, 0=auto-detect from
                template placeholder count).
            is_prefix: Whether this is a prefix operator (e.g. ``-x``).
        """
        if arity == 0:
            arity = sql_template.count("{") - sql_template.count("{{") * 2
            if arity <= 0:
                arity = 1  # e.g. "RANK()" with no placeholders
        self._operators[(vtl_token, arity)] = SQLOperator(
            sql_template=sql_template, is_prefix=is_prefix
        )
        return self

    def register_custom(
        self, vtl_token: str, operator: SQLOperator, *, arity: int = 0
    ) -> "OperatorRegistry":
        """Register a custom-generated operator."""
        self._operators[(vtl_token, arity)] = operator
        return self

    def register_typed(
        self, vtl_token: str, data_type: type, sql_template: str
    ) -> "OperatorRegistry":
        """Register a type-specific operator variant."""
        self._typed_overrides[(vtl_token, data_type)] = SQLOperator(sql_template=sql_template)
        return self

    def has_typed(self, vtl_token: str, data_type: type) -> bool:
        """Check if a type-specific override exists."""
        return (vtl_token, data_type) in self._typed_overrides

    def is_registered(self, vtl_token: str) -> bool:
        """Check if any operator variant is registered for this token."""
        return any(tok == vtl_token for tok, _ in self._operators)

    def sql(self, vtl_token: str, *operands: str, data_type: Optional[type] = None) -> str:
        """Generate SQL, resolving by type override → arity → fallback.

        For unregistered operators, falls back to ``TOKEN(operands)`` style.
        """
        if data_type is not None:
            typed_op = self._typed_overrides.get((vtl_token, data_type))
            if typed_op:
                return typed_op.sql(*operands)
        n = len(operands)
        # Try exact arity match first, then arity-0 (default / custom)
        operator = self._operators.get((vtl_token, n)) or self._operators.get((vtl_token, 0))
        if operator is not None:
            return operator.sql(*operands)
        # Fallback: function-call syntax for unregistered operators
        return f"{vtl_token.upper()}({', '.join(operands)})"


def _validate_int_param(value: Optional[str], *, op: str, param_name: str, min_val: int) -> None:
    """Validate a scalar integer parameter against a minimum value."""
    if value is None or value == "NULL":
        return
    try:
        if int(value) < min_val:
            raise SemanticError(
                "1-1-18-4", op=op, param_type=param_name, correct_type=f">= {min_val}"
            )
    except (ValueError, TypeError):
        pass  # Column reference, not a constant


def _create_default_registry() -> OperatorRegistry:
    ops = OperatorRegistry()

    # Binary operators
    # Arithmetic
    ops.register(tokens.PLUS, "({0} + {1})")
    ops.register(tokens.MINUS, "({0} - {1})")
    ops.register(tokens.MULT, "({0} * {1})")
    ops.register(tokens.DIV, "vtl_div({0}, {1})")
    ops.register(tokens.MOD, "({0} % {1})")
    # Comparison
    ops.register(tokens.EQ, "({0} = {1})")
    ops.register(tokens.NEQ, "({0} <> {1})")
    ops.register(tokens.GT, "({0} > {1})")
    ops.register(tokens.LT, "({0} < {1})")
    ops.register(tokens.GTE, "({0} >= {1})")
    ops.register(tokens.LTE, "({0} <= {1})")
    # Logical
    ops.register(tokens.AND, "({0} AND {1})")
    ops.register(tokens.OR, "({0} OR {1})")
    ops.register_custom(
        tokens.XOR,
        SQLOperator(
            sql_template="",
            custom_generator=lambda a, b: f"(({a} AND NOT {b}) OR (NOT {a} AND {b}))",
        ),
    )
    ops.register(tokens.IN, "({0} IN {1})")
    ops.register(tokens.NOT_IN, "({0} NOT IN {1})")
    # String
    ops.register(tokens.CONCAT, "({0} || {1})")
    # Numeric functions (come through BinOp AST)
    ops.register(tokens.POWER, "POWER({0}, {1})")
    ops.register(tokens.LOG, "LOG({1}, {0})")  # DuckDB: LOG(base, value)
    # Conditional (come through BinOp AST)
    ops.register(tokens.NVL, "COALESCE({0}, {1})")
    # Date/Time
    ops.register(tokens.DATEDIFF, "ABS(DATE_DIFF('day', {0}, {1}))")
    # String matching
    ops.register(tokens.CHARSET_MATCH, "regexp_full_match({0}, {1})")
    # TimePeriod ordering — vtl_period_* comparison macros
    _tp_ordering = [(tokens.GT, "gt"), (tokens.GTE, "ge"), (tokens.LT, "lt"), (tokens.LTE, "le")]
    for _tok, _suffix in _tp_ordering:
        ops.register_typed(
            _tok,
            TimePeriod,
            f"vtl_period_{_suffix}(vtl_period_parse({{0}}), vtl_period_parse({{1}}))",
        )
    # TimePeriod datediff
    ops.register_typed(
        tokens.DATEDIFF,
        TimePeriod,
        "vtl_tp_datediff(vtl_period_parse({0}), vtl_period_parse({1}))",
    )
    # Duration comparison — magnitude ordering via vtl_duration_to_int
    for _tok in [tokens.GT, tokens.GTE, tokens.LT, tokens.LTE, tokens.EQ, tokens.NEQ]:
        ops.register_typed(
            _tok,
            Duration,
            f"(vtl_duration_to_int({{0}}) {_tok} vtl_duration_to_int({{1}}))",
        )

    # Unary operators
    # Arithmetic functions
    ops.register(tokens.PLUS, "+{0}", is_prefix=True)
    ops.register(tokens.MINUS, "-{0}", is_prefix=True)
    ops.register(tokens.CEIL, "CEIL({0})")
    ops.register(tokens.FLOOR, "FLOOR({0})")
    ops.register(tokens.ABS, "ABS({0})")
    ops.register(tokens.EXP, "EXP({0})")
    ops.register(tokens.LN, "LN({0})")
    ops.register(tokens.SQRT, "SQRT({0})")
    # Logical
    ops.register(tokens.NOT, "NOT {0}", is_prefix=True)
    # String functions
    ops.register(tokens.LEN, "LENGTH({0})")
    ops.register(tokens.TRIM, "TRIM({0})")
    ops.register(tokens.LTRIM, "LTRIM({0})")
    ops.register(tokens.RTRIM, "RTRIM({0})")
    ops.register(tokens.UCASE, "UPPER({0})")
    ops.register(tokens.LCASE, "LOWER({0})")
    # Null check
    ops.register(tokens.ISNULL, "({0} IS NULL)")
    # Date extraction — generic (Date) and TimePeriod overrides
    ops.register(tokens.YEAR, "YEAR({0})")
    ops.register(tokens.MONTH, "MONTH({0})")
    ops.register(tokens.DAYOFMONTH, "DAY({0})")
    ops.register(tokens.DAYOFYEAR, "DAYOFYEAR({0})")
    ops.register_typed(tokens.YEAR, TimePeriod, "CAST(vtl_period_parse({0}).year AS BIGINT)")
    ops.register_typed(tokens.MONTH, TimePeriod, "vtl_tp_getmonth(vtl_period_parse({0}))")
    ops.register_typed(tokens.DAYOFMONTH, TimePeriod, "vtl_tp_dayofmonth(vtl_period_parse({0}))")
    ops.register_typed(tokens.DAYOFYEAR, TimePeriod, "vtl_tp_dayofyear(vtl_period_parse({0}))")
    # Duration conversion functions
    ops.register(tokens.DAYTOYEAR, "vtl_daytoyear({0})")
    ops.register(tokens.DAYTOMONTH, "vtl_daytomonth({0})")
    ops.register(tokens.YEARTODAY, "vtl_yeartoday({0})")
    ops.register(tokens.MONTHTODAY, "vtl_monthtoday({0})")

    # Aggregate and Analytic operators
    ops.register(tokens.SUM, "SUM({0})")
    ops.register(tokens.AVG, "AVG({0})")
    ops.register(tokens.COUNT, "COUNT({0})")
    ops.register(tokens.MIN, "MIN({0})")
    ops.register(tokens.MAX, "MAX({0})")
    ops.register(tokens.MEDIAN, "MEDIAN({0})")
    ops.register(tokens.STDDEV_POP, "STDDEV_POP({0})")
    ops.register(tokens.STDDEV_SAMP, "STDDEV_SAMP({0})")
    ops.register(tokens.VAR_POP, "VAR_POP({0})")
    ops.register(tokens.VAR_SAMP, "VAR_SAMP({0})")
    # Window-only analytics
    ops.register(tokens.FIRST_VALUE, "FIRST_VALUE({0})")
    ops.register(tokens.LAST_VALUE, "LAST_VALUE({0})")
    ops.register(tokens.LAG, "LAG({0})")
    ops.register(tokens.LEAD, "LEAD({0})")
    ops.register(tokens.RANK, "RANK()")
    ops.register(tokens.RATIO_TO_REPORT, "RATIO_TO_REPORT({0})")

    # Parameterized operators
    # Comparison
    ops.register(tokens.BETWEEN, "({0} BETWEEN {1} AND {2})")

    # ROUND/TRUNC require DOUBLE when precision is not constant in DuckDB.
    def _precision_generator(sql_fn: str) -> Callable[..., str]:
        def gen(*args: Optional[str]) -> str:
            precision = "0" if (len(args) < 2 or args[1] is None) else str(args[1])
            return f"{sql_fn}(CAST({args[0]} AS DOUBLE), COALESCE(CAST({precision} AS INTEGER), 0))"

        return gen

    for _tok, _fn in [(tokens.ROUND, "ROUND"), (tokens.TRUNC, "TRUNC")]:
        ops.register_custom(
            _tok,
            SQLOperator(
                sql_template=f"{_fn}({{0}}, CAST({{1}} AS INTEGER))",
                custom_generator=_precision_generator(_fn),
            ),
        )

    def _instr_generator(*args: Optional[str]) -> str:
        """Generate SQL for VTL instr(string, pattern, start, occurrence)."""
        params = []
        params.append(str(args[0]) if len(args) > 0 and args[0] is not None else "NULL")
        params.append(str(args[1]) if len(args) > 1 and args[1] is not None else "NULL")
        start_arg = args[2] if len(args) > 2 and args[2] is not None else None
        _validate_int_param(start_arg, op="instr", param_name="Start", min_val=1)
        params.append(str(start_arg) if start_arg is not None else "NULL")
        occur_arg = args[3] if len(args) > 3 and args[3] is not None else None
        _validate_int_param(occur_arg, op="instr", param_name="Occurrence", min_val=1)
        params.append(str(occur_arg) if occur_arg is not None else "NULL")

        return f"vtl_instr({', '.join(params)})"

    ops.register_custom(
        tokens.INSTR,
        SQLOperator(
            sql_template="INSTR({0}, {1})",
            custom_generator=_instr_generator,
        ),
    )
    ops.register(tokens.LOG, "LOG({1}, {0})")
    ops.register(tokens.POWER, "POWER({0}, {1})")

    # Multi-parameter operations
    def _substr_generator(*args: Optional[str]) -> str:
        """Generate SQL for VTL substr with defaulted start/length."""
        if len(args) == 1:
            return str(args[0])
        string_arg = str(args[0])
        start = args[1] if len(args) > 1 else None
        _validate_int_param(start, op="substr", param_name="Start", min_val=1)
        start_sql = "1" if start is None or start == "NULL" else f"COALESCE({start}, 1)"
        length = args[2] if len(args) > 2 else None
        _validate_int_param(length, op="substr", param_name="Length", min_val=0)
        if length is None or length == "NULL":
            return f"SUBSTR({string_arg}, {start_sql})"
        return f"SUBSTR({string_arg}, {start_sql}, COALESCE({length}, LENGTH({string_arg})))"

    ops.register_custom(
        tokens.SUBSTR,
        SQLOperator(
            sql_template="SUBSTR({0}, {1}, {2})",
            custom_generator=_substr_generator,
        ),
    )

    def _replace_generator(*args: Optional[str]) -> str:
        """Generate SQL for VTL replace with null/default handling."""
        if any(a == "NULL" for a in args if a is not None):
            return "CAST(NULL AS VARCHAR)"
        if len(args) < 2 or args[1] is None:
            return str(args[0]) if args else "''"
        string_arg = str(args[0])
        pattern_arg = str(args[1])
        if len(args) < 3 or args[2] is None:
            return f"REPLACE({string_arg}, {pattern_arg}, '')"
        return f"REPLACE({string_arg}, {pattern_arg}, {args[2]})"

    ops.register_custom(
        tokens.REPLACE,
        SQLOperator(
            sql_template="REPLACE({0}, {1}, {2})",
            custom_generator=_replace_generator,
        ),
    )

    # Set operations — join multiple subqueries with the SQL set operator
    def _set_op_generator(sql_keyword: str) -> Callable[..., str]:
        def gen(*queries: str) -> str:
            return f" {sql_keyword} ".join(f"({q})" for q in queries)

        return gen

    ops.register_custom(
        tokens.UNION,
        SQLOperator(sql_template="", custom_generator=_set_op_generator("UNION ALL")),
    )
    ops.register_custom(
        tokens.INTERSECT,
        SQLOperator(sql_template="", custom_generator=_set_op_generator("INTERSECT")),
    )
    ops.register_custom(
        tokens.SETDIFF,
        SQLOperator(sql_template="", custom_generator=_set_op_generator("EXCEPT")),
    )
    ops.register_custom(
        tokens.SYMDIFF,
        SQLOperator(sql_template="SYMDIFF", requires_context=True),
    )

    return ops


# Global registry instance
registry = _create_default_registry()
