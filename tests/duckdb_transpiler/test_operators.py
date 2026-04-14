"""Tests for the Operator Registry module."""

import pytest

from vtlengine.AST.Grammar.tokens import (
    ABS,
    AND,
    AVG,
    CEIL,
    CONCAT,
    COUNT,
    DIV,
    EQ,
    FIRST_VALUE,
    FLOOR,
    GT,
    INSTR,
    INTERSECT,
    LAG,
    LCASE,
    LEN,
    LN,
    LOG,
    LT,
    LTRIM,
    MAX,
    MIN,
    MINUS,
    MOD,
    MULT,
    NEQ,
    OR,
    PLUS,
    POWER,
    REPLACE,
    ROUND,
    SETDIFF,
    SQRT,
    STDDEV_POP,
    SUBSTR,
    SUM,
    SYMDIFF,
    TRIM,
    TRUNC,
    UCASE,
    UNION,
    VAR_POP,
    XOR,
)
from vtlengine.duckdb_transpiler.Transpiler.operators import (
    OperatorRegistry,
    SQLOperator,
    get_duckdb_type,
    registry,
)


class TestSQLOperator:
    """Tests for SQLOperator dataclass."""

    def test_template_generate(self):
        """Test SQL generation from template."""
        op = SQLOperator(sql_template="({0} + {1})")
        assert op.generate('"a"', '"b"') == '("a" + "b")'

    def test_unary_template(self):
        """Test unary function template."""
        op = SQLOperator(sql_template="CEIL({0})")
        assert op.generate('"x"') == 'CEIL("x")'

    def test_prefix_template(self):
        """Test prefix template."""
        op = SQLOperator(sql_template="-{0}", is_prefix=True)
        assert op.generate('"x"') == '-"x"'

    def test_custom_generator(self):
        """Test operator with custom generator function."""

        def custom_gen(a: str, b: str) -> str:
            return f"CUSTOM_FUNC({a}, {b})"

        op = SQLOperator(sql_template="", custom_generator=custom_gen)
        result = op.generate("x", "y")
        assert result == "CUSTOM_FUNC(x, y)"

    def test_custom_generator_takes_precedence(self):
        """Test that custom_generator overrides sql_template."""
        op = SQLOperator(
            sql_template="({0} + {1})",
            custom_generator=lambda a, b: f"CUSTOM({a}, {b})",
        )
        assert op.generate("a", "b") == "CUSTOM(a, b)"


class TestOperatorRegistry:
    """Tests for the unified OperatorRegistry."""

    def test_register_and_generate(self):
        """Test registering and generating an operator."""
        reg = OperatorRegistry()
        reg.register("plus", "({0} + {1})")
        assert reg.generate("plus", '"a"', '"b"') == '("a" + "b")'

    def test_arity_disambiguation(self):
        """Test same token with different arities."""
        reg = OperatorRegistry()
        reg.register("op", "({0} + {1})")  # arity=2 (auto-detected)
        reg.register("op", "-{0}")  # arity=1 (auto-detected)

        assert reg.generate("op", "a", "b") == "(a + b)"
        assert reg.generate("op", "x") == "-x"

    def test_is_registered(self):
        """Test checking if operator is registered."""
        reg = OperatorRegistry()
        reg.register("plus", "({0} + {1})")

        assert reg.is_registered("plus") is True
        assert reg.is_registered("minus") is False

    def test_fallback_for_unknown(self):
        """Test that unknown operators get function-call fallback."""
        reg = OperatorRegistry()
        result = reg.generate("year", "x")
        assert result == "YEAR(x)"

    def test_typed_override(self):
        """Test type-specific operator variant."""
        reg = OperatorRegistry()
        reg.register("gt", "({0} > {1})")
        reg.register_typed("gt", int, "CUSTOM_GT({0}, {1})")

        assert reg.generate("gt", "a", "b") == "(a > b)"
        assert reg.generate("gt", "a", "b", data_type=int) == "CUSTOM_GT(a, b)"

    def test_has_typed(self):
        """Test has_typed check."""
        reg = OperatorRegistry()
        reg.register_typed("gt", int, "CUSTOM({0}, {1})")

        assert reg.has_typed("gt", int) is True
        assert reg.has_typed("gt", str) is False

    def test_custom_registration(self):
        """Test custom operator registration."""
        reg = OperatorRegistry()
        reg.register_custom(
            "xor",
            SQLOperator(
                sql_template="",
                custom_generator=lambda a, b: f"({a} XOR {b})",
            ),
        )
        assert reg.generate("xor", "a", "b") == "(a XOR b)"

    def test_chaining(self):
        """Test that registration methods return self for chaining."""
        reg = OperatorRegistry()
        result = reg.register("plus", "({0} + {1})").register("minus", "({0} - {1})")
        assert result is reg


class TestGlobalRegistry:
    """Tests for the global pre-populated registry."""

    @pytest.mark.parametrize(
        "token,expected_output",
        [
            (PLUS, '("a" + "b")'),
            (MINUS, '("a" - "b")'),
            (MULT, '("a" * "b")'),
            (DIV, 'vtl_div("a", "b")'),
            (MOD, '("a" % "b")'),
            (EQ, '("a" = "b")'),
            (NEQ, '("a" <> "b")'),
            (GT, '("a" > "b")'),
            (LT, '("a" < "b")'),
            (AND, '("a" AND "b")'),
            (OR, '("a" OR "b")'),
            (XOR, '(("a" AND NOT "b") OR (NOT "a" AND "b"))'),
            (CONCAT, '("a" || "b")'),
        ],
    )
    def test_binary_operators(self, token, expected_output):
        """Test all binary operators produce correct SQL with 2 operands."""
        result = registry.generate(token, '"a"', '"b"')
        assert result == expected_output

    @pytest.mark.parametrize(
        "token,expected_output",
        [
            (CEIL, 'CEIL("x")'),
            (FLOOR, 'FLOOR("x")'),
            (ABS, 'ABS("x")'),
            (SQRT, 'SQRT("x")'),
            (LN, 'LN("x")'),
            (LEN, 'LENGTH("x")'),
            (TRIM, 'TRIM("x")'),
            (LTRIM, 'LTRIM("x")'),
            (UCASE, 'UPPER("x")'),
            (LCASE, 'LOWER("x")'),
        ],
    )
    def test_unary_operators(self, token, expected_output):
        """Test unary operators produce correct SQL with 1 operand."""
        result = registry.generate(token, '"x"')
        assert result == expected_output

    @pytest.mark.parametrize(
        "token,expected_output",
        [
            (SUM, 'SUM("Me_1")'),
            (AVG, 'AVG("Me_1")'),
            (COUNT, 'COUNT("Me_1")'),
            (MIN, 'MIN("Me_1")'),
            (MAX, 'MAX("Me_1")'),
            (STDDEV_POP, 'STDDEV_POP("Me_1")'),
            (VAR_POP, 'VAR_POP("Me_1")'),
            (FIRST_VALUE, 'FIRST_VALUE("Me_1")'),
            (LAG, 'LAG("Me_1")'),
        ],
    )
    def test_aggregate_and_analytic_operators(self, token, expected_output):
        """Test aggregate/analytic operators (shared templates)."""
        result = registry.generate(token, '"Me_1"')
        assert result == expected_output

    @pytest.mark.parametrize(
        "token,args,expected_output",
        [
            (ROUND, ('"x"', "2"), 'ROUND(CAST("x" AS DOUBLE), COALESCE(CAST(2 AS INTEGER), 0))'),
            (TRUNC, ('"x"', "0"), 'TRUNC(CAST("x" AS DOUBLE), COALESCE(CAST(0 AS INTEGER), 0))'),
            (INSTR, ('"str"', "'a'"), "vtl_instr(\"str\", 'a', NULL, NULL)"),
            (LOG, ('"x"', "10"), 'LOG(10, "x")'),
            (POWER, ('"x"', "2"), 'POWER("x", 2)'),
            (
                SUBSTR,
                ('"str"', "1", "5"),
                'SUBSTR("str", COALESCE(1, 1), COALESCE(5, LENGTH("str")))',
            ),
            (REPLACE, ('"str"', "'a'", "'b'"), "REPLACE(\"str\", 'a', 'b')"),
        ],
    )
    def test_parameterized_operators(self, token, args, expected_output):
        """Test parameterized operators."""
        result = registry.generate(token, *args)
        assert result == expected_output

    @pytest.mark.parametrize(
        "token,expected_keyword",
        [
            (UNION, "UNION ALL"),
            (INTERSECT, "INTERSECT"),
            (SETDIFF, "EXCEPT"),
        ],
    )
    def test_set_operators(self, token, expected_keyword):
        """Test set operators join subqueries correctly."""
        result = registry.generate(token, "SELECT * FROM a", "SELECT * FROM b")
        assert expected_keyword in result
        assert "(SELECT * FROM a)" in result
        assert "(SELECT * FROM b)" in result

    def test_symdiff_registered(self):
        """Test SYMDIFF is registered and marked as requiring context."""
        # SYMDIFF uses custom handling in the transpiler, just check it's registered
        assert registry.is_registered(SYMDIFF) is True


class TestTypeMappings:
    """Tests for VTL to DuckDB type mappings."""

    @pytest.mark.parametrize(
        "vtl_type,duckdb_type",
        [
            ("Integer", "BIGINT"),
            ("Number", "DOUBLE"),
            ("String", "VARCHAR"),
            ("Boolean", "BOOLEAN"),
            ("Date", "TIMESTAMP"),
            ("TimePeriod", "VARCHAR"),
            ("TimeInterval", "VARCHAR"),
            ("Duration", "VARCHAR"),
            ("Null", "VARCHAR"),
        ],
    )
    def test_type_mapping(self, vtl_type, duckdb_type):
        """Test VTL to DuckDB type mapping."""
        assert get_duckdb_type(vtl_type) == duckdb_type

    def test_unknown_type_defaults_to_varchar(self):
        """Test unknown types default to VARCHAR."""
        assert get_duckdb_type("UnknownType") == "VARCHAR"
