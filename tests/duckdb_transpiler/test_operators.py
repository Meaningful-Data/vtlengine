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
    RANK,
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
    OperatorCategory,
    OperatorRegistry,
    SQLOperator,
    SQLOperatorRegistries,
    get_aggregate_sql,
    get_binary_sql,
    get_duckdb_type,
    get_sql_operator_symbol,
    get_unary_sql,
    is_operator_registered,
    registry,
)


class TestSQLOperator:
    """Tests for SQLOperator dataclass."""

    def test_binary_operator_generate(self):
        """Test binary operator SQL generation."""
        op = SQLOperator(sql_template="({0} + {1})", category=OperatorCategory.BINARY)
        result = op.generate('"a"', '"b"')
        assert result == '("a" + "b")'

    def test_binary_operator_requires_two_operands(self):
        """Test binary operator raises error with insufficient operands."""
        op = SQLOperator(sql_template="({0} + {1})", category=OperatorCategory.BINARY)
        with pytest.raises(ValueError, match="Binary operator requires 2 operands"):
            op.generate('"a"')

    def test_unary_function_operator(self):
        """Test unary function operator SQL generation."""
        op = SQLOperator(sql_template="CEIL({0})", category=OperatorCategory.UNARY)
        result = op.generate('"x"')
        assert result == 'CEIL("x")'

    def test_unary_prefix_operator(self):
        """Test unary prefix operator SQL generation."""
        op = SQLOperator(sql_template="-{0}", category=OperatorCategory.UNARY, is_prefix=True)
        result = op.generate('"x"')
        assert result == '-"x"'

    def test_unary_operator_requires_one_operand(self):
        """Test unary operator raises error with no operands."""
        op = SQLOperator(sql_template="CEIL({0})", category=OperatorCategory.UNARY)
        with pytest.raises(ValueError, match="Unary operator requires 1 operand"):
            op.generate()

    def test_aggregate_operator(self):
        """Test aggregate operator SQL generation."""
        op = SQLOperator(sql_template="SUM({0})", category=OperatorCategory.AGGREGATE)
        result = op.generate('"Me_1"')
        assert result == 'SUM("Me_1")'

    def test_parameterized_operator(self):
        """Test parameterized operator SQL generation."""
        op = SQLOperator(sql_template="ROUND({0}, {1})", category=OperatorCategory.PARAMETERIZED)
        result = op.generate('"x"', "2")
        assert result == 'ROUND("x", 2)'

    def test_set_operator(self):
        """Test set operator SQL generation."""
        op = SQLOperator(sql_template="UNION ALL", category=OperatorCategory.SET)
        result = op.generate("SELECT * FROM a", "SELECT * FROM b")
        assert result == "(SELECT * FROM a) UNION ALL (SELECT * FROM b)"

    def test_custom_generator(self):
        """Test operator with custom generator function."""

        def custom_gen(a: str, b: str) -> str:
            return f"CUSTOM_FUNC({a}, {b})"

        op = SQLOperator(
            sql_template="",
            category=OperatorCategory.BINARY,
            custom_generator=custom_gen,
        )
        result = op.generate("x", "y")
        assert result == "CUSTOM_FUNC(x, y)"


class TestOperatorRegistry:
    """Tests for OperatorRegistry class."""

    def test_register_and_get(self):
        """Test registering and retrieving an operator."""
        reg = OperatorRegistry(OperatorCategory.BINARY)
        op = SQLOperator(sql_template="({0} + {1})", category=OperatorCategory.BINARY)
        reg.register("plus", op)

        retrieved = reg.get("plus")
        assert retrieved is op

    def test_register_simple(self):
        """Test simplified registration."""
        reg = OperatorRegistry(OperatorCategory.UNARY)
        reg.register_simple("ceil", "CEIL({0})")

        op = reg.get("ceil")
        assert op is not None
        assert op.sql_template == "CEIL({0})"
        assert op.category == OperatorCategory.UNARY

    def test_is_registered(self):
        """Test checking if operator is registered."""
        reg = OperatorRegistry(OperatorCategory.BINARY)
        reg.register_simple("plus", "({0} + {1})")

        assert reg.is_registered("plus") is True
        assert reg.is_registered("minus") is False

    def test_generate(self):
        """Test SQL generation through registry."""
        reg = OperatorRegistry(OperatorCategory.BINARY)
        reg.register_simple("plus", "({0} + {1})")

        result = reg.generate("plus", '"a"', '"b"')
        assert result == '("a" + "b")'

    def test_generate_unknown_operator(self):
        """Test that generating with unknown operator raises error."""
        reg = OperatorRegistry(OperatorCategory.BINARY)

        with pytest.raises(ValueError, match="Unknown operator: unknown"):
            reg.generate("unknown", "a", "b")

    def test_get_sql_symbol_binary(self):
        """Test extracting SQL symbol from binary operator."""
        reg = OperatorRegistry(OperatorCategory.BINARY)
        reg.register_simple("plus", "({0} + {1})")

        symbol = reg.get_sql_symbol("plus")
        assert symbol == "+"

    def test_get_sql_symbol_unary(self):
        """Test extracting SQL symbol from unary operator."""
        reg = OperatorRegistry(OperatorCategory.UNARY)
        reg.register_simple("ceil", "CEIL({0})")

        symbol = reg.get_sql_symbol("ceil")
        assert symbol == "CEIL"

    def test_list_operators(self):
        """Test listing all registered operators."""
        reg = OperatorRegistry(OperatorCategory.BINARY)
        reg.register_simple("plus", "({0} + {1})")
        reg.register_simple("minus", "({0} - {1})")

        operators = reg.list_operators()
        assert len(operators) == 2
        assert ("plus", "({0} + {1})") in operators
        assert ("minus", "({0} - {1})") in operators

    def test_chaining(self):
        """Test that registration methods return self for chaining."""
        reg = OperatorRegistry(OperatorCategory.BINARY)
        result = reg.register_simple("plus", "({0} + {1})").register_simple("minus", "({0} - {1})")

        assert result is reg
        assert reg.is_registered("plus")
        assert reg.is_registered("minus")


class TestSQLOperatorRegistries:
    """Tests for SQLOperatorRegistries collection."""

    def test_all_registries_exist(self):
        """Test that all category registries exist."""
        regs = SQLOperatorRegistries()
        assert regs.binary is not None
        assert regs.unary is not None
        assert regs.aggregate is not None
        assert regs.analytic is not None
        assert regs.parameterized is not None
        assert regs.set_ops is not None

    def test_get_by_category(self):
        """Test getting registry by category."""
        regs = SQLOperatorRegistries()
        assert regs.get_by_category(OperatorCategory.BINARY) is regs.binary
        assert regs.get_by_category(OperatorCategory.UNARY) is regs.unary
        assert regs.get_by_category(OperatorCategory.AGGREGATE) is regs.aggregate

    def test_find_operator(self):
        """Test finding operator across registries."""
        regs = SQLOperatorRegistries()
        regs.binary.register_simple("plus", "({0} + {1})")
        regs.unary.register_simple("ceil", "CEIL({0})")

        result = regs.find_operator("plus")
        assert result is not None
        assert result[0] == OperatorCategory.BINARY

        result = regs.find_operator("ceil")
        assert result is not None
        assert result[0] == OperatorCategory.UNARY

        result = regs.find_operator("unknown")
        assert result is None


class TestGlobalRegistry:
    """Tests for the global pre-populated registry."""

    @pytest.mark.parametrize(
        "token,expected_output",
        [
            (PLUS, '("a" + "b")'),
            (MINUS, '("a" - "b")'),
            (MULT, '("a" * "b")'),
            (DIV, '("a" / "b")'),
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
        """Test all binary operators are registered correctly."""
        result = registry.binary.generate(token, '"a"', '"b"')
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
    def test_unary_function_operators(self, token, expected_output):
        """Test unary function operators."""
        result = registry.unary.generate(token, '"x"')
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
        ],
    )
    def test_aggregate_operators(self, token, expected_output):
        """Test aggregate operators."""
        result = registry.aggregate.generate(token, '"Me_1"')
        assert result == expected_output

    @pytest.mark.parametrize(
        "token,expected_output",
        [
            (FIRST_VALUE, 'FIRST_VALUE("x")'),
            (LAG, 'LAG("x")'),
            (RANK, "RANK()"),
        ],
    )
    def test_analytic_operators(self, token, expected_output):
        """Test analytic operators."""
        result = registry.analytic.generate(token, '"x"')
        assert result == expected_output

    @pytest.mark.parametrize(
        "token,args,expected_output",
        [
            (ROUND, ('"x"', "2"), 'ROUND(CAST("x" AS DOUBLE), COALESCE(CAST(2 AS INTEGER), 0))'),
            (TRUNC, ('"x"', "0"), 'TRUNC(CAST("x" AS DOUBLE), COALESCE(CAST(0 AS INTEGER), 0))'),
            (INSTR, ('"str"', "'a'"), "vtl_instr(\"str\", 'a', 1, 1)"),
            (LOG, ('"x"', "10"), 'LOG(10, "x")'),  # Note: LOG has swapped args
            (POWER, ('"x"', "2"), 'POWER("x", 2)'),
            (SUBSTR, ('"str"', "1", "5"), 'SUBSTR("str", 1, 5)'),
            (REPLACE, ('"str"', "'a'", "'b'"), "REPLACE(\"str\", 'a', 'b')"),
        ],
    )
    def test_parameterized_operators(self, token, args, expected_output):
        """Test parameterized operators."""
        result = registry.parameterized.generate(token, *args)
        assert result == expected_output

    @pytest.mark.parametrize(
        "token,expected",
        [
            (UNION, "UNION ALL"),
            (INTERSECT, "INTERSECT"),
            (SETDIFF, "EXCEPT"),
        ],
    )
    def test_set_operators_registered(self, token, expected):
        """Test set operators are registered."""
        op = registry.set_ops.get(token)
        assert op is not None
        assert expected in op.sql_template

    def test_symdiff_requires_context(self):
        """Test SYMDIFF is marked as requiring context."""
        op = registry.set_ops.get(SYMDIFF)
        assert op is not None
        assert op.requires_context is True


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_binary_sql(self):
        """Test get_binary_sql helper."""
        result = get_binary_sql(PLUS, '"a"', '"b"')
        assert result == '("a" + "b")'

    def test_get_unary_sql(self):
        """Test get_unary_sql helper."""
        result = get_unary_sql(CEIL, '"x"')
        assert result == 'CEIL("x")'

    def test_get_aggregate_sql(self):
        """Test get_aggregate_sql helper."""
        result = get_aggregate_sql(SUM, '"Me_1"')
        assert result == 'SUM("Me_1")'

    def test_get_sql_operator_symbol(self):
        """Test get_sql_operator_symbol helper."""
        assert get_sql_operator_symbol(PLUS) == "+"
        assert get_sql_operator_symbol(CEIL) == "CEIL"
        assert get_sql_operator_symbol(SUM) == "SUM"
        assert get_sql_operator_symbol("nonexistent") is None

    def test_is_operator_registered(self):
        """Test is_operator_registered helper."""
        assert is_operator_registered(PLUS) is True
        assert is_operator_registered(CEIL) is True
        assert is_operator_registered(SUM) is True
        assert is_operator_registered("nonexistent") is False


class TestTypeMappings:
    """Tests for VTL to DuckDB type mappings."""

    @pytest.mark.parametrize(
        "vtl_type,duckdb_type",
        [
            ("Integer", "BIGINT"),
            ("Number", "DOUBLE"),
            ("String", "VARCHAR"),
            ("Boolean", "BOOLEAN"),
            ("Date", "DATE"),
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
