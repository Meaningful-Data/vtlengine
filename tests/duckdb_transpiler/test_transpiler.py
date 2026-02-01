"""
Transpiler Tests

Tests for VTL AST to SQL transpilation.
Uses pytest parametrize to test Dataset, Component, and Scalar evaluations.
Each test verifies the complete SQL SELECT query output using AST Start nodes.
"""

from typing import Any, Dict, List, Tuple

import pytest

from vtlengine.AST import (
    Assignment,
    BinOp,
    Collection,
    Constant,
    If,
    MulOp,
    ParamOp,
    RegularAggregation,
    Start,
    UnaryOp,
    Validation,
    VarID,
)
from vtlengine.DataTypes import Boolean, Number, String
from vtlengine.duckdb_transpiler.Transpiler import SQLTranspiler
from vtlengine.Model import Component, Dataset, Role


# =============================================================================
# Test Utilities
# =============================================================================


def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison (remove extra whitespace)."""
    return " ".join(sql.split()).strip()


def assert_sql_equal(actual: str, expected: str):
    """Assert that two SQL strings are equivalent (ignoring whitespace)."""
    assert normalize_sql(actual) == normalize_sql(expected), (
        f"\nActual SQL:\n{actual}\n\nExpected SQL:\n{expected}"
    )


def assert_sql_contains(actual: str, expected_parts: list):
    """Assert that SQL contains all expected parts."""
    normalized = normalize_sql(actual)
    for part in expected_parts:
        assert part in normalized, f"Expected '{part}' not found in SQL:\n{actual}"


def create_simple_dataset(name: str, id_cols: list, measure_cols: list) -> Dataset:
    """Helper to create a simple Dataset for testing."""
    components = {}
    for col in id_cols:
        components[col] = Component(
            name=col, data_type=String, role=Role.IDENTIFIER, nullable=False
        )
    for col in measure_cols:
        components[col] = Component(
            name=col, data_type=Number, role=Role.MEASURE, nullable=True
        )
    return Dataset(name=name, components=components, data=None)


def create_transpiler(
    input_datasets: Dict[str, Dataset] = None,
    output_datasets: Dict[str, Dataset] = None,
) -> SQLTranspiler:
    """Helper to create a SQLTranspiler instance."""
    return SQLTranspiler(
        input_datasets=input_datasets or {},
        output_datasets=output_datasets or {},
        input_scalars={},
        output_scalars={},
    )


def make_ast_node(**kwargs) -> Dict[str, Any]:
    """Create common AST node parameters."""
    return {"line_start": 1, "column_start": 1, "line_stop": 1, "column_stop": 10, **kwargs}


def create_start_with_assignment(result_name: str, expression) -> Start:
    """Create a Start node containing an Assignment."""
    left = VarID(**make_ast_node(value=result_name))
    assignment = Assignment(**make_ast_node(left=left, op=":=", right=expression))
    return Start(**make_ast_node(children=[assignment]))


def transpile_and_get_sql(
    transpiler: SQLTranspiler, ast: Start
) -> List[Tuple[str, str, bool]]:
    """Transpile AST and return list of (name, sql, is_persistent) tuples."""
    return transpiler.transpile(ast)


# =============================================================================
# IN / NOT_IN Operator Tests
# =============================================================================


class TestInOperator:
    """Tests for IN and NOT_IN operators."""

    @pytest.mark.parametrize(
        "op,sql_op",
        [
            ("in", "IN"),
            ("not_in", "NOT IN"),
            ("not in", "NOT IN"),
        ],
    )
    def test_dataset_in_collection(self, op: str, sql_op: str):
        """Test dataset-level IN operation with complete SQL output."""
        ds = create_simple_dataset("DS_1", ["Id_1"], ["Me_1", "Me_2"])
        transpiler = create_transpiler(
            input_datasets={"DS_1": ds},
            output_datasets={"DS_r": ds},
        )

        # Create AST: DS_r := DS_1 in {1, 2}
        left = VarID(**make_ast_node(value="DS_1"))
        right = Collection(
            **make_ast_node(
                name="",
                type="Set",
                children=[
                    Constant(**make_ast_node(type_="INTEGER_CONSTANT", value=1)),
                    Constant(**make_ast_node(type_="INTEGER_CONSTANT", value=2)),
                ],
            )
        )
        expr = BinOp(**make_ast_node(left=left, op=op, right=right))
        ast = create_start_with_assignment("DS_r", expr)

        results = transpile_and_get_sql(transpiler, ast)

        assert len(results) == 1
        name, sql, _ = results[0]
        assert name == "DS_r"

        expected_sql = f'''SELECT "Id_1", ("Me_1" {sql_op} (1, 2)) AS "Me_1", ("Me_2" {sql_op} (1, 2)) AS "Me_2" FROM (SELECT * FROM "DS_1") AS t'''
        assert_sql_equal(sql, expected_sql)


# =============================================================================
# BETWEEN Operator Tests
# =============================================================================


class TestBetweenOperator:
    """Tests for BETWEEN operator in filter clause."""

    @pytest.mark.parametrize(
        "low_value,high_value",
        [
            (1, 10),
            (0, 100),
            (-5, 5),
        ],
    )
    def test_between_in_filter(self, low_value: int, high_value: int):
        """Test BETWEEN in filter clause with complete SQL output."""
        ds = create_simple_dataset("DS_1", ["Id_1"], ["Me_1"])
        transpiler = create_transpiler(
            input_datasets={"DS_1": ds},
            output_datasets={"DS_r": ds},
        )

        # Create AST: DS_r := DS_1[filter Me_1 between low and high]
        operand = VarID(**make_ast_node(value="Me_1"))
        low = Constant(**make_ast_node(type_="INTEGER_CONSTANT", value=low_value))
        high = Constant(**make_ast_node(type_="INTEGER_CONSTANT", value=high_value))
        between_expr = MulOp(**make_ast_node(op="between", children=[operand, low, high]))

        dataset_ref = VarID(**make_ast_node(value="DS_1"))
        filter_clause = RegularAggregation(
            **make_ast_node(op="filter", dataset=dataset_ref, children=[between_expr])
        )
        ast = create_start_with_assignment("DS_r", filter_clause)

        results = transpile_and_get_sql(transpiler, ast)

        assert len(results) == 1
        name, sql, _ = results[0]
        assert name == "DS_r"

        expected_sql = f'''SELECT * FROM (SELECT * FROM "DS_1") AS t WHERE ("Me_1" BETWEEN {low_value} AND {high_value})'''
        assert_sql_equal(sql, expected_sql)


# =============================================================================
# MATCH_CHARACTERS Operator Tests
# =============================================================================


class TestMatchOperator:
    """Tests for MATCH_CHARACTERS (regex) operator."""

    def test_dataset_match(self):
        """Test dataset-level MATCH with complete SQL output."""
        ds = create_simple_dataset("DS_1", ["Id_1"], ["Me_1", "Me_2"])
        ds.components["Me_1"].data_type = String
        ds.components["Me_2"].data_type = String
        transpiler = create_transpiler(
            input_datasets={"DS_1": ds},
            output_datasets={"DS_r": ds},
        )

        # Create AST: DS_r := match_characters(DS_1, "[A-Z]+")
        left = VarID(**make_ast_node(value="DS_1"))
        right = Constant(**make_ast_node(type_="STRING_CONSTANT", value="[A-Z]+"))
        expr = BinOp(**make_ast_node(left=left, op="match_characters", right=right))
        ast = create_start_with_assignment("DS_r", expr)

        results = transpile_and_get_sql(transpiler, ast)

        assert len(results) == 1
        name, sql, _ = results[0]
        assert name == "DS_r"

        expected_sql = '''SELECT "Id_1", regexp_full_match("Me_1", '[A-Z]+') AS "Me_1", regexp_full_match("Me_2", '[A-Z]+') AS "Me_2" FROM (SELECT * FROM "DS_1") AS t'''
        assert_sql_equal(sql, expected_sql)


# =============================================================================
# EXIST_IN Operator Tests
# =============================================================================


class TestExistInOperator:
    """Tests for EXIST_IN operator."""

    def test_exist_in_with_common_identifiers(self):
        """Test exist_in with complete SQL output."""
        ds1 = create_simple_dataset("DS_1", ["Id_1", "Id_2"], ["Me_1"])
        ds2 = create_simple_dataset("DS_2", ["Id_1", "Id_2"], ["Me_2"])
        transpiler = create_transpiler(
            input_datasets={"DS_1": ds1, "DS_2": ds2},
            output_datasets={"DS_r": ds1},
        )

        # Create AST: DS_r := exists_in(DS_1, DS_2)
        left = VarID(**make_ast_node(value="DS_1"))
        right = VarID(**make_ast_node(value="DS_2"))
        expr = BinOp(**make_ast_node(left=left, op="exists_in", right=right))
        ast = create_start_with_assignment("DS_r", expr)

        results = transpile_and_get_sql(transpiler, ast)

        assert len(results) == 1
        name, sql, _ = results[0]
        assert name == "DS_r"

        # Verify complete SELECT structure
        assert_sql_contains(sql, [
            'SELECT l."Id_1", l."Id_2"',
            'EXISTS(SELECT 1 FROM (SELECT * FROM "DS_2") AS r',
            'WHERE l."Id_1" = r."Id_1" AND l."Id_2" = r."Id_2"',
            'AS "bool_var"',
            'FROM (SELECT * FROM "DS_1") AS l',
        ])


# =============================================================================
# SET Operations Tests
# =============================================================================


class TestSetOperations:
    """Tests for set operations (union, intersect, setdiff, symdiff)."""

    def test_intersect_two_datasets(self):
        """Test INTERSECT with complete SQL output."""
        ds1 = create_simple_dataset("DS_1", ["Id_1"], ["Me_1"])
        ds2 = create_simple_dataset("DS_2", ["Id_1"], ["Me_1"])
        transpiler = create_transpiler(
            input_datasets={"DS_1": ds1, "DS_2": ds2},
            output_datasets={"DS_r": ds1},
        )

        # Create AST: DS_r := intersect(DS_1, DS_2)
        children = [
            VarID(**make_ast_node(value="DS_1")),
            VarID(**make_ast_node(value="DS_2")),
        ]
        expr = MulOp(**make_ast_node(op="intersect", children=children))
        ast = create_start_with_assignment("DS_r", expr)

        results = transpile_and_get_sql(transpiler, ast)

        assert len(results) == 1
        name, sql, _ = results[0]
        assert name == "DS_r"

        expected_sql = '(SELECT * FROM "DS_1") INTERSECT (SELECT * FROM "DS_2")'
        assert_sql_equal(sql, expected_sql)

    def test_setdiff_two_datasets(self):
        """Test SETDIFF with complete SQL output."""
        ds1 = create_simple_dataset("DS_1", ["Id_1"], ["Me_1"])
        ds2 = create_simple_dataset("DS_2", ["Id_1"], ["Me_1"])
        transpiler = create_transpiler(
            input_datasets={"DS_1": ds1, "DS_2": ds2},
            output_datasets={"DS_r": ds1},
        )

        # Create AST: DS_r := setdiff(DS_1, DS_2)
        children = [
            VarID(**make_ast_node(value="DS_1")),
            VarID(**make_ast_node(value="DS_2")),
        ]
        expr = MulOp(**make_ast_node(op="setdiff", children=children))
        ast = create_start_with_assignment("DS_r", expr)

        results = transpile_and_get_sql(transpiler, ast)

        assert len(results) == 1
        name, sql, _ = results[0]
        assert name == "DS_r"

        expected_sql = '(SELECT * FROM "DS_1") EXCEPT (SELECT * FROM "DS_2")'
        assert_sql_equal(sql, expected_sql)

    def test_union_with_dedup(self):
        """Test union with complete SQL output including DISTINCT ON."""
        ds1 = create_simple_dataset("DS_1", ["Id_1"], ["Me_1"])
        ds2 = create_simple_dataset("DS_2", ["Id_1"], ["Me_1"])
        transpiler = create_transpiler(
            input_datasets={"DS_1": ds1, "DS_2": ds2},
            output_datasets={"DS_r": ds1},
        )

        # Create AST: DS_r := union(DS_1, DS_2)
        children = [
            VarID(**make_ast_node(value="DS_1")),
            VarID(**make_ast_node(value="DS_2")),
        ]
        expr = MulOp(**make_ast_node(op="union", children=children))
        ast = create_start_with_assignment("DS_r", expr)

        results = transpile_and_get_sql(transpiler, ast)

        assert len(results) == 1
        name, sql, _ = results[0]
        assert name == "DS_r"

        # Verify union structure with dedup
        assert_sql_contains(sql, [
            "SELECT DISTINCT ON",
            '"Id_1"',
            "UNION ALL",
            '"DS_1"',
            '"DS_2"',
        ])


# =============================================================================
# CAST Operator Tests
# =============================================================================


class TestCastOperator:
    """Tests for CAST operations."""

    @pytest.mark.parametrize(
        "target_type,expected_duckdb_type",
        [
            ("Integer", "BIGINT"),
            ("Number", "DOUBLE"),
            ("String", "VARCHAR"),
            ("Boolean", "BOOLEAN"),
        ],
    )
    def test_dataset_cast_without_mask(self, target_type: str, expected_duckdb_type: str):
        """Test dataset-level CAST with complete SQL output."""
        ds = create_simple_dataset("DS_1", ["Id_1"], ["Me_1", "Me_2"])
        transpiler = create_transpiler(
            input_datasets={"DS_1": ds},
            output_datasets={"DS_r": ds},
        )

        # Create AST: DS_r := cast(DS_1, Type)
        operand = VarID(**make_ast_node(value="DS_1"))
        type_node = VarID(**make_ast_node(value=target_type))
        expr = ParamOp(**make_ast_node(op="cast", children=[operand, type_node], params=[]))
        ast = create_start_with_assignment("DS_r", expr)

        results = transpile_and_get_sql(transpiler, ast)

        assert len(results) == 1
        name, sql, _ = results[0]
        assert name == "DS_r"

        expected_sql = f'''SELECT "Id_1", CAST("Me_1" AS {expected_duckdb_type}) AS "Me_1", CAST("Me_2" AS {expected_duckdb_type}) AS "Me_2" FROM (SELECT * FROM "DS_1") AS t'''
        assert_sql_equal(sql, expected_sql)

    def test_cast_with_date_mask(self):
        """Test CAST to Date with mask producing STRPTIME SQL."""
        ds = create_simple_dataset("DS_1", ["Id_1"], ["Me_1"])
        transpiler = create_transpiler(
            input_datasets={"DS_1": ds},
            output_datasets={"DS_r": ds},
        )

        # Create AST: DS_r := cast(DS_1, Date, "%Y-%m-%d")
        operand = VarID(**make_ast_node(value="DS_1"))
        type_node = VarID(**make_ast_node(value="Date"))
        mask = Constant(**make_ast_node(type_="STRING_CONSTANT", value="%Y-%m-%d"))
        expr = ParamOp(**make_ast_node(op="cast", children=[operand, type_node], params=[mask]))
        ast = create_start_with_assignment("DS_r", expr)

        results = transpile_and_get_sql(transpiler, ast)

        assert len(results) == 1
        name, sql, _ = results[0]
        assert name == "DS_r"

        expected_sql = '''SELECT "Id_1", STRPTIME("Me_1", '%Y-%m-%d')::DATE AS "Me_1" FROM (SELECT * FROM "DS_1") AS t'''
        assert_sql_equal(sql, expected_sql)


# =============================================================================
# CHECK Validation Operator Tests
# =============================================================================


class TestCheckOperator:
    """Tests for CHECK validation operator."""

    def test_check_invalid_output(self):
        """Test CHECK with invalid output producing complete SQL."""
        ds = create_simple_dataset("DS_1", ["Id_1"], ["Me_1"])
        ds.components["Me_1"].data_type = Boolean
        transpiler = create_transpiler(
            input_datasets={"DS_1": ds},
            output_datasets={"DS_r": ds},
        )

        # Create Validation node
        validation = VarID(**make_ast_node(value="DS_1"))
        expr = Validation(
            **make_ast_node(
                op="check",
                validation=validation,
                error_code="E001",
                error_level=1,
                imbalance=None,
                invalid=True,
            )
        )
        ast = create_start_with_assignment("DS_r", expr)

        results = transpile_and_get_sql(transpiler, ast)

        assert len(results) == 1
        name, sql, _ = results[0]
        assert name == "DS_r"

        # Verify complete SELECT structure for invalid output
        assert_sql_contains(sql, [
            "SELECT *",
            "'E001' AS errorcode",
            "1 AS errorlevel",
            "WHERE",
            '"Me_1" = FALSE',
        ])


# =============================================================================
# Binary Operations Tests
# =============================================================================


class TestBinaryOperations:
    """Tests for standard binary operations."""

    def test_dataset_dataset_binary_op(self):
        """Test dataset-dataset binary operation with complete SQL output."""
        ds1 = create_simple_dataset("DS_1", ["Id_1"], ["Me_1"])
        ds2 = create_simple_dataset("DS_2", ["Id_1"], ["Me_1"])
        transpiler = create_transpiler(
            input_datasets={"DS_1": ds1, "DS_2": ds2},
            output_datasets={"DS_r": ds1},
        )

        # Create AST: DS_r := DS_1 + DS_2
        left = VarID(**make_ast_node(value="DS_1"))
        right = VarID(**make_ast_node(value="DS_2"))
        expr = BinOp(**make_ast_node(left=left, op="+", right=right))
        ast = create_start_with_assignment("DS_r", expr)

        results = transpile_and_get_sql(transpiler, ast)

        assert len(results) == 1
        name, sql, _ = results[0]
        assert name == "DS_r"

        expected_sql = '''SELECT a."Id_1", (a."Me_1" + b."Me_1") AS "Me_1" FROM (SELECT * FROM "DS_1") AS a INNER JOIN (SELECT * FROM "DS_2") AS b ON a."Id_1" = b."Id_1"'''
        assert_sql_equal(sql, expected_sql)

    @pytest.mark.parametrize(
        "op,sql_op",
        [
            ("+", "+"),
            ("-", "-"),
            ("*", "*"),
            ("/", "/"),
        ],
    )
    def test_dataset_scalar_binary_op(self, op: str, sql_op: str):
        """Test dataset-scalar binary operation with complete SQL output."""
        ds = create_simple_dataset("DS_1", ["Id_1"], ["Me_1", "Me_2"])
        transpiler = create_transpiler(
            input_datasets={"DS_1": ds},
            output_datasets={"DS_r": ds},
        )

        # Create AST: DS_r := DS_1 op 10
        left = VarID(**make_ast_node(value="DS_1"))
        right = Constant(**make_ast_node(type_="INTEGER_CONSTANT", value=10))
        expr = BinOp(**make_ast_node(left=left, op=op, right=right))
        ast = create_start_with_assignment("DS_r", expr)

        results = transpile_and_get_sql(transpiler, ast)

        assert len(results) == 1
        name, sql, _ = results[0]
        assert name == "DS_r"

        expected_sql = f'''SELECT "Id_1", ("Me_1" {sql_op} 10) AS "Me_1", ("Me_2" {sql_op} 10) AS "Me_2" FROM (SELECT * FROM "DS_1") AS t'''
        assert_sql_equal(sql, expected_sql)


# =============================================================================
# Unary Operations Tests
# =============================================================================


class TestUnaryOperations:
    """Tests for unary operations."""

    @pytest.mark.parametrize(
        "op,expected_sql_func",
        [
            ("ceil", "CEIL"),
            ("floor", "FLOOR"),
            ("abs", "ABS"),
            ("exp", "EXP"),
            ("ln", "LN"),
            ("sqrt", "SQRT"),
        ],
    )
    def test_dataset_unary_op(self, op: str, expected_sql_func: str):
        """Test dataset-level unary operation with complete SQL output."""
        ds = create_simple_dataset("DS_1", ["Id_1"], ["Me_1", "Me_2"])
        transpiler = create_transpiler(
            input_datasets={"DS_1": ds},
            output_datasets={"DS_r": ds},
        )

        # Create AST: DS_r := op(DS_1)
        operand = VarID(**make_ast_node(value="DS_1"))
        expr = UnaryOp(**make_ast_node(op=op, operand=operand))
        ast = create_start_with_assignment("DS_r", expr)

        results = transpile_and_get_sql(transpiler, ast)

        assert len(results) == 1
        name, sql, _ = results[0]
        assert name == "DS_r"

        expected_sql = f'''SELECT "Id_1", {expected_sql_func}("Me_1") AS "Me_1", {expected_sql_func}("Me_2") AS "Me_2" FROM (SELECT * FROM "DS_1") AS t'''
        assert_sql_equal(sql, expected_sql)

    def test_isnull_dataset_op(self):
        """Test dataset-level isnull with complete SQL output."""
        ds = create_simple_dataset("DS_1", ["Id_1"], ["Me_1"])
        transpiler = create_transpiler(
            input_datasets={"DS_1": ds},
            output_datasets={"DS_r": ds},
        )

        # Create AST: DS_r := isnull(DS_1)
        operand = VarID(**make_ast_node(value="DS_1"))
        expr = UnaryOp(**make_ast_node(op="isnull", operand=operand))
        ast = create_start_with_assignment("DS_r", expr)

        results = transpile_and_get_sql(transpiler, ast)

        assert len(results) == 1
        name, sql, _ = results[0]
        assert name == "DS_r"

        expected_sql = '''SELECT "Id_1", ("Me_1" IS NULL) AS "Me_1" FROM (SELECT * FROM "DS_1") AS t'''
        assert_sql_equal(sql, expected_sql)


# =============================================================================
# Parameterized Operations Tests
# =============================================================================


class TestParameterizedOperations:
    """Tests for parameterized operations."""

    def test_round_dataset_operation(self):
        """Test dataset-level ROUND with complete SQL output."""
        ds = create_simple_dataset("DS_1", ["Id_1"], ["Me_1", "Me_2"])
        transpiler = create_transpiler(
            input_datasets={"DS_1": ds},
            output_datasets={"DS_r": ds},
        )

        # Create AST: DS_r := round(DS_1, 2)
        operand = VarID(**make_ast_node(value="DS_1"))
        param = Constant(**make_ast_node(type_="INTEGER_CONSTANT", value=2))
        expr = ParamOp(**make_ast_node(op="round", children=[operand], params=[param]))
        ast = create_start_with_assignment("DS_r", expr)

        results = transpile_and_get_sql(transpiler, ast)

        assert len(results) == 1
        name, sql, _ = results[0]
        assert name == "DS_r"

        expected_sql = '''SELECT "Id_1", ROUND("Me_1", 2) AS "Me_1", ROUND("Me_2", 2) AS "Me_2" FROM (SELECT * FROM "DS_1") AS t'''
        assert_sql_equal(sql, expected_sql)

    def test_nvl_dataset_operation(self):
        """Test dataset-level NVL with complete SQL output."""
        ds = create_simple_dataset("DS_1", ["Id_1"], ["Me_1"])
        transpiler = create_transpiler(
            input_datasets={"DS_1": ds},
            output_datasets={"DS_r": ds},
        )

        # Create AST: DS_r := nvl(DS_1, 0)
        operand = VarID(**make_ast_node(value="DS_1"))
        default = Constant(**make_ast_node(type_="INTEGER_CONSTANT", value=0))
        expr = ParamOp(**make_ast_node(op="nvl", children=[operand], params=[default]))
        ast = create_start_with_assignment("DS_r", expr)

        results = transpile_and_get_sql(transpiler, ast)

        assert len(results) == 1
        name, sql, _ = results[0]
        assert name == "DS_r"

        expected_sql = '''SELECT "Id_1", COALESCE("Me_1", 0) AS "Me_1" FROM (SELECT * FROM "DS_1") AS t'''
        assert_sql_equal(sql, expected_sql)


# =============================================================================
# Clause Operations Tests
# =============================================================================


class TestClauseOperations:
    """Tests for clause operations (filter, calc, keep, drop, rename)."""

    def test_filter_clause(self):
        """Test filter clause with complete SQL output."""
        ds = create_simple_dataset("DS_1", ["Id_1"], ["Me_1"])
        transpiler = create_transpiler(
            input_datasets={"DS_1": ds},
            output_datasets={"DS_r": ds},
        )

        # Create AST: DS_r := DS_1[filter Me_1 > 10]
        condition = BinOp(
            **make_ast_node(
                left=VarID(**make_ast_node(value="Me_1")),
                op=">",
                right=Constant(**make_ast_node(type_="INTEGER_CONSTANT", value=10)),
            )
        )
        dataset_ref = VarID(**make_ast_node(value="DS_1"))
        expr = RegularAggregation(
            **make_ast_node(op="filter", dataset=dataset_ref, children=[condition])
        )
        ast = create_start_with_assignment("DS_r", expr)

        results = transpile_and_get_sql(transpiler, ast)

        assert len(results) == 1
        name, sql, _ = results[0]
        assert name == "DS_r"

        expected_sql = '''SELECT * FROM (SELECT * FROM "DS_1") AS t WHERE ("Me_1" > 10)'''
        assert_sql_equal(sql, expected_sql)

    def test_calc_clause_new_column(self):
        """Test calc clause creating new column with complete SQL output."""
        ds = create_simple_dataset("DS_1", ["Id_1"], ["Me_1"])
        transpiler = create_transpiler(
            input_datasets={"DS_1": ds},
            output_datasets={"DS_r": ds},
        )

        # Create AST: DS_r := DS_1[calc Me_2 := Me_1 * 2]
        calc_expr = BinOp(
            **make_ast_node(
                left=VarID(**make_ast_node(value="Me_1")),
                op="*",
                right=Constant(**make_ast_node(type_="INTEGER_CONSTANT", value=2)),
            )
        )
        calc_assignment = Assignment(
            **make_ast_node(
                left=VarID(**make_ast_node(value="Me_2")),
                op=":=",
                right=calc_expr,
            )
        )
        dataset_ref = VarID(**make_ast_node(value="DS_1"))
        expr = RegularAggregation(
            **make_ast_node(op="calc", dataset=dataset_ref, children=[calc_assignment])
        )
        ast = create_start_with_assignment("DS_r", expr)

        results = transpile_and_get_sql(transpiler, ast)

        assert len(results) == 1
        name, sql, _ = results[0]
        assert name == "DS_r"

        # Verify SELECT contains original columns and new calculated column
        assert_sql_contains(sql, [
            "SELECT",
            '"Id_1"',
            '"Me_1"',
            '("Me_1" * 2) AS "Me_2"',
            'FROM (SELECT * FROM "DS_1") AS t',
        ])


# =============================================================================
# Conditional Operations Tests
# =============================================================================


class TestConditionalOperations:
    """Tests for conditional operations (if-then-else) in calc context."""

    def test_if_then_else_in_calc(self):
        """Test IF-THEN-ELSE in calc clause with complete SQL output."""
        ds = create_simple_dataset("DS_1", ["Id_1"], ["Me_1"])
        transpiler = create_transpiler(
            input_datasets={"DS_1": ds},
            output_datasets={"DS_r": ds},
        )

        # Create AST: DS_r := DS_1[calc Me_2 := if Me_1 > 5 then 1 else 0]
        condition = BinOp(
            **make_ast_node(
                left=VarID(**make_ast_node(value="Me_1")),
                op=">",
                right=Constant(**make_ast_node(type_="INTEGER_CONSTANT", value=5)),
            )
        )
        then_op = Constant(**make_ast_node(type_="INTEGER_CONSTANT", value=1))
        else_op = Constant(**make_ast_node(type_="INTEGER_CONSTANT", value=0))
        if_expr = If(**make_ast_node(condition=condition, thenOp=then_op, elseOp=else_op))

        calc_assignment = Assignment(
            **make_ast_node(
                left=VarID(**make_ast_node(value="Me_2")),
                op=":=",
                right=if_expr,
            )
        )
        dataset_ref = VarID(**make_ast_node(value="DS_1"))
        expr = RegularAggregation(
            **make_ast_node(op="calc", dataset=dataset_ref, children=[calc_assignment])
        )
        ast = create_start_with_assignment("DS_r", expr)

        results = transpile_and_get_sql(transpiler, ast)

        assert len(results) == 1
        name, sql, _ = results[0]
        assert name == "DS_r"

        # Verify CASE WHEN structure
        assert_sql_contains(sql, [
            "SELECT",
            "CASE WHEN",
            '("Me_1" > 5)',
            "THEN 1 ELSE 0 END",
            'AS "Me_2"',
        ])


# =============================================================================
# Multiple Assignments Tests
# =============================================================================


class TestMultipleAssignments:
    """Tests for multiple assignments in a single script."""

    def test_chained_assignments(self):
        """Test multiple chained assignments producing multiple SELECT statements."""
        ds1 = create_simple_dataset("DS_1", ["Id_1"], ["Me_1"])
        ds2 = create_simple_dataset("DS_2", ["Id_1"], ["Me_1"])
        transpiler = create_transpiler(
            input_datasets={"DS_1": ds1},
            output_datasets={"DS_2": ds2, "DS_3": ds2},
        )

        # Create AST with two assignments:
        # DS_2 := DS_1 * 2;
        # DS_3 := DS_2 + 10;
        expr1 = BinOp(
            **make_ast_node(
                left=VarID(**make_ast_node(value="DS_1")),
                op="*",
                right=Constant(**make_ast_node(type_="INTEGER_CONSTANT", value=2)),
            )
        )
        assign1 = Assignment(
            **make_ast_node(
                left=VarID(**make_ast_node(value="DS_2")),
                op=":=",
                right=expr1,
            )
        )

        expr2 = BinOp(
            **make_ast_node(
                left=VarID(**make_ast_node(value="DS_2")),
                op="+",
                right=Constant(**make_ast_node(type_="INTEGER_CONSTANT", value=10)),
            )
        )
        assign2 = Assignment(
            **make_ast_node(
                left=VarID(**make_ast_node(value="DS_3")),
                op=":=",
                right=expr2,
            )
        )

        ast = Start(**make_ast_node(children=[assign1, assign2]))

        results = transpile_and_get_sql(transpiler, ast)

        assert len(results) == 2

        # First assignment
        name1, sql1, _ = results[0]
        assert name1 == "DS_2"
        expected_sql1 = '''SELECT "Id_1", ("Me_1" * 2) AS "Me_1" FROM (SELECT * FROM "DS_1") AS t'''
        assert_sql_equal(sql1, expected_sql1)

        # Second assignment (now DS_2 is available)
        name2, sql2, _ = results[1]
        assert name2 == "DS_3"
        expected_sql2 = '''SELECT "Id_1", ("Me_1" + 10) AS "Me_1" FROM (SELECT * FROM "DS_2") AS t'''
        assert_sql_equal(sql2, expected_sql2)
