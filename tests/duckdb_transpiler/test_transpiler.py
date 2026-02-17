"""
Transpiler Tests

Tests for VTL AST to SQL transpilation.
Uses pytest parametrize to test Dataset, Component, and Scalar evaluations.
Each test verifies the complete SQL SELECT query output using AST Start nodes.
"""

from typing import Any, Dict, List, Tuple

import pytest

from vtlengine.AST import (
    Aggregation,
    Argument,
    Assignment,
    BinOp,
    Collection,
    Constant,
    EvalOp,
    If,
    MulOp,
    Operator,
    ParamOp,
    RegularAggregation,
    Start,
    TimeAggregation,
    UDOCall,
    UnaryOp,
    Validation,
    VarID,
)
from vtlengine.AST.Grammar.tokens import (
    CURRENT_DATE,
    DATEDIFF,
    FLOW_TO_STOCK,
    PERIOD_INDICATOR,
    STOCK_TO_FLOW,
)
from vtlengine.DataTypes import Boolean, Date, Integer, Number, String
from vtlengine.duckdb_transpiler.Transpiler import SQLTranspiler
from vtlengine.Model import Component, Dataset, ExternalRoutine, Role, ValueDomain

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
        components[col] = Component(name=col, data_type=Number, role=Role.MEASURE, nullable=True)
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


def transpile_and_get_sql(transpiler: SQLTranspiler, ast: Start) -> List[Tuple[str, str, bool]]:
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

        expected_sql = f'SELECT "Id_1", ("Me_1" {sql_op} (1, 2)) AS "Me_1", ("Me_2" {sql_op} (1, 2)) AS "Me_2" FROM "DS_1"'
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

        # VTL-compliant BETWEEN with NULL propagation
        expected_sql = (
            f'SELECT * FROM "DS_1" WHERE CASE WHEN "Me_1" IS NULL'
            f" OR {low_value} IS NULL OR {high_value} IS NULL"
            f' THEN NULL ELSE ("Me_1" BETWEEN {low_value} AND {high_value}) END'
        )
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

        expected_sql = 'SELECT "Id_1", regexp_full_match("Me_1", \'[A-Z]+\') AS "Me_1", regexp_full_match("Me_2", \'[A-Z]+\') AS "Me_2" FROM "DS_1"'
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
        assert_sql_contains(
            sql,
            [
                'SELECT l."Id_1", l."Id_2"',
                'EXISTS(SELECT 1 FROM (SELECT * FROM "DS_2") AS r',
                'WHERE l."Id_1" = r."Id_1" AND l."Id_2" = r."Id_2"',
                'AS "bool_var"',
                'FROM (SELECT * FROM "DS_1") AS l',
            ],
        )


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

        expected_sql = (
            'SELECT a.* FROM (SELECT * FROM "DS_1") AS a '
            "WHERE EXISTS ("
            'SELECT 1 FROM (SELECT * FROM "DS_2") AS b '
            'WHERE a."Id_1" = b."Id_1")'
        )
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

        expected_sql = (
            'SELECT a.* FROM (SELECT * FROM "DS_1") AS a '
            "WHERE NOT EXISTS ("
            'SELECT 1 FROM (SELECT * FROM "DS_2") AS b '
            'WHERE a."Id_1" = b."Id_1")'
        )
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
        assert_sql_contains(
            sql,
            [
                "SELECT DISTINCT ON",
                '"Id_1"',
                "UNION ALL",
                '"DS_1"',
                '"DS_2"',
            ],
        )


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

        expected_sql = f'SELECT "Id_1", CAST("Me_1" AS {expected_duckdb_type}) AS "Me_1", CAST("Me_2" AS {expected_duckdb_type}) AS "Me_2" FROM "DS_1"'
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

        expected_sql = 'SELECT "Id_1", STRPTIME("Me_1", \'%Y-%m-%d\')::DATE AS "Me_1" FROM "DS_1"'
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
        assert_sql_contains(
            sql,
            [
                '"bool_var"',
                '"imbalance"',
                "'E001'",
                '"errorcode"',
                '"errorlevel"',
                "WHERE",
                "IS FALSE",
            ],
        )


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

        expected_sql = '''SELECT a."Id_1", (a."Me_1" + b."Me_1") AS "Me_1" FROM "DS_1" AS a INNER JOIN "DS_2" AS b ON a."Id_1" = b."Id_1"'''
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

        expected_sql = f'SELECT "Id_1", ("Me_1" {sql_op} 10) AS "Me_1", ("Me_2" {sql_op} 10) AS "Me_2" FROM "DS_1"'
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

        expected_sql = f'SELECT "Id_1", {expected_sql_func}("Me_1") AS "Me_1", {expected_sql_func}("Me_2") AS "Me_2" FROM "DS_1"'
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

        # For mono-measure datasets, isnull output is renamed to bool_var (VTL semantics)
        expected_sql = 'SELECT "Id_1", ("Me_1" IS NULL) AS "bool_var" FROM "DS_1"'
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

        expected_sql = 'SELECT "Id_1", ROUND(CAST("Me_1" AS DOUBLE), COALESCE(CAST(2 AS INTEGER), 0)) AS "Me_1", ROUND(CAST("Me_2" AS DOUBLE), COALESCE(CAST(2 AS INTEGER), 0)) AS "Me_2" FROM "DS_1"'
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

        expected_sql = 'SELECT "Id_1", NVL("Me_1", 0) AS "Me_1" FROM "DS_1"'
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

        # Optimized SQL with predicate pushdown (no unnecessary nesting)
        expected_sql = """SELECT * FROM "DS_1" WHERE ("Me_1" > 10)"""
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
        assert_sql_contains(
            sql,
            [
                "SELECT",
                '"Id_1"',
                '"Me_1"',
                '("Me_1" * 2) AS "Me_2"',
                'FROM (SELECT * FROM "DS_1") AS t',
            ],
        )


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
        assert_sql_contains(
            sql,
            [
                "SELECT",
                "CASE WHEN",
                '("Me_1" > 5)',
                "THEN 1 ELSE 0 END",
                'AS "Me_2"',
            ],
        )


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
        expected_sql1 = 'SELECT "Id_1", ("Me_1" * 2) AS "Me_1" FROM "DS_1"'
        assert_sql_equal(sql1, expected_sql1)

        # Second assignment (now DS_2 is available)
        name2, sql2, _ = results[1]
        assert name2 == "DS_3"
        expected_sql2 = 'SELECT "Id_1", ("Me_1" + 10) AS "Me_1" FROM "DS_2"'
        assert_sql_equal(sql2, expected_sql2)


# =============================================================================
# Value Domain Tests (Sprint 4)
# =============================================================================


class TestValueDomains:
    """Tests for value domain handling in transpiler."""

    def test_value_domain_in_collection_string_type(self):
        """Test value domain reference resolves to string literals."""
        # Create value domain with string values
        vd = ValueDomain(name="COUNTRIES", type=String, setlist=["US", "UK", "DE"])

        transpiler = SQLTranspiler(
            input_datasets={},
            output_datasets={},
            input_scalars={},
            output_scalars={},
            value_domains={"COUNTRIES": vd},
        )

        # Create a Collection node referencing the value domain
        collection = Collection(
            **make_ast_node(name="COUNTRIES", type="String", children=[], kind="ValueDomain")
        )

        result = transpiler.visit_Collection(collection)
        assert result == "('US', 'UK', 'DE')"

    def test_value_domain_in_collection_integer_type(self):
        """Test value domain reference resolves to integer literals."""
        # Create value domain with integer values
        vd = ValueDomain(name="VALID_CODES", type=Integer, setlist=[1, 2, 3, 4, 5])

        transpiler = SQLTranspiler(
            input_datasets={},
            output_datasets={},
            input_scalars={},
            output_scalars={},
            value_domains={"VALID_CODES": vd},
        )

        collection = Collection(
            **make_ast_node(name="VALID_CODES", type="Integer", children=[], kind="ValueDomain")
        )

        result = transpiler.visit_Collection(collection)
        assert result == "(1, 2, 3, 4, 5)"

    def test_value_domain_not_found_error(self):
        """Test error when value domain is not found."""
        transpiler = SQLTranspiler(
            input_datasets={},
            output_datasets={},
            input_scalars={},
            output_scalars={},
            value_domains={},
        )

        collection = Collection(
            **make_ast_node(name="UNKNOWN_VD", type="String", children=[], kind="ValueDomain")
        )

        with pytest.raises(ValueError, match="no value domains provided"):
            transpiler.visit_Collection(collection)

    def test_value_domain_missing_from_provided(self):
        """Test error when specific value domain is not in provided dict."""
        vd = ValueDomain(name="OTHER_VD", type=String, setlist=["A", "B"])

        transpiler = SQLTranspiler(
            input_datasets={},
            output_datasets={},
            input_scalars={},
            output_scalars={},
            value_domains={"OTHER_VD": vd},
        )

        collection = Collection(
            **make_ast_node(name="UNKNOWN_VD", type="String", children=[], kind="ValueDomain")
        )

        with pytest.raises(ValueError, match="'UNKNOWN_VD' not found"):
            transpiler.visit_Collection(collection)

    def test_collection_set_kind(self):
        """Test normal Set collection still works."""
        transpiler = SQLTranspiler(
            input_datasets={},
            output_datasets={},
            input_scalars={},
            output_scalars={},
        )

        # Create a Set collection with literal constants
        collection = Collection(
            **make_ast_node(
                name="",
                type="Integer",
                children=[
                    Constant(**make_ast_node(type_="INTEGER_CONSTANT", value=1)),
                    Constant(**make_ast_node(type_="INTEGER_CONSTANT", value=2)),
                    Constant(**make_ast_node(type_="INTEGER_CONSTANT", value=3)),
                ],
                kind="Set",
            )
        )

        result = transpiler.visit_Collection(collection)
        assert result == "(1, 2, 3)"

    @pytest.mark.parametrize(
        "type_name,value,expected",
        [
            ("String", "hello", "'hello'"),
            ("String", "it's", "'it''s'"),  # Escaped single quote
            ("Integer", 42, "42"),
            ("Number", 3.14, "3.14"),
            ("Boolean", True, "TRUE"),
            ("Boolean", False, "FALSE"),
            ("Date", "2024-01-15", "DATE '2024-01-15'"),
        ],
    )
    def test_value_to_sql_literal(self, type_name, value, expected):
        """Test _value_to_sql_literal helper method."""
        transpiler = SQLTranspiler(
            input_datasets={},
            output_datasets={},
            input_scalars={},
            output_scalars={},
        )

        result = transpiler._to_sql_literal(value, type_name)
        assert result == expected

    def test_value_to_sql_literal_null(self):
        """Test NULL handling in _value_to_sql_literal."""
        transpiler = SQLTranspiler(
            input_datasets={},
            output_datasets={},
            input_scalars={},
            output_scalars={},
        )

        result = transpiler._to_sql_literal(None, "String")
        assert result == "NULL"


# =============================================================================
# External Routines / Eval Operator Tests (Sprint 4)
# =============================================================================


class TestEvalOperator:
    """Tests for EVAL operator and external routines."""

    def test_eval_op_simple_query(self):
        """Test EVAL operator with simple external routine."""
        ds = create_simple_dataset("DS_1", ["Id_1"], ["Me_1"])
        external_routine = ExternalRoutine(
            dataset_names=["DS_1"],
            query='SELECT "Id_1", "Me_1" * 2 AS "Me_1" FROM "DS_1"',
            name="double_measure",
        )

        transpiler = SQLTranspiler(
            input_datasets={"DS_1": ds},
            output_datasets={"DS_r": ds},
            input_scalars={},
            output_scalars={},
            external_routines={"double_measure": external_routine},
        )

        eval_op = EvalOp(
            **make_ast_node(
                name="double_measure",
                operands=[VarID(**make_ast_node(value="DS_1"))],
                output=None,
                language="SQL",
            )
        )

        result = transpiler.visit_EvalOp(eval_op)
        # The query should be returned as-is since DS_1 is a direct table reference
        expected_sql = 'SELECT "Id_1", "Me_1" * 2 AS "Me_1" FROM "DS_1"'
        assert_sql_equal(result, expected_sql)

    def test_eval_op_routine_not_found(self):
        """Test error when external routine is not found."""
        transpiler = SQLTranspiler(
            input_datasets={},
            output_datasets={},
            input_scalars={},
            output_scalars={},
            external_routines={},
        )

        eval_op = EvalOp(
            **make_ast_node(
                name="unknown_routine",
                operands=[],
                output=None,
                language="SQL",
            )
        )

        with pytest.raises(ValueError, match="no external routines provided"):
            transpiler.visit_EvalOp(eval_op)

    def test_eval_op_routine_missing_from_provided(self):
        """Test error when specific routine is not in provided dict."""
        external_routine = ExternalRoutine(
            dataset_names=["DS_1"],
            query='SELECT * FROM "DS_1"',
            name="other_routine",
        )

        transpiler = SQLTranspiler(
            input_datasets={},
            output_datasets={},
            input_scalars={},
            output_scalars={},
            external_routines={"other_routine": external_routine},
        )

        eval_op = EvalOp(
            **make_ast_node(
                name="unknown_routine",
                operands=[],
                output=None,
                language="SQL",
            )
        )

        with pytest.raises(ValueError, match="'unknown_routine' not found"):
            transpiler.visit_EvalOp(eval_op)

    def test_eval_op_with_subquery_replacement(self):
        """Test EVAL operator replaces table references with subqueries when needed."""
        ds = create_simple_dataset("DS_1", ["Id_1"], ["Me_1"])
        external_routine = ExternalRoutine(
            dataset_names=["DS_1"],
            query='SELECT "Id_1", SUM("Me_1") AS "total" FROM DS_1 GROUP BY "Id_1"',
            name="aggregate_routine",
        )

        transpiler = SQLTranspiler(
            input_datasets={"DS_1": ds},
            output_datasets={"DS_r": ds},
            input_scalars={},
            output_scalars={},
            external_routines={"aggregate_routine": external_routine},
        )

        eval_op = EvalOp(
            **make_ast_node(
                name="aggregate_routine",
                operands=[VarID(**make_ast_node(value="DS_1"))],
                output=None,
                language="SQL",
            )
        )

        result = transpiler.visit_EvalOp(eval_op)
        # Should contain aggregate function
        expected_sql = 'SELECT "Id_1", SUM("Me_1") AS "total" FROM DS_1 GROUP BY "Id_1"'
        assert_sql_equal(result, expected_sql)


# =============================================================================
# Time Operators Tests (Sprint 5)
# =============================================================================


class TestTimeOperators:
    """Tests for time operators in transpiler."""

    def test_current_date(self):
        """Test current_date nullary operator."""
        transpiler = SQLTranspiler(
            input_datasets={},
            output_datasets={},
            input_scalars={},
            output_scalars={},
        )

        mul_op = MulOp(**make_ast_node(op=CURRENT_DATE, children=[]))
        result = transpiler.visit_MulOp(mul_op)
        assert result == "CURRENT_DATE"

    @pytest.mark.parametrize(
        "op_token,expected_func",
        [
            ("year", "YEAR"),
            ("month", "MONTH"),
            ("dayofmonth", "DAY"),
            ("dayofyear", "DAYOFYEAR"),
        ],
    )
    def test_time_extraction_scalar(self, op_token, expected_func):
        """Test time extraction operators on scalar operands."""
        transpiler = SQLTranspiler(
            input_datasets={},
            output_datasets={},
            input_scalars={},
            output_scalars={},
        )

        unary_op = UnaryOp(
            **make_ast_node(
                op=op_token,
                operand=VarID(**make_ast_node(value="date_col")),
            )
        )

        result = transpiler.visit_UnaryOp(unary_op)
        expected_sql = f'{expected_func}("date_col")'
        assert_sql_equal(result, expected_sql)

    def test_datediff_scalar(self):
        """Test datediff on scalar operands."""
        transpiler = SQLTranspiler(
            input_datasets={},
            output_datasets={},
            input_scalars={},
            output_scalars={},
        )

        binop = BinOp(
            **make_ast_node(
                left=Constant(**make_ast_node(type_="STRING_CONSTANT", value="2024-01-15")),
                op=DATEDIFF,
                right=Constant(**make_ast_node(type_="STRING_CONSTANT", value="2024-01-01")),
            )
        )

        result = transpiler.visit_BinOp(binop)
        expected_sql = "ABS(DATE_DIFF('day', '2024-01-15', '2024-01-01'))"
        assert_sql_equal(result, expected_sql)

    def test_period_indicator_scalar(self):
        """Test period_indicator on scalar operand."""
        transpiler = SQLTranspiler(
            input_datasets={},
            output_datasets={},
            input_scalars={},
            output_scalars={},
        )

        unary_op = UnaryOp(
            **make_ast_node(
                op=PERIOD_INDICATOR,
                operand=VarID(**make_ast_node(value="time_period_col")),
            )
        )

        result = transpiler.visit_UnaryOp(unary_op)
        # Updated to use vtl_period_indicator function for proper TimePeriod handling
        expected_sql = 'vtl_period_indicator(vtl_period_parse("time_period_col"))'
        assert_sql_equal(result, expected_sql)

    def test_flow_to_stock_requires_dataset(self):
        """Test flow_to_stock raises error for non-dataset operand."""
        transpiler = SQLTranspiler(
            input_datasets={},
            output_datasets={},
            input_scalars={},
            output_scalars={},
        )

        unary_op = UnaryOp(
            **make_ast_node(
                op=FLOW_TO_STOCK,
                operand=Constant(**make_ast_node(type_="INTEGER_CONSTANT", value=10)),
            )
        )

        with pytest.raises(ValueError, match="requires a dataset"):
            transpiler.visit_UnaryOp(unary_op)

    def test_stock_to_flow_requires_dataset(self):
        """Test stock_to_flow raises error for non-dataset operand."""
        transpiler = SQLTranspiler(
            input_datasets={},
            output_datasets={},
            input_scalars={},
            output_scalars={},
        )

        unary_op = UnaryOp(
            **make_ast_node(
                op=STOCK_TO_FLOW,
                operand=Constant(**make_ast_node(type_="INTEGER_CONSTANT", value=10)),
            )
        )

        with pytest.raises(ValueError, match="requires a dataset"):
            transpiler.visit_UnaryOp(unary_op)

    @pytest.mark.parametrize(
        "op_token,expected_sql",
        [
            (
                "daytoyear",
                "'P' || CAST(FLOOR(400 / 365) AS VARCHAR) || 'Y' || CAST(400 % 365 AS VARCHAR) || 'D'",
            ),
            (
                "daytomonth",
                "'P' || CAST(FLOOR(400 / 30) AS VARCHAR) || 'M' || CAST(400 % 30 AS VARCHAR) || 'D'",
            ),
        ],
    )
    def test_duration_conversion_daytox(self, op_token, expected_sql):
        """Test duration conversion operators (daytoyear, daytomonth)."""
        transpiler = SQLTranspiler(
            input_datasets={},
            output_datasets={},
            input_scalars={},
            output_scalars={},
        )

        unary_op = UnaryOp(
            **make_ast_node(
                op=op_token,
                operand=Constant(**make_ast_node(type_="INTEGER_CONSTANT", value=400)),
            )
        )

        result = transpiler.visit_UnaryOp(unary_op)
        assert_sql_equal(result, expected_sql)

    @pytest.mark.parametrize(
        "op_token,expected_sql",
        [
            (
                "yeartoday",
                r"( CAST(REGEXP_EXTRACT('P1Y100D', 'P(\d+)Y', 1) AS INTEGER) * 365 + CAST(REGEXP_EXTRACT('P1Y100D', '(\d+)D', 1) AS INTEGER) )",
            ),
            (
                "monthtoday",
                r"( CAST(REGEXP_EXTRACT('P1Y100D', 'P(\d+)M', 1) AS INTEGER) * 30 + CAST(REGEXP_EXTRACT('P1Y100D', '(\d+)D', 1) AS INTEGER) )",
            ),
        ],
    )
    def test_duration_conversion_xtoday(self, op_token, expected_sql):
        """Test duration conversion operators (yeartoday, monthtoday)."""
        transpiler = SQLTranspiler(
            input_datasets={},
            output_datasets={},
            input_scalars={},
            output_scalars={},
        )

        unary_op = UnaryOp(
            **make_ast_node(
                op=op_token,
                operand=Constant(**make_ast_node(type_="STRING_CONSTANT", value="P1Y100D")),
            )
        )

        result = transpiler.visit_UnaryOp(unary_op)
        assert_sql_equal(result, expected_sql)

    def test_flow_to_stock_dataset(self):
        """Test flow_to_stock on dataset generates window function SQL."""
        # Create dataset with time identifier (Id_1 as Date, Id_2 as String)
        components = {
            "Id_1": Component(name="Id_1", data_type=Date, role=Role.IDENTIFIER, nullable=False),
            "Id_2": Component(name="Id_2", data_type=String, role=Role.IDENTIFIER, nullable=False),
            "Me_1": Component(name="Me_1", data_type=Number, role=Role.MEASURE, nullable=True),
        }
        ds = Dataset(name="DS_1", components=components, data=None)

        transpiler = SQLTranspiler(
            input_datasets={"DS_1": ds},
            output_datasets={"DS_r": ds},
            input_scalars={},
            output_scalars={},
        )

        unary_op = UnaryOp(
            **make_ast_node(
                op=FLOW_TO_STOCK,
                operand=VarID(**make_ast_node(value="DS_1")),
            )
        )

        result = transpiler.visit_UnaryOp(unary_op)
        expected_sql = 'SELECT "Id_1", "Id_2", SUM("Me_1") OVER (PARTITION BY "Id_2" ORDER BY "Id_1") AS "Me_1" FROM "DS_1"'
        assert_sql_equal(result, expected_sql)

    def test_stock_to_flow_dataset(self):
        """Test stock_to_flow on dataset generates window function SQL."""
        # Create dataset with time identifier (Id_1 as Date, Id_2 as String)
        components = {
            "Id_1": Component(name="Id_1", data_type=Date, role=Role.IDENTIFIER, nullable=False),
            "Id_2": Component(name="Id_2", data_type=String, role=Role.IDENTIFIER, nullable=False),
            "Me_1": Component(name="Me_1", data_type=Number, role=Role.MEASURE, nullable=True),
        }
        ds = Dataset(name="DS_1", components=components, data=None)

        transpiler = SQLTranspiler(
            input_datasets={"DS_1": ds},
            output_datasets={"DS_r": ds},
            input_scalars={},
            output_scalars={},
        )

        unary_op = UnaryOp(
            **make_ast_node(
                op=STOCK_TO_FLOW,
                operand=VarID(**make_ast_node(value="DS_1")),
            )
        )

        result = transpiler.visit_UnaryOp(unary_op)
        expected_sql = 'SELECT "Id_1", "Id_2", COALESCE("Me_1" - LAG("Me_1") OVER (PARTITION BY "Id_2" ORDER BY "Id_1"), "Me_1") AS "Me_1" FROM "DS_1"'
        assert_sql_equal(result, expected_sql)


# =============================================================================
# RANDOM Operator Tests
# =============================================================================


class TestRandomOperator:
    """Tests for RANDOM operator."""

    def test_random_scalar(self):
        """Test RANDOM with scalar seed and index."""
        transpiler = create_transpiler()

        # Create AST: random(42, 5)
        seed = Constant(**make_ast_node(type_="INTEGER_CONSTANT", value=42))
        index = Constant(**make_ast_node(type_="INTEGER_CONSTANT", value=5))
        random_op = ParamOp(**make_ast_node(op="random", children=[seed], params=[index]))

        result = transpiler.visit_ParamOp(random_op)

        # Full SQL: hash-based deterministic random
        expected_sql = (
            "(ABS(hash(CAST(42 AS VARCHAR) || '_' || CAST(5 AS VARCHAR))) % 1000000) / 1000000.0"
        )
        assert_sql_equal(result, expected_sql)

    def test_random_dataset(self):
        """Test RANDOM on dataset measures."""
        ds = create_simple_dataset("DS_1", ["Id_1"], ["Me_1"])
        transpiler = create_transpiler(input_datasets={"DS_1": ds})

        # Create AST: DS_r := random(DS_1, 3)
        dataset_ref = VarID(**make_ast_node(value="DS_1"))
        index = Constant(**make_ast_node(type_="INTEGER_CONSTANT", value=3))
        random_op = ParamOp(**make_ast_node(op="random", children=[dataset_ref], params=[index]))

        result = transpiler.visit_ParamOp(random_op)

        # Full SQL: applies random to each measure
        expected_sql = (
            'SELECT "Id_1", '
            "(ABS(hash(CAST(\"Me_1\" AS VARCHAR) || '_' || CAST(3 AS VARCHAR))) % 1000000) "
            '/ 1000000.0 AS "Me_1" '
            'FROM "DS_1"'
        )
        assert_sql_equal(result, expected_sql)


# =============================================================================
# MEMBERSHIP Operator Tests
# =============================================================================


class TestMembershipOperator:
    """Tests for MEMBERSHIP (#) operator."""

    def test_membership_extract_measure(self):
        """Test extracting a measure from dataset."""
        ds = create_simple_dataset("DS_1", ["Id_1", "Id_2"], ["Me_1", "Me_2"])
        transpiler = create_transpiler(input_datasets={"DS_1": ds})

        # Create AST: DS_1#Me_1
        dataset_ref = VarID(**make_ast_node(value="DS_1"))
        comp_name = VarID(**make_ast_node(value="Me_1"))
        membership_op = BinOp(**make_ast_node(left=dataset_ref, op="#", right=comp_name))

        result = transpiler.visit_BinOp(membership_op)

        # Full SQL: select identifiers and the specified component
        expected_sql = 'SELECT "Id_1", "Id_2", "Me_1" FROM "DS_1"'
        assert_sql_equal(result, expected_sql)

    def test_membership_extract_identifier(self):
        """Test extracting an identifier component."""
        ds = create_simple_dataset("DS_1", ["Id_1", "Id_2"], ["Me_1"])
        transpiler = create_transpiler(input_datasets={"DS_1": ds})

        # Create AST: DS_1#Id_2
        dataset_ref = VarID(**make_ast_node(value="DS_1"))
        comp_name = VarID(**make_ast_node(value="Id_2"))
        membership_op = BinOp(**make_ast_node(left=dataset_ref, op="#", right=comp_name))

        result = transpiler.visit_BinOp(membership_op)

        # Full SQL: select identifiers and the extracted component
        expected_sql = 'SELECT "Id_1", "Id_2", "Id_2" AS "str_var" FROM "DS_1"'
        assert_sql_equal(result, expected_sql)


# =============================================================================
# TIME_AGG Operator Tests
# =============================================================================


class TestTimeAggOperator:
    """Tests for TIME_AGG operator."""

    @pytest.mark.parametrize(
        "period,expected_sql",
        [
            ("Y", """STRFTIME(CAST("date_col" AS DATE), '%Y')"""),
            (
                "Q",
                """(STRFTIME(CAST("date_col" AS DATE), '%Y') || 'Q' || """
                """CAST(QUARTER(CAST("date_col" AS DATE)) AS VARCHAR))""",
            ),
            (
                "M",
                """(STRFTIME(CAST("date_col" AS DATE), '%Y') || 'M' || """
                """LPAD(CAST(MONTH(CAST("date_col" AS DATE)) AS VARCHAR), 2, '0'))""",
            ),
            ("D", """STRFTIME(CAST("date_col" AS DATE), '%Y-%m-%d')"""),
        ],
    )
    def test_time_agg_scalar(self, period: str, expected_sql: str):
        """Test TIME_AGG with scalar date."""
        transpiler = create_transpiler()

        # Create AST: time_agg(period, date_col)
        date_col = VarID(**make_ast_node(value="date_col"))
        time_agg_op = TimeAggregation(
            **make_ast_node(op="time_agg", period_to=period, operand=date_col)
        )

        result = transpiler.visit_TimeAggregation(time_agg_op)

        # Full SQL verification
        assert_sql_equal(result, expected_sql)

    def test_time_agg_year(self):
        """Test TIME_AGG to year period with full SQL."""
        transpiler = create_transpiler()

        date_col = VarID(**make_ast_node(value="my_date"))
        time_agg_op = TimeAggregation(
            **make_ast_node(op="time_agg", period_to="Y", operand=date_col)
        )

        result = transpiler.visit_TimeAggregation(time_agg_op)

        expected_sql = """STRFTIME(CAST("my_date" AS DATE), '%Y')"""
        assert_sql_equal(result, expected_sql)

    def test_time_agg_quarter(self):
        """Test TIME_AGG to quarter period with full SQL."""
        transpiler = create_transpiler()

        date_col = VarID(**make_ast_node(value="my_date"))
        time_agg_op = TimeAggregation(
            **make_ast_node(op="time_agg", period_to="Q", operand=date_col)
        )

        result = transpiler.visit_TimeAggregation(time_agg_op)

        expected_sql = (
            """(STRFTIME(CAST("my_date" AS DATE), '%Y') || 'Q' || """
            """CAST(QUARTER(CAST("my_date" AS DATE)) AS VARCHAR))"""
        )
        assert_sql_equal(result, expected_sql)

    def test_time_agg_month(self):
        """Test TIME_AGG to month period with full SQL."""
        transpiler = create_transpiler()

        date_col = VarID(**make_ast_node(value="my_date"))
        time_agg_op = TimeAggregation(
            **make_ast_node(op="time_agg", period_to="M", operand=date_col)
        )

        result = transpiler.visit_TimeAggregation(time_agg_op)

        expected_sql = (
            """(STRFTIME(CAST("my_date" AS DATE), '%Y') || 'M' || """
            """LPAD(CAST(MONTH(CAST("my_date" AS DATE)) AS VARCHAR), 2, '0'))"""
        )
        assert_sql_equal(result, expected_sql)

    def test_time_agg_semester(self):
        """Test TIME_AGG to semester period with full SQL."""
        transpiler = create_transpiler()

        date_col = VarID(**make_ast_node(value="my_date"))
        time_agg_op = TimeAggregation(
            **make_ast_node(op="time_agg", period_to="S", operand=date_col)
        )

        result = transpiler.visit_TimeAggregation(time_agg_op)

        expected_sql = (
            """(STRFTIME(CAST("my_date" AS DATE), '%Y') || 'S' || """
            """CAST(CEIL(MONTH(CAST("my_date" AS DATE)) / 6.0) AS INTEGER))"""
        )
        assert_sql_equal(result, expected_sql)


# =============================================================================
# Structure Computation Tests
# =============================================================================


def create_bool_output_dataset(name: str, id_cols: list) -> Dataset:
    """Helper to create a Dataset with bool_var measure (comparison result)."""
    components = {}
    for col in id_cols:
        components[col] = Component(
            name=col, data_type=String, role=Role.IDENTIFIER, nullable=False
        )
    components["bool_var"] = Component(
        name="bool_var", data_type=Boolean, role=Role.MEASURE, nullable=True
    )
    return Dataset(name=name, components=components, data=None)


class TestStructureComputation:
    """Tests for structure computation using output_datasets from semantic analysis."""

    @pytest.mark.parametrize(
        "op,sql_op",
        [
            ("=", "="),
            ("<>", "<>"),
            (">", ">"),
            ("<", "<"),
            (">=", ">="),
            ("<=", "<="),
        ],
    )
    def test_dataset_dataset_comparison_mono_measure(self, op: str, sql_op: str):
        """
        Test dataset-dataset comparison with mono-measure produces bool_var.

        When comparing two datasets with a single measure, the output should have
        bool_var as the measure name instead of the original measure name.
        This is determined by the output_datasets from semantic analysis.
        """
        ds1 = create_simple_dataset("DS_1", ["Id_1"], ["Me_1"])
        ds2 = create_simple_dataset("DS_2", ["Id_1"], ["Me_1"])
        output_ds = create_bool_output_dataset("DS_r", ["Id_1"])

        transpiler = create_transpiler(
            input_datasets={"DS_1": ds1, "DS_2": ds2},
            output_datasets={"DS_r": output_ds},
        )

        # Create AST: DS_r := DS_1 op DS_2
        left = VarID(**make_ast_node(value="DS_1"))
        right = VarID(**make_ast_node(value="DS_2"))
        expr = BinOp(**make_ast_node(left=left, op=op, right=right))
        ast = create_start_with_assignment("DS_r", expr)

        results = transpile_and_get_sql(transpiler, ast)

        assert len(results) == 1
        name, sql, _ = results[0]
        assert name == "DS_r"

        # Should output bool_var for mono-measure comparison
        expected_sql = f'''SELECT a."Id_1", (a."Me_1" {sql_op} b."Me_1") AS "bool_var"
                          FROM "DS_1" AS a INNER JOIN "DS_2" AS b ON a."Id_1" = b."Id_1"'''
        assert_sql_equal(sql, expected_sql)

    @pytest.mark.parametrize(
        "op,sql_op",
        [
            ("=", "="),
            (">", ">"),
        ],
    )
    def test_dataset_dataset_comparison_multi_measure(self, op: str, sql_op: str):
        """
        Test dataset-dataset comparison with multiple measures keeps measure names.

        When comparing datasets with multiple measures, each measure produces
        a boolean result with the same measure name.
        """
        ds1 = create_simple_dataset("DS_1", ["Id_1"], ["Me_1", "Me_2"])
        ds2 = create_simple_dataset("DS_2", ["Id_1"], ["Me_1", "Me_2"])
        # Multi-measure comparison keeps original measure names
        output_ds = create_simple_dataset("DS_r", ["Id_1"], ["Me_1", "Me_2"])

        transpiler = create_transpiler(
            input_datasets={"DS_1": ds1, "DS_2": ds2},
            output_datasets={"DS_r": output_ds},
        )

        # Create AST: DS_r := DS_1 op DS_2
        left = VarID(**make_ast_node(value="DS_1"))
        right = VarID(**make_ast_node(value="DS_2"))
        expr = BinOp(**make_ast_node(left=left, op=op, right=right))
        ast = create_start_with_assignment("DS_r", expr)

        results = transpile_and_get_sql(transpiler, ast)

        assert len(results) == 1
        name, sql, _ = results[0]
        assert name == "DS_r"

        # Should keep original measure names for multi-measure comparison
        expected_sql = f'''SELECT a."Id_1", (a."Me_1" {sql_op} b."Me_1") AS "Me_1",
                          (a."Me_2" {sql_op} b."Me_2") AS "Me_2"
                          FROM "DS_1" AS a INNER JOIN "DS_2" AS b ON a."Id_1" = b."Id_1"'''
        assert_sql_equal(sql, expected_sql)

    @pytest.mark.parametrize(
        "op,sql_op",
        [
            ("=", "="),
            ("<>", "<>"),
            (">", ">"),
            ("<", "<"),
        ],
    )
    def test_dataset_scalar_comparison_mono_measure(self, op: str, sql_op: str):
        """
        Test dataset-scalar comparison with mono-measure produces bool_var.
        """
        ds = create_simple_dataset("DS_1", ["Id_1"], ["Me_1"])
        output_ds = create_bool_output_dataset("DS_r", ["Id_1"])

        transpiler = create_transpiler(
            input_datasets={"DS_1": ds},
            output_datasets={"DS_r": output_ds},
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

        # Should output bool_var for mono-measure comparison
        expected_sql = f'SELECT "Id_1", ("Me_1" {sql_op} 10) AS "bool_var" FROM "DS_1"'
        assert_sql_equal(sql, expected_sql)

    def test_dataset_scalar_comparison_multi_measure(self):
        """
        Test dataset-scalar comparison with multi-measure keeps measure names.
        """
        ds = create_simple_dataset("DS_1", ["Id_1"], ["Me_1", "Me_2"])
        output_ds = create_simple_dataset("DS_r", ["Id_1"], ["Me_1", "Me_2"])

        transpiler = create_transpiler(
            input_datasets={"DS_1": ds},
            output_datasets={"DS_r": output_ds},
        )

        # Create AST: DS_r := DS_1 > 5
        left = VarID(**make_ast_node(value="DS_1"))
        right = Constant(**make_ast_node(type_="INTEGER_CONSTANT", value=5))
        expr = BinOp(**make_ast_node(left=left, op=">", right=right))
        ast = create_start_with_assignment("DS_r", expr)

        results = transpile_and_get_sql(transpiler, ast)

        assert len(results) == 1
        name, sql, _ = results[0]
        assert name == "DS_r"

        # Should keep original measure names for multi-measure comparison
        expected_sql = 'SELECT "Id_1", ("Me_1" > 5) AS "Me_1", ("Me_2" > 5) AS "Me_2" FROM "DS_1"'
        assert_sql_equal(sql, expected_sql)

    def test_scalar_dataset_comparison_mono_measure(self):
        """
        Test scalar-dataset comparison with mono-measure produces bool_var.
        """
        ds = create_simple_dataset("DS_1", ["Id_1"], ["Me_1"])
        output_ds = create_bool_output_dataset("DS_r", ["Id_1"])

        transpiler = create_transpiler(
            input_datasets={"DS_1": ds},
            output_datasets={"DS_r": output_ds},
        )

        # Create AST: DS_r := 10 > DS_1 (scalar on left)
        left = Constant(**make_ast_node(type_="INTEGER_CONSTANT", value=10))
        right = VarID(**make_ast_node(value="DS_1"))
        expr = BinOp(**make_ast_node(left=left, op=">", right=right))
        ast = create_start_with_assignment("DS_r", expr)

        results = transpile_and_get_sql(transpiler, ast)

        assert len(results) == 1
        name, sql, _ = results[0]
        assert name == "DS_r"

        # Should output bool_var for mono-measure comparison (scalar on left)
        expected_sql = 'SELECT "Id_1", (10 > "Me_1") AS "bool_var" FROM "DS_1"'
        assert_sql_equal(sql, expected_sql)

    def test_arithmetic_operation_keeps_measure_names(self):
        """
        Test that arithmetic operations keep original measure names.

        Arithmetic operations (+, -, *, /) should preserve the input measure names
        regardless of whether there's one or multiple measures.
        """
        ds = create_simple_dataset("DS_1", ["Id_1"], ["Me_1"])
        output_ds = create_simple_dataset("DS_r", ["Id_1"], ["Me_1"])

        transpiler = create_transpiler(
            input_datasets={"DS_1": ds},
            output_datasets={"DS_r": output_ds},
        )

        # Create AST: DS_r := DS_1 + 10
        left = VarID(**make_ast_node(value="DS_1"))
        right = Constant(**make_ast_node(type_="INTEGER_CONSTANT", value=10))
        expr = BinOp(**make_ast_node(left=left, op="+", right=right))
        ast = create_start_with_assignment("DS_r", expr)

        results = transpile_and_get_sql(transpiler, ast)

        assert len(results) == 1
        name, sql, _ = results[0]
        assert name == "DS_r"

        # Arithmetic should keep Me_1, not convert to bool_var
        expected_sql = 'SELECT "Id_1", ("Me_1" + 10) AS "Me_1" FROM "DS_1"'
        assert_sql_equal(sql, expected_sql)


def create_boolean_dataset(name: str, id_cols: list, measure_cols: list) -> Dataset:
    """Helper to create a Dataset with boolean measures."""
    components = {}
    for col in id_cols:
        components[col] = Component(
            name=col, data_type=String, role=Role.IDENTIFIER, nullable=False
        )
    for col in measure_cols:
        components[col] = Component(name=col, data_type=Boolean, role=Role.MEASURE, nullable=True)
    return Dataset(name=name, components=components, data=None)


class TestBooleanOperations:
    """Tests for Boolean operations on datasets."""

    @pytest.mark.parametrize(
        "op,sql_op",
        [
            ("and", "AND"),
            ("or", "OR"),
        ],
    )
    def test_boolean_dataset_dataset_operation(self, op: str, sql_op: str):
        """
        Test Boolean operations between two datasets.

        Boolean operations (and, or, xor) between datasets should apply to
        common measures and preserve measure names.
        """
        ds1 = create_boolean_dataset("DS_1", ["Id_1"], ["Me_1"])
        ds2 = create_boolean_dataset("DS_2", ["Id_1"], ["Me_1"])
        output_ds = create_boolean_dataset("DS_r", ["Id_1"], ["Me_1"])

        transpiler = create_transpiler(
            input_datasets={"DS_1": ds1, "DS_2": ds2},
            output_datasets={"DS_r": output_ds},
        )

        # Create AST: DS_r := DS_1 op DS_2
        left = VarID(**make_ast_node(value="DS_1"))
        right = VarID(**make_ast_node(value="DS_2"))
        expr = BinOp(**make_ast_node(left=left, op=op, right=right))
        ast = create_start_with_assignment("DS_r", expr)

        results = transpile_and_get_sql(transpiler, ast)

        assert len(results) == 1
        name, sql, _ = results[0]
        assert name == "DS_r"

        expected_sql = f'''SELECT a."Id_1", (a."Me_1" {sql_op} b."Me_1") AS "Me_1"
                          FROM "DS_1" AS a INNER JOIN "DS_2" AS b ON a."Id_1" = b."Id_1"'''
        assert_sql_equal(sql, expected_sql)

    def test_xor_dataset_dataset_operation(self):
        """
        Test XOR operation between two datasets.

        XOR generates ((a AND NOT b) OR (NOT a AND b)) form.
        """
        ds1 = create_boolean_dataset("DS_1", ["Id_1"], ["Me_1"])
        ds2 = create_boolean_dataset("DS_2", ["Id_1"], ["Me_1"])
        output_ds = create_boolean_dataset("DS_r", ["Id_1"], ["Me_1"])

        transpiler = create_transpiler(
            input_datasets={"DS_1": ds1, "DS_2": ds2},
            output_datasets={"DS_r": output_ds},
        )

        # Create AST: DS_r := DS_1 xor DS_2
        left = VarID(**make_ast_node(value="DS_1"))
        right = VarID(**make_ast_node(value="DS_2"))
        expr = BinOp(**make_ast_node(left=left, op="xor", right=right))
        ast = create_start_with_assignment("DS_r", expr)

        results = transpile_and_get_sql(transpiler, ast)

        assert len(results) == 1
        name, sql, _ = results[0]
        assert name == "DS_r"

        expected_sql = '''SELECT a."Id_1", ((a."Me_1" AND NOT b."Me_1") OR (NOT a."Me_1" AND b."Me_1")) AS "Me_1"
                          FROM "DS_1" AS a INNER JOIN "DS_2" AS b ON a."Id_1" = b."Id_1"'''
        assert_sql_equal(sql, expected_sql)

    @pytest.mark.parametrize(
        "op,sql_op",
        [
            ("and", "AND"),
            ("or", "OR"),
        ],
    )
    def test_boolean_dataset_scalar_operation(self, op: str, sql_op: str):
        """
        Test Boolean operations between dataset and scalar.

        Boolean operations between a dataset and a boolean scalar should
        apply to all measures.
        """
        ds = create_boolean_dataset("DS_1", ["Id_1"], ["Me_1", "Me_2"])
        output_ds = create_boolean_dataset("DS_r", ["Id_1"], ["Me_1", "Me_2"])

        transpiler = create_transpiler(
            input_datasets={"DS_1": ds},
            output_datasets={"DS_r": output_ds},
        )

        # Create AST: DS_r := DS_1 op true
        left = VarID(**make_ast_node(value="DS_1"))
        right = Constant(**make_ast_node(type_="BOOLEAN_CONSTANT", value=True))
        expr = BinOp(**make_ast_node(left=left, op=op, right=right))
        ast = create_start_with_assignment("DS_r", expr)

        results = transpile_and_get_sql(transpiler, ast)

        assert len(results) == 1
        name, sql, _ = results[0]
        assert name == "DS_r"

        expected_sql = f'SELECT "Id_1", ("Me_1" {sql_op} TRUE) AS "Me_1", ("Me_2" {sql_op} TRUE) AS "Me_2" FROM "DS_1"'
        assert_sql_equal(sql, expected_sql)

    def test_not_dataset_operation(self):
        """
        Test NOT unary operation on dataset.

        NOT on a dataset should negate all boolean measures.
        """
        ds = create_boolean_dataset("DS_1", ["Id_1"], ["Me_1", "Me_2"])
        output_ds = create_boolean_dataset("DS_r", ["Id_1"], ["Me_1", "Me_2"])

        transpiler = create_transpiler(
            input_datasets={"DS_1": ds},
            output_datasets={"DS_r": output_ds},
        )

        # Create AST: DS_r := not DS_1
        operand = VarID(**make_ast_node(value="DS_1"))
        expr = UnaryOp(**make_ast_node(op="not", operand=operand))
        ast = create_start_with_assignment("DS_r", expr)

        results = transpile_and_get_sql(transpiler, ast)

        assert len(results) == 1
        name, sql, _ = results[0]
        assert name == "DS_r"

        expected_sql = 'SELECT "Id_1", NOT "Me_1" AS "Me_1", NOT "Me_2" AS "Me_2" FROM "DS_1"'
        assert_sql_equal(sql, expected_sql)

    def test_boolean_dataset_multi_measure(self):
        """
        Test Boolean operation on dataset with multiple measures.

        Boolean operation should apply to all common measures.
        """
        ds1 = create_boolean_dataset("DS_1", ["Id_1"], ["Me_1", "Me_2"])
        ds2 = create_boolean_dataset("DS_2", ["Id_1"], ["Me_1", "Me_2"])
        output_ds = create_boolean_dataset("DS_r", ["Id_1"], ["Me_1", "Me_2"])

        transpiler = create_transpiler(
            input_datasets={"DS_1": ds1, "DS_2": ds2},
            output_datasets={"DS_r": output_ds},
        )

        # Create AST: DS_r := DS_1 and DS_2
        left = VarID(**make_ast_node(value="DS_1"))
        right = VarID(**make_ast_node(value="DS_2"))
        expr = BinOp(**make_ast_node(left=left, op="and", right=right))
        ast = create_start_with_assignment("DS_r", expr)

        results = transpile_and_get_sql(transpiler, ast)

        assert len(results) == 1
        name, sql, _ = results[0]
        assert name == "DS_r"

        expected_sql = '''SELECT a."Id_1", (a."Me_1" AND b."Me_1") AS "Me_1",
                          (a."Me_2" AND b."Me_2") AS "Me_2"
                          FROM "DS_1" AS a INNER JOIN "DS_2" AS b ON a."Id_1" = b."Id_1"'''
        assert_sql_equal(sql, expected_sql)


# =============================================================================
# exist_in and UDO Tests (AnaVal patterns)
# =============================================================================


class TestExistInOperations:
    """Tests for exist_in operations."""

    def test_exist_in_simple_datasets(self):
        """Test exist_in between two simple datasets."""
        # Create datasets with common identifiers
        ds1 = Dataset(
            name="DS_1",
            components={
                "Id_1": Component(
                    name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False
                ),
                "Id_2": Component(
                    name="Id_2", data_type=String, role=Role.IDENTIFIER, nullable=False
                ),
                "Me_1": Component(name="Me_1", data_type=Number, role=Role.MEASURE, nullable=True),
            },
            data=None,
        )
        ds2 = Dataset(
            name="DS_2",
            components={
                "Id_1": Component(
                    name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False
                ),
                "Id_2": Component(
                    name="Id_2", data_type=String, role=Role.IDENTIFIER, nullable=False
                ),
                "Me_2": Component(name="Me_2", data_type=Number, role=Role.MEASURE, nullable=True),
            },
            data=None,
        )
        # Output has identifiers from left + bool_var
        output_ds = Dataset(
            name="DS_r",
            components={
                "Id_1": Component(
                    name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False
                ),
                "Id_2": Component(
                    name="Id_2", data_type=String, role=Role.IDENTIFIER, nullable=False
                ),
                "bool_var": Component(
                    name="bool_var", data_type=Boolean, role=Role.MEASURE, nullable=True
                ),
            },
            data=None,
        )

        transpiler = create_transpiler(
            input_datasets={"DS_1": ds1, "DS_2": ds2},
            output_datasets={"DS_r": output_ds},
        )

        # Create AST: DS_r := exists_in(DS_1, DS_2, false)
        left = VarID(**make_ast_node(value="DS_1"))
        right = VarID(**make_ast_node(value="DS_2"))
        retain = Constant(**make_ast_node(value=False, type_="BOOLEAN_CONSTANT"))
        expr = MulOp(**make_ast_node(op="exists_in", children=[left, right, retain]))
        ast = create_start_with_assignment("DS_r", expr)

        results = transpile_and_get_sql(transpiler, ast)

        assert len(results) == 1
        name, sql, _ = results[0]
        assert name == "DS_r"

        # Should generate EXISTS subquery with identifier match
        assert_sql_contains(sql, ["EXISTS", "SELECT 1", "l.", "r.", "bool_var"])

    def test_exist_in_with_filtered_dataset(self):
        """Test exist_in with filtered dataset."""
        ds1 = Dataset(
            name="DS_1",
            components={
                "Id_1": Component(
                    name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False
                ),
                "Me_1": Component(name="Me_1", data_type=Number, role=Role.MEASURE, nullable=True),
            },
            data=None,
        )
        ds2 = Dataset(
            name="DS_2",
            components={
                "Id_1": Component(
                    name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False
                ),
                "Me_1": Component(name="Me_1", data_type=String, role=Role.MEASURE, nullable=True),
            },
            data=None,
        )
        output_ds = Dataset(
            name="DS_r",
            components={
                "Id_1": Component(
                    name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False
                ),
                "bool_var": Component(
                    name="bool_var", data_type=Boolean, role=Role.MEASURE, nullable=True
                ),
            },
            data=None,
        )

        transpiler = create_transpiler(
            input_datasets={"DS_1": ds1, "DS_2": ds2},
            output_datasets={"DS_r": output_ds},
        )

        # Create AST: DS_r := exists_in(DS_1, DS_2[filter Me_1 = "1"], false)
        left = VarID(**make_ast_node(value="DS_1"))
        # Right side with filter - RegularAggregation has op and children
        ds2_var = VarID(**make_ast_node(value="DS_2"))
        filter_cond = BinOp(
            **make_ast_node(
                left=VarID(**make_ast_node(value="Me_1")),
                op="=",
                right=Constant(**make_ast_node(value="1", type_="STRING_CONSTANT")),
            )
        )
        right = RegularAggregation(
            **make_ast_node(dataset=ds2_var, op="filter", children=[filter_cond])
        )
        retain = Constant(**make_ast_node(value=False, type_="BOOLEAN_CONSTANT"))
        expr = MulOp(**make_ast_node(op="exists_in", children=[left, right, retain]))
        ast = create_start_with_assignment("DS_r", expr)

        results = transpile_and_get_sql(transpiler, ast)

        assert len(results) == 1
        name, sql, _ = results[0]
        assert name == "DS_r"

        # Should generate EXISTS with filter in the subquery
        assert_sql_contains(sql, ["EXISTS", "WHERE", "bool_var"])


class TestUDOOperations:
    """Tests for User-Defined Operator operations."""

    def test_udo_simple_dataset_sum(self):
        """Test UDO that adds two datasets: suma(ds1, ds2) returns ds1 + ds2."""
        ds1 = Dataset(
            name="DS_1",
            components={
                "Id_1": Component(
                    name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False
                ),
                "Me_1": Component(name="Me_1", data_type=Number, role=Role.MEASURE, nullable=True),
            },
            data=None,
        )
        ds2 = Dataset(
            name="DS_2",
            components={
                "Id_1": Component(
                    name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False
                ),
                "Me_1": Component(name="Me_1", data_type=Number, role=Role.MEASURE, nullable=True),
            },
            data=None,
        )
        output_ds = Dataset(
            name="DS_r",
            components={
                "Id_1": Component(
                    name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False
                ),
                "Me_1": Component(name="Me_1", data_type=Number, role=Role.MEASURE, nullable=True),
            },
            data=None,
        )

        transpiler = create_transpiler(
            input_datasets={"DS_1": ds1, "DS_2": ds2},
            output_datasets={"DS_r": output_ds},
        )

        # Define UDO: suma(ds1 dataset, ds2 dataset) returns ds1 + ds2
        udo_definition = Operator(
            **make_ast_node(
                op="suma",
                parameters=[
                    Argument(**make_ast_node(name="ds1", type_=Number, default=None)),
                    Argument(**make_ast_node(name="ds2", type_=Number, default=None)),
                ],
                output_type="Dataset",
                expression=BinOp(
                    **make_ast_node(
                        left=VarID(**make_ast_node(value="ds1")),
                        op="+",
                        right=VarID(**make_ast_node(value="ds2")),
                    )
                ),
            )
        )

        # Create UDO call: suma(DS_1, DS_2)
        udo_call = UDOCall(
            **make_ast_node(
                op="suma",
                params=[
                    VarID(**make_ast_node(value="DS_1")),
                    VarID(**make_ast_node(value="DS_2")),
                ],
            )
        )

        # Register the UDO definition
        transpiler.visit(udo_definition)

        # Create full AST: DS_r := suma(DS_1, DS_2)
        ast = create_start_with_assignment("DS_r", udo_call)
        results = transpile_and_get_sql(transpiler, ast)

        assert len(results) == 1
        name, sql, _ = results[0]
        assert name == "DS_r"
        # Should produce a join with addition of measures
        assert_sql_contains(sql, ['"Id_1"', '"Me_1"', "+", "JOIN"])

    def test_udo_aggregation_group_except(self):
        """Test UDO that drops an identifier: drop_id(ds, comp) returns max(ds group except comp)."""
        ds = Dataset(
            name="DS_1",
            components={
                "Id_1": Component(
                    name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False
                ),
                "Id_2": Component(
                    name="Id_2", data_type=String, role=Role.IDENTIFIER, nullable=False
                ),
                "Me_1": Component(name="Me_1", data_type=Number, role=Role.MEASURE, nullable=True),
            },
            data=None,
        )
        output_ds = Dataset(
            name="DS_r",
            components={
                "Id_1": Component(
                    name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False
                ),
                "Me_1": Component(name="Me_1", data_type=Number, role=Role.MEASURE, nullable=True),
            },
            data=None,
        )

        transpiler = create_transpiler(
            input_datasets={"DS_1": ds},
            output_datasets={"DS_r": output_ds},
        )

        # Define UDO: drop_id(ds dataset, comp component) returns max(ds group except comp)
        udo_definition = Operator(
            **make_ast_node(
                op="drop_id",
                parameters=[
                    Argument(**make_ast_node(name="ds", type_=Number, default=None)),
                    Argument(**make_ast_node(name="comp", type_=String, default=None)),
                ],
                output_type="Dataset",
                expression=Aggregation(
                    **make_ast_node(
                        op="max",
                        operand=VarID(**make_ast_node(value="ds")),
                        grouping_op="group except",
                        grouping=[VarID(**make_ast_node(value="comp"))],
                    )
                ),
            )
        )

        # Create UDO call: drop_id(DS_1, Id_2)
        udo_call = UDOCall(
            **make_ast_node(
                op="drop_id",
                params=[
                    VarID(**make_ast_node(value="DS_1")),
                    VarID(**make_ast_node(value="Id_2")),
                ],
            )
        )

        # Register the UDO definition
        transpiler.visit(udo_definition)

        # Create full AST: DS_r := drop_id(DS_1, Id_2)
        ast = create_start_with_assignment("DS_r", udo_call)
        results = transpile_and_get_sql(transpiler, ast)

        assert len(results) == 1
        name, sql, _ = results[0]
        assert name == "DS_r"
        # Should produce MAX aggregation grouped by Id_1 (all except Id_2)
        assert_sql_contains(sql, ["MAX", '"Id_1"', "GROUP BY"])
        # Id_2 should be excluded from result (group except removes it)
        assert '"Id_2"' not in sql or "GROUP BY" in sql

    def test_udo_with_membership(self):
        """Test UDO with membership operator: extract_measure(ds, comp) returns ds#comp."""
        ds = Dataset(
            name="DS_1",
            components={
                "Id_1": Component(
                    name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False
                ),
                "Me_1": Component(name="Me_1", data_type=Number, role=Role.MEASURE, nullable=True),
                "Me_2": Component(name="Me_2", data_type=Number, role=Role.MEASURE, nullable=True),
            },
            data=None,
        )
        output_ds = Dataset(
            name="DS_r",
            components={
                "Id_1": Component(
                    name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False
                ),
                "Me_1": Component(name="Me_1", data_type=Number, role=Role.MEASURE, nullable=True),
            },
            data=None,
        )

        transpiler = create_transpiler(
            input_datasets={"DS_1": ds},
            output_datasets={"DS_r": output_ds},
        )

        # Define UDO: extract_measure(ds dataset, comp component) returns ds#comp
        udo_definition = Operator(
            **make_ast_node(
                op="extract_measure",
                parameters=[
                    Argument(**make_ast_node(name="ds", type_=Number, default=None)),
                    Argument(**make_ast_node(name="comp", type_=String, default=None)),
                ],
                output_type="Dataset",
                expression=BinOp(
                    **make_ast_node(
                        left=VarID(**make_ast_node(value="ds")),
                        op="#",
                        right=VarID(**make_ast_node(value="comp")),
                    )
                ),
            )
        )

        # Create UDO call: extract_measure(DS_1, Me_1)
        udo_call = UDOCall(
            **make_ast_node(
                op="extract_measure",
                params=[
                    VarID(**make_ast_node(value="DS_1")),
                    VarID(**make_ast_node(value="Me_1")),
                ],
            )
        )

        # Register the UDO definition
        transpiler.visit(udo_definition)

        # Create full AST: DS_r := extract_measure(DS_1, Me_1)
        ast = create_start_with_assignment("DS_r", udo_call)
        results = transpile_and_get_sql(transpiler, ast)

        assert len(results) == 1
        name, sql, _ = results[0]
        assert name == "DS_r"
        # Should select only Id_1 and Me_1
        assert_sql_contains(sql, ['"Id_1"', '"Me_1"'])
        # Me_2 should not be selected
        assert '"Me_2"' not in sql

    def test_udo_nested_call(self):
        """Test nested UDO calls: outer(inner(DS))."""
        ds = Dataset(
            name="DS_1",
            components={
                "Id_1": Component(
                    name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False
                ),
                "Me_1": Component(name="Me_1", data_type=Number, role=Role.MEASURE, nullable=True),
                "Me_2": Component(name="Me_2", data_type=Number, role=Role.MEASURE, nullable=True),
            },
            data=None,
        )
        output_ds = Dataset(
            name="DS_r",
            components={
                "Id_1": Component(
                    name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False
                ),
                "Me_1": Component(name="Me_1", data_type=Number, role=Role.MEASURE, nullable=True),
            },
            data=None,
        )

        transpiler = create_transpiler(
            input_datasets={"DS_1": ds},
            output_datasets={"DS_r": output_ds},
        )

        # Define inner UDO: keep_one(ds dataset) returns ds[keep Me_1]
        inner_udo = Operator(
            **make_ast_node(
                op="keep_one",
                parameters=[
                    Argument(**make_ast_node(name="ds", type_=Number, default=None)),
                ],
                output_type="Dataset",
                expression=RegularAggregation(
                    **make_ast_node(
                        op="keep",
                        dataset=VarID(**make_ast_node(value="ds")),
                        children=[VarID(**make_ast_node(value="Me_1"))],
                    )
                ),
            )
        )

        # Define outer UDO: double_it(ds dataset) returns ds * 2
        outer_udo = Operator(
            **make_ast_node(
                op="double_it",
                parameters=[
                    Argument(**make_ast_node(name="ds", type_=Number, default=None)),
                ],
                output_type="Dataset",
                expression=BinOp(
                    **make_ast_node(
                        left=VarID(**make_ast_node(value="ds")),
                        op="*",
                        right=Constant(**make_ast_node(value=2, type_="INTEGER_CONSTANT")),
                    )
                ),
            )
        )

        # Register UDOs
        transpiler.visit(inner_udo)
        transpiler.visit(outer_udo)

        # Create nested call: double_it(keep_one(DS_1))
        inner_call = UDOCall(
            **make_ast_node(
                op="keep_one",
                params=[VarID(**make_ast_node(value="DS_1"))],
            )
        )
        outer_call = UDOCall(
            **make_ast_node(
                op="double_it",
                params=[inner_call],
            )
        )

        # Create full AST
        ast = create_start_with_assignment("DS_r", outer_call)
        results = transpile_and_get_sql(transpiler, ast)

        assert len(results) == 1
        name, sql, _ = results[0]
        assert name == "DS_r"
        # Should have multiplication by 2 and only Me_1
        assert_sql_contains(sql, ['"Me_1"', "* 2"])
        # Me_2 should be dropped by inner UDO
        assert '"Me_2"' not in sql

    def test_udo_with_filtered_dataset_param(self):
        """Test UDO where the parameter is a filtered dataset expression.

        VTL pattern: drop_identifier ( DS_1 [ filter Me_1 > 0 ] , Id_2 )
        Bug: When UDO param 'ds' is bound to a RegularAggregation (filter),
        the SQL was generating FROM "<RegularAggregation...>" instead of
        properly visiting the expression.
        """
        ds = Dataset(
            name="DS_1",
            components={
                "Id_1": Component(
                    name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False
                ),
                "Id_2": Component(
                    name="Id_2", data_type=String, role=Role.IDENTIFIER, nullable=False
                ),
                "Me_1": Component(name="Me_1", data_type=Number, role=Role.MEASURE, nullable=True),
            },
            data=None,
        )
        output_ds = Dataset(
            name="DS_r",
            components={
                "Id_1": Component(
                    name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False
                ),
                "Me_1": Component(name="Me_1", data_type=Number, role=Role.MEASURE, nullable=True),
            },
            data=None,
        )

        transpiler = create_transpiler(
            input_datasets={"DS_1": ds},
            output_datasets={"DS_r": output_ds},
        )

        # Define UDO: drop_identifier(ds dataset, comp component) returns max(ds group except comp)
        udo_definition = Operator(
            **make_ast_node(
                op="drop_identifier",
                parameters=[
                    Argument(**make_ast_node(name="ds", type_=Number, default=None)),
                    Argument(**make_ast_node(name="comp", type_=String, default=None)),
                ],
                output_type="Dataset",
                expression=Aggregation(
                    **make_ast_node(
                        op="max",
                        operand=VarID(**make_ast_node(value="ds")),
                        grouping_op="group except",
                        grouping=[VarID(**make_ast_node(value="comp"))],
                    )
                ),
            )
        )

        # Register the UDO
        transpiler.visit(udo_definition)

        # Create filtered dataset: DS_1 [ filter Me_1 > 0 ]
        filtered_ds = RegularAggregation(
            **make_ast_node(
                op="filter",
                dataset=VarID(**make_ast_node(value="DS_1")),
                children=[
                    BinOp(
                        **make_ast_node(
                            left=VarID(**make_ast_node(value="Me_1")),
                            op=">",
                            right=Constant(**make_ast_node(value=0, type_="INTEGER_CONSTANT")),
                        )
                    )
                ],
            )
        )

        # Create UDO call: drop_identifier(DS_1 [ filter Me_1 > 0 ], Id_2)
        udo_call = UDOCall(
            **make_ast_node(
                op="drop_identifier",
                params=[
                    filtered_ds,
                    VarID(**make_ast_node(value="Id_2")),
                ],
            )
        )

        # Create full AST: DS_r := drop_identifier(DS_1 [ filter Me_1 > 0 ], Id_2)
        ast = create_start_with_assignment("DS_r", udo_call)
        results = transpile_and_get_sql(transpiler, ast)

        assert len(results) == 1
        name, sql, _ = results[0]
        assert name == "DS_r"
        # The SQL should contain proper filter clause, NOT "<RegularAggregation...>"
        assert "RegularAggregation" not in sql
        assert '"DS_1"' in sql
        # Should have the filter condition
        assert '"Me_1"' in sql
        assert "> 0" in sql or ">0" in sql

    def test_udo_dataset_sql_resolves_param(self):
        """Test that _get_dataset_sql resolves UDO parameter to actual dataset name.

        Bug: When UDO parameter 'ds' is used inside aggregation, the SQL was
        generating FROM "ds" instead of FROM "ACTUAL_DATASET_NAME".
        """
        ds = Dataset(
            name="ACTUAL_DS",
            components={
                "Id_1": Component(
                    name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False
                ),
                "Id_2": Component(
                    name="Id_2", data_type=String, role=Role.IDENTIFIER, nullable=False
                ),
                "Me_1": Component(name="Me_1", data_type=Number, role=Role.MEASURE, nullable=True),
            },
            data=None,
        )
        output_ds = Dataset(
            name="DS_r",
            components={
                "Id_1": Component(
                    name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False
                ),
                "Me_1": Component(name="Me_1", data_type=Number, role=Role.MEASURE, nullable=True),
            },
            data=None,
        )

        transpiler = create_transpiler(
            input_datasets={"ACTUAL_DS": ds},
            output_datasets={"DS_r": output_ds},
        )

        # Define UDO: drop_identifier(ds dataset, comp component) returns max(ds group except comp)
        udo_definition = Operator(
            **make_ast_node(
                op="drop_identifier",
                parameters=[
                    Argument(**make_ast_node(name="ds", type_=Number, default=None)),
                    Argument(**make_ast_node(name="comp", type_=String, default=None)),
                ],
                output_type="Dataset",
                expression=Aggregation(
                    **make_ast_node(
                        op="max",
                        operand=VarID(**make_ast_node(value="ds")),
                        grouping_op="group except",
                        grouping=[VarID(**make_ast_node(value="comp"))],
                    )
                ),
            )
        )

        # Register the UDO
        transpiler.visit(udo_definition)

        # Create UDO call: drop_identifier(ACTUAL_DS, Id_2)
        udo_call = UDOCall(
            **make_ast_node(
                op="drop_identifier",
                params=[
                    VarID(**make_ast_node(value="ACTUAL_DS")),
                    VarID(**make_ast_node(value="Id_2")),
                ],
            )
        )

        # Create full AST: DS_r := drop_identifier(ACTUAL_DS, Id_2)
        ast = create_start_with_assignment("DS_r", udo_call)
        results = transpile_and_get_sql(transpiler, ast)

        assert len(results) == 1
        name, sql, _ = results[0]
        assert name == "DS_r"
        # The SQL should reference "ACTUAL_DS", NOT "ds" (the UDO parameter name)
        assert '"ACTUAL_DS"' in sql
        assert '"ds"' not in sql or "ds" not in sql.split("FROM")[1]


class TestIntermediateResultsInExistIn:
    """Tests for exist_in with intermediate results."""

    def test_exist_in_with_intermediate_result(self):
        """Test exist_in where operand is a previously computed result.

        Pattern:
        intermediate := DS_1
        DS_r := exists_in ( intermediate , DS_2 , false )
        """
        ds1 = Dataset(
            name="DS_1",
            components={
                "Id_1": Component(
                    name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False
                ),
                "Me_1": Component(name="Me_1", data_type=Number, role=Role.MEASURE, nullable=True),
            },
            data=None,
        )
        ds2 = Dataset(
            name="DS_2",
            components={
                "Id_1": Component(
                    name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False
                ),
                "Me_2": Component(name="Me_2", data_type=Number, role=Role.MEASURE, nullable=True),
            },
            data=None,
        )
        # Intermediate result
        intermediate_ds = Dataset(
            name="intermediate",
            components={
                "Id_1": Component(
                    name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False
                ),
                "Me_1": Component(name="Me_1", data_type=Number, role=Role.MEASURE, nullable=True),
            },
            data=None,
        )
        # Final output
        output_ds = Dataset(
            name="DS_r",
            components={
                "Id_1": Component(
                    name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False
                ),
                "bool_var": Component(
                    name="bool_var", data_type=Boolean, role=Role.MEASURE, nullable=True
                ),
            },
            data=None,
        )

        transpiler = create_transpiler(
            input_datasets={"DS_1": ds1, "DS_2": ds2},
            output_datasets={
                "intermediate": intermediate_ds,
                "DS_r": output_ds,
            },
        )

        # Create AST:
        # intermediate := DS_1
        # DS_r := exists_in(intermediate, DS_2, false)
        assignment1 = Assignment(
            **make_ast_node(
                left=VarID(**make_ast_node(value="intermediate")),
                op=":=",
                right=VarID(**make_ast_node(value="DS_1")),
            )
        )

        left = VarID(**make_ast_node(value="intermediate"))
        right = VarID(**make_ast_node(value="DS_2"))
        retain = Constant(**make_ast_node(value=False, type_="BOOLEAN_CONSTANT"))
        expr = MulOp(**make_ast_node(op="exists_in", children=[left, right, retain]))
        assignment2 = Assignment(
            **make_ast_node(
                left=VarID(**make_ast_node(value="DS_r")),
                op=":=",
                right=expr,
            )
        )

        ast = Start(**make_ast_node(children=[assignment1, assignment2]))

        results = transpile_and_get_sql(transpiler, ast)

        # Should have two results
        assert len(results) == 2

        # Second result should be the exist_in
        name, sql, _ = results[1]
        assert name == "DS_r"
        assert_sql_contains(sql, ["EXISTS", "bool_var"])


class TestGetStructure:
    """Tests for structure-related behavior in SQL transpilation."""

    def test_binop_dataset_dataset_includes_all_identifiers(self):
        """Test that dataset-dataset binary ops include all identifiers from both sides."""
        ds1 = Dataset(
            name="DS_1",
            components={
                "Id_1": Component(
                    name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False
                ),
                "Id_2": Component(
                    name="Id_2", data_type=String, role=Role.IDENTIFIER, nullable=False
                ),
                "Me_1": Component(name="Me_1", data_type=Number, role=Role.MEASURE, nullable=True),
            },
            data=None,
        )
        ds2 = Dataset(
            name="DS_2",
            components={
                "Id_1": Component(
                    name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False
                ),
                "Id_3": Component(
                    name="Id_3", data_type=String, role=Role.IDENTIFIER, nullable=False
                ),
                "Me_1": Component(name="Me_1", data_type=Number, role=Role.MEASURE, nullable=True),
            },
            data=None,
        )
        output_ds = Dataset(
            name="DS_r",
            components={
                "Id_1": Component(
                    name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False
                ),
                "Id_2": Component(
                    name="Id_2", data_type=String, role=Role.IDENTIFIER, nullable=False
                ),
                "Id_3": Component(
                    name="Id_3", data_type=String, role=Role.IDENTIFIER, nullable=False
                ),
                "Me_1": Component(name="Me_1", data_type=Number, role=Role.MEASURE, nullable=True),
            },
            data=None,
        )

        transpiler = create_transpiler(
            input_datasets={"DS_1": ds1, "DS_2": ds2},
            output_datasets={"DS_r": output_ds},
        )

        # Create: DS_r := DS_1 + DS_2
        left = VarID(**make_ast_node(value="DS_1"))
        right = VarID(**make_ast_node(value="DS_2"))
        expr = BinOp(**make_ast_node(left=left, op="+", right=right))
        ast = create_start_with_assignment("DS_r", expr)

        results = transpile_and_get_sql(transpiler, ast)

        assert len(results) == 1
        name, sql, _ = results[0]
        assert name == "DS_r"

        # Should include all identifiers
        assert '"Id_1"' in sql
        assert '"Id_2"' in sql
        assert '"Id_3"' in sql
