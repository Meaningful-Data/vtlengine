"""
Transpiler Time Type Integration Tests

Tests for TimePeriod and TimeInterval handling in the VTL-to-SQL transpiler.
Tests verify the generated SQL uses proper time type functions.
"""

from typing import Any, Dict

import duckdb
import pytest

from vtlengine.AST import (
    Assignment,
    BinOp,
    Constant,
    RegularAggregation,
    Start,
    TimeAggregation,
    UnaryOp,
    VarID,
)
from vtlengine.AST.Grammar.tokens import (
    PERIOD_INDICATOR,
    TIMESHIFT,
    YEAR,
)
from vtlengine.DataTypes import Integer, Number, String, TimePeriod, TimeInterval
from vtlengine.duckdb_transpiler.sql import initialize_time_types
from vtlengine.duckdb_transpiler.Transpiler import SQLTranspiler
from vtlengine.Model import Component, Dataset, Role


# =============================================================================
# Test Utilities
# =============================================================================


def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison (remove extra whitespace)."""
    return " ".join(sql.split()).strip()


def assert_sql_contains(actual: str, expected_parts: list):
    """Assert that SQL contains all expected parts."""
    normalized = normalize_sql(actual)
    for part in normalized_parts(expected_parts):
        assert part in normalized, f"Expected '{part}' not found in SQL:\n{actual}"


def normalized_parts(parts: list) -> list:
    """Normalize expected parts for comparison."""
    return [normalize_sql(p) for p in parts]


def create_time_period_dataset(
    name: str, time_col: str = "time_id", measure_cols: list = None
) -> Dataset:
    """Create a Dataset with a TimePeriod identifier."""
    measure_cols = measure_cols or ["Me_1"]
    components = {
        time_col: Component(
            name=time_col, data_type=TimePeriod, role=Role.IDENTIFIER, nullable=False
        )
    }
    for col in measure_cols:
        components[col] = Component(name=col, data_type=Number, role=Role.MEASURE, nullable=True)
    return Dataset(name=name, components=components, data=None)


def create_time_interval_dataset(
    name: str, time_col: str = "time_id", measure_cols: list = None
) -> Dataset:
    """Create a Dataset with a TimeInterval identifier."""
    measure_cols = measure_cols or ["Me_1"]
    components = {
        time_col: Component(
            name=time_col, data_type=TimeInterval, role=Role.IDENTIFIER, nullable=False
        )
    }
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


def transpile_and_get_sql(transpiler: SQLTranspiler, ast: Start) -> list:
    """Transpile AST and return results list."""
    return transpiler.transpile(ast)


# =============================================================================
# Tests: TIMESHIFT with TimePeriod
# =============================================================================


class TestTimeshiftTimePeriod:
    """Tests for TIMESHIFT operation with TimePeriod identifiers."""

    def test_timeshift_generates_vtl_period_shift(self):
        """Verify TIMESHIFT uses vtl_period_shift for TimePeriod columns."""
        ds = create_time_period_dataset("DS_1", "time_id", ["Me_1"])
        transpiler = create_transpiler(
            input_datasets={"DS_1": ds},
            output_datasets={"DS_r": ds},
        )

        # Create AST: DS_r := timeshift(DS_1, 1)
        dataset_ref = VarID(**make_ast_node(value="DS_1"))
        shift_val = Constant(**make_ast_node(type_="INTEGER_CONSTANT", value=1))
        expr = BinOp(**make_ast_node(left=dataset_ref, op=TIMESHIFT, right=shift_val))
        ast = create_start_with_assignment("DS_r", expr)

        results = transpile_and_get_sql(transpiler, ast)

        assert len(results) == 1
        _, sql, _ = results[0]

        # Should use vtl_period_shift function
        assert "vtl_period_shift" in sql
        assert "vtl_period_parse" in sql
        assert "vtl_period_to_string" in sql

    def test_timeshift_execution(self):
        """Test that TIMESHIFT SQL actually executes correctly."""
        conn = duckdb.connect(":memory:")
        initialize_time_types(conn)

        # Create test data
        conn.execute("""
            CREATE TABLE DS_1 (time_id VARCHAR, Me_1 DOUBLE);
            INSERT INTO DS_1 VALUES ('2020-Q1', 10.0), ('2020-Q2', 20.0);
        """)

        # Run the timeshift query
        sql = """
            SELECT vtl_period_to_string(vtl_period_shift(vtl_period_parse(time_id), 1)) AS time_id, Me_1
            FROM DS_1
        """
        result = conn.execute(sql).fetchall()

        # Should shift by 1 quarter
        assert result[0][0] == "2020-Q2"
        assert result[1][0] == "2020-Q3"

        conn.close()


# =============================================================================
# Tests: PERIOD_INDICATOR
# =============================================================================


class TestPeriodIndicator:
    """Tests for PERIOD_INDICATOR operation."""

    def test_period_indicator_generates_vtl_function(self):
        """Verify PERIOD_INDICATOR uses vtl_period_indicator function."""
        ds = create_time_period_dataset("DS_1", "time_id", ["Me_1"])
        transpiler = create_transpiler(
            input_datasets={"DS_1": ds},
            output_datasets={"DS_r": ds},
        )

        # Create AST: DS_r := period_indicator(DS_1)
        dataset_ref = VarID(**make_ast_node(value="DS_1"))
        expr = UnaryOp(**make_ast_node(op=PERIOD_INDICATOR, operand=dataset_ref))
        ast = create_start_with_assignment("DS_r", expr)

        results = transpile_and_get_sql(transpiler, ast)

        assert len(results) == 1
        _, sql, _ = results[0]

        # Should use vtl_period_indicator function
        assert "vtl_period_indicator" in sql
        assert "vtl_period_parse" in sql

    def test_period_indicator_execution(self):
        """Test that PERIOD_INDICATOR SQL actually executes correctly."""
        conn = duckdb.connect(":memory:")
        initialize_time_types(conn)

        # Create test data
        conn.execute("""
            CREATE TABLE DS_1 (time_id VARCHAR);
            INSERT INTO DS_1 VALUES ('2020-Q1'), ('2020M06'), ('2020');
        """)

        # Run the period_indicator query
        sql = """
            SELECT time_id, vtl_period_indicator(vtl_period_parse(time_id)) AS indicator
            FROM DS_1
        """
        result = conn.execute(sql).fetchall()

        assert result[0][1] == "Q"
        assert result[1][1] == "M"
        assert result[2][1] == "A"

        conn.close()


# =============================================================================
# Tests: TIME_AGG with TimePeriod
# =============================================================================


class TestTimeAggTimePeriod:
    """Tests for TIME_AGG operation with TimePeriod."""

    def test_time_agg_execution_with_time_period(self):
        """Test that TIME_AGG SQL executes correctly for TimePeriod input."""
        conn = duckdb.connect(":memory:")
        initialize_time_types(conn)

        # Create test data
        conn.execute("""
            CREATE TABLE DS_1 (time_id VARCHAR, Me_1 DOUBLE);
            INSERT INTO DS_1 VALUES ('2020-Q1', 10.0), ('2020-Q2', 20.0), ('2020-Q3', 30.0);
        """)

        # Run time_agg to aggregate to annual
        sql = """
            SELECT vtl_period_to_string(vtl_time_agg(vtl_period_parse(time_id), 'A')) AS time_id, Me_1
            FROM DS_1
        """
        result = conn.execute(sql).fetchall()

        # All should aggregate to 2020 (annual)
        for row in result:
            assert row[0] == "2020"

        conn.close()


# =============================================================================
# Tests: TimePeriod Comparison
# =============================================================================


class TestTimePeriodComparison:
    """Tests for TimePeriod comparison operations."""

    @pytest.mark.parametrize(
        "op,left,right,expected",
        [
            ("<", "2020-Q1", "2020-Q2", True),
            ("<", "2020-Q2", "2020-Q1", False),
            ("<=", "2020-Q1", "2020-Q1", True),
            (">", "2020-Q2", "2020-Q1", True),
            (">=", "2020-Q2", "2020-Q2", True),
            ("=", "2020-Q1", "2020-Q1", True),
            ("=", "2020-Q1", "2020-Q2", False),
            ("<>", "2020-Q1", "2020-Q2", True),
        ],
    )
    def test_time_period_comparison_execution(self, op, left, right, expected):
        """Test TimePeriod comparison functions execute correctly."""
        conn = duckdb.connect(":memory:")
        initialize_time_types(conn)

        # Map operator to function
        op_map = {
            "<": "vtl_period_lt",
            "<=": "vtl_period_le",
            ">": "vtl_period_gt",
            ">=": "vtl_period_ge",
            "=": "vtl_period_eq",
            "<>": "vtl_period_ne",
        }
        func = op_map[op]

        sql = f"SELECT {func}(vtl_period_parse('{left}'), vtl_period_parse('{right}'))"
        result = conn.execute(sql).fetchone()[0]

        assert result == expected

        conn.close()


# =============================================================================
# Tests: TimeInterval Comparison
# =============================================================================


class TestTimeIntervalComparison:
    """Tests for TimeInterval comparison operations."""

    @pytest.mark.parametrize(
        "op,left,right,expected",
        [
            ("<", "2020-01-01/2020-06-30", "2021-01-01/2021-06-30", True),
            (">", "2021-01-01/2021-12-31", "2020-01-01/2020-12-31", True),
            ("=", "2020-01-01/2020-12-31", "2020-01-01/2020-12-31", True),
            ("=", "2020-01-01/2020-12-31", "2021-01-01/2021-12-31", False),
        ],
    )
    def test_time_interval_comparison_execution(self, op, left, right, expected):
        """Test TimeInterval comparison functions execute correctly."""
        conn = duckdb.connect(":memory:")
        initialize_time_types(conn)

        # Map operator to function
        op_map = {
            "<": "vtl_interval_lt",
            "<=": "vtl_interval_le",
            ">": "vtl_interval_gt",
            ">=": "vtl_interval_ge",
            "=": "vtl_interval_eq",
            "<>": "vtl_interval_ne",
        }
        func = op_map[op]

        sql = f"SELECT {func}(vtl_interval_parse('{left}'), vtl_interval_parse('{right}'))"
        result = conn.execute(sql).fetchone()[0]

        assert result == expected

        conn.close()


# =============================================================================
# Tests: Year Extraction from TimePeriod
# =============================================================================


class TestYearExtraction:
    """Tests for YEAR extraction from TimePeriod."""

    def test_year_extraction_execution(self):
        """Test that YEAR extraction works for TimePeriod."""
        conn = duckdb.connect(":memory:")
        initialize_time_types(conn)

        # Test year extraction from various period formats
        test_cases = [
            ("2020", 2020),
            ("2020-Q1", 2020),
            ("2021-M06", 2021),
            ("2022-W15", 2022),
        ]

        for period, expected_year in test_cases:
            sql = f"SELECT vtl_period_year(vtl_period_parse('{period}'))"
            result = conn.execute(sql).fetchone()[0]
            assert result == expected_year, f"YEAR({period}) should be {expected_year}"

        conn.close()


# =============================================================================
# Tests: SQL Initialization
# =============================================================================


class TestSQLInitialization:
    """Tests for SQL initialization of time types."""

    def test_initialization_is_idempotent(self):
        """Test that initialize_time_types can be called multiple times."""
        conn = duckdb.connect(":memory:")

        # Call multiple times
        initialize_time_types(conn)
        initialize_time_types(conn)
        initialize_time_types(conn)

        # Should still work
        result = conn.execute(
            "SELECT vtl_period_to_string(vtl_period_parse('2020-Q1'))"
        ).fetchone()[0]
        assert result == "2020-Q1"

        conn.close()

    def test_all_functions_available(self):
        """Test that all time type functions are available after initialization."""
        conn = duckdb.connect(":memory:")
        initialize_time_types(conn)

        # Test each function exists and works
        functions_to_test = [
            "SELECT vtl_period_parse('2020-Q1').start_date",
            "SELECT vtl_period_to_string(vtl_period_parse('2020-Q1'))",
            "SELECT vtl_period_indicator(vtl_period_parse('2020-Q1'))",
            "SELECT vtl_period_year(vtl_period_parse('2020-Q1'))",
            "SELECT vtl_period_number(vtl_period_parse('2020-Q1'))",
            "SELECT vtl_period_lt(vtl_period_parse('2020-Q1'), vtl_period_parse('2020-Q2'))",
            "SELECT vtl_period_shift(vtl_period_parse('2020-Q1'), 1).period_indicator",
            "SELECT vtl_period_diff(vtl_period_parse('2020-Q1'), vtl_period_parse('2020-Q2'))",
            "SELECT vtl_time_agg(vtl_period_parse('2020-Q1'), 'A').period_indicator",
            "SELECT vtl_interval_parse('2020-01-01/2020-12-31').start_date",
            "SELECT vtl_interval_to_string(vtl_interval_parse('2020-01-01/2020-12-31'))",
            "SELECT vtl_interval_lt(vtl_interval_parse('2020-01-01/2020-12-31'), vtl_interval_parse('2021-01-01/2021-12-31'))",
        ]

        for sql in functions_to_test:
            try:
                conn.execute(sql).fetchone()
            except Exception as e:
                pytest.fail(f"Function test failed: {sql}\nError: {e}")

        conn.close()
