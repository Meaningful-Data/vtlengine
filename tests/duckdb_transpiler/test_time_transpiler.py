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
    Start,
    VarID,
)
from vtlengine.DataTypes import Number, TimeInterval, TimePeriod
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


# NOTE: Time operator tests (timeshift, period_indicator, time_agg,
# flow_to_stock, stock_to_flow, fill_time_series, duration conversions)
# are deferred to #519: (Duckdb) Implement time operators.


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

        # Equality uses VARCHAR directly; ordering uses STRUCT comparison macros
        ordering_map = {
            "<": "vtl_period_lt",
            "<=": "vtl_period_le",
            ">": "vtl_period_gt",
            ">=": "vtl_period_ge",
        }
        if op in ordering_map:
            func = ordering_map[op]
            sql = f"SELECT {func}(vtl_period_parse('{left}'), vtl_period_parse('{right}'))"
        else:
            # Equality/inequality: compare canonical VARCHAR directly
            sql = f"SELECT '{left}' {op} '{right}'"
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

        # TimeInterval uses VARCHAR comparison directly
        sql = f"SELECT '{left}' {op} '{right}'"
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

        # Test year extraction from various period formats (canonical)
        test_cases = [
            ("2020A", 2020),
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
            "SELECT vtl_period_parse('2020-Q1').year",
            "SELECT vtl_period_to_string(vtl_period_parse('2020-Q1'))",
            "SELECT vtl_period_indicator(vtl_period_parse('2020-Q1'))",
            "SELECT vtl_period_year(vtl_period_parse('2020-Q1'))",
            "SELECT vtl_period_number(vtl_period_parse('2020-Q1'))",
            "SELECT vtl_period_lt(vtl_period_parse('2020-Q1'), vtl_period_parse('2020-Q2'))",
            "SELECT vtl_period_normalize('2020Q1')",
            "SELECT vtl_interval_parse('2020-01-01/2020-12-31').date1",
            "SELECT vtl_interval_to_string(vtl_interval_parse('2020-01-01/2020-12-31'))",
        ]

        for sql in functions_to_test:
            try:
                conn.execute(sql).fetchone()
            except Exception as e:
                pytest.fail(f"Function test failed: {sql}\nError: {e}")

        conn.close()
