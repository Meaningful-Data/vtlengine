"""Tests for VTL Time Type SQL functions."""

from pathlib import Path

import duckdb
import pytest

SQL_DIR = Path(__file__).parent.parent.parent / "src/vtlengine/duckdb_transpiler/sql"


def load_sql_files(conn: duckdb.DuckDBPyConnection, *filenames: str) -> None:
    """Load SQL files into connection."""
    for filename in filenames:
        sql_path = SQL_DIR / filename
        if sql_path.exists():
            conn.execute(sql_path.read_text())


@pytest.fixture
def conn():
    """Create DuckDB connection with time types loaded."""
    connection = duckdb.connect(":memory:")
    load_sql_files(connection, "types.sql", "functions_period_parse.sql")
    return connection


class TestPeriodParse:
    """Tests for vtl_period_parse function."""

    @pytest.mark.parametrize(
        "input_str,expected_start,expected_end,expected_indicator",
        [
            # Annual
            ("2022", "2022-01-01", "2022-12-31", "A"),
            ("2022A", "2022-01-01", "2022-12-31", "A"),
            # Semester
            ("2022-S1", "2022-01-01", "2022-06-30", "S"),
            ("2022-S2", "2022-07-01", "2022-12-31", "S"),
            ("2022S1", "2022-01-01", "2022-06-30", "S"),
            # Quarter
            ("2022-Q1", "2022-01-01", "2022-03-31", "Q"),
            ("2022-Q2", "2022-04-01", "2022-06-30", "Q"),
            ("2022-Q3", "2022-07-01", "2022-09-30", "Q"),
            ("2022-Q4", "2022-10-01", "2022-12-31", "Q"),
            ("2022Q3", "2022-07-01", "2022-09-30", "Q"),
            # Month
            ("2022-M01", "2022-01-01", "2022-01-31", "M"),
            ("2022-M06", "2022-06-01", "2022-06-30", "M"),
            ("2022-M12", "2022-12-01", "2022-12-31", "M"),
            ("2022M06", "2022-06-01", "2022-06-30", "M"),
            # Week (ISO week)
            ("2022-W01", "2022-01-03", "2022-01-09", "W"),
            ("2022-W52", "2022-12-26", "2023-01-01", "W"),
            ("2022W15", "2022-04-11", "2022-04-17", "W"),
            # Day
            ("2022-D001", "2022-01-01", "2022-01-01", "D"),
            ("2022-D100", "2022-04-10", "2022-04-10", "D"),
            ("2022-D365", "2022-12-31", "2022-12-31", "D"),
            ("2022D100", "2022-04-10", "2022-04-10", "D"),
        ],
    )
    def test_period_parse_valid(
        self, conn, input_str, expected_start, expected_end, expected_indicator
    ):
        """Test parsing valid TimePeriod strings."""
        result = conn.execute(f"SELECT vtl_period_parse('{input_str}')").fetchone()[0]

        assert result["start_date"].isoformat() == expected_start
        assert result["end_date"].isoformat() == expected_end
        assert result["period_indicator"] == expected_indicator

    def test_period_parse_null(self, conn):
        """Test parsing NULL returns NULL."""
        result = conn.execute("SELECT vtl_period_parse(NULL)").fetchone()[0]
        assert result is None


@pytest.fixture
def conn_with_format():
    """Create DuckDB connection with format functions loaded."""
    connection = duckdb.connect(":memory:")
    load_sql_files(
        connection,
        "types.sql",
        "functions_period_parse.sql",
        "functions_period_format.sql",
    )
    return connection


class TestPeriodFormat:
    """Tests for vtl_period_to_string function."""

    @pytest.mark.parametrize(
        "input_str,expected_output",
        [
            # Annual - outputs just year
            ("2022", "2022"),
            ("2022A", "2022"),
            # Semester
            ("2022-S1", "2022-S1"),
            ("2022-S2", "2022-S2"),
            # Quarter
            ("2022-Q1", "2022-Q1"),
            ("2022-Q3", "2022-Q3"),
            # Month - with leading zero
            ("2022-M01", "2022-M01"),
            ("2022-M06", "2022-M06"),
            ("2022-M12", "2022-M12"),
            # Week - with leading zero
            ("2022-W01", "2022-W01"),
            ("2022-W15", "2022-W15"),
            # Day - with leading zeros
            ("2022-D001", "2022-D001"),
            ("2022-D100", "2022-D100"),
        ],
    )
    def test_period_format_roundtrip(self, conn_with_format, input_str, expected_output):
        """Test formatting TimePeriod back to string."""
        result = conn_with_format.execute(
            f"SELECT vtl_period_to_string(vtl_period_parse('{input_str}'))"
        ).fetchone()[0]
        assert result == expected_output

    def test_period_format_null(self, conn_with_format):
        """Test formatting NULL returns NULL."""
        result = conn_with_format.execute(
            "SELECT vtl_period_to_string(NULL::vtl_time_period)"
        ).fetchone()[0]
        assert result is None


@pytest.fixture
def conn_with_compare():
    """Create DuckDB connection with comparison functions loaded."""
    connection = duckdb.connect(":memory:")
    load_sql_files(
        connection,
        "types.sql",
        "functions_period_parse.sql",
        "functions_period_compare.sql",
    )
    return connection


class TestPeriodCompare:
    """Tests for TimePeriod comparison functions."""

    @pytest.mark.parametrize(
        "a,b,expected_lt,expected_eq",
        [
            # Same quarter
            ("2022-Q1", "2022-Q2", True, False),
            ("2022-Q2", "2022-Q1", False, False),
            ("2022-Q2", "2022-Q2", False, True),
            # Different years
            ("2021-Q4", "2022-Q1", True, False),
            ("2023-M01", "2022-M12", False, False),
            # Annual
            ("2021", "2022", True, False),
            ("2022", "2022", False, True),
        ],
    )
    def test_period_compare_same_indicator(self, conn_with_compare, a, b, expected_lt, expected_eq):
        """Test comparison of periods with same indicator."""
        lt_result = conn_with_compare.execute(
            f"SELECT vtl_period_lt(vtl_period_parse('{a}'), vtl_period_parse('{b}'))"
        ).fetchone()[0]
        eq_result = conn_with_compare.execute(
            f"SELECT vtl_period_eq(vtl_period_parse('{a}'), vtl_period_parse('{b}'))"
        ).fetchone()[0]

        assert lt_result == expected_lt
        assert eq_result == expected_eq

    def test_period_compare_different_indicator_raises(self, conn_with_compare):
        """Test comparison of periods with different indicators raises error."""
        with pytest.raises(duckdb.InvalidInputException, match="same period indicator"):
            conn_with_compare.execute(
                "SELECT vtl_period_lt(vtl_period_parse('2022-Q1'), vtl_period_parse('2022-M06'))"
            ).fetchone()

    def test_period_compare_null(self, conn_with_compare):
        """Test comparison with NULL returns NULL."""
        result = conn_with_compare.execute(
            "SELECT vtl_period_lt(vtl_period_parse('2022-Q1'), NULL)"
        ).fetchone()[0]
        assert result is None


@pytest.fixture
def conn_with_extract():
    """Create DuckDB connection with extraction functions loaded."""
    connection = duckdb.connect(":memory:")
    load_sql_files(
        connection,
        "types.sql",
        "functions_period_parse.sql",
        "functions_period_extract.sql",
    )
    return connection


@pytest.fixture
def conn_with_ops():
    """Create DuckDB connection with operation functions loaded."""
    connection = duckdb.connect(":memory:")
    load_sql_files(
        connection,
        "types.sql",
        "functions_period_parse.sql",
        "functions_period_format.sql",
        "functions_period_ops.sql",
    )
    return connection


class TestPeriodExtract:
    """Tests for TimePeriod extraction functions."""

    @pytest.mark.parametrize(
        "input_str,expected_year,expected_indicator,expected_number",
        [
            ("2022", 2022, "A", 1),
            ("2022-Q3", 2022, "Q", 3),
            ("2022-M06", 2022, "M", 6),
            ("2022-S2", 2022, "S", 2),
            ("2022-W15", 2022, "W", 15),
            ("2022-D100", 2022, "D", 100),
        ],
    )
    def test_period_extract(
        self, conn_with_extract, input_str, expected_year, expected_indicator, expected_number
    ):
        """Test extracting components from TimePeriod."""
        result = conn_with_extract.execute(f"""
            SELECT
                vtl_period_year(vtl_period_parse('{input_str}')),
                vtl_period_indicator(vtl_period_parse('{input_str}')),
                vtl_period_number(vtl_period_parse('{input_str}'))
        """).fetchone()

        assert result[0] == expected_year
        assert result[1] == expected_indicator
        assert result[2] == expected_number

    def test_period_extract_null(self, conn_with_extract):
        """Test extracting from NULL returns NULL."""
        result = conn_with_extract.execute("""
            SELECT
                vtl_period_year(NULL::vtl_time_period),
                vtl_period_indicator(NULL::vtl_time_period),
                vtl_period_number(NULL::vtl_time_period)
        """).fetchone()

        assert result[0] is None
        assert result[1] is None
        assert result[2] is None


class TestPeriodShift:
    """Tests for vtl_period_shift function."""

    @pytest.mark.parametrize(
        "input_str,shift,expected_output",
        [
            # Forward shifts
            ("2022-Q1", 1, "2022-Q2"),
            ("2022-Q4", 1, "2023-Q1"),
            ("2022-M06", 3, "2022-M09"),
            ("2022-M11", 2, "2023-M01"),
            ("2022", 1, "2023"),
            # Backward shifts
            ("2022-Q2", -1, "2022-Q1"),
            ("2022-Q1", -1, "2021-Q4"),
            ("2022-M03", -3, "2021-M12"),
            ("2022", -2, "2020"),
            # Zero shift
            ("2022-Q3", 0, "2022-Q3"),
        ],
    )
    def test_period_shift(self, conn_with_ops, input_str, shift, expected_output):
        """Test shifting TimePeriod by N periods."""
        result = conn_with_ops.execute(f"""
            SELECT vtl_period_to_string(vtl_period_shift(vtl_period_parse('{input_str}'), {shift}))
        """).fetchone()[0]
        assert result == expected_output

    def test_period_shift_null(self, conn_with_ops):
        """Test shifting NULL returns NULL."""
        # Use table-based approach to test NULL handling, as literal NULLs in macros
        # have different expansion behavior
        conn_with_ops.execute(
            "CREATE TEMP TABLE null_period AS SELECT NULL::vtl_time_period AS period"
        )
        result = conn_with_ops.execute(
            "SELECT vtl_period_shift(period, 1) FROM null_period"
        ).fetchone()[0]
        assert result is None


class TestPeriodDiff:
    """Tests for vtl_period_diff function."""

    @pytest.mark.parametrize(
        "a,b,expected_diff",
        [
            # Same period
            ("2022-Q1", "2022-Q1", 0),
            # Adjacent quarters
            ("2022-Q1", "2022-Q2", 91),  # Q1 ends March 31, Q2 ends June 30
            # Different years
            ("2021", "2022", 365),  # Full year difference
            # Months
            ("2022-M01", "2022-M02", 28),  # Jan 31 to Feb 28
        ],
    )
    def test_period_diff(self, conn_with_ops, a, b, expected_diff):
        """Test difference in days between two TimePeriods."""
        result = conn_with_ops.execute(f"""
            SELECT vtl_period_diff(vtl_period_parse('{a}'), vtl_period_parse('{b}'))
        """).fetchone()[0]
        assert result == expected_diff

    def test_period_diff_null(self, conn_with_ops):
        """Test diff with NULL returns NULL."""
        result = conn_with_ops.execute(
            "SELECT vtl_period_diff(vtl_period_parse('2022-Q1'), NULL)"
        ).fetchone()[0]
        assert result is None


@pytest.fixture
def conn_with_interval():
    """Create DuckDB connection with interval functions loaded."""
    connection = duckdb.connect(":memory:")
    load_sql_files(
        connection,
        "types.sql",
        "functions_interval.sql",
    )
    return connection


class TestIntervalFunctions:
    """Tests for TimeInterval functions."""

    @pytest.mark.parametrize(
        "input_str,expected_start,expected_end",
        [
            ("2021-01-01/2022-01-01", "2021-01-01", "2022-01-01"),
            ("2022-06-15/2022-12-31", "2022-06-15", "2022-12-31"),
        ],
    )
    def test_interval_parse(self, conn_with_interval, input_str, expected_start, expected_end):
        """Test parsing TimeInterval strings."""
        result = conn_with_interval.execute(f"SELECT vtl_interval_parse('{input_str}')").fetchone()[
            0
        ]

        assert result["start_date"].isoformat() == expected_start
        assert result["end_date"].isoformat() == expected_end

    def test_interval_to_string(self, conn_with_interval):
        """Test formatting TimeInterval to string."""
        result = conn_with_interval.execute(
            "SELECT vtl_interval_to_string(vtl_interval_parse('2021-01-01/2022-01-01'))"
        ).fetchone()[0]
        assert result == "2021-01-01/2022-01-01"

    def test_interval_eq(self, conn_with_interval):
        """Test TimeInterval equality."""
        result = conn_with_interval.execute("""
            SELECT
                vtl_interval_eq(
                    vtl_interval_parse('2021-01-01/2022-01-01'),
                    vtl_interval_parse('2021-01-01/2022-01-01')
                ),
                vtl_interval_eq(
                    vtl_interval_parse('2021-01-01/2022-01-01'),
                    vtl_interval_parse('2021-01-01/2022-06-30')
                )
        """).fetchone()
        assert result[0] is True
        assert result[1] is False

    def test_interval_days(self, conn_with_interval):
        """Test TimeInterval days calculation."""
        result = conn_with_interval.execute(
            "SELECT vtl_interval_days(vtl_interval_parse('2022-01-01/2022-01-31'))"
        ).fetchone()[0]
        assert result == 30
