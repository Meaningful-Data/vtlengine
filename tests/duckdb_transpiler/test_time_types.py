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
