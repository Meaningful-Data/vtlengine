"""Tests for VTL Time Type SQL macros (new STRUCT-based implementation)."""

import duckdb
import pytest

from vtlengine.duckdb_transpiler.sql import initialize_time_types


@pytest.fixture
def conn():
    """Create DuckDB connection with time types and macros loaded."""
    connection = duckdb.connect(":memory:")
    initialize_time_types(connection)
    return connection


# =========================================================================
# vtl_period_normalize: any input format (#505) → canonical internal VARCHAR
# =========================================================================


class TestPeriodNormalize:
    """Tests for vtl_period_normalize macro."""

    @pytest.mark.parametrize(
        "input_str,expected",
        [
            # Annual
            ("2020", "2020A"),
            ("2020A", "2020A"),
            ("2020-A1", "2020A"),
            # Semester
            ("2020S1", "2020-S1"),
            ("2020-S1", "2020-S1"),
            ("2020S2", "2020-S2"),
            ("2020-S2", "2020-S2"),
            # Quarter
            ("2020Q3", "2020-Q3"),
            ("2020-Q3", "2020-Q3"),
            ("2020Q1", "2020-Q1"),
            ("2020-Q4", "2020-Q4"),
            # Month
            ("2020M1", "2020-M01"),
            ("2020M12", "2020-M12"),
            ("2020-M01", "2020-M01"),
            ("2020-M06", "2020-M06"),
            # Week
            ("2020W1", "2020-W01"),
            ("2020W53", "2020-W53"),
            ("2020-W01", "2020-W01"),
            ("2020-W15", "2020-W15"),
            # Day
            ("2020D1", "2020-D001"),
            ("2020D100", "2020-D100"),
            ("2020D366", "2020-D366"),
            ("2020-D001", "2020-D001"),
            ("2020-D100", "2020-D100"),
            # ISO month (YYYY-MM)
            ("2020-01", "2020-M01"),
            ("2020-06", "2020-M06"),
            ("2020-12", "2020-M12"),
            # ISO single-digit month (YYYY-M)
            ("2020-1", "2020-M01"),
            # ISO date (YYYY-MM-DD) → Day
            ("2020-01-01", "2020-D001"),
            ("2020-01-15", "2020-D015"),
            ("2020-12-31", "2020-D366"),  # 2020 is leap year
        ],
    )
    def test_normalize(self, conn, input_str, expected):
        result = conn.execute(f"SELECT vtl_period_normalize('{input_str}')").fetchone()[0]
        assert result == expected

    def test_normalize_null(self, conn):
        result = conn.execute("SELECT vtl_period_normalize(NULL)").fetchone()[0]
        assert result is None


# =========================================================================
# vtl_period_parse: internal VARCHAR → vtl_time_period STRUCT
# =========================================================================


class TestPeriodParse:
    """Tests for vtl_period_parse macro (only handles canonical format)."""

    @pytest.mark.parametrize(
        "input_str,expected_year,expected_indicator,expected_number",
        [
            ("2022A", 2022, "A", 1),
            ("2022-S1", 2022, "S", 1),
            ("2022-S2", 2022, "S", 2),
            ("2022-Q3", 2022, "Q", 3),
            ("2022-M01", 2022, "M", 1),
            ("2022-M06", 2022, "M", 6),
            ("2022-M12", 2022, "M", 12),
            ("2022-W01", 2022, "W", 1),
            ("2022-W52", 2022, "W", 52),
            ("2022-D001", 2022, "D", 1),
            ("2022-D100", 2022, "D", 100),
            ("2022-D365", 2022, "D", 365),
        ],
    )
    def test_parse(self, conn, input_str, expected_year, expected_indicator, expected_number):
        result = conn.execute(f"SELECT vtl_period_parse('{input_str}')").fetchone()[0]
        assert result["year"] == expected_year
        assert result["period_indicator"] == expected_indicator
        assert result["period_number"] == expected_number

    def test_parse_null(self, conn):
        result = conn.execute("SELECT vtl_period_parse(NULL)").fetchone()[0]
        assert result is None


# =========================================================================
# vtl_period_to_string: vtl_time_period STRUCT → internal VARCHAR (roundtrip)
# =========================================================================


class TestPeriodToString:
    """Tests for vtl_period_to_string macro."""

    @pytest.mark.parametrize(
        "internal_str",
        [
            "2022A",
            "2022-S1",
            "2022-S2",
            "2022-Q1",
            "2022-Q4",
            "2022-M01",
            "2022-M06",
            "2022-M12",
            "2022-W01",
            "2022-W15",
            "2022-W52",
            "2022-D001",
            "2022-D100",
            "2022-D365",
        ],
    )
    def test_roundtrip(self, conn, internal_str):
        """vtl_period_to_string(vtl_period_parse(x)) == x for all indicator types."""
        result = conn.execute(
            f"SELECT vtl_period_to_string(vtl_period_parse('{internal_str}'))"
        ).fetchone()[0]
        assert result == internal_str

    def test_format_null(self, conn):
        result = conn.execute("SELECT vtl_period_to_string(NULL::vtl_time_period)").fetchone()[0]
        assert result is None


# =========================================================================
# Ordering comparisons: vtl_period_lt/le/gt/ge
# =========================================================================


class TestPeriodCompare:
    """Tests for TimePeriod ordering comparison macros."""

    @pytest.mark.parametrize(
        "a,b,expected",
        [
            # Same quarter
            ("2022-Q1", "2022-Q2", True),
            ("2022-Q2", "2022-Q1", False),
            ("2022-Q2", "2022-Q2", False),
            # Cross-year
            ("2021-Q4", "2022-Q1", True),
            ("2023-M01", "2022-M12", False),
            # Month
            ("2020-M03", "2020-M06", True),
            ("2020-M06", "2020-M03", False),
            # Annual
            ("2021A", "2022A", True),
            ("2022A", "2022A", False),
        ],
    )
    def test_lt(self, conn, a, b, expected):
        result = conn.execute(
            f"SELECT vtl_period_lt(vtl_period_parse('{a}'), vtl_period_parse('{b}'))"
        ).fetchone()[0]
        assert result == expected

    @pytest.mark.parametrize(
        "a,b,expected",
        [
            ("2022-Q1", "2022-Q2", True),
            ("2022-Q2", "2022-Q2", True),
            ("2022-Q3", "2022-Q2", False),
        ],
    )
    def test_le(self, conn, a, b, expected):
        result = conn.execute(
            f"SELECT vtl_period_le(vtl_period_parse('{a}'), vtl_period_parse('{b}'))"
        ).fetchone()[0]
        assert result == expected

    @pytest.mark.parametrize(
        "a,b,expected",
        [
            ("2022-M06", "2022-M03", True),
            ("2022-M03", "2022-M06", False),
            ("2022-M06", "2022-M06", False),
        ],
    )
    def test_gt(self, conn, a, b, expected):
        result = conn.execute(
            f"SELECT vtl_period_gt(vtl_period_parse('{a}'), vtl_period_parse('{b}'))"
        ).fetchone()[0]
        assert result == expected

    @pytest.mark.parametrize(
        "a,b,expected",
        [
            ("2022-M06", "2022-M03", True),
            ("2022-M06", "2022-M06", True),
            ("2022-M03", "2022-M06", False),
        ],
    )
    def test_ge(self, conn, a, b, expected):
        result = conn.execute(
            f"SELECT vtl_period_ge(vtl_period_parse('{a}'), vtl_period_parse('{b}'))"
        ).fetchone()[0]
        assert result == expected

    def test_different_indicator_raises(self, conn):
        """Ordering comparison of different indicators must raise error."""
        with pytest.raises(duckdb.InvalidInputException, match="different indicators"):
            conn.execute(
                "SELECT vtl_period_lt(vtl_period_parse('2022-Q1'), vtl_period_parse('2022-M06'))"
            ).fetchone()

    def test_null_propagation(self, conn):
        result = conn.execute("SELECT vtl_period_lt(vtl_period_parse('2022-Q1'), NULL)").fetchone()[
            0
        ]
        assert result is None


# =========================================================================
# Equality on VARCHAR (no STRUCT needed)
# =========================================================================


class TestPeriodEquality:
    """Tests that canonical VARCHAR strings compare correctly with = / <>."""

    @pytest.mark.parametrize(
        "a,b,expected_eq",
        [
            ("2022-M06", "2022-M06", True),
            ("2022-M06", "2022-M03", False),
            ("2022A", "2022A", True),
            ("2022-S1", "2022-S2", False),
            # Different indicators are simply not equal
            ("2022-Q1", "2022-M01", False),
        ],
    )
    def test_varchar_equality(self, conn, a, b, expected_eq):
        result = conn.execute(f"SELECT '{a}' = '{b}'").fetchone()[0]
        assert result == expected_eq


# =========================================================================
# MIN/MAX with vtl_period_parse and vtl_period_to_string
# =========================================================================


class TestPeriodMinMax:
    """Tests for MIN/MAX aggregation on TimePeriod STRUCT."""

    def test_min_months(self, conn):
        conn.execute("""
            CREATE TABLE test_periods AS
            SELECT * FROM (VALUES ('2022-M06'), ('2022-M03'), ('2022-M12'), ('2022-M01')) t(p)
        """)
        result = conn.execute(
            "SELECT vtl_period_to_string(MIN(vtl_period_parse(p))) FROM test_periods"
        ).fetchone()[0]
        assert result == "2022-M01"

    def test_max_months(self, conn):
        conn.execute("""
            CREATE TABLE test_periods AS
            SELECT * FROM (VALUES ('2022-M06'), ('2022-M03'), ('2022-M12'), ('2022-M01')) t(p)
        """)
        result = conn.execute(
            "SELECT vtl_period_to_string(MAX(vtl_period_parse(p))) FROM test_periods"
        ).fetchone()[0]
        assert result == "2022-M12"

    def test_min_quarters_cross_year(self, conn):
        conn.execute("""
            CREATE TABLE test_periods AS
            SELECT * FROM (VALUES ('2023-Q2'), ('2022-Q4'), ('2023-Q1')) t(p)
        """)
        result = conn.execute(
            "SELECT vtl_period_to_string(MIN(vtl_period_parse(p))) FROM test_periods"
        ).fetchone()[0]
        assert result == "2022-Q4"

    def test_max_annual(self, conn):
        conn.execute("""
            CREATE TABLE test_periods AS
            SELECT * FROM (VALUES ('2020A'), ('2023A'), ('2021A')) t(p)
        """)
        result = conn.execute(
            "SELECT vtl_period_to_string(MAX(vtl_period_parse(p))) FROM test_periods"
        ).fetchone()[0]
        assert result == "2023A"


# =========================================================================
# TimeInterval parse/format
# =========================================================================


class TestIntervalParse:
    """Tests for TimeInterval parse and format macros."""

    @pytest.mark.parametrize(
        "input_str,expected_start,expected_end",
        [
            ("2021-01-01/2022-01-01", "2021-01-01", "2022-01-01"),
            ("2022-06-15/2022-12-31", "2022-06-15", "2022-12-31"),
        ],
    )
    def test_interval_parse(self, conn, input_str, expected_start, expected_end):
        result = conn.execute(f"SELECT vtl_interval_parse('{input_str}')").fetchone()[0]
        assert result["date1"].isoformat() == expected_start
        assert result["date2"].isoformat() == expected_end

    def test_interval_roundtrip(self, conn):
        result = conn.execute(
            "SELECT vtl_interval_to_string(vtl_interval_parse('2021-01-01/2022-01-01'))"
        ).fetchone()[0]
        assert result == "2021-01-01/2022-01-01"

    def test_interval_null(self, conn):
        result = conn.execute("SELECT vtl_interval_parse(NULL)").fetchone()[0]
        assert result is None

    def test_interval_varchar_equality(self, conn):
        """TimeInterval equality works on VARCHAR directly."""
        result = conn.execute(
            "SELECT '2021-01-01/2022-01-01' = '2021-01-01/2022-01-01'"
        ).fetchone()[0]
        assert result is True
        result = conn.execute(
            "SELECT '2021-01-01/2022-01-01' = '2021-01-01/2022-06-30'"
        ).fetchone()[0]
        assert result is False


# =========================================================================
# vtl_interval_freq / vtl_interval_shift: TimeInterval frequency and shifting
# =========================================================================


class TestIntervalFrequency:
    """The frequency of a TimeInterval must match Time._classify_interval_period."""

    # (months, days) for every case of test_classify_interval_period, with the
    # P<n>Y<n>M<n>D codes flattened into the offset they stand for.
    @pytest.mark.parametrize(
        "interval,months,days",
        [
            # Daily and weekly
            ("2020-01-01/2020-01-02", 0, 1),
            ("2020-01-01/2020-01-01", 0, 1),
            ("2020-01-01/2020-01-08", 0, 7),
            ("2020-01-01/2020-01-07", 0, 7),
            # Monthly, both the "start of next" and "end of current" conventions
            ("2020-01-01/2020-02-01", 1, 0),
            ("2020-01-01/2020-01-31", 1, 0),
            ("2020-02-01/2020-03-01", 1, 0),
            ("2020-02-01/2020-02-29", 1, 0),  # leap year February
            ("2021-02-01/2021-02-28", 1, 0),  # non-leap year February
            # Quarterly and semesterly
            ("2020-01-01/2020-04-01", 3, 0),
            ("2020-01-01/2020-03-31", 3, 0),
            ("2020-01-01/2020-07-01", 6, 0),
            ("2020-01-01/2020-06-30", 6, 0),
            # Yearly, not anchored to January
            ("2020-01-01/2021-01-01", 12, 0),
            ("2020-01-01/2020-12-31", 12, 0),
            ("2001-06-15/2002-06-14", 12, 0),
            # Multi-period spans, which have no canonical period indicator
            ("2020-01-01/2022-01-01", 24, 0),  # P2Y
            ("2020-01-01/2021-12-31", 24, 0),  # P2Y, "end of period"
            ("2020-01-01/2027-01-01", 84, 0),  # P7Y
            ("2020-01-01/2020-08-01", 7, 0),  # P7M
            ("2020-01-01/2020-07-31", 7, 0),  # P7M, "end of period"
            ("2020-01-01/2021-03-01", 14, 0),  # P1Y2M
            ("2020-01-01/2020-01-15", 0, 14),  # P14D
            ("2020-01-01/2020-01-16", 0, 15),  # P15D
        ],
    )
    def test_interval_freq(self, conn, interval, months, days):
        result = conn.execute(f"SELECT vtl_interval_freq('{interval}')").fetchone()[0]
        assert (result["months"], result["days"]) == (months, days)

    def test_interval_freq_matches_pandas(self, conn):
        """Cross-check against the pandas classifier the macro mirrors."""
        from vtlengine.Operators.Time import Time

        for interval in ("2020-01-01/2020-12-31", "2020-01-01/2022-01-01", "2020-01-01/2020-01-15"):
            code = Time._classify_interval_period(interval)
            offset = Time._period_offset(code).kwds
            expected = (
                offset.get("years", 0) * 12 + offset.get("months", 0),
                offset.get("days", 0),
            )
            result = conn.execute(f"SELECT vtl_interval_freq('{interval}')").fetchone()[0]
            assert (result["months"], result["days"]) == expected, interval

    def test_interval_step(self, conn):
        """The step drives generate_series, so it must stay a native INTERVAL."""
        result = conn.execute(
            "SELECT CAST(DATE '2020-01-31' + vtl_interval_step('2020-01-01/2020-01-31') AS DATE)"
        ).fetchone()[0]
        assert result.isoformat() == "2020-02-29"

    @pytest.mark.parametrize(
        "interval,shift,expected",
        [
            ("2001-01-01/2001-12-31", 1, "2002-01-01/2002-12-31"),
            ("2001-01-01/2001-12-31", -2, "1999-01-01/1999-12-31"),
            ("2001-01-01/2001-12-31", 0, "2001-01-01/2001-12-31"),
            # A month end clamps but is not snapped to the target month end
            ("2020-01-31/2020-02-29", 1, "2020-02-29/2020-03-29"),
            # Non-canonical durations shift by their own length
            ("2020-01-01/2022-01-01", 2, "2024-01-01/2026-01-01"),
            ("2020-01-01/2020-01-15", 3, "2020-02-12/2020-02-26"),
        ],
    )
    def test_interval_shift(self, conn, interval, shift, expected):
        result = conn.execute(
            f"SELECT vtl_interval_shift('{interval}', {shift}, vtl_interval_step('{interval}'))"
        ).fetchone()[0]
        assert result == expected

    def test_interval_shift_keeps_time_component(self, conn):
        result = conn.execute(
            "SELECT vtl_interval_shift('2020-01-01T00:00:00/2020-12-31T00:00:00', 1, "
            "vtl_interval_step('2020-01-01T00:00:00/2020-12-31T00:00:00'))"
        ).fetchone()[0]
        assert result == "2021-01-01T00:00:00/2021-12-31T00:00:00"

    def test_interval_shift_null(self, conn):
        result = conn.execute("SELECT vtl_interval_shift(NULL, 1, INTERVAL 1 YEAR)").fetchone()[0]
        assert result is None
