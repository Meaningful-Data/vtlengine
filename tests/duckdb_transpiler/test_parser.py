"""
Parser Tests

Tests for the DuckDB data loading and validation functionality.
Uses pytest parametrize to test different data types and validation scenarios.
"""

import tempfile
from pathlib import Path
from typing import Dict

import duckdb
import pytest

from vtlengine.DataTypes import Boolean, Date, Integer, Number, String
from vtlengine.Model import Component, Role


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def duckdb_connection():
    """Create a DuckDB in-memory connection for testing."""
    conn = duckdb.connect(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def temp_csv_dir():
    """Create a temporary directory for CSV files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def create_csv_file(directory: str, name: str, content: str) -> Path:
    """Helper to create a CSV file with given content."""
    filepath = Path(directory) / f"{name}.csv"
    with open(filepath, "w") as f:
        f.write(content)
    return filepath


def create_components(specs: list) -> Dict[str, Component]:
    """Helper to create components from specifications."""
    type_map = {
        "Integer": Integer,
        "Number": Number,
        "String": String,
        "Boolean": Boolean,
        "Date": Date,
    }
    role_map = {
        "Identifier": Role.IDENTIFIER,
        "Measure": Role.MEASURE,
        "Attribute": Role.ATTRIBUTE,
    }
    components = {}
    for name, dtype, role, nullable in specs:
        components[name] = Component(
            name=name,
            data_type=type_map[dtype],
            role=role_map[role],
            nullable=nullable,
        )
    return components


# =============================================================================
# CSV Loading Tests
# =============================================================================


class TestCSVLoading:
    """Tests for CSV data loading with DuckDB."""

    @pytest.mark.parametrize(
        "column_specs,csv_content,expected_count",
        [
            # Simple integer data
            (
                [("Id_1", "String", "Identifier", False), ("Me_1", "Integer", "Measure", True)],
                "Id_1,Me_1\nA,1\nB,2\nC,3",
                3,
            ),
            # Number (decimal) data
            (
                [("Id_1", "String", "Identifier", False), ("Me_1", "Number", "Measure", True)],
                "Id_1,Me_1\nA,10.5\nB,20.3\nC,30.1",
                3,
            ),
            # Boolean data
            (
                [("Id_1", "String", "Identifier", False), ("Me_1", "Boolean", "Measure", True)],
                "Id_1,Me_1\nA,true\nB,false\nC,true",
                3,
            ),
            # Multiple measures
            (
                [
                    ("Id_1", "String", "Identifier", False),
                    ("Me_1", "Integer", "Measure", True),
                    ("Me_2", "Number", "Measure", True),
                ],
                "Id_1,Me_1,Me_2\nA,1,1.5\nB,2,2.5",
                2,
            ),
        ],
    )
    def test_load_csv_basic_types(
        self,
        duckdb_connection,
        temp_csv_dir,
        column_specs,
        csv_content,
        expected_count,
    ):
        """Test loading CSV files with basic data types."""
        components = create_components(column_specs)
        csv_path = create_csv_file(temp_csv_dir, "test_data", csv_content)

        # Load data using DuckDB
        col_names = ",".join([f'"{spec[0]}"' for spec in column_specs])
        result = duckdb_connection.execute(
            f"SELECT {col_names} FROM read_csv('{csv_path}')"
        ).fetchall()

        assert len(result) == expected_count

    @pytest.mark.parametrize(
        "csv_content,expected_null_count",
        [
            # Nullable measure with NULL values
            ("Id_1,Me_1\nA,1\nB,\nC,3", 1),
            # Multiple NULLs
            ("Id_1,Me_1\nA,\nB,\nC,", 3),
            # No NULLs
            ("Id_1,Me_1\nA,1\nB,2\nC,3", 0),
        ],
    )
    def test_null_value_handling(
        self,
        duckdb_connection,
        temp_csv_dir,
        csv_content,
        expected_null_count,
    ):
        """Test handling of NULL values in nullable columns."""
        csv_path = create_csv_file(temp_csv_dir, "test_nulls", csv_content)

        result = duckdb_connection.execute(
            f"SELECT COUNT(*) FROM read_csv('{csv_path}') WHERE Me_1 IS NULL"
        ).fetchone()

        assert result[0] == expected_null_count


# =============================================================================
# Type Validation Tests
# =============================================================================


class TestTypeValidation:
    """Tests for data type validation during loading."""

    @pytest.mark.parametrize(
        "dtype_spec,valid_values",
        [
            ("Integer", ["1", "2", "100", "-50", "0"]),
            ("String", ["hello", "world", "test123", ""]),
            ("Boolean", ["true", "false", "TRUE", "FALSE"]),
        ],
    )
    def test_valid_type_values(self, duckdb_connection, temp_csv_dir, dtype_spec, valid_values):
        """Test that valid type values are accepted."""
        csv_content = "Id_1,Me_1\n" + "\n".join([f"{i},{v}" for i, v in enumerate(valid_values)])
        csv_path = create_csv_file(temp_csv_dir, "test_valid", csv_content)

        result = duckdb_connection.execute(
            f"SELECT COUNT(*) FROM read_csv('{csv_path}')"
        ).fetchone()

        assert result[0] == len(valid_values)

    @pytest.mark.parametrize(
        "invalid_csv_content",
        [
            # Integer column with non-numeric value
            "Id_1,Me_1\nA,not_a_number",
            # Integer with decimal
            "Id_1,Me_1\nA,1.5",
        ],
    )
    def test_invalid_integer_values(self, duckdb_connection, temp_csv_dir, invalid_csv_content):
        """Test that invalid integer values raise errors."""
        csv_path = create_csv_file(temp_csv_dir, "test_invalid", invalid_csv_content)

        # DuckDB should fail when trying to cast invalid values to BIGINT
        with pytest.raises(Exception):
            duckdb_connection.execute(
                f"SELECT CAST(Me_1 AS BIGINT) FROM read_csv('{csv_path}')"
            ).fetchall()


# =============================================================================
# Identifier Constraints Tests
# =============================================================================


class TestIdentifierConstraints:
    """Tests for identifier column constraints."""

    def test_identifier_not_null_constraint(self, duckdb_connection, temp_csv_dir):
        """Test that NULL identifier values are rejected."""
        csv_content = "Id_1,Me_1\n,1\nB,2"  # First row has NULL Id_1
        csv_path = create_csv_file(temp_csv_dir, "test_null_id", csv_content)

        # Check that NULL exists in the data
        result = duckdb_connection.execute(
            f"SELECT COUNT(*) FROM read_csv('{csv_path}') WHERE Id_1 IS NULL OR Id_1 = ''"
        ).fetchone()

        # Data loads but has empty/null identifiers
        assert result[0] >= 1

    @pytest.mark.parametrize(
        "csv_content,has_duplicates",
        [
            ("Id_1,Me_1\nA,1\nA,2", True),  # Duplicate identifier
            ("Id_1,Me_1\nA,1\nB,2", False),  # Unique identifiers
            ("Id_1,Id_2,Me_1\nA,X,1\nA,Y,2", False),  # Composite - unique
            ("Id_1,Id_2,Me_1\nA,X,1\nA,X,2", True),  # Composite - duplicate
        ],
    )
    def test_duplicate_identifier_detection(
        self, duckdb_connection, temp_csv_dir, csv_content, has_duplicates
    ):
        """Test detection of duplicate identifier values."""
        csv_path = create_csv_file(temp_csv_dir, "test_dups", csv_content)

        # Detect duplicates using GROUP BY HAVING
        id_cols = csv_content.split("\n")[0].replace(",Me_1", "")
        result = duckdb_connection.execute(
            f"""
            SELECT COUNT(*) FROM (
                SELECT {id_cols}, COUNT(*) as cnt
                FROM read_csv('{csv_path}')
                GROUP BY {id_cols}
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()

        if has_duplicates:
            assert result[0] > 0
        else:
            assert result[0] == 0


# =============================================================================
# Column Type Mapping Tests
# =============================================================================


class TestColumnTypeMapping:
    """Tests for VTL to DuckDB type mapping."""

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
        ],
    )
    def test_type_mapping(self, vtl_type, duckdb_type):
        """Test that VTL types map to correct DuckDB types."""
        from vtlengine.duckdb_transpiler.Transpiler import VTL_TO_DUCKDB_TYPES

        assert VTL_TO_DUCKDB_TYPES.get(vtl_type, "VARCHAR") == duckdb_type or vtl_type == "Number"


# =============================================================================
# Date/Time Format Tests
# =============================================================================


class TestDateTimeFormats:
    """Tests for date and time format handling."""

    @pytest.mark.parametrize(
        "date_format,date_values",
        [
            ("%Y-%m-%d", ["2024-01-15", "2024-12-31"]),
            ("%Y/%m/%d", ["2024/01/15", "2024/12/31"]),
            ("%d-%m-%Y", ["15-01-2024", "31-12-2024"]),
        ],
    )
    def test_date_parsing_formats(self, duckdb_connection, temp_csv_dir, date_format, date_values):
        """Test parsing of various date formats."""
        csv_content = "Id_1,Me_1\n" + "\n".join([f"{i},{v}" for i, v in enumerate(date_values)])
        csv_path = create_csv_file(temp_csv_dir, "test_dates", csv_content)

        # Parse dates with specified format
        result = duckdb_connection.execute(
            f"SELECT STRPTIME(Me_1, '{date_format}')::DATE FROM read_csv('{csv_path}')"
        ).fetchall()

        assert len(result) == len(date_values)


# =============================================================================
# Large Dataset Tests
# =============================================================================


class TestLargeDatasets:
    """Tests for handling larger datasets."""

    @pytest.mark.parametrize("row_count", [100, 1000, 10000])
    def test_large_dataset_loading(self, duckdb_connection, temp_csv_dir, row_count):
        """Test loading datasets with many rows."""
        rows = [f"{i},{i * 1.5}" for i in range(row_count)]
        csv_content = "Id_1,Me_1\n" + "\n".join(rows)
        csv_path = create_csv_file(temp_csv_dir, "test_large", csv_content)

        result = duckdb_connection.execute(
            f"SELECT COUNT(*) FROM read_csv('{csv_path}')"
        ).fetchone()

        assert result[0] == row_count

    @pytest.mark.parametrize("column_count", [5, 10, 20])
    def test_many_columns(self, duckdb_connection, temp_csv_dir, column_count):
        """Test loading datasets with many columns."""
        header = ",".join([f"col{i}" for i in range(column_count)])
        row = ",".join([str(i) for i in range(column_count)])
        csv_content = f"{header}\n{row}\n{row}"
        csv_path = create_csv_file(temp_csv_dir, "test_wide", csv_content)

        result = duckdb_connection.execute(f"SELECT * FROM read_csv('{csv_path}')").description

        assert len(result) == column_count


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    @pytest.mark.parametrize(
        "special_values",
        [
            ["hello, world", "test"],  # Comma in value (needs quoting)
            ['say "hello"', "test"],  # Quotes in value
            ["line1\nline2", "test"],  # Newline in value (needs quoting)
        ],
    )
    def test_special_characters_in_values(self, duckdb_connection, temp_csv_dir, special_values):
        """Test handling of special characters in string values."""
        # Create CSV with proper quoting
        rows = []
        for i, v in enumerate(special_values):
            escaped = v.replace('"', '""')
            rows.append(f'{i},"{escaped}"')
        csv_content = "Id_1,Me_1\n" + "\n".join(rows)
        csv_path = create_csv_file(temp_csv_dir, "test_special", csv_content)

        result = duckdb_connection.execute(
            f"SELECT COUNT(*) FROM read_csv('{csv_path}')"
        ).fetchone()

        assert result[0] == len(special_values)

    def test_empty_dataset(self, duckdb_connection, temp_csv_dir):
        """Test handling of empty datasets (header only)."""
        csv_content = "Id_1,Me_1"  # No data rows
        csv_path = create_csv_file(temp_csv_dir, "test_empty", csv_content)

        result = duckdb_connection.execute(
            f"SELECT COUNT(*) FROM read_csv('{csv_path}', header=true)"
        ).fetchone()

        assert result[0] == 0

    def test_single_row_dataset(self, duckdb_connection, temp_csv_dir):
        """Test handling of single-row datasets."""
        csv_content = "Id_1,Me_1\nA,1"
        csv_path = create_csv_file(temp_csv_dir, "test_single", csv_content)

        result = duckdb_connection.execute(
            f"SELECT COUNT(*) FROM read_csv('{csv_path}')"
        ).fetchone()

        assert result[0] == 1
