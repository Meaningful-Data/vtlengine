"""
Tests for efficient CSV IO operations in DuckDB transpiler.

Sprint 6: Datapoint Loading/Saving Optimization
- Tests for save_datapoints_duckdb using COPY TO
- Tests for load_datapoints_duckdb using read_csv
- Tests for run() with use_duckdb=True and output_folder parameter
- Tests for table deletion after save
"""

import tempfile
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from vtlengine.DataTypes import Number, String
from vtlengine.Model import Component, Role

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for output files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def duckdb_conn():
    """Create an in-memory DuckDB connection."""
    conn = duckdb.connect(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def sample_components():
    """Create sample component definitions."""
    return {
        "Id_1": Component(name="Id_1", data_type=String, role=Role.IDENTIFIER, nullable=False),
        "Me_1": Component(name="Me_1", data_type=Number, role=Role.MEASURE, nullable=True),
    }


@pytest.fixture
def sample_table(duckdb_conn):
    """Create a sample table with test data."""
    duckdb_conn.execute("""
        CREATE TABLE "DS_1" (
            "Id_1" VARCHAR NOT NULL,
            "Me_1" DOUBLE
        )
    """)
    duckdb_conn.execute("""
        INSERT INTO "DS_1" VALUES
        ('A', 10.0),
        ('B', 20.0),
        ('C', 30.0)
    """)
    return "DS_1"


# =============================================================================
# Tests for save_datapoints_duckdb
# =============================================================================


class TestSaveDatapointsDuckdb:
    """Tests for save_datapoints_duckdb function."""

    def test_saves_csv_with_header(self, duckdb_conn, sample_table, temp_output_dir):
        """Test that save_datapoints_duckdb creates CSV with header."""
        from vtlengine.duckdb_transpiler.io import save_datapoints_duckdb

        save_datapoints_duckdb(
            conn=duckdb_conn,
            dataset_name="DS_1",
            output_path=temp_output_dir,
            delete_after_save=False,
        )

        output_file = temp_output_dir / "DS_1.csv"
        assert output_file.exists()

        # Read and verify header is present
        df = pd.read_csv(output_file)
        assert list(df.columns) == ["Id_1", "Me_1"]

    def test_saves_correct_data(self, duckdb_conn, sample_table, temp_output_dir):
        """Test that save_datapoints_duckdb saves correct data."""
        from vtlengine.duckdb_transpiler.io import save_datapoints_duckdb

        save_datapoints_duckdb(
            conn=duckdb_conn,
            dataset_name="DS_1",
            output_path=temp_output_dir,
            delete_after_save=False,
        )

        output_file = temp_output_dir / "DS_1.csv"
        df = pd.read_csv(output_file)

        assert len(df) == 3
        assert set(df["Id_1"].tolist()) == {"A", "B", "C"}
        assert set(df["Me_1"].tolist()) == {10.0, 20.0, 30.0}

    def test_no_index_column(self, duckdb_conn, sample_table, temp_output_dir):
        """Test that CSV has no index column."""
        from vtlengine.duckdb_transpiler.io import save_datapoints_duckdb

        save_datapoints_duckdb(
            conn=duckdb_conn,
            dataset_name="DS_1",
            output_path=temp_output_dir,
            delete_after_save=False,
        )

        output_file = temp_output_dir / "DS_1.csv"
        with open(output_file) as f:
            header = f.readline().strip()

        # Header should not have unnamed index column
        assert "Unnamed" not in header
        assert header == "Id_1,Me_1"

    def test_deletes_table_after_save(self, duckdb_conn, sample_table, temp_output_dir):
        """Test that table is deleted after save when delete_after_save=True."""
        from vtlengine.duckdb_transpiler.io import save_datapoints_duckdb

        save_datapoints_duckdb(
            conn=duckdb_conn,
            dataset_name="DS_1",
            output_path=temp_output_dir,
            delete_after_save=True,
        )

        # Table should no longer exist
        result = duckdb_conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'DS_1'"
        ).fetchone()
        assert result[0] == 0

    def test_keeps_table_when_delete_false(self, duckdb_conn, sample_table, temp_output_dir):
        """Test that table is kept when delete_after_save=False."""
        from vtlengine.duckdb_transpiler.io import save_datapoints_duckdb

        save_datapoints_duckdb(
            conn=duckdb_conn,
            dataset_name="DS_1",
            output_path=temp_output_dir,
            delete_after_save=False,
        )

        # Table should still exist
        result = duckdb_conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'DS_1'"
        ).fetchone()
        assert result[0] == 1


# =============================================================================
# Tests for load_datapoints_duckdb with CSV path
# =============================================================================


class TestLoadDatapointsDuckdbFromCSV:
    """Tests for load_datapoints_duckdb loading from CSV files."""

    def test_loads_csv_into_table(self, duckdb_conn, sample_components, temp_output_dir):
        """Test that load_datapoints_duckdb creates table from CSV."""
        from vtlengine.duckdb_transpiler.io import load_datapoints_duckdb

        # Create test CSV
        csv_path = temp_output_dir / "DS_1.csv"
        pd.DataFrame({"Id_1": ["A", "B"], "Me_1": [10.0, 20.0]}).to_csv(csv_path, index=False)

        load_datapoints_duckdb(
            conn=duckdb_conn,
            components=sample_components,
            dataset_name="DS_1",
            csv_path=csv_path,
        )

        # Verify table exists and has correct data
        result = duckdb_conn.execute('SELECT * FROM "DS_1" ORDER BY "Id_1"').fetchall()
        assert result == [("A", 10.0), ("B", 20.0)]

    def test_validates_duplicates(self, duckdb_conn, sample_components, temp_output_dir):
        """Test that duplicate rows are detected."""
        from vtlengine.duckdb_transpiler.io import load_datapoints_duckdb
        from vtlengine.Exceptions import DataLoadError

        # Create CSV with duplicate keys
        csv_path = temp_output_dir / "DS_1.csv"
        pd.DataFrame({"Id_1": ["A", "A"], "Me_1": [10.0, 20.0]}).to_csv(csv_path, index=False)

        with pytest.raises(DataLoadError):
            load_datapoints_duckdb(
                conn=duckdb_conn,
                components=sample_components,
                dataset_name="DS_1",
                csv_path=csv_path,
            )


# =============================================================================
# Tests for run() function with use_duckdb=True and output_folder
# =============================================================================


class TestRunWithOutputFolder:
    """Tests for run() function with use_duckdb=True and efficient CSV IO."""

    @pytest.fixture
    def simple_data_structure(self):
        """Create a simple data structure for testing."""
        return {
            "datasets": [
                {
                    "name": "DS_1",
                    "DataStructure": [
                        {"name": "Id_1", "type": "String", "role": "Identifier", "nullable": False},
                        {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                    ],
                }
            ]
        }

    @pytest.fixture
    def input_csv(self, temp_output_dir):
        """Create an input CSV file for testing."""
        csv_path = temp_output_dir / "DS_1.csv"
        pd.DataFrame({"Id_1": ["A", "B", "C"], "Me_1": [10.0, 20.0, 30.0]}).to_csv(
            csv_path, index=False
        )
        return csv_path

    def test_run_saves_output_to_folder(self, temp_output_dir, simple_data_structure, input_csv):
        """Test that run() with use_duckdb=True saves outputs to specified folder."""
        from vtlengine.API import run

        output_dir = temp_output_dir / "output"
        output_dir.mkdir()

        vtl_script = "DS_r <- DS_1 * 2;"

        run(
            script=vtl_script,
            data_structures=simple_data_structure,
            datapoints={"DS_1": input_csv},
            output_folder=output_dir,
            use_duckdb=True,
        )

        # Check that output CSV was created
        output_file = output_dir / "DS_r.csv"
        assert output_file.exists()

        # Verify the output data
        result_df = pd.read_csv(output_file)
        assert list(result_df["Me_1"]) == [20.0, 40.0, 60.0]

    def test_run_without_output_folder_returns_datasets(
        self, temp_output_dir, simple_data_structure, input_csv
    ):
        """Test that run() with use_duckdb=True returns Datasets when no output_folder."""
        from vtlengine.API import run
        from vtlengine.Model import Dataset

        vtl_script = "DS_r <- DS_1 + 5;"

        results = run(
            script=vtl_script,
            data_structures=simple_data_structure,
            datapoints={"DS_1": input_csv},
            output_folder=None,
            use_duckdb=True,
        )

        assert "DS_r" in results
        assert isinstance(results["DS_r"], Dataset)
        assert list(results["DS_r"].data.sort_values("Id_1")["Me_1"]) == [15.0, 25.0, 35.0]

    def test_run_deletes_intermediate_tables(
        self, temp_output_dir, simple_data_structure, input_csv
    ):
        """Test that run() with use_duckdb=True deletes tables after saving."""
        from vtlengine.API import run

        output_dir = temp_output_dir / "output"
        output_dir.mkdir()

        # Multi-step script with intermediate result
        vtl_script = """
        DS_temp := DS_1 * 2;
        DS_r <- DS_temp + 10;
        """

        run(
            script=vtl_script,
            data_structures=simple_data_structure,
            datapoints={"DS_1": input_csv},
            output_folder=output_dir,
            use_duckdb=True,
        )

        # Only persistent result should be saved
        assert (output_dir / "DS_r.csv").exists()
        # Intermediate result should not be saved (it's not persistent)
        assert not (output_dir / "DS_temp.csv").exists()

    def test_run_only_persistent_results(self, temp_output_dir, simple_data_structure, input_csv):
        """Test that only persistent assignments are saved."""
        from vtlengine.API import run

        output_dir = temp_output_dir / "output"
        output_dir.mkdir()

        # DS_temp uses := (temporary), DS_r uses <- (persistent)
        vtl_script = """
        DS_temp := DS_1 * 2;
        DS_r <- DS_temp;
        """

        run(
            script=vtl_script,
            data_structures=simple_data_structure,
            datapoints={"DS_1": input_csv},
            output_folder=output_dir,
            return_only_persistent=True,
            use_duckdb=True,
        )

        # Only DS_r (persistent) should be saved
        assert (output_dir / "DS_r.csv").exists()
        assert not (output_dir / "DS_temp.csv").exists()
