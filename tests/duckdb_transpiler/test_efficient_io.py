"""
Tests for efficient CSV IO operations in DuckDB transpiler.

Sprint 6: Datapoint Loading/Saving Optimization
- Tests for save_datapoints_duckdb using COPY TO
- Tests for load_datapoints_duckdb using read_csv
- Tests for run() and output_folder parameter
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
            file_path=csv_path,
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
                file_path=csv_path,
            )


# =============================================================================
# Tests for run() function and output_folder
# =============================================================================


class TestRunWithOutputFolder:
    """Tests for run() function and efficient CSV IO."""

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
        """Test that run() with DuckDB saves outputs to specified folder."""
        from vtlengine.API import run

        output_dir = temp_output_dir / "output"
        output_dir.mkdir()

        vtl_script = "DS_r <- DS_1 * 2;"

        run(
            script=vtl_script,
            data_structures=simple_data_structure,
            datapoints={"DS_1": input_csv},
            output_folder=output_dir,
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
        """Test that run() with DuckDB returns Datasets when no output_folder."""
        from vtlengine.API import run
        from vtlengine.Model import Dataset

        vtl_script = "DS_r <- DS_1 + 5;"

        results = run(
            script=vtl_script,
            data_structures=simple_data_structure,
            datapoints={"DS_1": input_csv},
            output_folder=None,
        )

        assert "DS_r" in results
        assert isinstance(results["DS_r"], Dataset)
        assert list(results["DS_r"].data.sort_values("Id_1")["Me_1"]) == [15.0, 25.0, 35.0]

    def test_run_deletes_intermediate_tables(
        self, temp_output_dir, simple_data_structure, input_csv
    ):
        """Test that run() with DuckDB deletes tables after saving."""
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
        )

        # Only DS_r (persistent) should be saved
        assert (output_dir / "DS_r.csv").exists()
        assert not (output_dir / "DS_temp.csv").exists()


# =============================================================================
# Tests for register_dataframes validation
# =============================================================================


class TestRegisterDataframesValidation:
    """Tests for register_dataframes post-load validation."""

    def test_validates_duplicates(self, duckdb_conn, sample_components):
        """Test that register_dataframes detects duplicate identifier rows."""
        from vtlengine.duckdb_transpiler.io._io import register_dataframes
        from vtlengine.Exceptions import DataLoadError
        from vtlengine.Model import Dataset

        df = pd.DataFrame({"Id_1": ["A", "A"], "Me_1": [10.0, 20.0]})
        input_datasets = {"DS_1": Dataset(name="DS_1", components=sample_components)}

        with pytest.raises(DataLoadError):
            register_dataframes(duckdb_conn, {"DS_1": df}, input_datasets)

    def test_drops_table_on_validation_failure(self, duckdb_conn, sample_components):
        """Test that table is dropped when validation fails."""
        from vtlengine.duckdb_transpiler.io._io import register_dataframes
        from vtlengine.Exceptions import DataLoadError
        from vtlengine.Model import Dataset

        df = pd.DataFrame({"Id_1": ["A", "A"], "Me_1": [10.0, 20.0]})
        input_datasets = {"DS_1": Dataset(name="DS_1", components=sample_components)}

        with pytest.raises(DataLoadError):
            register_dataframes(duckdb_conn, {"DS_1": df}, input_datasets)

        # Table should have been dropped on failure
        result = duckdb_conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'DS_1'"
        ).fetchone()
        assert result[0] == 0

    def test_valid_dataframe_passes(self, duckdb_conn, sample_components):
        """Test that valid DataFrames pass validation and create tables."""
        from vtlengine.duckdb_transpiler.io._io import register_dataframes
        from vtlengine.Model import Dataset

        df = pd.DataFrame({"Id_1": ["A", "B"], "Me_1": [10.0, 20.0]})
        input_datasets = {"DS_1": Dataset(name="DS_1", components=sample_components)}

        register_dataframes(duckdb_conn, {"DS_1": df}, input_datasets)

        result = duckdb_conn.execute('SELECT * FROM "DS_1" ORDER BY "Id_1"').fetchall()
        assert result == [("A", 10.0), ("B", 20.0)]


# =============================================================================
# Tests for extract_datapoint_paths SDMX file detection
# =============================================================================


class TestExtractDatapointPathsSDMX:
    """Tests for SDMX file detection in extract_datapoint_paths."""

    def test_csv_file_routes_to_path_dict(self, sample_components, temp_output_dir):
        """Test that CSV files still route to path_dict."""
        from vtlengine.duckdb_transpiler.io._io import extract_datapoint_paths
        from vtlengine.Model import Dataset

        csv_path = temp_output_dir / "DS_1.csv"
        pd.DataFrame({"Id_1": ["A"], "Me_1": [10.0]}).to_csv(csv_path, index=False)

        input_datasets = {"DS_1": Dataset(name="DS_1", components=sample_components)}

        path_dict, df_dict = extract_datapoint_paths({"DS_1": csv_path}, input_datasets)

        assert path_dict is not None
        assert "DS_1" in path_dict
        assert len(df_dict) == 0

    def test_dataframe_routes_to_df_dict(self, sample_components):
        """Test that DataFrames route to df_dict."""
        from vtlengine.duckdb_transpiler.io._io import extract_datapoint_paths
        from vtlengine.Model import Dataset

        df = pd.DataFrame({"Id_1": ["A"], "Me_1": [10.0]})
        input_datasets = {"DS_1": Dataset(name="DS_1", components=sample_components)}

        path_dict, df_dict = extract_datapoint_paths({"DS_1": df}, input_datasets)

        assert path_dict is None
        assert "DS_1" in df_dict


class TestRegisterDataframesPartialTime:
    """DuckDB DataFrame-register path must reject truncated time components."""

    _STRUCT = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                    {"name": "Me_1", "type": "Date", "role": "Measure", "nullable": True},
                ],
            }
        ]
    }

    @pytest.mark.parametrize(
        "value",
        [
            "2020-01-01 12:30",  # partial: no seconds (space separator)
            "2020-01-01T12:30",  # partial: no seconds (T separator)
            "2020-01-01T12",  # partial: hour only
            "2020-01-01X12:30:45",  # bad separator (must not be truncated to date)
            "2020-01-01T25:00:00",  # time value out of range
        ],
    )
    def test_rejects_invalid_datetime(self, value):
        from vtlengine.API import run
        from vtlengine.Exceptions import DataLoadError

        df = pd.DataFrame({"Id_1": [1], "Me_1": [value]})
        with pytest.raises(DataLoadError) as exc_info:
            run(
                script="DS_A <- DS_1;",
                data_structures=self._STRUCT,
                datapoints={"DS_1": df},
            )
        # The offending value must be named in the error, never "unknown".
        assert "unknown" not in str(exc_info.value.args[0])

    @pytest.mark.parametrize("value", ["2020-01-01T12:30:45", "2020-01-01 12:30:45", "2020-01-01"])
    def test_accepts_full_or_date_only(self, value):
        from vtlengine.API import run

        df = pd.DataFrame({"Id_1": [1], "Me_1": [value]})
        result = run(
            script="DS_A <- DS_1;",
            data_structures=self._STRUCT,
            datapoints={"DS_1": df},
        )
        assert result["DS_A"].data["Me_1"].notna().all()

    @pytest.mark.parametrize(
        "values",
        [
            pd.to_datetime(["2020-01-01 12:30:45"]).tz_localize("Europe/Madrid"),
            pd.to_datetime(["2020-01-01 12:30:45"]),
            pd.to_datetime(["2020-01-01"]).date,
        ],
        ids=["tz-aware-datetime64", "naive-datetime64", "date-objects"],
    )
    def test_typed_columns_bypass_string_guard(self, values):
        """Typed temporal columns cannot hold malformed strings; the regex guard must
        not reject them (a TIMESTAMPTZ renders to VARCHAR with a '+01' offset that is
        not the string input format)."""
        from vtlengine.API import run

        df = pd.DataFrame({"Id_1": [1], "Me_1": values})
        result = run(
            script="DS_A <- DS_1;",
            data_structures=self._STRUCT,
            datapoints={"DS_1": df},
        )
        assert result["DS_A"].data["Me_1"].notna().all()

    def test_categorical_strings_are_still_validated(self):
        """A pandas categorical of strings registers as ENUM in DuckDB; the guard must
        treat it as a string source and reject malformed values."""
        from vtlengine.API import run
        from vtlengine.Exceptions import DataLoadError

        df = pd.DataFrame({"Id_1": [1], "Me_1": pd.Series(["2020-01-01T12:30"], dtype="category")})
        with pytest.raises(DataLoadError):
            run(
                script="DS_A <- DS_1;",
                data_structures=self._STRUCT,
                datapoints={"DS_1": df},
            )


class TestCsvLoadDateFormats:
    """DuckDB CSV loader must accept the same Date formats as pandas (T, tz, Z)."""

    _STRUCT = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                    {"name": "Me_1", "type": "Date", "role": "Measure", "nullable": True},
                ],
            }
        ]
    }

    def test_csv_accepts_t_separator_and_timezone(self, tmp_path):
        from vtlengine.API import run

        csv = tmp_path / "DS_1.csv"
        csv.write_text(
            "Id_1,Me_1\n"
            "1,2020-01-01T12:30:45\n"
            "2,2020-01-01T12:30:45+02:00\n"
            "3,2020-01-01T12:30:45Z\n"
        )
        res = run(
            script="DS_A <- DS_1;",
            data_structures=self._STRUCT,
            datapoints={"DS_1": str(csv)},
        )
        # All three normalize to the same naive datetime (offset dropped).
        assert res["DS_A"].data["Me_1"].tolist() == [
            "2020-01-01T12:30:45",
            "2020-01-01T12:30:45",
            "2020-01-01T12:30:45",
        ]

    def test_csv_rejects_partial_time(self, tmp_path):
        from vtlengine.API import run
        from vtlengine.Exceptions import DataLoadError

        csv = tmp_path / "DS_1.csv"
        csv.write_text("Id_1,Me_1\n1,2020-01-01T12:30\n")
        with pytest.raises(DataLoadError):
            run(
                script="DS_A <- DS_1;",
                data_structures=self._STRUCT,
                datapoints={"DS_1": str(csv)},
            )
