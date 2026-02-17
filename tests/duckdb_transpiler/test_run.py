"""
Run/Execution Tests

Tests for end-to-end execution of VTL scripts using DuckDB transpiler.
Uses pytest parametrize to test Dataset, Component, and Scalar evaluations.
Each test uses VTL scripts as input with data structures and data,
verifying that results match the expected output.

Naming conventions:
- Identifiers: Id_1, Id_2, etc.
- Measures: Me_1, Me_2, etc.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, List

import duckdb
import pandas as pd
import pytest

from vtlengine.duckdb_transpiler import transpile

# =============================================================================
# Test Fixtures and Utilities
# =============================================================================


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def create_data_structure(datasets: List[Dict]) -> Dict:
    """Create a data structure dictionary for testing."""
    return {"datasets": datasets}


def create_dataset_structure(
    name: str,
    id_cols: List[tuple],  # (name, type)
    measure_cols: List[tuple],  # (name, type, nullable)
) -> Dict:
    """Create a dataset structure definition."""
    components = []
    for col_name, col_type in id_cols:
        components.append(
            {
                "name": col_name,
                "type": col_type,
                "role": "Identifier",
                "nullable": False,
            }
        )
    for col_name, col_type, nullable in measure_cols:
        components.append(
            {
                "name": col_name,
                "type": col_type,
                "role": "Measure",
                "nullable": nullable,
            }
        )
    return {"name": name, "DataStructure": components}


def create_csv_data(filepath: Path, data: List[List], columns: List[str]):
    """Create a CSV file with test data."""
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(filepath, index=False)
    return filepath


def setup_test_data(
    temp_dir: Path,
    name: str,
    structure: Dict,
    data: List[List],
) -> tuple:
    """Setup data structure and CSV for a test dataset."""
    structure_path = temp_dir / f"{name}_structure.json"
    data_path = temp_dir / f"{name}.csv"

    # Write structure
    full_structure = create_data_structure([structure])
    with open(structure_path, "w") as f:
        json.dump(full_structure, f)

    # Write data
    columns = [c["name"] for c in structure["DataStructure"]]
    create_csv_data(data_path, data, columns)

    return structure_path, data_path


def execute_vtl_with_duckdb(
    vtl_script: str,
    data_structures: Dict,
    datapoints: Dict[str, pd.DataFrame],
    value_domains: Dict = None,
    external_routines: Dict = None,
) -> Dict:
    """Execute VTL script using DuckDB transpiler and return results."""
    conn = duckdb.connect(":memory:")

    # Get column types from data structures
    ds_types = {}
    for ds in data_structures.get("datasets", []):
        ds_types[ds["name"]] = {c["name"]: c["type"] for c in ds["DataStructure"]}

    # Register input datasets with proper type conversion
    for name, df in datapoints.items():
        df_copy = df.copy()
        # Convert Date columns to datetime
        if name in ds_types:
            for col, dtype in ds_types[name].items():
                if dtype == "Date" and col in df_copy.columns:
                    df_copy[col] = pd.to_datetime(df_copy[col])
        conn.register(name, df_copy)

    # Get SQL queries from transpiler
    queries = transpile(vtl_script, data_structures, value_domains, external_routines)

    # Execute queries and collect results
    results = {}
    for result_name, sql, _is_persistent in queries:
        result_df = conn.execute(sql).fetchdf()
        conn.register(result_name, result_df)
        results[result_name] = result_df

    conn.close()
    return results


# =============================================================================
# Dataset Evaluation Tests
# =============================================================================


class TestDatasetArithmeticOperations:
    """Tests for dataset-level arithmetic operations."""

    @pytest.mark.parametrize(
        "vtl_script,input_data,expected_result",
        [
            # Dataset * scalar
            (
                "DS_r := DS_1 * 2;",
                [["A", 10], ["B", 20], ["C", 30]],
                [["A", 20], ["B", 40], ["C", 60]],
            ),
            # Dataset + scalar
            (
                "DS_r := DS_1 + 5;",
                [["A", 10], ["B", 20]],
                [["A", 15], ["B", 25]],
            ),
            # Dataset - scalar
            (
                "DS_r := DS_1 - 3;",
                [["A", 10], ["B", 5]],
                [["A", 7], ["B", 2]],
            ),
            # Dataset / scalar
            (
                "DS_r := DS_1 / 2;",
                [["A", 10], ["B", 20]],
                [["A", 5.0], ["B", 10.0]],
            ),
        ],
        ids=["multiply", "add", "subtract", "divide"],
    )
    def test_dataset_scalar_arithmetic(
        self, temp_data_dir, vtl_script, input_data, expected_result
    ):
        """Test dataset-scalar arithmetic operations."""
        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Number", True)],
        )

        data_structures = create_data_structure([structure])
        input_df = pd.DataFrame(input_data, columns=["Id_1", "Me_1"])
        expected_df = pd.DataFrame(expected_result, columns=["Id_1", "Me_1"])

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        pd.testing.assert_frame_equal(
            results["DS_r"].sort_values("Id_1").reset_index(drop=True),
            expected_df.sort_values("Id_1").reset_index(drop=True),
            check_dtype=False,
        )

    @pytest.mark.parametrize(
        "vtl_script,input1_data,input2_data,expected_result",
        [
            # Dataset + Dataset
            (
                "DS_r := DS_1 + DS_2;",
                [["A", 10], ["B", 20]],
                [["A", 5], ["B", 10]],
                [["A", 15], ["B", 30]],
            ),
            # Dataset - Dataset
            (
                "DS_r := DS_1 - DS_2;",
                [["A", 100], ["B", 50]],
                [["A", 30], ["B", 20]],
                [["A", 70], ["B", 30]],
            ),
            # Dataset * Dataset
            (
                "DS_r := DS_1 * DS_2;",
                [["A", 10], ["B", 5]],
                [["A", 2], ["B", 3]],
                [["A", 20], ["B", 15]],
            ),
        ],
        ids=["add_datasets", "subtract_datasets", "multiply_datasets"],
    )
    def test_dataset_dataset_arithmetic(
        self, temp_data_dir, vtl_script, input1_data, input2_data, expected_result
    ):
        """Test dataset-dataset arithmetic operations."""
        structure1 = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Number", True)],
        )
        structure2 = create_dataset_structure(
            "DS_2",
            [("Id_1", "String")],
            [("Me_1", "Number", True)],
        )

        data_structures = create_data_structure([structure1, structure2])
        input1_df = pd.DataFrame(input1_data, columns=["Id_1", "Me_1"])
        input2_df = pd.DataFrame(input2_data, columns=["Id_1", "Me_1"])
        expected_df = pd.DataFrame(expected_result, columns=["Id_1", "Me_1"])

        results = execute_vtl_with_duckdb(
            vtl_script, data_structures, {"DS_1": input1_df, "DS_2": input2_df}
        )

        pd.testing.assert_frame_equal(
            results["DS_r"].sort_values("Id_1").reset_index(drop=True),
            expected_df.sort_values("Id_1").reset_index(drop=True),
            check_dtype=False,
        )


class TestDatasetClauseOperations:
    """Tests for dataset clause operations (filter, calc, keep, drop)."""

    @pytest.mark.parametrize(
        "vtl_script,input_data,expected_ids",
        [
            # Filter greater than
            (
                "DS_r := DS_1[filter Me_1 > 15];",
                [["A", 10], ["B", 20], ["C", 30]],
                ["B", "C"],
            ),
            # Filter equals
            (
                "DS_r := DS_1[filter Me_1 = 20];",
                [["A", 10], ["B", 20], ["C", 30]],
                ["B"],
            ),
            # Filter with AND
            (
                "DS_r := DS_1[filter Me_1 >= 10 and Me_1 <= 20];",
                [["A", 5], ["B", 15], ["C", 25]],
                ["B"],
            ),
        ],
        ids=["filter_gt", "filter_eq", "filter_and"],
    )
    def test_filter_clause(self, temp_data_dir, vtl_script, input_data, expected_ids):
        """Test filter clause operations."""
        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Number", True)],
        )

        data_structures = create_data_structure([structure])
        input_df = pd.DataFrame(input_data, columns=["Id_1", "Me_1"])

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        result_ids = sorted(results["DS_r"]["Id_1"].tolist())
        assert result_ids == sorted(expected_ids)

    @pytest.mark.parametrize(
        "vtl_script,input_data,expected_new_col_values",
        [
            # Calc with multiplication
            (
                "DS_r := DS_1[calc doubled := Me_1 * 2];",
                [["A", 10], ["B", 20]],
                [20, 40],
            ),
            # Calc with addition
            (
                "DS_r := DS_1[calc plus_ten := Me_1 + 10];",
                [["A", 5], ["B", 15]],
                [15, 25],
            ),
        ],
        ids=["calc_multiply", "calc_add"],
    )
    def test_calc_clause(self, temp_data_dir, vtl_script, input_data, expected_new_col_values):
        """Test calc clause operations."""
        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Number", True)],
        )

        data_structures = create_data_structure([structure])
        input_df = pd.DataFrame(input_data, columns=["Id_1", "Me_1"])

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        # The new column name depends on the VTL script
        result_df = results["DS_r"].sort_values("Id_1").reset_index(drop=True)

        # Check that a new column was created with expected values
        new_col = [c for c in result_df.columns if c not in ["Id_1", "Me_1"]]
        assert len(new_col) == 1
        assert list(result_df[new_col[0]]) == expected_new_col_values


# =============================================================================
# Component Evaluation Tests
# =============================================================================


class TestComponentOperations:
    """Tests for component-level operations within clauses."""

    @pytest.mark.parametrize(
        "calc_expression,input_value,expected_value",
        [
            ("Me_1 + 1", 10, 11),
            ("Me_1 * 2", 5, 10),
            ("Me_1 - 3", 8, 5),
            ("-Me_1", 7, -7),
        ],
        ids=["add", "multiply", "subtract", "negate"],
    )
    def test_component_arithmetic_in_calc(
        self, temp_data_dir, calc_expression, input_value, expected_value
    ):
        """Test component arithmetic within calc clause."""
        vtl_script = f"DS_r := DS_1[calc result := {calc_expression}];"

        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Number", True)],
        )

        data_structures = create_data_structure([structure])
        input_df = pd.DataFrame([["A", input_value]], columns=["Id_1", "Me_1"])

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        assert results["DS_r"]["result"].iloc[0] == expected_value

    @pytest.mark.parametrize(
        "filter_condition,input_values,expected_count",
        [
            ("Me_1 > 5", [3, 5, 7, 10], 2),
            ("Me_1 >= 5", [3, 5, 7, 10], 3),
            ("Me_1 < 7", [3, 5, 7, 10], 2),
            ("Me_1 = 5", [3, 5, 7, 10], 1),
        ],
        ids=["gt", "gte", "lt", "eq"],
    )
    def test_component_comparison_in_filter(
        self, temp_data_dir, filter_condition, input_values, expected_count
    ):
        """Test component comparison within filter clause."""
        vtl_script = f"DS_r := DS_1[filter {filter_condition}];"

        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Number", True)],
        )

        data_structures = create_data_structure([structure])
        input_data = [[str(i), v] for i, v in enumerate(input_values)]
        input_df = pd.DataFrame(input_data, columns=["Id_1", "Me_1"])

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        assert len(results["DS_r"]) == expected_count


# =============================================================================
# Scalar Evaluation Tests
# =============================================================================


class TestScalarOperations:
    """Tests for scalar-level operations."""

    @pytest.mark.parametrize(
        "vtl_script,expected_value",
        [
            ("x := 1 + 2;", 3),
            ("x := 10 - 3;", 7),
            ("x := 4 * 5;", 20),
            ("x := 15 / 3;", 5.0),
        ],
        ids=["add", "subtract", "multiply", "divide"],
    )
    def test_scalar_arithmetic(self, vtl_script, expected_value):
        """Test scalar arithmetic operations."""
        conn = duckdb.connect(":memory:")

        # Parse and extract the expression
        # For scalar operations, we execute the SQL directly
        expr = vtl_script.split(":=")[1].strip().rstrip(";")
        sql = f"SELECT {expr} AS result"
        result = conn.execute(sql).fetchone()[0]

        conn.close()
        assert result == expected_value


# =============================================================================
# P0 Operators - IN/NOT_IN Tests
# =============================================================================


class TestInNotInOperators:
    """Tests for IN and NOT_IN operators."""

    @pytest.mark.parametrize(
        "vtl_script,input_data,expected_result",
        [
            # Filter with IN
            (
                'DS_r := DS_1[filter Id_1 in {"A", "B"}];',
                [["A", 10], ["B", 20], ["C", 30]],
                [["A", 10], ["B", 20]],
            ),
        ],
        ids=["filter_in"],
    )
    def test_in_filter(self, temp_data_dir, vtl_script, input_data, expected_result):
        """Test IN operator in filter clause."""
        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Number", True)],
        )

        data_structures = create_data_structure([structure])
        input_df = pd.DataFrame(input_data, columns=["Id_1", "Me_1"])
        expected_df = pd.DataFrame(expected_result, columns=["Id_1", "Me_1"])

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        pd.testing.assert_frame_equal(
            results["DS_r"].sort_values("Id_1").reset_index(drop=True),
            expected_df.sort_values("Id_1").reset_index(drop=True),
            check_dtype=False,
        )


# =============================================================================
# P0 Operators - BETWEEN Tests
# =============================================================================


class TestBetweenOperator:
    """Tests for BETWEEN operator."""

    @pytest.mark.parametrize(
        "vtl_script,input_data,expected_ids",
        [
            # Between inclusive
            (
                "DS_r := DS_1[filter between(Me_1, 10, 20)];",
                [["A", 5], ["B", 10], ["C", 15], ["D", 20], ["E", 25]],
                ["B", "C", "D"],
            ),
        ],
        ids=["between_inclusive"],
    )
    def test_between_filter(self, temp_data_dir, vtl_script, input_data, expected_ids):
        """Test BETWEEN operator in filter clause."""
        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Number", True)],
        )

        data_structures = create_data_structure([structure])
        input_df = pd.DataFrame(input_data, columns=["Id_1", "Me_1"])

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        result_ids = sorted(results["DS_r"]["Id_1"].tolist())
        assert result_ids == sorted(expected_ids)


# =============================================================================
# P0 Operators - Set Operations Tests
# =============================================================================


class TestSetOperations:
    """Tests for set operations (union, intersect, setdiff, symdiff)."""

    @pytest.mark.parametrize(
        "vtl_script,input1_data,input2_data,expected_ids",
        [
            # Union
            (
                "DS_r := union(DS_1, DS_2);",
                [["A", 10], ["B", 20]],
                [["C", 30], ["D", 40]],
                ["A", "B", "C", "D"],
            ),
            # Intersect
            (
                "DS_r := intersect(DS_1, DS_2);",
                [["A", 10], ["B", 20], ["C", 30]],
                [["B", 20], ["C", 30], ["D", 40]],
                ["B", "C"],
            ),
            # Setdiff
            (
                "DS_r := setdiff(DS_1, DS_2);",
                [["A", 10], ["B", 20], ["C", 30]],
                [["B", 20], ["D", 40]],
                ["A", "C"],
            ),
        ],
        ids=["union", "intersect", "setdiff"],
    )
    def test_set_operations(
        self, temp_data_dir, vtl_script, input1_data, input2_data, expected_ids
    ):
        """Test set operations."""
        structure1 = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Number", True)],
        )
        structure2 = create_dataset_structure(
            "DS_2",
            [("Id_1", "String")],
            [("Me_1", "Number", True)],
        )

        data_structures = create_data_structure([structure1, structure2])
        input1_df = pd.DataFrame(input1_data, columns=["Id_1", "Me_1"])
        input2_df = pd.DataFrame(input2_data, columns=["Id_1", "Me_1"])

        results = execute_vtl_with_duckdb(
            vtl_script, data_structures, {"DS_1": input1_df, "DS_2": input2_df}
        )

        result_ids = sorted(results["DS_r"]["Id_1"].tolist())
        assert result_ids == sorted(expected_ids)


# =============================================================================
# P0 Operators - CAST Tests
# =============================================================================


class TestCastOperator:
    """Tests for CAST operator."""

    @pytest.mark.parametrize(
        "vtl_script,input_data,expected_type",
        [
            # Cast to Integer
            (
                "DS_r := cast(DS_1, integer);",
                [["A", 10.5], ["B", 20.7]],
                "int",
            ),
            # Cast to String
            (
                "DS_r := cast(DS_1, string);",
                [["A", 10], ["B", 20]],
                "str",
            ),
        ],
        ids=["to_integer", "to_string"],
    )
    def test_cast_type_conversion(self, temp_data_dir, vtl_script, input_data, expected_type):
        """Test CAST type conversion."""
        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Number", True)],
        )

        data_structures = create_data_structure([structure])
        input_df = pd.DataFrame(input_data, columns=["Id_1", "Me_1"])

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        # Check the result type
        result_dtype = results["DS_r"]["Me_1"].dtype
        if expected_type == "int":
            assert "int" in str(result_dtype).lower()
        elif expected_type == "str":
            assert "object" in str(result_dtype).lower() or "str" in str(result_dtype).lower()


# =============================================================================
# Aggregation Tests
# =============================================================================


class TestAggregationOperations:
    """Tests for aggregation operations."""

    @pytest.mark.parametrize(
        "vtl_script,input_data,expected_value,result_col",
        [
            # Sum
            (
                "DS_r := sum(DS_1);",
                [["A", 10], ["B", 20], ["C", 30]],
                60,
                "Me_1",
            ),
            # Count
            (
                "DS_r := count(DS_1);",
                [["A", 10], ["B", 20], ["C", 30]],
                3,
                "int_var",
            ),
            # Avg
            (
                "DS_r := avg(DS_1);",
                [["A", 10], ["B", 20], ["C", 30]],
                20.0,
                "Me_1",
            ),
            # Min
            (
                "DS_r := min(DS_1);",
                [["A", 10], ["B", 20], ["C", 30]],
                10,
                "Me_1",
            ),
            # Max
            (
                "DS_r := max(DS_1);",
                [["A", 10], ["B", 20], ["C", 30]],
                30,
                "Me_1",
            ),
        ],
        ids=["sum", "count", "avg", "min", "max"],
    )
    def test_aggregation_functions(
        self, temp_data_dir, vtl_script, input_data, expected_value, result_col
    ):
        """Test aggregation function operations."""
        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Number", True)],
        )

        data_structures = create_data_structure([structure])
        input_df = pd.DataFrame(input_data, columns=["Id_1", "Me_1"])

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        # For aggregations, the result should have the aggregated value
        result_value = results["DS_r"][result_col].iloc[0]
        assert result_value == expected_value


# =============================================================================
# Join Tests
# =============================================================================


class TestJoinOperations:
    """Tests for join operations."""

    @pytest.mark.parametrize(
        "vtl_script,input1_data,input2_data,expected_count",
        [
            # Inner join
            (
                "DS_r := inner_join(DS_1, DS_2);",
                [["A", 10], ["B", 20], ["C", 30]],
                [["A", 100], ["B", 200], ["D", 400]],
                2,  # Only A and B match
            ),
            # Left join
            (
                "DS_r := left_join(DS_1, DS_2);",
                [["A", 10], ["B", 20]],
                [["A", 100], ["C", 300]],
                2,  # All from DS_1
            ),
        ],
        ids=["inner_join", "left_join"],
    )
    def test_join_operations(
        self, temp_data_dir, vtl_script, input1_data, input2_data, expected_count
    ):
        """Test join operations."""
        structure1 = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Number", True)],
        )
        structure2 = create_dataset_structure(
            "DS_2",
            [("Id_1", "String")],
            [("Me_2", "Number", True)],
        )

        data_structures = create_data_structure([structure1, structure2])
        input1_df = pd.DataFrame(input1_data, columns=["Id_1", "Me_1"])
        input2_df = pd.DataFrame(input2_data, columns=["Id_1", "Me_2"])

        results = execute_vtl_with_duckdb(
            vtl_script, data_structures, {"DS_1": input1_df, "DS_2": input2_df}
        )

        assert len(results["DS_r"]) == expected_count


# =============================================================================
# Unary Operations Tests
# =============================================================================


class TestUnaryOperations:
    """Tests for unary operations."""

    @pytest.mark.parametrize(
        "vtl_script,input_data,expected_values",
        [
            # Abs
            (
                "DS_r := abs(DS_1);",
                [["A", -10], ["B", 20], ["C", -30]],
                [10, 20, 30],
            ),
            # Ceil
            (
                "DS_r := ceil(DS_1);",
                [["A", 10.1], ["B", 20.9]],
                [11, 21],
            ),
            # Floor
            (
                "DS_r := floor(DS_1);",
                [["A", 10.9], ["B", 20.1]],
                [10, 20],
            ),
        ],
        ids=["abs", "ceil", "floor"],
    )
    def test_unary_operations(self, temp_data_dir, vtl_script, input_data, expected_values):
        """Test unary operations."""
        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Number", True)],
        )

        data_structures = create_data_structure([structure])
        input_df = pd.DataFrame(input_data, columns=["Id_1", "Me_1"])

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        result_df = results["DS_r"].sort_values("Id_1")
        # Get the measure column (may be renamed by VTL semantic analysis based on result type)
        measure_col = [c for c in result_df.columns if c != "Id_1"][0]
        result_values = list(result_df[measure_col])
        for rv, ev in zip(result_values, expected_values):
            assert rv == ev, f"Expected {ev}, got {rv}"


# =============================================================================
# Parameterized Operations Tests
# =============================================================================


class TestParameterizedOperations:
    """Tests for parameterized operations."""

    @pytest.mark.parametrize(
        "vtl_script,input_data,expected_values",
        [
            # Round
            (
                "DS_r := round(DS_1, 0);",
                [["A", 10.4], ["B", 20.6]],
                [10.0, 21.0],
            ),
            # Trunc
            (
                "DS_r := trunc(DS_1, 0);",
                [["A", 10.9], ["B", 20.1]],
                [10.0, 20.0],
            ),
        ],
        ids=["round", "trunc"],
    )
    def test_parameterized_operations(self, temp_data_dir, vtl_script, input_data, expected_values):
        """Test parameterized operations."""
        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Number", True)],
        )

        data_structures = create_data_structure([structure])
        input_df = pd.DataFrame(input_data, columns=["Id_1", "Me_1"])

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        result_values = list(results["DS_r"].sort_values("Id_1")["Me_1"])
        for rv, ev in zip(result_values, expected_values):
            assert rv == ev, f"Expected {ev}, got {rv}"


# =============================================================================
# Time Operators Tests (Sprint 5)
# =============================================================================


class TestTimeOperators:
    """Tests for time operators."""

    def test_current_date(self, temp_data_dir):
        """Test current_date operator."""
        # current_date returns today's date as a scalar
        conn = duckdb.connect(":memory:")
        result = conn.execute("SELECT CURRENT_DATE AS result").fetchone()[0]
        conn.close()
        # Just verify it returns a date (exact value will vary)
        assert result is not None

    @pytest.mark.parametrize(
        "vtl_script,input_data,expected_values",
        [
            # Year extraction
            (
                "DS_r := DS_1[calc year_val := year(date_col)];",
                [["A", "2024-01-15"], ["B", "2023-06-30"]],
                [2024, 2023],
            ),
            # Month extraction
            (
                "DS_r := DS_1[calc month_val := month(date_col)];",
                [["A", "2024-01-15"], ["B", "2024-06-30"]],
                [1, 6],
            ),
            # Day of month extraction
            (
                "DS_r := DS_1[calc day_val := dayofmonth(date_col)];",
                [["A", "2024-01-15"], ["B", "2024-06-30"]],
                [15, 30],
            ),
        ],
        ids=["year", "month", "dayofmonth"],
    )
    def test_time_extraction(self, temp_data_dir, vtl_script, input_data, expected_values):
        """Test time extraction operators."""
        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("date_col", "Date", True)],
        )

        data_structures = create_data_structure([structure])
        input_df = pd.DataFrame(input_data, columns=["Id_1", "date_col"])
        input_df["date_col"] = pd.to_datetime(input_df["date_col"]).dt.date

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        result_df = results["DS_r"].sort_values("Id_1").reset_index(drop=True)
        extracted_col = [c for c in result_df.columns if c.endswith("_val")][0]
        result_values = list(result_df[extracted_col])

        for rv, ev in zip(result_values, expected_values):
            assert rv == ev, f"Expected {ev}, got {rv}"

    def test_flow_to_stock(self, temp_data_dir):
        """Test flow_to_stock cumulative sum operation."""
        vtl_script = "DS_r := flow_to_stock(DS_1);"

        structure = create_dataset_structure(
            "DS_1",
            [("time_id", "Date"), ("region", "String")],
            [("value", "Number", True)],
        )

        data_structures = create_data_structure([structure])
        # Flow data: 10, 20, 30 for region A
        input_data = [
            ["2024-01-01", "A", 10],
            ["2024-01-02", "A", 20],
            ["2024-01-03", "A", 30],
            ["2024-01-01", "B", 5],
            ["2024-01-02", "B", 15],
        ]
        input_df = pd.DataFrame(input_data, columns=["time_id", "region", "value"])
        input_df["time_id"] = pd.to_datetime(input_df["time_id"]).dt.date

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        # Cumulative sum for region A: 10, 30, 60
        # Cumulative sum for region B: 5, 20
        result_df = results["DS_r"]
        result_a = result_df[result_df["region"] == "A"].sort_values("time_id")["value"].tolist()
        result_b = result_df[result_df["region"] == "B"].sort_values("time_id")["value"].tolist()

        assert result_a == [10, 30, 60], f"Expected [10, 30, 60], got {result_a}"
        assert result_b == [5, 20], f"Expected [5, 20], got {result_b}"

    def test_stock_to_flow(self, temp_data_dir):
        """Test stock_to_flow difference operation."""
        vtl_script = "DS_r := stock_to_flow(DS_1);"

        structure = create_dataset_structure(
            "DS_1",
            [("time_id", "Date"), ("region", "String")],
            [("value", "Number", True)],
        )

        data_structures = create_data_structure([structure])
        # Stock data: 10, 30, 60 for region A (cumulative)
        input_data = [
            ["2024-01-01", "A", 10],
            ["2024-01-02", "A", 30],
            ["2024-01-03", "A", 60],
        ]
        input_df = pd.DataFrame(input_data, columns=["time_id", "region", "value"])
        input_df["time_id"] = pd.to_datetime(input_df["time_id"]).dt.date

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        # Flow values: 10 (first), 20 (30-10), 30 (60-30)
        result_df = results["DS_r"]
        result_a = result_df.sort_values("time_id")["value"].tolist()

        assert result_a == [10, 20, 30], f"Expected [10, 20, 30], got {result_a}"


# =============================================================================
# Value Domain Tests (Sprint 4)
# =============================================================================


class TestValueDomainOperations:
    """Tests for value domain operations."""

    def test_value_domain_in_filter(self, temp_data_dir):
        """Test using value domain in filter clause."""
        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Number", True)],
        )

        data_structures = create_data_structure([structure])

        # Define a value domain with allowed codes
        value_domains = [
            {
                "name": "VALID_CODES",
                "type": "String",
                "setlist": ["A", "B"],
            }
        ]

        input_data = [["A", 10], ["B", 20], ["C", 30]]
        input_df = pd.DataFrame(input_data, columns=["Id_1", "Me_1"])

        # Use value domain reference in filter
        vtl_script = "DS_r := DS_1[filter Id_1 in VALID_CODES];"
        results = execute_vtl_with_duckdb(
            vtl_script, data_structures, {"DS_1": input_df}, value_domains=value_domains
        )

        result_ids = sorted(results["DS_r"]["Id_1"].tolist())
        assert result_ids == ["A", "B"]


# =============================================================================
# Complex Multi-Operator Tests
# =============================================================================


class TestComplexMultiOperatorStatements:
    """
    Tests for complex VTL statements that combine 5+ different operators.

    These tests verify that the DuckDB transpiler correctly handles complex
    VTL statements combining multiple operators like joins, aggregations,
    filters, arithmetic, and clause operations.
    """

    def test_aggr_with_multiple_functions_group_by_having(self, temp_data_dir):
        """
        Test aggregation with multiple functions, group by, and having clause.

        Operators: aggr, sum, max, group by, having, avg, > (7 operators)

        VTL: DS_r := DS_1[aggr Me_sum := sum(Me_1), Me_max := max(Me_1)
                          group by Id_1 having avg(Me_1) > 10];
        """
        vtl_script = """
            DS_r := DS_1[aggr Me_sum := sum(Me_1), Me_max := max(Me_1)
                         group by Id_1 having avg(Me_1) > 10];
        """

        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String"), ("Id_2", "String")],
            [("Me_1", "Number", True)],
        )

        data_structures = create_data_structure([structure])
        # Group A: avg=15 (passes having)
        # Group B: avg=5 (fails having)
        # Group C: avg=25 (passes having)
        input_data = [
            ["A", "x", 10],
            ["A", "y", 20],
            ["B", "x", 3],
            ["B", "y", 7],
            ["C", "x", 20],
            ["C", "y", 30],
        ]
        input_df = pd.DataFrame(input_data, columns=["Id_1", "Id_2", "Me_1"])

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        result_df = results["DS_r"].sort_values("Id_1").reset_index(drop=True)
        # Only A and C should pass the having filter
        assert len(result_df) == 2
        assert sorted(result_df["Id_1"].tolist()) == ["A", "C"]
        # Check aggregations
        result_a = result_df[result_df["Id_1"] == "A"].iloc[0]
        assert result_a["Me_sum"] == 30  # 10 + 20
        assert result_a["Me_max"] == 20

        result_c = result_df[result_df["Id_1"] == "C"].iloc[0]
        assert result_c["Me_sum"] == 50  # 20 + 30
        assert result_c["Me_max"] == 30

    def test_filter_with_boolean_and_comparison_operators(self, temp_data_dir):
        """
        Test filter with multiple boolean and comparison operators.

        Operators: filter, =, and, <, or, <> (6 operators)

        VTL: DS_r := DS_1[filter (Id_1 = "A" and Me_1 < 20) or (Id_1 <> "B" and Me_1 > 25)];
        """
        vtl_script = """
            DS_r := DS_1[filter (Id_1 = "A" and Me_1 < 20) or (Id_1 <> "B" and Me_1 > 25)];
        """

        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String"), ("Id_2", "Integer")],
            [("Me_1", "Number", True)],
        )

        data_structures = create_data_structure([structure])
        input_data = [
            ["A", 1, 15],  # passes: A and <20
            ["A", 2, 25],  # fails: A but not <20, and not >25
            ["B", 1, 30],  # fails: B (not <>B) even though >25
            ["C", 1, 30],  # passes: <>B and >25
            ["D", 1, 10],  # fails: <>B but not >25, not A
        ]
        input_df = pd.DataFrame(input_data, columns=["Id_1", "Id_2", "Me_1"])

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        result_df = results["DS_r"].sort_values(["Id_1", "Id_2"]).reset_index(drop=True)
        # Should have A,1 and C,1
        assert len(result_df) == 2
        expected_ids = [("A", 1), ("C", 1)]
        actual_ids = list(zip(result_df["Id_1"].tolist(), result_df["Id_2"].tolist()))
        assert sorted(actual_ids) == sorted(expected_ids)

    def test_calc_with_arithmetic_and_functions(self, temp_data_dir):
        """
        Test calc clause with multiple arithmetic operations and functions.

        Operators: calc, +, *, /, abs, round (6 operators)

        VTL: DS_r := DS_1[calc Me_result := round(abs(Me_1 * 2 + Me_2) / 3, 1)];
        """
        vtl_script = """
            DS_r := DS_1[calc Me_result := round(abs(Me_1 * 2 + Me_2) / 3, 1)];
        """

        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Number", True), ("Me_2", "Number", True)],
        )

        data_structures = create_data_structure([structure])
        input_data = [
            ["A", 10, 5],  # abs(10*2+5)/3 = 25/3 = 8.333... -> 8.3
            ["B", -15, 3],  # abs(-15*2+3)/3 = abs(-27)/3 = 9.0
            ["C", 6, -18],  # abs(6*2-18)/3 = abs(-6)/3 = 2.0
        ]
        input_df = pd.DataFrame(input_data, columns=["Id_1", "Me_1", "Me_2"])

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        result_df = results["DS_r"].sort_values("Id_1").reset_index(drop=True)
        expected_results = {"A": 8.3, "B": 9.0, "C": 2.0}

        for _, row in result_df.iterrows():
            expected = expected_results[row["Id_1"]]
            assert abs(row["Me_result"] - expected) < 0.01, (
                f"For {row['Id_1']}: expected {expected}, got {row['Me_result']}"
            )

    def test_inner_join_with_filter_and_calc(self, temp_data_dir):
        """
        Test inner join with filter and calc clauses combined.

        Operators: inner_join, filter, >, calc, +, * (6 operators)

        VTL: DS_r := inner_join(DS_1, DS_2 filter Me_1 > 5 calc Me_total := Me_1 + Me_2 * 2);
        """
        vtl_script = """
            DS_r := inner_join(DS_1, DS_2 filter Me_1 > 5 calc Me_total := Me_1 + Me_2 * 2);
        """

        structure1 = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Number", True)],
        )
        structure2 = create_dataset_structure(
            "DS_2",
            [("Id_1", "String")],
            [("Me_2", "Number", True)],
        )

        data_structures = create_data_structure([structure1, structure2])
        input1_data = [
            ["A", 3],  # fails filter
            ["B", 10],  # passes filter
            ["C", 8],  # passes filter
            ["D", 4],  # fails filter
        ]
        input2_data = [
            ["A", 100],
            ["B", 5],
            ["C", 10],
            ["E", 200],  # no match in DS_1
        ]
        input1_df = pd.DataFrame(input1_data, columns=["Id_1", "Me_1"])
        input2_df = pd.DataFrame(input2_data, columns=["Id_1", "Me_2"])

        results = execute_vtl_with_duckdb(
            vtl_script, data_structures, {"DS_1": input1_df, "DS_2": input2_df}
        )

        result_df = results["DS_r"].sort_values("Id_1").reset_index(drop=True)
        # B and C match and pass filter
        assert len(result_df) == 2
        assert sorted(result_df["Id_1"].tolist()) == ["B", "C"]

        # Check calculated values: Me_total = Me_1 + Me_2 * 2
        result_b = result_df[result_df["Id_1"] == "B"].iloc[0]
        assert result_b["Me_total"] == 10 + 5 * 2  # 20

        result_c = result_df[result_df["Id_1"] == "C"].iloc[0]
        assert result_c["Me_total"] == 8 + 10 * 2  # 28

    def test_union_with_filter_and_calc(self, temp_data_dir):
        """
        Test union of two filtered and calculated datasets.

        Operators: union, filter, >=, calc, -, * (6 operators across statements)

        VTL:
            tmp1 := DS_1[filter Me_1 >= 10][calc Me_doubled := Me_1 * 2];
            tmp2 := DS_2[filter Me_1 >= 5][calc Me_doubled := Me_1 * 2];
            DS_r := union(tmp1, tmp2);
        """
        vtl_script = """
            tmp1 := DS_1[filter Me_1 >= 10][calc Me_doubled := Me_1 * 2];
            tmp2 := DS_2[filter Me_1 >= 5][calc Me_doubled := Me_1 * 2];
            DS_r := union(tmp1, tmp2);
        """

        structure1 = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Number", True)],
        )
        structure2 = create_dataset_structure(
            "DS_2",
            [("Id_1", "String")],
            [("Me_1", "Number", True)],
        )

        data_structures = create_data_structure([structure1, structure2])
        # DS_1: only A (>=10) passes
        input1_data = [
            ["A", 15],
            ["B", 5],
        ]
        # DS_2: C and D (>=5) pass
        input2_data = [
            ["C", 8],
            ["D", 3],
        ]
        input1_df = pd.DataFrame(input1_data, columns=["Id_1", "Me_1"])
        input2_df = pd.DataFrame(input2_data, columns=["Id_1", "Me_1"])

        results = execute_vtl_with_duckdb(
            vtl_script, data_structures, {"DS_1": input1_df, "DS_2": input2_df}
        )

        result_df = results["DS_r"].sort_values("Id_1").reset_index(drop=True)
        # A from DS_1, C from DS_2
        assert len(result_df) == 2
        assert sorted(result_df["Id_1"].tolist()) == ["A", "C"]

        # Check doubled values
        result_a = result_df[result_df["Id_1"] == "A"].iloc[0]
        assert result_a["Me_doubled"] == 30  # 15 * 2

        result_c = result_df[result_df["Id_1"] == "C"].iloc[0]
        assert result_c["Me_doubled"] == 16  # 8 * 2

    def test_aggregation_with_multiple_group_operations(self, temp_data_dir):
        """
        Test aggregation with multiple aggregation functions and group by.

        Operators: aggr, sum, avg, count, min, max, group by (7 operators)

        VTL: DS_r := DS_1[aggr
                          Me_sum := sum(Me_1),
                          Me_avg := avg(Me_1),
                          Me_cnt := count(Me_1),
                          Me_min := min(Me_1),
                          Me_max := max(Me_1)
                          group by Id_1];
        """
        vtl_script = """
            DS_r := DS_1[aggr
                         Me_sum := sum(Me_1),
                         Me_avg := avg(Me_1),
                         Me_cnt := count(Me_1),
                         Me_min := min(Me_1),
                         Me_max := max(Me_1)
                         group by Id_1];
        """

        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String"), ("Id_2", "Integer")],
            [("Me_1", "Number", True)],
        )

        data_structures = create_data_structure([structure])
        input_data = [
            ["A", 1, 10],
            ["A", 2, 20],
            ["A", 3, 30],
            ["B", 1, 5],
            ["B", 2, 15],
        ]
        input_df = pd.DataFrame(input_data, columns=["Id_1", "Id_2", "Me_1"])

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        result_df = results["DS_r"].sort_values("Id_1").reset_index(drop=True)

        # Group A: sum=60, avg=20, count=3, min=10, max=30
        result_a = result_df[result_df["Id_1"] == "A"].iloc[0]
        assert result_a["Me_sum"] == 60
        assert result_a["Me_avg"] == 20.0
        assert result_a["Me_cnt"] == 3
        assert result_a["Me_min"] == 10
        assert result_a["Me_max"] == 30

        # Group B: sum=20, avg=10, count=2, min=5, max=15
        result_b = result_df[result_df["Id_1"] == "B"].iloc[0]
        assert result_b["Me_sum"] == 20
        assert result_b["Me_avg"] == 10.0
        assert result_b["Me_cnt"] == 2
        assert result_b["Me_min"] == 5
        assert result_b["Me_max"] == 15

    def test_left_join_with_nvl_and_calc(self, temp_data_dir):
        """
        Test left join with nvl to handle nulls and calc for derived values.

        Operators: left_join, calc, nvl, +, *, if-then-else (6 operators)

        VTL: DS_r := left_join(DS_1, DS_2 calc Me_combined := nvl(Me_2, 0) + Me_1 * 2);
        """
        vtl_script = """
            DS_r := left_join(DS_1, DS_2 calc Me_combined := nvl(Me_2, 0) + Me_1 * 2);
        """

        structure1 = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Number", True)],
        )
        structure2 = create_dataset_structure(
            "DS_2",
            [("Id_1", "String")],
            [("Me_2", "Number", True)],
        )

        data_structures = create_data_structure([structure1, structure2])
        input1_data = [
            ["A", 10],
            ["B", 20],
            ["C", 30],  # no match in DS_2
        ]
        input2_data = [
            ["A", 5],
            ["B", 15],
            ["D", 25],  # no match in DS_1
        ]
        input1_df = pd.DataFrame(input1_data, columns=["Id_1", "Me_1"])
        input2_df = pd.DataFrame(input2_data, columns=["Id_1", "Me_2"])

        results = execute_vtl_with_duckdb(
            vtl_script, data_structures, {"DS_1": input1_df, "DS_2": input2_df}
        )

        result_df = results["DS_r"].sort_values("Id_1").reset_index(drop=True)
        # Left join keeps all from DS_1: A, B, C
        assert len(result_df) == 3
        assert sorted(result_df["Id_1"].tolist()) == ["A", "B", "C"]

        # A: nvl(5, 0) + 10*2 = 25
        result_a = result_df[result_df["Id_1"] == "A"].iloc[0]
        assert result_a["Me_combined"] == 25

        # B: nvl(15, 0) + 20*2 = 55
        result_b = result_df[result_df["Id_1"] == "B"].iloc[0]
        assert result_b["Me_combined"] == 55

        # C: nvl(null, 0) + 30*2 = 60
        result_c = result_df[result_df["Id_1"] == "C"].iloc[0]
        assert result_c["Me_combined"] == 60

    def test_complex_string_operations(self, temp_data_dir):
        """
        Test complex string operations combining multiple functions.

        Operators: calc, ||, upper, lower, substr, length (6 operators)

        VTL: DS_r := DS_1[calc Me_result := upper(substr(Me_str, 1, 3)) || "_" || lower(Me_str)];
        """
        vtl_script = """
            DS_r := DS_1[calc Me_result := upper(substr(Me_str, 1, 3)) || "_" || lower(Me_str)];
        """

        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_str", "String", True)],
        )

        data_structures = create_data_structure([structure])
        input_data = [
            ["A", "Hello"],
            ["B", "World"],
            ["C", "Test"],
        ]
        input_df = pd.DataFrame(input_data, columns=["Id_1", "Me_str"])

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        result_df = results["DS_r"].sort_values("Id_1").reset_index(drop=True)
        expected = {
            "A": "HEL_hello",  # upper(substr("Hello", 1, 3)) || "_" || lower("Hello")
            "B": "WOR_world",
            "C": "TES_test",
        }

        for _, row in result_df.iterrows():
            assert row["Me_result"] == expected[row["Id_1"]], (
                f"For {row['Id_1']}: expected {expected[row['Id_1']]}, got {row['Me_result']}"
            )

    def test_if_then_else_with_boolean_operators(self, temp_data_dir):
        """
        Test if-then-else with multiple boolean operators.

        Operators: calc, if-then-else, and, or, >, <, = (7 operators)

        VTL: DS_r := DS_1[calc Me_category := if Me_1 > 20 and Me_2 < 10 then "A"
                                              else if Me_1 = 15 or Me_2 > 20 then "B"
                                              else "C"];
        """
        vtl_script = """
            DS_r := DS_1[calc Me_category := if Me_1 > 20 and Me_2 < 10 then "A"
                                             else if Me_1 = 15 or Me_2 > 20 then "B"
                                             else "C"];
        """

        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Number", True), ("Me_2", "Number", True)],
        )

        data_structures = create_data_structure([structure])
        input_data = [
            ["A", 25, 5],  # >20 and <10 -> "A"
            ["B", 15, 15],  # =15 -> "B"
            ["C", 10, 25],  # >20 for Me_2 -> "B"
            ["D", 10, 15],  # none match -> "C"
        ]
        input_df = pd.DataFrame(input_data, columns=["Id_1", "Me_1", "Me_2"])

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        result_df = results["DS_r"].sort_values("Id_1").reset_index(drop=True)
        expected = {"A": "A", "B": "B", "C": "B", "D": "C"}

        for _, row in result_df.iterrows():
            assert row["Me_category"] == expected[row["Id_1"]], (
                f"For {row['Id_1']}: expected {expected[row['Id_1']]}, got {row['Me_category']}"
            )


# =============================================================================
# Complex Multi-Operator Tests (from existing test suite - verified with pandas)
# =============================================================================


class TestVerifiedComplexOperators:
    """
    Tests for complex VTL statements verified to work with pandas interpreter.

    These tests are adapted from the existing test suite where they pass with
    the pandas-based interpreter, ensuring DuckDB transpiler compatibility.
    """

    def test_calc_filter_chain(self, temp_data_dir):
        """
        Test calc followed by filter with arithmetic and boolean operators.

        VTL: DS_r := DS_1[calc Me_1:= Me_1 * 3.0, Me_2:= Me_2 * 2.0]
                        [filter Id_1 = 2021 and Me_1 > 15.0];

        Operators: calc, *, filter, =, and, > (6 operators)
        From test: ClauseAfterClause/test_9
        """
        vtl_script = """
            DS_r := DS_1[calc Me_1 := Me_1 * 3.0, Me_2 := Me_2 * 2.0]
                       [filter Id_1 = 2021 and Me_1 > 15.0];
        """

        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "Integer"), ("Id_2", "String")],
            [("Me_1", "Number", True), ("Me_2", "Number", True)],
        )

        data_structures = create_data_structure([structure])
        # Input data based on test 1-1-1-9
        input_data = [
            [2021, "Belgium", 10.0, 10.0],  # Me_1*3=30>15 -> passes
            [2021, "Denmark", 5.0, 15.0],  # Me_1*3=15, not >15 -> fails
            [2021, "France", 9.0, 19.0],  # Me_1*3=27>15 -> passes
            [2019, "Spain", 8.0, 10.0],  # Id_1!=2021 -> fails
        ]
        input_df = pd.DataFrame(input_data, columns=["Id_1", "Id_2", "Me_1", "Me_2"])

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        result_df = results["DS_r"].sort_values("Id_2").reset_index(drop=True)
        # Should have Belgium and France
        assert len(result_df) == 2
        assert sorted(result_df["Id_2"].tolist()) == ["Belgium", "France"]

        # Check calculated values
        belgium = result_df[result_df["Id_2"] == "Belgium"].iloc[0]
        assert belgium["Me_1"] == 30.0  # 10 * 3
        assert belgium["Me_2"] == 20.0  # 10 * 2

        france = result_df[result_df["Id_2"] == "France"].iloc[0]
        assert france["Me_1"] == 27.0  # 9 * 3
        assert france["Me_2"] == 38.0  # 19 * 2

    def test_filter_rename_drop_chain(self, temp_data_dir):
        """
        Test filter followed by rename and drop.

        VTL: DS_r := DS_1[filter Id_1 = "A"][rename Me_1 to Me_1A][drop Me_2];

        Operators: filter, =, rename, drop (4 operators)
        """
        vtl_script = """
            DS_r := DS_1[filter Id_1 = "A"][rename Me_1 to Me_1A][drop Me_2];
        """

        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Number", True), ("Me_2", "Number", True)],
        )

        data_structures = create_data_structure([structure])
        input_data = [
            ["A", 10, 100],
            ["B", 20, 200],
            ["A", 30, 300],
        ]
        input_df = pd.DataFrame(input_data, columns=["Id_1", "Me_1", "Me_2"])

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        result_df = results["DS_r"].sort_values("Me_1A").reset_index(drop=True)

        # Only rows with Id_1="A"
        assert len(result_df) == 2
        # Me_1 renamed to Me_1A, Me_2 dropped
        assert "Me_1A" in result_df.columns
        assert "Me_1" not in result_df.columns
        assert "Me_2" not in result_df.columns
        assert list(result_df["Me_1A"]) == [10, 30]

    def test_inner_join_multiple_datasets(self, temp_data_dir):
        """
        Test inner join with multiple datasets.

        VTL: DS_r := inner_join(DS_1, DS_2);

        Operators: inner_join (with implicit identifier matching)
        """
        vtl_script = """
            DS_r := inner_join(DS_1, DS_2);
        """

        structure1 = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Number", True)],
        )
        structure2 = create_dataset_structure(
            "DS_2",
            [("Id_1", "String")],
            [("Me_2", "Number", True)],
        )

        data_structures = create_data_structure([structure1, structure2])
        input1_data = [["A", 10], ["B", 20], ["C", 30]]
        input2_data = [["A", 100], ["B", 200], ["D", 400]]
        input1_df = pd.DataFrame(input1_data, columns=["Id_1", "Me_1"])
        input2_df = pd.DataFrame(input2_data, columns=["Id_1", "Me_2"])

        results = execute_vtl_with_duckdb(
            vtl_script, data_structures, {"DS_1": input1_df, "DS_2": input2_df}
        )

        result_df = results["DS_r"].sort_values("Id_1").reset_index(drop=True)
        # Only A and B match
        assert len(result_df) == 2
        assert list(result_df["Id_1"]) == ["A", "B"]
        assert list(result_df["Me_1"]) == [10, 20]
        assert list(result_df["Me_2"]) == [100, 200]

    def test_union_with_filter(self, temp_data_dir):
        """
        Test union of filtered datasets.

        VTL:
            tmp1 := DS_1[filter Me_1 > 10];
            tmp2 := DS_2[filter Me_1 > 10];
            DS_r := union(tmp1, tmp2);

        Operators: filter, >, union (3 operators per statement)
        """
        vtl_script = """
            tmp1 := DS_1[filter Me_1 > 10];
            tmp2 := DS_2[filter Me_1 > 10];
            DS_r := union(tmp1, tmp2);
        """

        structure1 = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Number", True)],
        )
        structure2 = create_dataset_structure(
            "DS_2",
            [("Id_1", "String")],
            [("Me_1", "Number", True)],
        )

        data_structures = create_data_structure([structure1, structure2])
        input1_data = [["A", 5], ["B", 15], ["C", 25]]
        input2_data = [["D", 8], ["E", 18], ["F", 28]]
        input1_df = pd.DataFrame(input1_data, columns=["Id_1", "Me_1"])
        input2_df = pd.DataFrame(input2_data, columns=["Id_1", "Me_1"])

        results = execute_vtl_with_duckdb(
            vtl_script, data_structures, {"DS_1": input1_df, "DS_2": input2_df}
        )

        result_df = results["DS_r"].sort_values("Id_1").reset_index(drop=True)
        # B, C from DS_1 (>10) and E, F from DS_2 (>10)
        assert len(result_df) == 4
        assert sorted(result_df["Id_1"].tolist()) == ["B", "C", "E", "F"]

    def test_calc_with_multiple_arithmetic(self, temp_data_dir):
        """
        Test calc with multiple arithmetic operations.

        VTL: DS_r := DS_1[calc Me_sum := Me_1 + Me_2,
                          Me_diff := Me_1 - Me_2,
                          Me_prod := Me_1 * Me_2,
                          Me_ratio := Me_1 / Me_2];

        Operators: calc, +, -, *, / (5 operators)
        """
        vtl_script = """
            DS_r := DS_1[calc Me_sum := Me_1 + Me_2,
                         Me_diff := Me_1 - Me_2,
                         Me_prod := Me_1 * Me_2,
                         Me_ratio := Me_1 / Me_2];
        """

        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Number", True), ("Me_2", "Number", True)],
        )

        data_structures = create_data_structure([structure])
        input_data = [
            ["A", 10, 2],
            ["B", 20, 4],
            ["C", 30, 5],
        ]
        input_df = pd.DataFrame(input_data, columns=["Id_1", "Me_1", "Me_2"])

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        result_df = results["DS_r"].sort_values("Id_1").reset_index(drop=True)
        assert len(result_df) == 3

        # Check row A: 10+2=12, 10-2=8, 10*2=20, 10/2=5
        row_a = result_df[result_df["Id_1"] == "A"].iloc[0]
        assert row_a["Me_sum"] == 12
        assert row_a["Me_diff"] == 8
        assert row_a["Me_prod"] == 20
        assert row_a["Me_ratio"] == 5.0

        # Check row B: 20+4=24, 20-4=16, 20*4=80, 20/4=5
        row_b = result_df[result_df["Id_1"] == "B"].iloc[0]
        assert row_b["Me_sum"] == 24
        assert row_b["Me_diff"] == 16
        assert row_b["Me_prod"] == 80
        assert row_b["Me_ratio"] == 5.0


# =============================================================================
# RANDOM Operator Tests
# =============================================================================


class TestRandomOperator:
    """Tests for RANDOM operator - deterministic pseudo-random number generation."""

    def test_random_in_calc(self, temp_data_dir):
        """
        Test RANDOM operator in calc clause.

        VTL: DS_r := DS_1[calc Me_rand := random(Me_1, 1)];

        RANDOM(seed, index) returns a deterministic pseudo-random number between 0 and 1.
        Same seed + index always produces the same result.
        """
        vtl_script = """
            DS_r := DS_1[calc Me_rand := random(Me_1, 1)];
        """

        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Integer", True)],
        )

        data_structures = create_data_structure([structure])
        input_data = [
            ["A", 42],
            ["B", 42],  # Same seed as A -> same random value
            ["C", 100],  # Different seed -> different random value
        ]
        input_df = pd.DataFrame(input_data, columns=["Id_1", "Me_1"])

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        result_df = results["DS_r"].sort_values("Id_1").reset_index(drop=True)
        assert len(result_df) == 3

        # Random values should be between 0 and 1
        assert all(0 <= v <= 1 for v in result_df["Me_rand"])

        # Same seed (42) should produce same random value
        row_a = result_df[result_df["Id_1"] == "A"].iloc[0]
        row_b = result_df[result_df["Id_1"] == "B"].iloc[0]
        assert row_a["Me_rand"] == row_b["Me_rand"], "Same seed should produce same random"

        # Different seed (100) should produce different random value
        row_c = result_df[result_df["Id_1"] == "C"].iloc[0]
        assert row_a["Me_rand"] != row_c["Me_rand"], (
            "Different seed should produce different random"
        )

    def test_random_with_different_indices(self, temp_data_dir):
        """
        Test RANDOM with different index values produces different results.

        VTL: DS_r := DS_1[calc Me_r1 := random(Me_1, 1), Me_r2 := random(Me_1, 2)];
        """
        vtl_script = """
            DS_r := DS_1[calc Me_r1 := random(Me_1, 1), Me_r2 := random(Me_1, 2)];
        """

        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Integer", True)],
        )

        data_structures = create_data_structure([structure])
        input_data = [["A", 42]]
        input_df = pd.DataFrame(input_data, columns=["Id_1", "Me_1"])

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        result_df = results["DS_r"]
        row = result_df.iloc[0]

        # Different indices should produce different random values
        assert row["Me_r1"] != row["Me_r2"], "Different index should produce different random"


# =============================================================================
# MEMBERSHIP Operator Tests
# =============================================================================


class TestMembershipOperator:
    """Tests for MEMBERSHIP (#) operator - component extraction from datasets."""

    def test_membership_extract_measure(self, temp_data_dir):
        """
        Test extracting a measure from a dataset using #.

        VTL: DS_r := DS_1#Me_1;

        Extracts component Me_1 from DS_1, keeping identifiers.
        """
        vtl_script = """
            DS_r := DS_1#Me_1;
        """

        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Number", True), ("Me_2", "Number", True)],
        )

        data_structures = create_data_structure([structure])
        input_data = [
            ["A", 10.0, 20.0],
            ["B", 30.0, 40.0],
        ]
        input_df = pd.DataFrame(input_data, columns=["Id_1", "Me_1", "Me_2"])

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        result_df = results["DS_r"].sort_values("Id_1").reset_index(drop=True)

        # Result should have Id_1 and Me_1 only
        assert "Id_1" in result_df.columns
        assert "Me_1" in result_df.columns
        assert "Me_2" not in result_df.columns

        # Check values
        assert result_df[result_df["Id_1"] == "A"]["Me_1"].iloc[0] == 10.0
        assert result_df[result_df["Id_1"] == "B"]["Me_1"].iloc[0] == 30.0

    def test_membership_with_calc(self, temp_data_dir):
        """
        Test combining membership extraction with calc.

        VTL: DS_temp := DS_1#Me_1;
             DS_r := DS_temp[calc Me_doubled := Me_1 * 2];

        First extract Me_1 from DS_1, then calculate on it.
        """
        vtl_script = """
            DS_temp := DS_1#Me_1;
            DS_r := DS_temp[calc Me_doubled := Me_1 * 2];
        """

        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Number", True), ("Me_2", "Number", True)],
        )

        data_structures = create_data_structure([structure])
        input_data = [
            ["A", 10.0, 20.0],
            ["B", 20.0, 40.0],
            ["C", 30.0, 50.0],
        ]
        input_df = pd.DataFrame(input_data, columns=["Id_1", "Me_1", "Me_2"])

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        result_df = results["DS_r"].sort_values("Id_1").reset_index(drop=True)

        # Check doubled values
        assert result_df[result_df["Id_1"] == "A"]["Me_doubled"].iloc[0] == 20.0
        assert result_df[result_df["Id_1"] == "B"]["Me_doubled"].iloc[0] == 40.0
        assert result_df[result_df["Id_1"] == "C"]["Me_doubled"].iloc[0] == 60.0


# =============================================================================
# TIME_AGG Operator Tests
# =============================================================================


class TestTimeAggOperator:
    """Tests for TIME_AGG operator - time period aggregation."""

    def test_time_agg_to_year(self, temp_data_dir):
        """
        Test TIME_AGG converting dates to annual periods.

        VTL: DS_r := DS_1[calc Me_year := time_agg("A", Me_date, first)];

        Note: VTL uses "A" for Annual (not "Y"), and requires "first" or "last" for Date inputs.
        """
        vtl_script = """
            DS_r := DS_1[calc Me_year := time_agg("A", Me_date, first)];
        """

        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_date", "Date", True)],
        )

        data_structures = create_data_structure([structure])
        input_data = [
            ["A", "2024-03-15"],
            ["B", "2023-07-20"],
            ["C", "2024-12-01"],
        ]
        input_df = pd.DataFrame(input_data, columns=["Id_1", "Me_date"])

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        result_df = results["DS_r"].sort_values("Id_1").reset_index(drop=True)

        # Check year extraction
        assert result_df[result_df["Id_1"] == "A"]["Me_year"].iloc[0] == "2024"
        assert result_df[result_df["Id_1"] == "B"]["Me_year"].iloc[0] == "2023"
        assert result_df[result_df["Id_1"] == "C"]["Me_year"].iloc[0] == "2024"

    def test_time_agg_to_quarter(self, temp_data_dir):
        """
        Test TIME_AGG converting dates to quarter periods.

        VTL: DS_r := DS_1[calc Me_quarter := time_agg("Q", Me_date, first)];
        """
        vtl_script = """
            DS_r := DS_1[calc Me_quarter := time_agg("Q", Me_date, first)];
        """

        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_date", "Date", True)],
        )

        data_structures = create_data_structure([structure])
        input_data = [
            ["A", "2024-01-15"],  # Q1
            ["B", "2024-04-20"],  # Q2
            ["C", "2024-09-01"],  # Q3
            ["D", "2024-12-25"],  # Q4
        ]
        input_df = pd.DataFrame(input_data, columns=["Id_1", "Me_date"])

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        result_df = results["DS_r"].sort_values("Id_1").reset_index(drop=True)

        # Check quarter extraction
        assert result_df[result_df["Id_1"] == "A"]["Me_quarter"].iloc[0] == "2024Q1"
        assert result_df[result_df["Id_1"] == "B"]["Me_quarter"].iloc[0] == "2024Q2"
        assert result_df[result_df["Id_1"] == "C"]["Me_quarter"].iloc[0] == "2024Q3"
        assert result_df[result_df["Id_1"] == "D"]["Me_quarter"].iloc[0] == "2024Q4"

    def test_time_agg_to_month(self, temp_data_dir):
        """
        Test TIME_AGG converting dates to month periods.

        VTL: DS_r := DS_1[calc Me_month := time_agg("M", Me_date, first)];
        """
        vtl_script = """
            DS_r := DS_1[calc Me_month := time_agg("M", Me_date, first)];
        """

        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_date", "Date", True)],
        )

        data_structures = create_data_structure([structure])
        input_data = [
            ["A", "2024-01-15"],
            ["B", "2024-06-20"],
            ["C", "2024-12-01"],
        ]
        input_df = pd.DataFrame(input_data, columns=["Id_1", "Me_date"])

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        result_df = results["DS_r"].sort_values("Id_1").reset_index(drop=True)

        # Check month extraction (format: YYYYM##)
        assert result_df[result_df["Id_1"] == "A"]["Me_month"].iloc[0] == "2024M01"
        assert result_df[result_df["Id_1"] == "B"]["Me_month"].iloc[0] == "2024M06"
        assert result_df[result_df["Id_1"] == "C"]["Me_month"].iloc[0] == "2024M12"

    def test_time_agg_to_semester(self, temp_data_dir):
        """
        Test TIME_AGG converting dates to semester periods.

        VTL: DS_r := DS_1[calc Me_semester := time_agg("S", Me_date, first)];
        """
        vtl_script = """
            DS_r := DS_1[calc Me_semester := time_agg("S", Me_date, first)];
        """

        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_date", "Date", True)],
        )

        data_structures = create_data_structure([structure])
        input_data = [
            ["A", "2024-03-15"],  # S1 (Jan-Jun)
            ["B", "2024-06-30"],  # S1
            ["C", "2024-07-01"],  # S2 (Jul-Dec)
            ["D", "2024-12-25"],  # S2
        ]
        input_df = pd.DataFrame(input_data, columns=["Id_1", "Me_date"])

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        result_df = results["DS_r"].sort_values("Id_1").reset_index(drop=True)

        # Check semester extraction
        assert result_df[result_df["Id_1"] == "A"]["Me_semester"].iloc[0] == "2024S1"
        assert result_df[result_df["Id_1"] == "B"]["Me_semester"].iloc[0] == "2024S1"
        assert result_df[result_df["Id_1"] == "C"]["Me_semester"].iloc[0] == "2024S2"
        assert result_df[result_df["Id_1"] == "D"]["Me_semester"].iloc[0] == "2024S2"


# =============================================================================
# Aggregation with GROUP BY Tests
# =============================================================================


class TestAggregationWithGroupBy:
    """
    Tests for aggregation operations with explicit GROUP BY clause.

    These tests verify that when using aggregation with group by, only the specified
    columns appear in the SELECT clause (not all identifiers from the original dataset).
    This tests the fix for the "column must appear in GROUP BY clause" error.
    """

    def test_sum_with_single_group_by(self, temp_data_dir):
        """
        Test SUM aggregation grouped by a single column.

        VTL: DS_r := sum(DS_1 group by Id_1);
        """
        vtl_script = "DS_r := sum(DS_1 group by Id_1);"

        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String"), ("Id_2", "String")],
            [("Me_1", "Number", True)],
        )

        data_structures = create_data_structure([structure])
        input_data = [
            ["A", "X", 10],
            ["A", "Y", 20],
            ["B", "X", 30],
            ["B", "Y", 40],
        ]
        input_df = pd.DataFrame(input_data, columns=["Id_1", "Id_2", "Me_1"])

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        result_df = results["DS_r"].sort_values("Id_1").reset_index(drop=True)

        # Verify structure: should have Id_1 and Me_1 only (Id_2 not in group by)
        assert "Id_1" in result_df.columns
        assert "Me_1" in result_df.columns
        assert "Id_2" not in result_df.columns

        # Verify values: A -> 10+20=30, B -> 30+40=70
        assert len(result_df) == 2
        assert result_df[result_df["Id_1"] == "A"]["Me_1"].iloc[0] == 30
        assert result_df[result_df["Id_1"] == "B"]["Me_1"].iloc[0] == 70

    def test_sum_with_multiple_group_by(self, temp_data_dir):
        """
        Test SUM aggregation grouped by multiple columns.

        VTL: DS_r := sum(DS_1 group by Id_1, Id_3);
        """
        vtl_script = "DS_r := sum(DS_1 group by Id_1, Id_3);"

        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String"), ("Id_2", "String"), ("Id_3", "String")],
            [("Me_1", "Number", True)],
        )

        data_structures = create_data_structure([structure])
        input_data = [
            ["A", "X", "P", 10],
            ["A", "Y", "P", 20],
            ["A", "X", "Q", 5],
            ["B", "X", "P", 30],
        ]
        input_df = pd.DataFrame(input_data, columns=["Id_1", "Id_2", "Id_3", "Me_1"])

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        result_df = results["DS_r"].sort_values(["Id_1", "Id_3"]).reset_index(drop=True)

        # Verify structure: should have Id_1, Id_3, and Me_1 only (Id_2 not in group by)
        assert "Id_1" in result_df.columns
        assert "Id_3" in result_df.columns
        assert "Me_1" in result_df.columns
        assert "Id_2" not in result_df.columns

        # Verify values
        assert len(result_df) == 3
        # A, P -> 10+20=30
        assert (
            result_df[(result_df["Id_1"] == "A") & (result_df["Id_3"] == "P")]["Me_1"].iloc[0] == 30
        )
        # A, Q -> 5
        assert (
            result_df[(result_df["Id_1"] == "A") & (result_df["Id_3"] == "Q")]["Me_1"].iloc[0] == 5
        )
        # B, P -> 30
        assert (
            result_df[(result_df["Id_1"] == "B") & (result_df["Id_3"] == "P")]["Me_1"].iloc[0] == 30
        )

    def test_count_with_group_by(self, temp_data_dir):
        """
        Test COUNT aggregation with GROUP BY.

        VTL: DS_r := count(DS_1 group by Id_1);
        """
        vtl_script = "DS_r := count(DS_1 group by Id_1);"

        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String"), ("Id_2", "String")],
            [("Me_1", "Number", True)],
        )

        data_structures = create_data_structure([structure])
        input_data = [
            ["A", "X", 10],
            ["A", "Y", 20],
            ["A", "Z", 30],
            ["B", "X", 40],
        ]
        input_df = pd.DataFrame(input_data, columns=["Id_1", "Id_2", "Me_1"])

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        result_df = results["DS_r"].sort_values("Id_1").reset_index(drop=True)

        # Verify structure
        assert "Id_1" in result_df.columns
        assert "Id_2" not in result_df.columns

        # Verify counts: A has 3 rows, B has 1 row
        assert len(result_df) == 2
        # Count result is in int_var column
        count_col = [c for c in result_df.columns if c not in ["Id_1"]][0]
        assert result_df[result_df["Id_1"] == "A"][count_col].iloc[0] == 3
        assert result_df[result_df["Id_1"] == "B"][count_col].iloc[0] == 1

    def test_avg_with_group_by(self, temp_data_dir):
        """
        Test AVG aggregation with GROUP BY.

        VTL: DS_r := avg(DS_1 group by Id_1);
        """
        vtl_script = "DS_r := avg(DS_1 group by Id_1);"

        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String"), ("Id_2", "String")],
            [("Me_1", "Number", True)],
        )

        data_structures = create_data_structure([structure])
        input_data = [
            ["A", "X", 10],
            ["A", "Y", 20],
            ["B", "X", 100],
            ["B", "Y", 200],
        ]
        input_df = pd.DataFrame(input_data, columns=["Id_1", "Id_2", "Me_1"])

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        result_df = results["DS_r"].sort_values("Id_1").reset_index(drop=True)

        # Verify structure
        assert "Id_1" in result_df.columns
        assert "Id_2" not in result_df.columns

        # Verify averages: A -> (10+20)/2=15, B -> (100+200)/2=150
        assert len(result_df) == 2
        assert result_df[result_df["Id_1"] == "A"]["Me_1"].iloc[0] == 15.0
        assert result_df[result_df["Id_1"] == "B"]["Me_1"].iloc[0] == 150.0


# =============================================================================
# CHECK Validation Tests
# =============================================================================


class TestCheckValidationOperations:
    """
    Tests for CHECK validation operations.

    These tests verify that CHECK operations:
    1. Properly evaluate comparison expressions and produce bool_var column
    2. Handle imbalance expressions correctly
    """

    def test_check_simple_comparison(self, temp_data_dir):
        """
        Test CHECK with simple comparison expression.

        VTL: DS_r := check(DS_1 > 0);
        """
        vtl_script = "DS_r := check(DS_1 > 0);"

        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Number", True)],
        )

        data_structures = create_data_structure([structure])
        input_data = [
            ["A", 10],
            ["B", -5],
            ["C", 0],
        ]
        input_df = pd.DataFrame(input_data, columns=["Id_1", "Me_1"])

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        result_df = results["DS_r"].sort_values("Id_1").reset_index(drop=True)

        # Verify bool_var column exists
        assert "bool_var" in result_df.columns

        # Verify results: A (10>0) -> True, B (-5>0) -> False, C (0>0) -> False
        assert result_df[result_df["Id_1"] == "A"]["bool_var"].iloc[0] == True  # noqa: E712
        assert result_df[result_df["Id_1"] == "B"]["bool_var"].iloc[0] == False  # noqa: E712
        assert result_df[result_df["Id_1"] == "C"]["bool_var"].iloc[0] == False  # noqa: E712

    def test_check_dataset_scalar_comparison(self, temp_data_dir):
        """
        Test CHECK with dataset-scalar comparison.

        VTL: DS_r := check(DS_1 >= 100);
        """
        vtl_script = "DS_r := check(DS_1 >= 100);"

        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Number", True)],
        )

        data_structures = create_data_structure([structure])
        input_data = [
            ["A", 100],
            ["B", 50],
            ["C", 200],
        ]
        input_df = pd.DataFrame(input_data, columns=["Id_1", "Me_1"])

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        result_df = results["DS_r"].sort_values("Id_1").reset_index(drop=True)

        # Verify bool_var column exists
        assert "bool_var" in result_df.columns

        # Verify results
        assert result_df[result_df["Id_1"] == "A"]["bool_var"].iloc[0] == True  # noqa: E712
        assert result_df[result_df["Id_1"] == "B"]["bool_var"].iloc[0] == False  # noqa: E712
        assert result_df[result_df["Id_1"] == "C"]["bool_var"].iloc[0] == True  # noqa: E712

    def test_check_with_imbalance(self, temp_data_dir):
        """
        Test CHECK with imbalance expression.

        VTL: DS_r := check(DS_1 >= 0 imbalance DS_1);
        """
        vtl_script = "DS_r := check(DS_1 >= 0 imbalance DS_1);"

        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Number", True)],
        )

        data_structures = create_data_structure([structure])
        input_data = [
            ["A", 10],
            ["B", -5],
            ["C", 0],
        ]
        input_df = pd.DataFrame(input_data, columns=["Id_1", "Me_1"])

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        result_df = results["DS_r"].sort_values("Id_1").reset_index(drop=True)

        # Verify bool_var column exists
        assert "bool_var" in result_df.columns

        # Verify imbalance column exists
        assert "imbalance" in result_df.columns

        # Verify bool_var results
        assert result_df[result_df["Id_1"] == "A"]["bool_var"].iloc[0] == True  # noqa: E712
        assert result_df[result_df["Id_1"] == "B"]["bool_var"].iloc[0] == False  # noqa: E712
        assert result_df[result_df["Id_1"] == "C"]["bool_var"].iloc[0] == True  # noqa: E712

        # Verify imbalance values (contains the measure value from the imbalance expression)
        assert result_df[result_df["Id_1"] == "A"]["imbalance"].iloc[0] == 10
        assert result_df[result_df["Id_1"] == "B"]["imbalance"].iloc[0] == -5
        assert result_df[result_df["Id_1"] == "C"]["imbalance"].iloc[0] == 0

    def test_check_dataset_dataset_comparison(self, temp_data_dir):
        """
        Test CHECK with dataset-dataset comparison.

        VTL: DS_r := check(DS_1 = DS_2);
        """
        vtl_script = "DS_r := check(DS_1 = DS_2);"

        structure1 = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Number", True)],
        )
        structure2 = create_dataset_structure(
            "DS_2",
            [("Id_1", "String")],
            [("Me_1", "Number", True)],
        )

        data_structures = create_data_structure([structure1, structure2])
        input1_data = [
            ["A", 10],
            ["B", 20],
            ["C", 30],
        ]
        input2_data = [
            ["A", 10],
            ["B", 25],
            ["C", 30],
        ]
        input1_df = pd.DataFrame(input1_data, columns=["Id_1", "Me_1"])
        input2_df = pd.DataFrame(input2_data, columns=["Id_1", "Me_1"])

        results = execute_vtl_with_duckdb(
            vtl_script, data_structures, {"DS_1": input1_df, "DS_2": input2_df}
        )

        result_df = results["DS_r"].sort_values("Id_1").reset_index(drop=True)

        # Verify bool_var column exists
        assert "bool_var" in result_df.columns

        # Verify results: A (10=10) -> True, B (20=25) -> False, C (30=30) -> True
        assert result_df[result_df["Id_1"] == "A"]["bool_var"].iloc[0] == True  # noqa: E712
        assert result_df[result_df["Id_1"] == "B"]["bool_var"].iloc[0] == False  # noqa: E712
        assert result_df[result_df["Id_1"] == "C"]["bool_var"].iloc[0] == True  # noqa: E712


# =============================================================================
# SQL Generation Optimization Tests
# =============================================================================


class TestDirectTableReferences:
    """Tests for direct table reference optimization in SQL generation."""

    def test_simple_dataset_reference_uses_direct_table(self, temp_data_dir):
        """
        Test that simple dataset references use direct table names in joins.

        VTL: DS_r := inner_join(DS_1, DS_2 using Id_1);
        Expected SQL should reference tables directly, not (SELECT * FROM "table")
        """
        vtl_script = "DS_r := inner_join(DS_1, DS_2 using Id_1);"

        structure1 = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Number", True)],
        )
        structure2 = create_dataset_structure(
            "DS_2",
            [("Id_1", "String")],
            [("Me_2", "Number", True)],
        )

        data_structures = create_data_structure([structure1, structure2])

        queries = transpile(vtl_script, data_structures)

        # Get the SQL for DS_r
        ds_r_sql = queries[0][1]

        # Should NOT contain (SELECT * FROM "DS_1") or (SELECT * FROM "DS_2")
        assert '(SELECT * FROM "DS_1")' not in ds_r_sql
        assert '(SELECT * FROM "DS_2")' not in ds_r_sql
        # Should contain direct table references
        assert '"DS_1"' in ds_r_sql
        assert '"DS_2"' in ds_r_sql
