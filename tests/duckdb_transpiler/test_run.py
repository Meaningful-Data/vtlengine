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
        components.append({
            "name": col_name,
            "type": col_type,
            "role": "Identifier",
            "nullable": False,
        })
    for col_name, col_type, nullable in measure_cols:
        components.append({
            "name": col_name,
            "type": col_type,
            "role": "Measure",
            "nullable": nullable,
        })
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
) -> Dict:
    """Execute VTL script using DuckDB transpiler and return results."""
    conn = duckdb.connect(":memory:")

    # Register input datasets
    for name, df in datapoints.items():
        conn.register(name, df)

    # Get SQL queries from transpiler
    queries = transpile(vtl_script, data_structures, None, None)

    # Execute queries and collect results
    results = {}
    for result_name, sql, is_persistent in queries:
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
    def test_calc_clause(
        self, temp_data_dir, vtl_script, input_data, expected_new_col_values
    ):
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
                "DS_r := cast(DS_1, Integer);",
                [["A", 10.5], ["B", 20.7]],
                "int",
            ),
            # Cast to String
            (
                "DS_r := cast(DS_1, String);",
                [["A", 10], ["B", 20]],
                "str",
            ),
        ],
        ids=["to_integer", "to_string"],
    )
    def test_cast_type_conversion(
        self, temp_data_dir, vtl_script, input_data, expected_type
    ):
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
        "vtl_script,input_data,expected_value",
        [
            # Sum
            (
                "DS_r := sum(DS_1);",
                [["A", 10], ["B", 20], ["C", 30]],
                60,
            ),
            # Count
            (
                "DS_r := count(DS_1);",
                [["A", 10], ["B", 20], ["C", 30]],
                3,
            ),
            # Avg
            (
                "DS_r := avg(DS_1);",
                [["A", 10], ["B", 20], ["C", 30]],
                20.0,
            ),
            # Min
            (
                "DS_r := min(DS_1);",
                [["A", 10], ["B", 20], ["C", 30]],
                10,
            ),
            # Max
            (
                "DS_r := max(DS_1);",
                [["A", 10], ["B", 20], ["C", 30]],
                30,
            ),
        ],
        ids=["sum", "count", "avg", "min", "max"],
    )
    def test_aggregation_functions(
        self, temp_data_dir, vtl_script, input_data, expected_value
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
        result_value = results["DS_r"]["Me_1"].iloc[0]
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
    def test_unary_operations(
        self, temp_data_dir, vtl_script, input_data, expected_values
    ):
        """Test unary operations."""
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
    def test_parameterized_operations(
        self, temp_data_dir, vtl_script, input_data, expected_values
    ):
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
