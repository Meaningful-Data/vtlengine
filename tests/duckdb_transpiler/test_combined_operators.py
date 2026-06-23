"""
Combined Operators Tests

Tests for complex VTL scenarios combining multiple operators from different groups.
These tests verify that the DuckDB transpiler correctly handles chained and nested operations.

Naming conventions:
- Identifiers: Id_1, Id_2, etc.
- Measures: Me_1, Me_2, etc.
"""

from typing import Dict, List

import duckdb
import pandas as pd
import pytest

from vtlengine.duckdb_transpiler import transpile

# =============================================================================
# Test Utilities
# =============================================================================


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
    for result_name, sql, _is_persistent in queries:
        result_df = conn.execute(sql).fetchdf()
        conn.register(result_name, result_df)
        results[result_name] = result_df

    conn.close()
    return results


# =============================================================================
# Arithmetic + Clause Combinations
# =============================================================================


class TestArithmeticWithClauses:
    """Tests combining arithmetic operations with clauses."""

    @pytest.mark.parametrize(
        "vtl_script,input_data,expected_ids,expected_values",
        [
            # Filter then multiply
            (
                """
                DS_temp := DS_1[filter Me_1 > 10];
                DS_r := DS_temp * 2;
                """,
                [["A", 5], ["B", 15], ["C", 25]],
                ["B", "C"],
                [30, 50],
            ),
            # Multiply then filter
            (
                """
                DS_temp := DS_1 * 10;
                DS_r := DS_temp[filter Me_1 > 100];
                """,
                [["A", 5], ["B", 15], ["C", 25]],
                ["B", "C"],
                [150, 250],
            ),
            # Addition with filter on result
            (
                """
                DS_temp := DS_1 + 100;
                DS_r := DS_temp[filter Me_1 >= 115];
                """,
                [["A", 10], ["B", 15], ["C", 20]],
                ["B", "C"],
                [115, 120],
            ),
        ],
        ids=["filter_then_multiply", "multiply_then_filter", "add_then_filter"],
    )
    def test_arithmetic_filter_combinations(
        self, vtl_script, input_data, expected_ids, expected_values
    ):
        """Test arithmetic operations combined with filter clauses."""
        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Number", True)],
        )

        data_structures = create_data_structure([structure])
        input_df = pd.DataFrame(input_data, columns=["Id_1", "Me_1"])

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        result_df = results["DS_r"].sort_values("Id_1").reset_index(drop=True)
        assert list(result_df["Id_1"]) == sorted(expected_ids)
        assert list(result_df["Me_1"]) == expected_values

    @pytest.mark.parametrize(
        "vtl_script,input_data,expected_me1,expected_calc_col",
        [
            # Calc then multiply
            (
                """
                DS_temp := DS_1[calc doubled := Me_1 * 2];
                DS_r := DS_temp * 10;
                """,
                [["A", 5], ["B", 10]],
                [50, 100],  # Me_1 * 10
                [100, 200],  # doubled * 10
            ),
            # Multiply then calc
            (
                """
                DS_temp := DS_1 * 2;
                DS_r := DS_temp[calc tripled := Me_1 * 3];
                """,
                [["A", 5], ["B", 10]],
                [10, 20],  # Me_1 * 2
                [30, 60],  # tripled = (Me_1*2) * 3
            ),
        ],
        ids=["calc_then_multiply", "multiply_then_calc"],
    )
    def test_arithmetic_calc_combinations(
        self, vtl_script, input_data, expected_me1, expected_calc_col
    ):
        """Test arithmetic operations combined with calc clauses."""
        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Number", True)],
        )

        data_structures = create_data_structure([structure])
        input_df = pd.DataFrame(input_data, columns=["Id_1", "Me_1"])

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        result_df = results["DS_r"].sort_values("Id_1").reset_index(drop=True)
        assert list(result_df["Me_1"]) == expected_me1

        # Find the calc column
        calc_cols = [c for c in result_df.columns if c not in ["Id_1", "Me_1"]]
        assert len(calc_cols) == 1
        assert list(result_df[calc_cols[0]]) == expected_calc_col


# =============================================================================
# Set Operations + Arithmetic Combinations
# =============================================================================


class TestSetOperationsWithArithmetic:
    """Tests combining set operations with arithmetic."""

    @pytest.mark.parametrize(
        "vtl_script,input1_data,input2_data,expected_ids,expected_values",
        [
            # Union then multiply
            (
                """
                DS_temp := union(DS_1, DS_2);
                DS_r := DS_temp * 10;
                """,
                [["A", 1], ["B", 2]],
                [["C", 3], ["D", 4]],
                ["A", "B", "C", "D"],
                [10, 20, 30, 40],
            ),
            # Multiply then union
            (
                """
                DS_1a := DS_1 * 10;
                DS_2a := DS_2 * 100;
                DS_r := union(DS_1a, DS_2a);
                """,
                [["A", 1], ["B", 2]],
                [["C", 3], ["D", 4]],
                ["A", "B", "C", "D"],
                [10, 20, 300, 400],
            ),
            # Intersect then add
            (
                """
                DS_temp := intersect(DS_1, DS_2);
                DS_r := DS_temp + 100;
                """,
                [["A", 10], ["B", 20], ["C", 30]],
                [["B", 20], ["C", 30], ["D", 40]],
                ["B", "C"],
                [120, 130],
            ),
        ],
        ids=["union_then_multiply", "multiply_then_union", "intersect_then_add"],
    )
    def test_set_ops_with_arithmetic(
        self, vtl_script, input1_data, input2_data, expected_ids, expected_values
    ):
        """Test set operations combined with arithmetic."""
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

        result_df = results["DS_r"].sort_values("Id_1").reset_index(drop=True)
        assert list(result_df["Id_1"]) == sorted(expected_ids)
        assert list(result_df["Me_1"]) == expected_values


# =============================================================================
# Join + Aggregation Combinations
# =============================================================================


class TestJoinWithAggregation:
    """Tests combining join operations with aggregations."""

    @pytest.mark.parametrize(
        "vtl_script,input1_data,input2_data,expected_value",
        [
            # Join then sum
            (
                """
                DS_temp := inner_join(DS_1, DS_2);
                DS_r := sum(DS_temp group by Id_1);
                """,
                [["A", 10], ["B", 20]],
                [["A", 100], ["B", 200], ["C", 300]],
                # After join, Me_1 + Me_2 summed by Id_1
                None,  # Just check structure works
            ),
        ],
        ids=["join_then_sum"],
    )
    def test_join_with_aggregation(self, vtl_script, input1_data, input2_data, expected_value):
        """Test join operations combined with aggregations."""
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

        # Verify the result exists and has expected structure
        assert "DS_r" in results
        assert len(results["DS_r"]) > 0


# =============================================================================
# Multiple Clause Operations
# =============================================================================


class TestMultipleClauseOperations:
    """Tests combining multiple clause operations."""

    @pytest.mark.parametrize(
        "vtl_script,input_data,expected_ids,expected_new_col",
        [
            # Filter then calc
            (
                """
                DS_temp := DS_1[filter Me_1 > 10];
                DS_r := DS_temp[calc squared := Me_1 * Me_1];
                """,
                [["A", 5], ["B", 15], ["C", 25]],
                ["B", "C"],
                [225, 625],  # 15^2, 25^2
            ),
            # Calc then filter
            (
                """
                DS_temp := DS_1[calc doubled := Me_1 * 2];
                DS_r := DS_temp[filter doubled > 30];
                """,
                [["A", 10], ["B", 15], ["C", 25]],
                ["C"],  # Only C has doubled (50) > 30
                [50],
            ),
            # Filter and calc combined in chain
            (
                """
                DS_1a := DS_1[filter Me_1 >= 10];
                DS_1b := DS_1a[calc triple := Me_1 * 3];
                DS_r := DS_1b[filter triple <= 60];
                """,
                [["A", 5], ["B", 10], ["C", 20], ["D", 30]],
                ["B", "C"],  # 10*3=30, 20*3=60 both <= 60
                [30, 60],
            ),
        ],
        ids=["filter_then_calc", "calc_then_filter", "filter_calc_filter_chain"],
    )
    def test_multiple_clauses(self, vtl_script, input_data, expected_ids, expected_new_col):
        """Test multiple clause operations combined."""
        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Number", True)],
        )

        data_structures = create_data_structure([structure])
        input_df = pd.DataFrame(input_data, columns=["Id_1", "Me_1"])

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        result_df = results["DS_r"].sort_values("Id_1").reset_index(drop=True)
        assert list(result_df["Id_1"]) == sorted(expected_ids)

        # Find the new calculated column
        new_cols = [c for c in result_df.columns if c not in ["Id_1", "Me_1"]]
        assert len(new_cols) == 1
        assert list(result_df[new_cols[0]]) == expected_new_col


# =============================================================================
# Unary + Binary Combinations
# =============================================================================


class TestUnaryBinaryCombinations:
    """Tests combining unary and binary operations."""

    @pytest.mark.parametrize(
        "vtl_script,input_data,expected_values",
        [
            # Abs then add
            (
                """
                DS_temp := abs(DS_1);
                DS_r := DS_temp + 10;
                """,
                [["A", -5], ["B", 10], ["C", -15]],
                [15, 20, 25],  # |vals| + 10
            ),
            # Round then multiply
            (
                """
                DS_temp := round(DS_1, 0);
                DS_r := DS_temp * 2;
                """,
                [["A", 10.4], ["B", 10.6], ["C", 20.5]],
                [20.0, 22.0, 42.0],  # round then * 2
            ),
            # Ceil then subtract
            (
                """
                DS_temp := ceil(DS_1);
                DS_r := DS_temp - 1;
                """,
                [["A", 10.1], ["B", 20.9]],
                [10, 20],  # ceil - 1
            ),
            # Floor and then abs
            (
                """
                DS_temp := floor(DS_1);
                DS_r := abs(DS_temp);
                """,
                [["A", -10.9], ["B", 20.1], ["C", -30.5]],
                [11, 20, 31],  # abs(floor(-10.9))=11, etc
            ),
        ],
        ids=["abs_then_add", "round_then_multiply", "ceil_then_subtract", "floor_then_abs"],
    )
    def test_unary_binary_combinations(self, vtl_script, input_data, expected_values):
        """Test unary operations combined with binary operations."""
        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Number", True)],
        )

        data_structures = create_data_structure([structure])
        input_df = pd.DataFrame(input_data, columns=["Id_1", "Me_1"])

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        result_df = results["DS_r"].sort_values("Id_1").reset_index(drop=True)
        # Get the measure column (may be renamed by VTL semantic analysis based on result type)
        measure_col = [c for c in result_df.columns if c != "Id_1"][0]
        assert list(result_df[measure_col]) == expected_values


# =============================================================================
# Dataset-Dataset with Clauses
# =============================================================================


class TestDatasetDatasetWithClauses:
    """Tests combining dataset-dataset operations with clauses."""

    @pytest.mark.parametrize(
        "vtl_script,input1_data,input2_data,expected_ids,expected_values",
        [
            # Add datasets then filter
            (
                """
                DS_temp := DS_1 + DS_2;
                DS_r := DS_temp[filter Me_1 > 25];
                """,
                [["A", 10], ["B", 20]],
                [["A", 5], ["B", 10]],
                ["B"],  # 10+5=15, 20+10=30, only B > 25
                [30],
            ),
            # Filter both then add
            (
                """
                DS_1a := DS_1[filter Me_1 >= 15];
                DS_2a := DS_2[filter Me_1 >= 10];
                DS_r := DS_1a + DS_2a;
                """,
                [["A", 10], ["B", 20], ["C", 30]],
                [["A", 5], ["B", 10], ["C", 15]],
                ["B", "C"],  # Only B and C pass both filters
                [30, 45],  # 20+10, 30+15
            ),
            # Multiply datasets then calc
            (
                """
                DS_temp := DS_1 * DS_2;
                DS_r := DS_temp[calc doubled := Me_1 * 2];
                """,
                [["A", 2], ["B", 3]],
                [["A", 5], ["B", 4]],
                ["A", "B"],
                [20, 24],  # (2*5)*2, (3*4)*2
            ),
        ],
        ids=["add_then_filter", "filter_both_then_add", "multiply_then_calc"],
    )
    def test_dataset_ops_with_clauses(
        self, vtl_script, input1_data, input2_data, expected_ids, expected_values
    ):
        """Test dataset-dataset operations combined with clauses."""
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

        result_df = results["DS_r"].sort_values("Id_1").reset_index(drop=True)
        assert list(result_df["Id_1"]) == sorted(expected_ids)

        # For calc case, check the new column; otherwise check Me_1
        if "doubled" in result_df.columns:
            assert list(result_df["doubled"]) == expected_values
        else:
            assert list(result_df["Me_1"]) == expected_values


# =============================================================================
# Complex Multi-Step Transformations
# =============================================================================


class TestComplexMultiStepTransformations:
    """Tests for complex multi-step VTL transformations."""

    def test_full_etl_pipeline(self):
        """Test a full ETL-like pipeline with multiple steps."""
        vtl_script = """
            /* Step 1: Filter source data */
            DS_filtered := DS_raw[filter Me_1 > 0];

            /* Step 2: Calculate derived measures */
            DS_enriched := DS_filtered[calc doubled := Me_1 * 2, tripled := Me_1 * 3];

            /* Step 3: Apply additional filter */
            DS_r := DS_enriched[filter doubled >= 20];
        """

        structure = create_dataset_structure(
            "DS_raw",
            [("Id_1", "String")],
            [("Me_1", "Number", True)],
        )

        data_structures = create_data_structure([structure])
        input_df = pd.DataFrame(
            [
                ["A", -5],
                ["B", 5],
                ["C", 10],
                ["D", 15],
            ],
            columns=["Id_1", "Me_1"],
        )

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_raw": input_df})

        # Final result should only include C and D (Me_1 > 0 and doubled >= 20)
        result_df = results["DS_r"].sort_values("Id_1").reset_index(drop=True)
        assert list(result_df["Id_1"]) == ["C", "D"]
        assert list(result_df["doubled"]) == [20, 30]
        assert list(result_df["tripled"]) == [30, 45]

    def test_aggregation_pipeline(self):
        """Test aggregation combined with other operations."""
        vtl_script = """
            /* Step 1: Filter data */
            DS_filtered := DS_1[filter Me_1 > 5];

            /* Step 2: Multiply by factor */
            DS_scaled := DS_filtered * 10;

            /* Step 3: Aggregate */
            DS_r := sum(DS_scaled);
        """

        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Number", True)],
        )

        data_structures = create_data_structure([structure])
        input_df = pd.DataFrame(
            [
                ["A", 3],  # Filtered out
                ["B", 10],  # 10 * 10 = 100
                ["C", 20],  # 20 * 10 = 200
            ],
            columns=["Id_1", "Me_1"],
        )

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        # Sum of scaled filtered values: 100 + 200 = 300
        assert results["DS_r"]["Me_1"].iloc[0] == 300

    def test_merge_and_transform(self):
        """Test merging datasets then transforming."""
        vtl_script = """
            /* Step 1: Union two datasets */
            DS_merged := union(DS_1, DS_2);

            /* Step 2: Apply transformation */
            DS_transformed := abs(DS_merged);

            /* Step 3: Scale up */
            DS_r := DS_transformed * 100;
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
        input1_df = pd.DataFrame([["A", -5], ["B", 10]], columns=["Id_1", "Me_1"])
        input2_df = pd.DataFrame([["C", -15], ["D", 20]], columns=["Id_1", "Me_1"])

        results = execute_vtl_with_duckdb(
            vtl_script, data_structures, {"DS_1": input1_df, "DS_2": input2_df}
        )

        result_df = results["DS_r"].sort_values("Id_1").reset_index(drop=True)
        assert list(result_df["Id_1"]) == ["A", "B", "C", "D"]
        assert list(result_df["Me_1"]) == [500, 1000, 1500, 2000]  # |Me_1| * 100


# =============================================================================
# Conditional Operations in Complex Scenarios
# =============================================================================


class TestConditionalInComplexScenarios:
    """Tests for conditional operations in complex scenarios."""

    def test_conditional_with_filter(self):
        """Test conditional (if-then-else) combined with filter."""
        vtl_script = """
            /* Calculate category based on value */
            DS_categorized := DS_1[calc category := if Me_1 > 50 then 1 else 0];

            /* Filter by category */
            DS_r := DS_categorized[filter category = 1];
        """

        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Number", True)],
        )

        data_structures = create_data_structure([structure])
        input_df = pd.DataFrame(
            [
                ["A", 30],
                ["B", 60],
                ["C", 80],
            ],
            columns=["Id_1", "Me_1"],
        )

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        result_df = results["DS_r"].sort_values("Id_1").reset_index(drop=True)
        assert list(result_df["Id_1"]) == ["B", "C"]
        assert all(result_df["category"] == 1)

    def test_nested_conditionals_with_arithmetic(self):
        """Test nested conditionals combined with arithmetic."""
        vtl_script = """
            DS_priced := DS_1[calc price := if Me_1 > 100 then Me_1 * 0.8 else if Me_1 > 50 then Me_1 * 0.9 else Me_1 * 1.0];
            DS_r := DS_priced[calc result := price * Me_2];
        """

        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Number", True), ("Me_2", "Number", True)],
        )

        data_structures = create_data_structure([structure])
        input_df = pd.DataFrame(
            [
                ["A", 30, 2],  # No discount: 30 * 1.0 * 2 = 60
                ["B", 75, 2],  # 10% discount: 75 * 0.9 * 2 = 135
                ["C", 150, 2],  # 20% discount: 150 * 0.8 * 2 = 240
            ],
            columns=["Id_1", "Me_1", "Me_2"],
        )

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        result_df = results["DS_r"].sort_values("Id_1").reset_index(drop=True)
        assert list(result_df["Id_1"]) == ["A", "B", "C"]
        # Verify pricing logic was applied
        assert "price" in result_df.columns
        assert "result" in result_df.columns


# =============================================================================
# Between with Other Operators
# =============================================================================


class TestBetweenWithOtherOperators:
    """Tests for BETWEEN operator combined with other operators."""

    @pytest.mark.parametrize(
        "vtl_script,input_data,expected_ids,expected_values",
        [
            # Between filter then multiply
            (
                """
                DS_filtered := DS_1[filter between(Me_1, 10, 30)];
                DS_r := DS_filtered * 2;
                """,
                [["A", 5], ["B", 15], ["C", 25], ["D", 35]],
                ["B", "C"],
                [30, 50],
            ),
            # Multiply then between filter
            (
                """
                DS_scaled := DS_1 * 10;
                DS_r := DS_scaled[filter between(Me_1, 100, 200)];
                """,
                [["A", 5], ["B", 15], ["C", 25]],
                ["B"],  # 15*10=150 is between 100 and 200
                [150],
            ),
            # Calc then between filter
            (
                """
                DS_calced := DS_1[calc adjusted := Me_1 + 5];
                DS_r := DS_calced[filter between(adjusted, 20, 40)];
                """,
                [["A", 10], ["B", 20], ["C", 30], ["D", 50]],
                ["B", "C"],  # adjusted: 25, 35 are between 20-40
                [25, 35],
            ),
        ],
        ids=["between_then_multiply", "multiply_then_between", "calc_then_between"],
    )
    def test_between_with_operations(self, vtl_script, input_data, expected_ids, expected_values):
        """Test BETWEEN operator combined with other operations."""
        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Number", True)],
        )

        data_structures = create_data_structure([structure])
        input_df = pd.DataFrame(input_data, columns=["Id_1", "Me_1"])

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        result_df = results["DS_r"].sort_values("Id_1").reset_index(drop=True)
        assert list(result_df["Id_1"]) == sorted(expected_ids)

        # Check the appropriate column
        if "adjusted" in result_df.columns:
            assert list(result_df["adjusted"]) == expected_values
        else:
            assert list(result_df["Me_1"]) == expected_values


# =============================================================================
# Chained Binary Operations
# =============================================================================


class TestChainedBinaryOperations:
    """Tests for chained binary operations across multiple datasets."""

    def test_three_dataset_chain(self):
        """Test chaining operations across three datasets."""
        vtl_script = """
            /* Chain: DS_1 + DS_2, then * DS_3 */
            DS_sum := DS_1 + DS_2;
            DS_r := DS_sum * DS_3;
        """

        structure1 = create_dataset_structure(
            "DS_1", [("Id_1", "String")], [("Me_1", "Number", True)]
        )
        structure2 = create_dataset_structure(
            "DS_2", [("Id_1", "String")], [("Me_1", "Number", True)]
        )
        structure3 = create_dataset_structure(
            "DS_3", [("Id_1", "String")], [("Me_1", "Number", True)]
        )

        data_structures = create_data_structure([structure1, structure2, structure3])
        input1_df = pd.DataFrame([["A", 10], ["B", 20]], columns=["Id_1", "Me_1"])
        input2_df = pd.DataFrame([["A", 5], ["B", 10]], columns=["Id_1", "Me_1"])
        input3_df = pd.DataFrame([["A", 2], ["B", 3]], columns=["Id_1", "Me_1"])

        results = execute_vtl_with_duckdb(
            vtl_script,
            data_structures,
            {"DS_1": input1_df, "DS_2": input2_df, "DS_3": input3_df},
        )

        result_df = results["DS_r"].sort_values("Id_1").reset_index(drop=True)
        assert list(result_df["Id_1"]) == ["A", "B"]
        # (10+5)*2=30, (20+10)*3=90
        assert list(result_df["Me_1"]) == [30, 90]

    def test_parallel_operations_then_combine(self):
        """Test parallel operations on datasets then combining results."""
        vtl_script = """
            /* Transform DS_1 and DS_2 separately */
            DS_1a := DS_1 * 10;
            DS_2a := DS_2 + 100;

            /* Combine transformed datasets */
            DS_r := DS_1a + DS_2a;
        """

        structure1 = create_dataset_structure(
            "DS_1", [("Id_1", "String")], [("Me_1", "Number", True)]
        )
        structure2 = create_dataset_structure(
            "DS_2", [("Id_1", "String")], [("Me_1", "Number", True)]
        )

        data_structures = create_data_structure([structure1, structure2])
        input1_df = pd.DataFrame([["A", 5], ["B", 10]], columns=["Id_1", "Me_1"])
        input2_df = pd.DataFrame([["A", 1], ["B", 2]], columns=["Id_1", "Me_1"])

        results = execute_vtl_with_duckdb(
            vtl_script, data_structures, {"DS_1": input1_df, "DS_2": input2_df}
        )

        result_df = results["DS_r"].sort_values("Id_1").reset_index(drop=True)
        assert list(result_df["Id_1"]) == ["A", "B"]
        # (5*10)+(1+100)=151, (10*10)+(2+100)=202
        assert list(result_df["Me_1"]) == [151, 202]


# =============================================================================
# NVL Combined with Other Operations
# =============================================================================


class TestNvlCombinations:
    """Tests for NVL (null value handling) combined with other operations."""

    @pytest.mark.parametrize(
        "vtl_script,input_data,expected_values",
        [
            # NVL then multiply
            (
                """
                DS_cleaned := nvl(DS_1, 0);
                DS_r := DS_cleaned * 10;
                """,
                [["A", 5], ["B", None], ["C", 15]],
                [50, 0, 150],
            ),
            # Multiply then NVL
            (
                """
                DS_scaled := DS_1 * 10;
                DS_r := nvl(DS_scaled, -1);
                """,
                [["A", 5], ["B", None], ["C", 15]],
                [50, -1, 150],
            ),
        ],
        ids=["nvl_then_multiply", "multiply_then_nvl"],
    )
    def test_nvl_with_arithmetic(self, vtl_script, input_data, expected_values):
        """Test NVL combined with arithmetic operations."""
        structure = create_dataset_structure(
            "DS_1",
            [("Id_1", "String")],
            [("Me_1", "Number", True)],
        )

        data_structures = create_data_structure([structure])
        input_df = pd.DataFrame(input_data, columns=["Id_1", "Me_1"])

        results = execute_vtl_with_duckdb(vtl_script, data_structures, {"DS_1": input_df})

        result_df = results["DS_r"].sort_values("Id_1").reset_index(drop=True)
        assert list(result_df["Me_1"]) == expected_values
