#!/usr/bin/env python3
"""
Compare VTL execution results between Pandas and DuckDB engines.

This script executes a VTL script using both engines and compares the results
for each output dataset, including column-by-column value comparison.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from vtlengine import run


def compare_dataframes(
    df_pandas: pd.DataFrame,
    df_duckdb: pd.DataFrame,
    dataset_name: str,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> Tuple[bool, List[str]]:
    """
    Compare two DataFrames and return detailed differences.

    Args:
        df_pandas: DataFrame from Pandas engine
        df_duckdb: DataFrame from DuckDB engine
        dataset_name: Name of the dataset being compared
        rtol: Relative tolerance for numeric comparison
        atol: Absolute tolerance for numeric comparison

    Returns:
        Tuple of (is_equal, list_of_differences)
    """
    differences: List[str] = []

    # Compare columns
    pandas_cols = set(df_pandas.columns)
    duckdb_cols = set(df_duckdb.columns)

    if pandas_cols != duckdb_cols:
        only_pandas = pandas_cols - duckdb_cols
        only_duckdb = duckdb_cols - pandas_cols
        if only_pandas:
            differences.append(f"Columns only in Pandas: {sorted(only_pandas)}")
        if only_duckdb:
            differences.append(f"Columns only in DuckDB: {sorted(only_duckdb)}")

    # Use common columns for comparison
    common_cols = sorted(pandas_cols & duckdb_cols)
    if not common_cols:
        differences.append("No common columns to compare")
        return False, differences

    # Sort both dataframes by common columns for consistent comparison
    # Try to sort by identifier columns first
    sort_cols = [c for c in common_cols if not c.startswith(("Me_", "bool_", "error", "imbalance"))]
    if not sort_cols:
        sort_cols = common_cols[:3]  # Use first 3 columns

    try:
        df_p = df_pandas[common_cols].sort_values(sort_cols).reset_index(drop=True)
        df_d = df_duckdb[common_cols].sort_values(sort_cols).reset_index(drop=True)
    except Exception as e:
        differences.append(f"Error sorting dataframes: {e}")
        df_p = df_pandas[common_cols].reset_index(drop=True)
        df_d = df_duckdb[common_cols].reset_index(drop=True)

    # Compare row counts
    if len(df_p) != len(df_d):
        differences.append(f"Row count mismatch: Pandas={len(df_p)}, DuckDB={len(df_d)}")

    # Compare values column by column
    min_rows = min(len(df_p), len(df_d))
    for col in common_cols:
        col_p = df_p[col].iloc[:min_rows]
        col_d = df_d[col].iloc[:min_rows]

        # Handle different types
        if pd.api.types.is_numeric_dtype(col_p) and pd.api.types.is_numeric_dtype(col_d):
            # Numeric comparison with tolerance
            try:
                # Handle NaN values
                nan_mask_p = pd.isna(col_p)
                nan_mask_d = pd.isna(col_d)

                if not (nan_mask_p == nan_mask_d).all():
                    nan_diff_count = (nan_mask_p != nan_mask_d).sum()
                    differences.append(f"Column '{col}': {nan_diff_count} rows differ in NaN values")

                # Compare non-NaN values
                valid_mask = ~nan_mask_p & ~nan_mask_d
                if valid_mask.any():
                    vals_p = col_p[valid_mask].values
                    vals_d = col_d[valid_mask].values

                    if not np.allclose(vals_p, vals_d, rtol=rtol, atol=atol, equal_nan=True):
                        diff_mask = ~np.isclose(vals_p, vals_d, rtol=rtol, atol=atol, equal_nan=True)
                        diff_count = diff_mask.sum()
                        if diff_count > 0:
                            max_diff = np.max(np.abs(vals_p[diff_mask] - vals_d[diff_mask]))
                            differences.append(
                                f"Column '{col}': {diff_count} values differ (max diff: {max_diff:.6e})"
                            )
            except Exception as e:
                differences.append(f"Column '{col}': Error comparing numeric values: {e}")

        elif pd.api.types.is_bool_dtype(col_p) or pd.api.types.is_bool_dtype(col_d):
            # Boolean comparison
            try:
                # Convert to boolean for comparison
                vals_p = col_p.astype(bool)
                vals_d = col_d.astype(bool)
                diff_count = (vals_p != vals_d).sum()
                if diff_count > 0:
                    differences.append(f"Column '{col}': {diff_count} boolean values differ")
            except Exception as e:
                differences.append(f"Column '{col}': Error comparing boolean values: {e}")

        else:
            # String/object comparison
            try:
                # Convert to string for comparison
                str_p = col_p.astype(str)
                str_d = col_d.astype(str)
                diff_count = (str_p != str_d).sum()
                if diff_count > 0:
                    differences.append(f"Column '{col}': {diff_count} string values differ")
            except Exception as e:
                differences.append(f"Column '{col}': Error comparing string values: {e}")

    return len(differences) == 0, differences


def run_comparison(
    script_path: Path,
    data_structures_path: Path,
    data_path: Path,
    dataset_name: str,
    verbose: bool = False,
) -> Dict[str, Tuple[bool, List[str]]]:
    """
    Run VTL script with both engines and compare results.

    Args:
        script_path: Path to VTL script
        data_structures_path: Path to data structures JSON
        data_path: Path to CSV data file
        dataset_name: Name of the input dataset
        verbose: Whether to print verbose output

    Returns:
        Dictionary mapping dataset names to (is_equal, differences) tuples
    """
    print(f"Loading data from {data_path}...")

    # Run with Pandas
    print("Running with Pandas engine...")
    result_pandas = run(
        script=script_path,
        data_structures=data_structures_path,
        datapoints={dataset_name: data_path},
        use_duckdb=False,
    )

    # Run with DuckDB
    print("Running with DuckDB engine...")
    result_duckdb = run(
        script=script_path,
        data_structures=data_structures_path,
        datapoints={dataset_name: data_path},
        use_duckdb=True,
    )

    # Compare results
    print("\nComparing results...")
    print("=" * 80)

    comparison_results: Dict[str, Tuple[bool, List[str]]] = {}

    # Get all output dataset names
    pandas_datasets = set(result_pandas.keys())
    duckdb_datasets = set(result_duckdb.keys())

    if pandas_datasets != duckdb_datasets:
        print(f"\nWARNING: Dataset mismatch!")
        if pandas_datasets - duckdb_datasets:
            print(f"  Only in Pandas: {sorted(pandas_datasets - duckdb_datasets)}")
        if duckdb_datasets - pandas_datasets:
            print(f"  Only in DuckDB: {sorted(duckdb_datasets - pandas_datasets)}")

    common_datasets = sorted(pandas_datasets & duckdb_datasets)

    for ds_name in common_datasets:
        df_pandas = result_pandas[ds_name].data
        df_duckdb = result_duckdb[ds_name].data

        is_equal, differences = compare_dataframes(df_pandas, df_duckdb, ds_name)
        comparison_results[ds_name] = (is_equal, differences)

        if is_equal:
            status = "MATCH"
            color = "\033[92m"  # Green
        else:
            status = "DIFFER"
            color = "\033[91m"  # Red

        reset = "\033[0m"
        print(f"\n{color}[{status}]{reset} {ds_name} (Pandas: {len(df_pandas)} rows, DuckDB: {len(df_duckdb)} rows)")

        if not is_equal:
            for diff in differences:
                print(f"       - {diff}")

        if verbose and is_equal:
            print(f"       Columns: {sorted(df_pandas.columns.tolist())}")

    return comparison_results


def print_summary(comparison_results: Dict[str, Tuple[bool, List[str]]]) -> bool:
    """Print summary of comparison results and return True if all match."""
    total = len(comparison_results)
    matches = sum(1 for is_equal, _ in comparison_results.values() if is_equal)
    differs = total - matches

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total datasets compared: {total}")
    print(f"  Matching: {matches}")
    print(f"  Differing: {differs}")

    if differs == 0:
        print("\n\033[92mSUCCESS: All datasets match!\033[0m")
        return True
    else:
        print(f"\n\033[91mFAILURE: {differs} dataset(s) have differences\033[0m")
        print("\nDatasets with differences:")
        for ds_name, (is_equal, differences) in comparison_results.items():
            if not is_equal:
                print(f"  - {ds_name}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Compare VTL execution results between Pandas and DuckDB engines."
    )
    parser.add_argument(
        "--script",
        type=Path,
        default=Path(__file__).parent / "test_bdi.vtl",
        help="Path to VTL script (default: test_bdi.vtl)",
    )
    parser.add_argument(
        "--structures",
        type=Path,
        default=Path(__file__).parent / "PoC_Dataset.json",
        help="Path to data structures JSON (default: PoC_Dataset.json)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parent / "PoC_10K.csv",
        help="Path to CSV data file (default: PoC_10K.csv)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="PoC_Dataset",
        help="Name of the input dataset (default: PoC_Dataset)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Validate paths
    for path, name in [(args.script, "script"), (args.structures, "structures"), (args.data, "data")]:
        if not path.exists():
            print(f"Error: {name} file not found: {path}")
            sys.exit(1)

    print(f"VTL Script: {args.script}")
    print(f"Data Structures: {args.structures}")
    print(f"Data File: {args.data}")
    print(f"Dataset Name: {args.dataset_name}")
    print()

    comparison_results = run_comparison(
        args.script,
        args.structures,
        args.data,
        args.dataset_name,
        args.verbose,
    )

    success = print_summary(comparison_results)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
