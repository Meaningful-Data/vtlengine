#!/usr/bin/env python3
"""
Compare VTL execution results between Pandas and DuckDB engines.

This script executes a VTL script using both engines and compares the results
for each output dataset, including column-by-column value comparison and
performance metrics (time and memory usage).
"""

import argparse
import gc
import os
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from vtlengine import run

# =============================================================================
# CONFIGURATION - Adjust these values as needed
# =============================================================================
DEFAULT_THREADS = 4  # Number of threads for DuckDB
DEFAULT_MEMORY_LIMIT = "8GB"  # Memory limit for DuckDB (e.g., "4GB", "8GB", "16GB")
DEFAULT_RUNS = 3  # Number of runs for performance averaging


@dataclass
class PerformanceMetrics:
    """Container for performance metrics from a single run."""

    time_seconds: float
    peak_memory_mb: float
    current_memory_mb: float


@dataclass
class PerformanceStats:
    """Aggregated performance statistics across multiple runs."""

    engine: str
    num_rows: int
    runs: int
    time_min: float = 0.0
    time_max: float = 0.0
    time_avg: float = 0.0
    memory_min_mb: float = 0.0
    memory_max_mb: float = 0.0
    memory_avg_mb: float = 0.0
    all_times: List[float] = field(default_factory=list)
    all_memories: List[float] = field(default_factory=list)

    def calculate_stats(self) -> None:
        """Calculate min/max/avg from collected metrics."""
        if self.all_times:
            self.time_min = min(self.all_times)
            self.time_max = max(self.all_times)
            self.time_avg = sum(self.all_times) / len(self.all_times)
        if self.all_memories:
            self.memory_min_mb = min(self.all_memories)
            self.memory_max_mb = max(self.all_memories)
            self.memory_avg_mb = sum(self.all_memories) / len(self.all_memories)


def configure_duckdb(threads: int, memory_limit: str) -> None:
    """Configure DuckDB settings via environment variables.

    vtlengine uses VTL_* environment variables (see Config/config.py):
    - VTL_THREADS: Number of threads for DuckDB
    - VTL_MEMORY_LIMIT: Max memory (e.g., "8GB", "80%")
    """
    os.environ["VTL_THREADS"] = str(threads)
    os.environ["VTL_MEMORY_LIMIT"] = memory_limit


def measure_execution(
    script_path: Path,
    data_structures_path: Path,
    data_path: Path,
    dataset_name: str,
    use_duckdb: bool,
    threads: int,
    memory_limit: str,
) -> Tuple[Dict[str, Any], PerformanceMetrics]:
    """
    Execute VTL script and measure performance.

    Returns:
        Tuple of (result_dict, PerformanceMetrics)
    """
    # Force garbage collection before measurement
    gc.collect()

    # Start memory tracking
    tracemalloc.start()

    # Configure DuckDB if needed
    if use_duckdb:
        configure_duckdb(threads, memory_limit)

    # Measure execution time
    start_time = time.perf_counter()

    result = run(
        script=script_path,
        data_structures=data_structures_path,
        datapoints={dataset_name: data_path},
        use_duckdb=use_duckdb,
    )

    end_time = time.perf_counter()

    # Get memory stats
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    metrics = PerformanceMetrics(
        time_seconds=end_time - start_time,
        peak_memory_mb=peak / (1024 * 1024),
        current_memory_mb=current / (1024 * 1024),
    )

    return result, metrics


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
                    differences.append(
                        f"Column '{col}': {nan_diff_count} rows differ in NaN values"
                    )

                # Compare non-NaN values
                valid_mask = ~nan_mask_p & ~nan_mask_d
                if valid_mask.any():
                    vals_p = col_p[valid_mask].values
                    vals_d = col_d[valid_mask].values

                    if not np.allclose(vals_p, vals_d, rtol=rtol, atol=atol, equal_nan=True):
                        diff_mask = ~np.isclose(
                            vals_p, vals_d, rtol=rtol, atol=atol, equal_nan=True
                        )
                        diff_count = diff_mask.sum()
                        if diff_count > 0:
                            max_diff = np.max(np.abs(vals_p[diff_mask] - vals_d[diff_mask]))
                            msg = f"Column '{col}': {diff_count} values differ "
                            msg += f"(max diff: {max_diff:.6e})"
                            differences.append(msg)
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


def run_performance_comparison(
    script_path: Path,
    data_structures_path: Path,
    data_path: Path,
    dataset_name: str,
    num_runs: int,
    threads: int,
    memory_limit: str,
    verbose: bool = False,
    duckdb_only: bool = False,
) -> Tuple[Dict[str, Tuple[bool, List[str]]], Optional[PerformanceStats], PerformanceStats]:
    """
    Run VTL script with both engines multiple times and collect performance stats.

    Returns:
        Tuple of (comparison_results, pandas_stats, duckdb_stats)
    """
    # Count rows in input file
    with open(data_path) as f:
        num_rows = sum(1 for _ in f) - 1  # Subtract header

    print(f"Input file: {data_path}")
    print(f"Number of rows: {num_rows:,}")
    print(f"Number of runs: {num_runs}")
    print(f"DuckDB threads: {threads}")
    print(f"DuckDB memory limit: {memory_limit}")
    print()

    pandas_stats: Optional[PerformanceStats] = None
    duckdb_stats = PerformanceStats(engine="DuckDB", num_rows=num_rows, runs=num_runs)

    result_pandas: Optional[Dict[str, Any]] = None
    result_duckdb: Optional[Dict[str, Any]] = None

    # Run Pandas engine multiple times (skip if duckdb_only)
    if not duckdb_only:
        pandas_stats = PerformanceStats(engine="Pandas", num_rows=num_rows, runs=num_runs)
        print(f"Running Pandas engine ({num_runs} runs)...")
        for i in range(num_runs):
            result, metrics = measure_execution(
                script_path,
                data_structures_path,
                data_path,
                dataset_name,
                use_duckdb=False,
                threads=threads,
                memory_limit=memory_limit,
            )
            pandas_stats.all_times.append(metrics.time_seconds)
            pandas_stats.all_memories.append(metrics.peak_memory_mb)
            if result_pandas is None:
                result_pandas = result
            if verbose:
                mem_mb = metrics.peak_memory_mb
                print(f"  Run {i + 1}: {metrics.time_seconds:.2f}s, {mem_mb:.1f} MB")
            gc.collect()

        pandas_stats.calculate_stats()
    else:
        print("Skipping Pandas engine (--duckdb-only mode)")

    # Run DuckDB engine multiple times
    print(f"Running DuckDB engine ({num_runs} runs)...")
    for i in range(num_runs):
        result, metrics = measure_execution(
            script_path,
            data_structures_path,
            data_path,
            dataset_name,
            use_duckdb=True,
            threads=threads,
            memory_limit=memory_limit,
        )
        duckdb_stats.all_times.append(metrics.time_seconds)
        duckdb_stats.all_memories.append(metrics.peak_memory_mb)
        if result_duckdb is None:
            result_duckdb = result
        if verbose:
            print(f"  Run {i + 1}: {metrics.time_seconds:.2f}s, {metrics.peak_memory_mb:.1f} MB")
        gc.collect()

    duckdb_stats.calculate_stats()

    comparison_results: Dict[str, Tuple[bool, List[str]]] = {}

    # Skip comparison in duckdb_only mode
    if duckdb_only:
        if result_duckdb is None:
            raise RuntimeError("Failed to get results from DuckDB engine")
        print(f"\nDuckDB produced {len(result_duckdb)} datasets")
        return comparison_results, pandas_stats, duckdb_stats

    # Compare results from last run
    print("\nComparing results...")
    print("=" * 80)

    if result_pandas is None or result_duckdb is None:
        raise RuntimeError("Failed to get results from one or both engines")

    pandas_datasets = set(result_pandas.keys())
    duckdb_datasets = set(result_duckdb.keys())

    if pandas_datasets != duckdb_datasets:
        print("\nWARNING: Dataset mismatch!")
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
        print(
            f"\n{color}[{status}]{reset} {ds_name} "
            f"(Pandas: {len(df_pandas)} rows, DuckDB: {len(df_duckdb)} rows)"
        )

        if not is_equal:
            for diff in differences:
                print(f"       - {diff}")

        if verbose and is_equal:
            print(f"       Columns: {sorted(df_pandas.columns.tolist())}")

    return comparison_results, pandas_stats, duckdb_stats


def print_performance_table(
    pandas_stats: Optional[PerformanceStats], duckdb_stats: PerformanceStats
) -> None:
    """Print a formatted performance comparison table."""
    print("\n" + "=" * 100)
    print("PERFORMANCE COMPARISON")
    print("=" * 100)

    # DuckDB-only mode
    if pandas_stats is None:
        print(f"{'Metric':<25} {'DuckDB':>20}")
        print("-" * 50)
        print(f"{'Input Rows':<25} {duckdb_stats.num_rows:>20,}")
        print(f"{'Number of Runs':<25} {duckdb_stats.runs:>20}")
        print()
        print(f"{'Time (min)':<25} {duckdb_stats.time_min:>19.2f}s")
        print(f"{'Time (max)':<25} {duckdb_stats.time_max:>19.2f}s")
        print(f"{'Time (avg)':<25} {duckdb_stats.time_avg:>19.2f}s")
        print()
        print(f"{'Peak Memory (min)':<25} {duckdb_stats.memory_min_mb:>18.1f}MB")
        print(f"{'Peak Memory (max)':<25} {duckdb_stats.memory_max_mb:>18.1f}MB")
        print(f"{'Peak Memory (avg)':<25} {duckdb_stats.memory_avg_mb:>18.1f}MB")
        print("=" * 100)
        return

    # Full comparison mode
    print(f"{'Metric':<25} {'Pandas':>20} {'DuckDB':>20} {'Speedup':>15} {'Memory Ratio':>15}")
    print("-" * 100)

    # Rows
    print(f"{'Input Rows':<25} {pandas_stats.num_rows:>20,}")
    print(f"{'Number of Runs':<25} {pandas_stats.runs:>20}")
    print()

    # Time metrics
    speedup_min = pandas_stats.time_min / duckdb_stats.time_min if duckdb_stats.time_min > 0 else 0
    speedup_max = pandas_stats.time_max / duckdb_stats.time_max if duckdb_stats.time_max > 0 else 0
    speedup_avg = pandas_stats.time_avg / duckdb_stats.time_avg if duckdb_stats.time_avg > 0 else 0

    print(
        f"{'Time (min)':<25} {pandas_stats.time_min:>19.2f}s "
        f"{duckdb_stats.time_min:>19.2f}s {speedup_min:>14.2f}x"
    )
    print(
        f"{'Time (max)':<25} {pandas_stats.time_max:>19.2f}s "
        f"{duckdb_stats.time_max:>19.2f}s {speedup_max:>14.2f}x"
    )
    print(
        f"{'Time (avg)':<25} {pandas_stats.time_avg:>19.2f}s "
        f"{duckdb_stats.time_avg:>19.2f}s {speedup_avg:>14.2f}x"
    )
    print()

    # Memory metrics
    mem_ratio_min = (
        duckdb_stats.memory_min_mb / pandas_stats.memory_min_mb
        if pandas_stats.memory_min_mb > 0
        else 0
    )
    mem_ratio_max = (
        duckdb_stats.memory_max_mb / pandas_stats.memory_max_mb
        if pandas_stats.memory_max_mb > 0
        else 0
    )
    mem_ratio_avg = (
        duckdb_stats.memory_avg_mb / pandas_stats.memory_avg_mb
        if pandas_stats.memory_avg_mb > 0
        else 0
    )

    print(
        f"{'Peak Memory (min)':<25} {pandas_stats.memory_min_mb:>18.1f}MB "
        f"{duckdb_stats.memory_min_mb:>18.1f}MB {'':<14} {mem_ratio_min:>14.2f}x"
    )
    print(
        f"{'Peak Memory (max)':<25} {pandas_stats.memory_max_mb:>18.1f}MB "
        f"{duckdb_stats.memory_max_mb:>18.1f}MB {'':<14} {mem_ratio_max:>14.2f}x"
    )
    print(
        f"{'Peak Memory (avg)':<25} {pandas_stats.memory_avg_mb:>18.1f}MB "
        f"{duckdb_stats.memory_avg_mb:>18.1f}MB {'':<14} {mem_ratio_avg:>14.2f}x"
    )

    print("=" * 100)

    # Summary
    speedup = pandas_stats.time_avg / duckdb_stats.time_avg if duckdb_stats.time_avg > 0 else 0
    if speedup > 1:
        print(f"\n\033[92mDuckDB is {speedup:.2f}x faster than Pandas (avg)\033[0m")
    elif speedup < 1 and speedup > 0:
        print(f"\n\033[93mPandas is {1 / speedup:.2f}x faster than DuckDB (avg)\033[0m")
    else:
        print("\nPerformance is similar")


def print_summary(comparison_results: Dict[str, Tuple[bool, List[str]]]) -> bool:
    """Print summary of comparison results and return True if all match."""
    total = len(comparison_results)
    matches = sum(1 for is_equal, _ in comparison_results.values() if is_equal)
    differs = total - matches

    print("\n" + "=" * 80)
    print("CORRECTNESS SUMMARY")
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
        for ds_name, (is_equal, _) in comparison_results.items():
            if not is_equal:
                print(f"  - {ds_name}")
        return False


def main() -> None:
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
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"Number of runs for performance averaging (default: {DEFAULT_RUNS})",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help=f"Number of threads for DuckDB (default: {DEFAULT_THREADS})",
    )
    parser.add_argument(
        "--memory-limit",
        type=str,
        default=DEFAULT_MEMORY_LIMIT,
        help=f"Memory limit for DuckDB (default: {DEFAULT_MEMORY_LIMIT})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--skip-correctness",
        action="store_true",
        help="Skip correctness comparison (only show performance)",
    )
    parser.add_argument(
        "--duckdb-only",
        action="store_true",
        help="Run DuckDB only (skip Pandas engine)",
    )

    args = parser.parse_args()

    # Validate paths
    for path, name in [
        (args.script, "script"),
        (args.structures, "structures"),
        (args.data, "data"),
    ]:
        if not path.exists():
            print(f"Error: {name} file not found: {path}")
            sys.exit(1)

    print("=" * 80)
    if args.duckdb_only:
        print("VTL ENGINE BENCHMARK: DuckDB Only")
    else:
        print("VTL ENGINE COMPARISON: Pandas vs DuckDB")
    print("=" * 80)
    print(f"VTL Script: {args.script}")
    print(f"Data Structures: {args.structures}")
    print(f"Data File: {args.data}")
    print(f"Dataset Name: {args.dataset_name}")
    print()

    comparison_results, pandas_stats, duckdb_stats = run_performance_comparison(
        args.script,
        args.structures,
        args.data,
        args.dataset_name,
        args.runs,
        args.threads,
        args.memory_limit,
        args.verbose,
        args.duckdb_only,
    )

    # Print performance table
    print_performance_table(pandas_stats, duckdb_stats)

    # Print correctness summary
    if not args.skip_correctness:
        success = print_summary(comparison_results)
        sys.exit(0 if success else 1)
    else:
        print("\n(Correctness comparison skipped)")
        sys.exit(0)


if __name__ == "__main__":
    main()
