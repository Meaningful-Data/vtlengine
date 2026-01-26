"""
Benchmark script for DuckDB parser performance testing.

Tests CSV loading and validation with different file sizes.
"""

import gc
import json
import os
import sys
import time
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from duckdb_transpiler.config import create_configured_connection, get_system_info
from duckdb_transpiler.DuckdbPerformance.file_generator import (
    DP_PATH,
    DS_PATH,
    generate_big_csv,
    generate_big_csv_fast,
    get_suffix,
)
from duckdb_transpiler.Parser import load_datapoints_duckdb
from vtlengine.DataTypes import Integer, Number, String
from vtlengine.Model import Component, Role


def load_datastructure(json_path: Path) -> dict[str, Component]:
    """Load data structure from JSON file and convert to Component dict."""
    with open(json_path) as f:
        data = json.load(f)

    components = {}
    ds = data["datasets"][0]["DataStructure"]

    type_map = {
        "Integer": Integer,
        "Number": Number,
        "String": String,
    }

    for comp in ds:
        name = comp["name"]
        dtype = type_map[comp["type"]]
        role = Role.IDENTIFIER if comp["role"] == "Identifier" else Role.MEASURE
        nullable = comp["nullable"]
        components[name] = Component(name=name, data_type=dtype, role=role, nullable=nullable)

    return components


def benchmark_load(csv_path: Path, json_path: Path, dataset_name: str = "DS_1") -> dict:
    """
    Benchmark DuckDB CSV loading and validation.

    Returns dict with timing metrics.
    """
    components = load_datastructure(json_path)
    conn = create_configured_connection()

    # Time the load
    start_total = time.perf_counter()
    load_datapoints_duckdb(conn, components, dataset_name, csv_path)
    end_load = time.perf_counter()

    # Count rows (forces materialization)
    row_count = conn.execute(f"SELECT COUNT(*) FROM {dataset_name}").fetchone()[0]
    end_count = time.perf_counter()

    # Get file size
    file_size_mb = os.path.getsize(csv_path) / (1024**2)

    results = {
        "file": csv_path.name,
        "rows": row_count,
        "size_mb": file_size_mb,
        "load_time_s": end_load - start_total,
        "total_time_s": end_count - start_total,
        "rows_per_sec": row_count / (end_load - start_total),
        "mb_per_sec": file_size_mb / (end_load - start_total),
    }

    conn.close()
    gc.collect()

    return results


def format_results(results: dict) -> str:
    """Format benchmark results for display."""
    lines = [
        f"File: {results['file']}",
        f"  Rows: {results['rows']:,}",
        f"  Size: {results['size_mb']:.2f} MB",
        f"  Load time: {results['load_time_s']:.3f}s",
        f"  Total time: {results['total_time_s']:.3f}s",
        f"  Throughput: {results['rows_per_sec']:,.0f} rows/s",
        f"  Throughput: {results['mb_per_sec']:.1f} MB/s",
    ]
    return "\n".join(lines)


def run_benchmark_suite(sizes: list[int], shuffle: bool = False, generator: str = "c"):
    """
    Run full benchmark suite.

    Args:
        sizes: List of row counts to test
        shuffle: Whether to shuffle rows
        generator: Which generator to use ("python" or "c")
    """
    dtypes = {
        "Id_1": "Integer",
        "Id_2": "String",
        "Id_3": "Integer",
        "Me_1": "Number",
        "Me_2": "Integer",
    }

    print("=" * 70)
    print("DuckDB Parser Benchmark")
    print("=" * 70)
    print(f"Columns: {list(dtypes.keys())}")
    print(f"Shuffle: {shuffle}")
    print(f"Generator: {generator.capitalize()}")

    # Show system info
    info = get_system_info()
    print("\nSystem Info:")
    print(f"  Total RAM: {info['total_ram_gb']:.1f} GB")
    print(f"  Available RAM: {info['available_ram_gb']:.1f} GB")
    print(f"  Memory Limit: {info['configured_limit_str']}")
    print(f"  Threads: {info['threads']}")
    print(f"  Temp Dir: {info['temp_directory']}")
    print()

    all_results = []

    for rows in sizes:
        print(f"\n{'=' * 70}")
        print(f"Testing {rows:,} rows")
        print("=" * 70)

        # Generate CSV
        suffix = get_suffix(rows)
        csv_path = DP_PATH / f"BF_{suffix}"
        json_path = DS_PATH / suffix.replace(".csv", ".json")

        # Check if file exists
        if csv_path.exists():
            print(f"Using existing file: {csv_path}")
        else:
            print("Generating CSV...")
            gen_start = time.time()
            if generator == "python":
                generate_big_csv(dtypes=dtypes, length=rows, shuffle=shuffle)
            else:
                generate_big_csv_fast(dtypes=dtypes, length=rows, shuffle=shuffle)
            gen_time = time.time() - gen_start
            print(f"Generation time: {gen_time:.2f}s")

        # Run benchmark
        print("\nBenchmarking DuckDB parser...")
        results = benchmark_load(csv_path, json_path)
        print(format_results(results))

        all_results.append(results)

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Rows':>12} | {'Size (MB)':>10} | {'Time (s)':>10} | {'rows/s':>12} | {'MB/s':>8}")
    print("-" * 70)
    for r in all_results:
        print(
            f"{r['rows']:>12,} | {r['size_mb']:>10.1f} | "
            f"{r['load_time_s']:>10.3f} | {r['rows_per_sec']:>12,.0f} | "
            f"{r['mb_per_sec']:>8.1f}"
        )

    return all_results


if __name__ == "__main__":
    args = sys.argv[1:]
    shuffle = False
    generator = "c"
    sizes = [1_000_000]

    if "--shuffle" in args:
        shuffle = True
    if "--generator" in args:
        idx = args.index("--generator")
        if idx + 1 < len(args) and args[idx + 1] in ("python", "c"):
            generator = args[idx + 1]
    if "--sizes" in args:
        idx = args.index("--sizes")
        if idx + 1 < len(args):
            sizes_str = args[idx + 1]
            sizes = [int(s.strip()) for s in sizes_str.split(",") if s.strip().isdigit()]

    run_benchmark_suite(sizes=sizes, shuffle=shuffle, generator=generator)
