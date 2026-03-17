import os
import time

import numpy as np
import pandas as pd

from vtlengine import run

# Force DuckDB to use 1 thread for fair comparison
os.environ["VTL_THREADS"] = "1"

COMPARISON_SCRIPT = "DS_r <- DS_1[calc Me_3 := Me_1 > Me_2];"
MINMAX_SCRIPT = """
    DS_min <- min(DS_1 group by Id_2);
    DS_max <- max(DS_1 group by Id_2);
"""

COMPARISON_STRUCTURES = {
    "datasets": [
        {
            "name": "DS_1",
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Time_Period", "role": "Measure", "nullable": False},
                {"name": "Me_2", "type": "Time_Period", "role": "Measure", "nullable": False},
            ],
        }
    ]
}

MINMAX_STRUCTURES = {
    "datasets": [
        {
            "name": "DS_1",
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Id_2", "type": "String", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Time_Period", "role": "Measure", "nullable": False},
            ],
        }
    ]
}


def generate_comparison_data(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    years = rng.integers(2000, 2025, size=n)
    months1 = rng.integers(1, 13, size=n)
    months2 = rng.integers(1, 13, size=n)
    tp1 = [f"{y}-M{m:02d}" for y, m in zip(years, months1)]
    tp2 = [f"{y}-M{m:02d}" for y, m in zip(years, months2)]
    return pd.DataFrame({"Id_1": np.arange(1, n + 1), "Me_1": tp1, "Me_2": tp2})


def generate_agg_data(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    groups = [str(g) for g in rng.integers(1, 101, size=n)]
    years = rng.integers(2000, 2025, size=n)
    months = rng.integers(1, 13, size=n)
    tp = [f"{y}-M{m:02d}" for y, m in zip(years, months)]
    return pd.DataFrame({"Id_1": np.arange(1, n + 1), "Id_2": groups, "Me_1": tp})


def bench_comparison(rows: int) -> dict:
    print(f"\n{'=' * 70}")
    print(f"  COMPARISON: DS_r <- DS_1[calc Me_3 := Me_1 > Me_2]  |  {rows:,} rows")
    print(f"{'=' * 70}")

    df = generate_comparison_data(rows)
    datapoints = {"DS_1": df}

    t0 = time.perf_counter()
    rp = run(script=COMPARISON_SCRIPT, data_structures=COMPARISON_STRUCTURES, datapoints=datapoints)
    t_pandas = time.perf_counter() - t0

    t0 = time.perf_counter()
    rd = run(
        script=COMPARISON_SCRIPT,
        data_structures=COMPARISON_STRUCTURES,
        datapoints=datapoints,
        use_duckdb=True,
    )
    t_duckdb = time.perf_counter() - t0

    match = (
        rp["DS_r"].data.sort_values("Id_1").reset_index(drop=True)["Me_3"]
        == rd["DS_r"].data.sort_values("Id_1").reset_index(drop=True)["Me_3"]
    ).all()

    speedup = t_pandas / t_duckdb if t_duckdb > 0 else float("inf")
    print(f"  Pandas:  {t_pandas:.2f}s")
    print(f"  DuckDB:  {t_duckdb:.2f}s  (1 thread)")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"  Match:   {match}")

    return {
        "rows": rows,
        "pandas": t_pandas,
        "duckdb": t_duckdb,
        "speedup": speedup,
        "match": match,
    }


def bench_minmax(rows: int) -> dict:
    print(f"\n{'=' * 70}")
    print(f"  MIN/MAX: min/max(DS_1 group by Id_2)  |  {rows:,} rows, 100 groups")
    print(f"{'=' * 70}")

    df = generate_agg_data(rows)
    datapoints = {"DS_1": df}

    t0 = time.perf_counter()
    rp = run(script=MINMAX_SCRIPT, data_structures=MINMAX_STRUCTURES, datapoints=datapoints)
    t_pandas = time.perf_counter() - t0

    t0 = time.perf_counter()
    rd = run(
        script=MINMAX_SCRIPT,
        data_structures=MINMAX_STRUCTURES,
        datapoints=datapoints,
        use_duckdb=True,
    )
    t_duckdb = time.perf_counter() - t0

    match_min = (
        rp["DS_min"].data.sort_values("Id_2").reset_index(drop=True)["Me_1"].values
        == rd["DS_min"].data.sort_values("Id_2").reset_index(drop=True)["Me_1"].values
    ).all()
    match_max = (
        rp["DS_max"].data.sort_values("Id_2").reset_index(drop=True)["Me_1"].values
        == rd["DS_max"].data.sort_values("Id_2").reset_index(drop=True)["Me_1"].values
    ).all()

    speedup = t_pandas / t_duckdb if t_duckdb > 0 else float("inf")
    print(f"  Pandas:  {t_pandas:.2f}s")
    print(f"  DuckDB:  {t_duckdb:.2f}s  (1 thread)")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"  Match:   min={match_min}, max={match_max}")

    return {
        "rows": rows,
        "pandas": t_pandas,
        "duckdb": t_duckdb,
        "speedup": speedup,
        "match": match_min and match_max,
    }


if __name__ == "__main__":
    results = []

    for n in [1_000_000, 5_000_000]:
        results.append(("comparison", bench_comparison(n)))
        results.append(("minmax", bench_minmax(n)))

    print(f"\n{'=' * 70}")
    print("  SUMMARY (DuckDB with 1 thread)")
    print(f"{'=' * 70}")
    header = f"  {'Benchmark':<15} {'Rows':>10} {'Pandas':>10}"
    header += f" {'DuckDB':>10} {'Speedup':>10} {'Match':>7}"
    print(header)
    print(f"  {'-' * 62}")
    for name, r in results:
        print(
            f"  {name:<15} {r['rows']:>10,} {r['pandas']:>9.2f}s {r['duckdb']:>9.2f}s"
            f" {r['speedup']:>9.1f}x {str(r['match']):>7}"
        )
