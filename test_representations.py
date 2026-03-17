"""Test that all four TimePeriod output representations produce matching results
between Pandas and DuckDB engines."""

import pandas as pd

from vtlengine import run

SCRIPT = """
    DS_r <- DS_1;
"""

DATA_STRUCTURES = {
    "datasets": [
        {
            "name": "DS_1",
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Time_Period", "role": "Measure", "nullable": False},
            ],
        }
    ]
}

# Test data with various period indicators
DF = pd.DataFrame(
    {
        "Id_1": list(range(1, 9)),
        "Me_1": [
            "2020A",  # Annual
            "2020S1",  # Semester
            "2020Q3",  # Quarter
            "2020M06",  # Month
            "2020M1",  # Month (no padding)
            "2020W15",  # Week
            "2020D100",  # Day
            "2020D1",  # Day (no padding)
        ],
    }
)


def run_both(representation: str) -> tuple:
    """Run with both engines and return (pandas_df, duckdb_df)."""
    result_pandas = run(
        script=SCRIPT,
        data_structures=DATA_STRUCTURES,
        datapoints={"DS_1": DF.copy()},
        time_period_output_format=representation,
    )
    result_duckdb = run(
        script=SCRIPT,
        data_structures=DATA_STRUCTURES,
        datapoints={"DS_1": DF.copy()},
        use_duckdb=True,
        time_period_output_format=representation,
    )
    df_p = result_pandas["DS_r"].data.sort_values("Id_1").reset_index(drop=True)
    df_d = result_duckdb["DS_r"].data.sort_values("Id_1").reset_index(drop=True)
    return df_p, df_d


def test_representation(name: str) -> bool:
    print(f"\n{'=' * 60}")
    print(f"Representation: {name}")
    print(f"{'=' * 60}")

    df_p, df_d = run_both(name)

    print(f"\nPandas:\n{df_p.to_string(index=False)}")
    print(f"\nDuckDB:\n{df_d.to_string(index=False)}")

    match = (df_p["Me_1"].values == df_d["Me_1"].values).all()
    print(f"\nMatch: {match}")

    if not match:
        for i in range(len(df_p)):
            p, d = df_p["Me_1"].iloc[i], df_d["Me_1"].iloc[i]
            status = "OK" if p == d else "MISMATCH"
            print(f"  [{status}] pandas={p!r:20s} duckdb={d!r:20s}")

    return match


def test_sdmx_gregorian_amd_only() -> bool:
    """SDMX Gregorian only supports A, M, D — test with those only."""
    print(f"\n{'=' * 60}")
    print("Representation: sdmx_gregorian (A/M/D only)")
    print(f"{'=' * 60}")

    df_amd = pd.DataFrame(
        {
            "Id_1": [1, 2, 3, 4],
            "Me_1": ["2020A", "2020M06", "2020M1", "2020D100"],
        }
    )

    result_pandas = run(
        script=SCRIPT,
        data_structures=DATA_STRUCTURES,
        datapoints={"DS_1": df_amd.copy()},
        time_period_output_format="sdmx_gregorian",
    )
    result_duckdb = run(
        script=SCRIPT,
        data_structures=DATA_STRUCTURES,
        datapoints={"DS_1": df_amd.copy()},
        use_duckdb=True,
        time_period_output_format="sdmx_gregorian",
    )
    df_p = result_pandas["DS_r"].data.sort_values("Id_1").reset_index(drop=True)
    df_d = result_duckdb["DS_r"].data.sort_values("Id_1").reset_index(drop=True)

    print(f"\nPandas:\n{df_p.to_string(index=False)}")
    print(f"\nDuckDB:\n{df_d.to_string(index=False)}")

    match = (df_p["Me_1"].values == df_d["Me_1"].values).all()
    print(f"\nMatch: {match}")
    if not match:
        for i in range(len(df_p)):
            p, d = df_p["Me_1"].iloc[i], df_d["Me_1"].iloc[i]
            status = "OK" if p == d else "MISMATCH"
            print(f"  [{status}] pandas={p!r:20s} duckdb={d!r:20s}")
    return match


if __name__ == "__main__":
    results = {}
    for repr_name in ["vtl", "sdmx_reporting", "natural"]:
        try:
            results[repr_name] = test_representation(repr_name)
        except Exception as e:
            print(f"\n  ERROR: {e}")
            results[repr_name] = False

    results["sdmx_gregorian"] = test_sdmx_gregorian_amd_only()

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:20s} {status}")
