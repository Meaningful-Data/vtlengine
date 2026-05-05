"""
Integration tests verifying that TimePeriod output representations produce
matching results between Pandas and DuckDB engines via the run() API.
"""

import pandas as pd
import pytest

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
                {"name": "Me_1", "type": "Time_Period", "role": "Measure", "nullable": True},
            ],
        }
    ]
}

ALL_PERIODS_DF = pd.DataFrame(
    {
        "Id_1": list(range(1, 9)),
        "Me_1": [
            "2020A",
            "2020S1",
            "2020Q3",
            "2020M06",
            "2020M1",
            "2020W15",
            "2020D100",
            "2020D1",
        ],
    }
)

# SDMX Gregorian only supports A, M, D indicators
AMD_ONLY_DF = pd.DataFrame(
    {
        "Id_1": [1, 2, 3, 4],
        "Me_1": ["2020A", "2020M06", "2020M1", "2020D100"],
    }
)


def _run_and_compare(datapoints: pd.DataFrame, representation: str) -> None:
    """Run with both engines and assert Me_1 values match."""
    result_pandas = run(
        script=SCRIPT,
        data_structures=DATA_STRUCTURES,
        datapoints={"DS_1": datapoints.copy()},
        time_period_output_format=representation,
    )
    result_duckdb = run(
        script=SCRIPT,
        data_structures=DATA_STRUCTURES,
        datapoints={"DS_1": datapoints.copy()},
        use_duckdb=True,
        time_period_output_format=representation,
    )
    df_p = result_pandas["DS_r"].data.sort_values("Id_1").reset_index(drop=True)
    df_d = result_duckdb["DS_r"].data.sort_values("Id_1").reset_index(drop=True)

    pd.testing.assert_series_equal(
        df_p["Me_1"],
        df_d["Me_1"],
        check_names=True,
        check_dtype=False,
        obj=f"{representation} Me_1",
    )


@pytest.mark.parametrize("representation", ["vtl", "sdmx_reporting", "natural"])
def test_representation_pandas_duckdb_match(representation: str) -> None:
    _run_and_compare(ALL_PERIODS_DF, representation)


def test_sdmx_gregorian_pandas_duckdb_match() -> None:
    _run_and_compare(AMD_ONLY_DF, "sdmx_gregorian")
