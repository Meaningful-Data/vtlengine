"""
Integration tests verifying that TimePeriod output representations produce
matching results between Pandas and DuckDB engines via the run() API.
"""

from typing import List

import pandas as pd
import pytest

from tests.Helper import engine_params
from vtlengine import run
from vtlengine.Exceptions import InputValidationException, RunTimeError

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


def test_invalid_time_period_output_format() -> None:
    with pytest.raises(InputValidationException) as ctx:
        run(
            script=SCRIPT,
            data_structures=DATA_STRUCTURES,
            datapoints={"DS_1": AMD_ONLY_DF.copy()},
            time_period_output_format="not_a_valid_format",
        )
    assert ctx.value.args[1] == "0-1-1-15"


@pytest.mark.parametrize("use_duckdb", engine_params())
@pytest.mark.parametrize(
    "representation, expected",
    [
        ("vtl", ["2020", "2020S2", "2020Q1", "2020M2", "2020W1", "2020D15"]),
        ("sdmx_reporting", ["2020-A1", "2020-S2", "2020-Q1", "2020-M02", "2020-W01", "2020-D015"]),
        ("natural", ["2020", "2020-S2", "2020-Q1", "2020-02", "2020-W01", "2020-01-15"]),
    ],
)
def test_GH_1352_13f(use_duckdb: bool, representation: str, expected: List[str]) -> None:
    """Extra of case 13: a Time cast to time_period is written as a loaded Time_Period is, in
    every output format, on both engines. The DuckDB engine held a quarter cast from a Time as
    2020-Q1.0, so it raised a raw ValueError writing it in sdmx_reporting (issue #1352)."""
    structures = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                    {"name": "Me_1", "type": "Time", "role": "Measure", "nullable": True},
                ],
            }
        ]
    }
    data = pd.DataFrame(
        {
            "Id_1": list(range(1, 7)),
            "Me_1": [
                "2020-01-01/2020-12-31",
                "2020-07-01/2020-12-31",
                "2020-01-01/2020-03-31",
                "2020-02-01/2020-02-29",
                "2019-12-30/2020-01-05",
                "2020-01-15/2020-01-15",
            ],
        }
    )
    result = run(
        script="DS_r <- DS_1 [ calc Me_2 := cast ( Me_1 , time_period ) ];",
        data_structures=structures,
        datapoints={"DS_1": data},
        use_duckdb=use_duckdb,
        time_period_output_format=representation,
    )
    assert result["DS_r"].data.sort_values("Id_1")["Me_2"].tolist() == expected


@pytest.mark.parametrize("use_duckdb", engine_params())
@pytest.mark.parametrize(
    "script", ["DS_r <- DS_1;", "DS_r <- DS_1 [ calc Me_2 := cast ( Me_1 , string ) ];"]
)
@pytest.mark.parametrize("period", ["2020S1", "2020Q3", "2020W15"])
def test_GH_1352_13g(use_duckdb: bool, script: str, period: str) -> None:
    """Extra of case 13: the sdmx_gregorian format writes only a year, a month or a day, so a
    semester, a quarter or a week raises 2-1-19-21 on both engines, in a written Time_Period
    and in a cast to string. The DuckDB engine raised a raw InvalidInputException (issue
    #1352)."""
    with pytest.raises(RunTimeError) as context:
        run(
            script=script,
            data_structures=DATA_STRUCTURES,
            datapoints={"DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [period]})},
            use_duckdb=use_duckdb,
            time_period_output_format="sdmx_gregorian",
        )
    assert context.value.args[1] == "2-1-19-21"
    assert context.value.args[0].startswith(
        f"SDMX Gregorian representation is not supported for period indicator '{period[4]}'."
    )


@pytest.mark.parametrize("use_duckdb", engine_params())
@pytest.mark.parametrize("written", [False, True], ids=["in_memory", "output_folder"])
def test_GH_1356_39(use_duckdb: bool, written: bool, tmp_path) -> None:
    """The sdmx_gregorian format cannot write the quarter 2020Q1: both engines raise 2-1-19-21,
    in memory and with an output_folder. The DuckDB engine raised a raw InvalidInputException
    (issue #1356)."""
    with pytest.raises(RunTimeError) as context:
        run(
            script="DS_r <- DS_1;",
            data_structures=DATA_STRUCTURES,
            datapoints={"DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": ["2020Q1", "2020A"]})},
            use_duckdb=use_duckdb,
            time_period_output_format="sdmx_gregorian",
            output_folder=tmp_path if written else None,
        )
    assert context.value.args[1] == "2-1-19-21"
