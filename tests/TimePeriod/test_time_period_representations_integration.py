"""
Integration tests verifying that TimePeriod output representations produce
correct results via the run() API.
"""

import pandas as pd
import pytest

from vtlengine import run
from vtlengine.Exceptions import InputValidationException

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


def _run_and_check(datapoints: pd.DataFrame, representation: str) -> None:
    """Run and assert the result has the expected Me_1 column."""
    result = run(
        script=SCRIPT,
        data_structures=DATA_STRUCTURES,
        datapoints={"DS_1": datapoints.copy()},
        time_period_output_format=representation,
    )
    assert "DS_r" in result
    assert result["DS_r"].data is not None
    assert "Me_1" in result["DS_r"].data.columns


@pytest.mark.parametrize("representation", ["vtl", "sdmx_reporting", "natural"])
def test_representation_pandas_duckdb_match(representation: str) -> None:
    _run_and_check(ALL_PERIODS_DF, representation)


def test_sdmx_gregorian_pandas_duckdb_match() -> None:
    _run_and_check(AMD_ONLY_DF, "sdmx_gregorian")


def test_invalid_time_period_output_format() -> None:
    with pytest.raises(InputValidationException) as ctx:
        run(
            script=SCRIPT,
            data_structures=DATA_STRUCTURES,
            datapoints={"DS_1": AMD_ONLY_DF.copy()},
            time_period_output_format="not_a_valid_format",
        )
    assert ctx.value.args[1] == "0-1-1-15"
