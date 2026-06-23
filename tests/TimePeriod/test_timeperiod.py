import json
import os
import warnings
from pathlib import Path

import pytest
from pytest import mark

from vtlengine.API import run, semantic_analysis
from vtlengine.DataTypes import Date, TimePeriod
from vtlengine.DataTypes.TimeHandling import (
    TimeIntervalHandler,
    TimePeriodHandler,
    from_input_customer_support_to_internal,
    generate_period_range,
    period_to_date,
)
from vtlengine.Exceptions import RunTimeError as RT
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import Component, Dataset, Role
from vtlengine.Operators.Time import Time, Year_to_Day

pytestmark = mark.input_path(Path(__file__).parent / "data")

ds_param = [
    ("1", 'DS_r := DSD_EXR[filter TIME_PERIOD = cast("2002M1", time_period)];'),
    ("2", 'DS_r := DSD_EXR[filter TIME_PERIOD <> cast("2002Q1", time_period)];'),
    # ("3", 'DS_r := DSD_EXR[filter TIME_PERIOD < cast("2002S2", time_period)];'),
    ("4", 'DS_r := DSD_EXR[filter TIME_PERIOD > cast("2002M1", time_period)];'),
    # ("5", 'DS_r := DSD_EXR[filter TIME_PERIOD <= cast("2002Q3", time_period)];'),
    # ("6", 'DS_r := DSD_EXR[filter TIME_PERIOD >= cast("2002W26", time_period)];'),
    (
        "GL_416",
        'test2_1 := BE2_DF_NICP[filter FREQ = "M" and TIME_PERIOD = cast("2020-01", time_period)];',
    ),
    ("GL_417_1", 'test := avg (BE2_DF_NICP group all time_agg ("Q"));'),
    ("GL_417_2", 'test := avg (BE2_DF_NICP group all time_agg ("A"));'),
    ("GL_417_4", 'test := avg (BE2_DF_NICP group all time_agg ("A"));'),
    (
        "GL_418",
        'test2_1 := BE2_DF_NICP[sub DERIVATION = "INDICES"][filter FREQ = "M"][keep OBS_VALUE]; \
                test2_2 := timeshift(test2_1,-12); \
                test2_result <- inner_join(test2_1[rename OBS_VALUE to CURRENT] as C, test2_2 \
                    [rename OBS_VALUE to PREVIOUS] as P calc GROWTH :=(CURRENT - PREVIOUS) / PREVIOUS * 100, \
                    identifier DERIVATION := "GROWTH_RATE");',
    ),
    (
        "GL_421_1",
        'test2_1 := BE2_DF_NICP[calc FREQ_2 := TIME_PERIOD in {cast("2020-01", time_period), cast("2021-01", time_period)}];',
    ),
    # ("GL_421_2", 'test := avg (BE2_DF_NICP group all time_agg ("A"));'),
    ("GL_440_1", "DS_r := DS_1;"),
    ("GL_462_1", "added := demo_data_structure;"),
    ("GL_462_2", "added := demo_data_structure; DS_r := added+ ds_2;"),
    ("GL_462_3", "sc_result := sc_1;"),
    ("GL_462_4", "DS_r := ds_2;"),
    (
        "GL_563_1",
        """ds_with_year  := DSD_AN_HOUSE_PRICES[calc identifier year_id  := cast(time_agg("A", TIME_PERIOD), string)]; ds_with_year2 := ds_with_year[calc identifier year_id2 := cast(time_agg("A", TIME_PERIOD), string)];""",
    ),
    (
        "GH_487",
        """
        DS_r <- timeshift(DS_1, 12);
        """,
    ),
]

error_param = [
    ("GL_440_2", "DS_r := DS_1;", "0-3-1-6"),
]


@pytest.mark.parametrize("code, expression", ds_param)
def test_case_ds(request, load_reference, code, expression):
    warnings.filterwarnings("ignore", category=FutureWarning)
    base_path = request.node.get_closest_marker("input_path").args[0]

    ds_dir = base_path / "DataStructure" / "input"
    prefix = f"{code}-"
    data_structures = sorted(ds_dir / f for f in os.listdir(ds_dir) if f.startswith(prefix))

    datapoints = {}
    for ds_file in data_structures:
        with open(ds_file) as f:
            structure = json.load(f)
        if "datasets" in structure:
            ds_name = structure["datasets"][0]["name"]
            csv_path = base_path / "DataSet" / "input" / f"{code}-{ds_file.stem.split('-')[-1]}.csv"
            if csv_path.exists():
                datapoints[ds_name] = csv_path

    result = run(
        script=expression,
        data_structures=data_structures,
        datapoints=datapoints,
        return_only_persistent=False,
    )
    reference = {**load_reference[0], **load_reference[1]}
    assert result == reference


@pytest.mark.parametrize("code, expression, error_code", error_param)
def test_errors(load_error, code, expression, error_code):
    warnings.filterwarnings("ignore", category=FutureWarning)
    result = error_code == load_error
    if result is False:
        print(f"\n{error_code} != {load_error}")
    assert result


def test_get_time_id_error_len_identifiers():
    dataset = Dataset(name="test_dataset", components={}, data=None)
    with pytest.raises(SemanticError, match="1-1-19-8"):
        Time._get_time_id(dataset)


def test_get_time_id_error_reference_id():
    components = {
        "Id_1": Component(name="Id_1", data_type=Date, role=Role.IDENTIFIER, nullable=False),
        "Id_2": Component(name="Id_2", data_type=TimePeriod, role=Role.IDENTIFIER, nullable=False),
    }
    dataset = Dataset(name="test_dataset", components=components, data=None)

    with pytest.raises(SemanticError, match="1-1-19-8"):
        Time._get_time_id(dataset)


def _run_semantic(script: str, data_structures: dict) -> None:
    semantic_analysis(script=script, data_structures=data_structures)


def test_GH_676_1():
    """time_agg with an invalid duration indicator triggers 1-1-19-3."""
    script = 'DS_r := time_agg("X", DS_1);'
    structures = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {
                        "name": "TIME_PERIOD",
                        "type": "Time_Period",
                        "role": "Identifier",
                        "nullable": False,
                    },
                    {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                ],
            }
        ]
    }
    with pytest.raises(SemanticError) as ctx:
        _run_semantic(script, structures)
    assert ctx.value.args[1] == "1-1-19-3"


def test_GH_676_2():
    """time_agg with period_to <= period_from triggers 1-1-19-4."""
    script = 'DS_r := time_agg("M", "A", DS_1);'
    structures = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {
                        "name": "TIME_PERIOD",
                        "type": "Time_Period",
                        "role": "Identifier",
                        "nullable": False,
                    },
                    {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                ],
            }
        ]
    }
    with pytest.raises(SemanticError) as ctx:
        _run_semantic(script, structures)
    assert ctx.value.args[1] == "1-1-19-4"


def test_GH_676_3():
    """period_to_date with an unknown period indicator triggers runtime 2-1-19-2."""
    with pytest.raises(RT) as ctx:
        period_to_date(2024, "X", 1)
    assert ctx.value.args[1] == "2-1-19-2"


def test_GH_676_4():
    """generate_period_range with mismatched period indicators triggers 2-1-19-3."""
    start = TimePeriodHandler("2020A")
    end = TimePeriodHandler("2020M01")
    with pytest.raises(RT) as ctx:
        generate_period_range(start, end)
    assert ctx.value.args[1] == "2-1-19-3"


def test_GH_676_5():
    """Period string with a too-long second term triggers 2-1-19-6."""
    with pytest.raises(RT) as ctx:
        from_input_customer_support_to_internal("2020-XYZWX")
    assert ctx.value.args[1] == "2-1-19-6"


def test_GH_676_6():
    """A monthly period number outside [1, 12] triggers 2-1-19-7."""
    with pytest.raises(RT) as ctx:
        TimePeriodHandler("2020M13")
    assert ctx.value.args[1] == "2-1-19-7"


def test_GH_676_7():
    """Year out of [0, 9999] triggers 2-1-19-10."""
    handler = TimePeriodHandler("2020A")
    with pytest.raises(RT) as ctx:
        handler.year = 10000
    assert ctx.value.args[1] == "2-1-19-10"


def test_GH_676_8():
    """A daily period number > 365 in a non-leap year triggers 2-1-19-9."""
    # 2021 is a non-leap year; D366 is past the 365-day range.
    with pytest.raises(RT) as ctx:
        TimePeriodHandler("2021D366")
    assert ctx.value.args[1] == "2-1-19-9"


def test_GH_676_9():
    """set_date1 with a value greater than date2 triggers 2-1-19-4."""
    interval = TimeIntervalHandler("2020-01-01", "2020-12-31")
    with pytest.raises(RT) as ctx:
        interval.set_date1("2021-06-01")
    assert ctx.value.args[1] == "2-1-19-4"


def test_GH_676_10():
    """set_date2 with a value lower than date1 triggers 2-1-19-5."""
    interval = TimeIntervalHandler("2020-01-01", "2020-12-31")
    with pytest.raises(RT) as ctx:
        interval.set_date2("2019-06-01")
    assert ctx.value.args[1] == "2-1-19-5"


def test_GH_676_11():
    """year_to_day with a malformed duration string triggers 2-1-19-22."""
    with pytest.raises(RT) as ctx:
        Year_to_Day.py_op("not-a-duration")
    assert ctx.value.args[1] == "2-1-19-22"
