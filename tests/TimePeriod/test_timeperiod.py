import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest
from pytest import mark

from tests.Helper import _use_duckdb_backend, engine_params
from vtlengine.API import create_ast, run
from vtlengine.API._InternalApi import load_datasets_with_data
from vtlengine.DataTypes import Date, TimePeriod
from vtlengine.DataTypes.TimeHandling import (
    TimeIntervalHandler,
    TimePeriodHandler,
    from_input_customer_support_to_internal,
    generate_period_range,
    period_to_date,
)
from vtlengine.Exceptions import DataLoadError, SemanticError, VTLEngineException
from vtlengine.Exceptions import RunTimeError as RT
from vtlengine.Interpreter import InterpreterAnalyzer
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
    ("GH_1062_1", "DS_r := DS_1;"),
    # The period number at the edge of what each indicator holds, leap year included
    ("GH_1063_1", "DS_r := DS_1;"),
]

error_param = [
    ("GL_440_2", "DS_r := DS_1;", "0-3-1-6"),
    ("GH_1062_2", "DS_r := DS_1;", "0-3-1-6"),
    ("GH_1062_3", "DS_r := DS_1;", "0-3-1-6"),
    ("GH_1062_4", "DS_r := DS_1;", "0-3-1-6"),
    ("GH_1062_5", "DS_r := DS_1;", "0-3-1-6"),
    ("GH_1062_6", "DS_r := DS_1;", "0-3-1-6"),
    # A period number past what its indicator holds is no period at all. The DuckDb
    # loader checked the shape of a Time_Period and never its range (issue #1063)
    ("GH_1063_2", "DS_r := DS_1;", "0-3-1-6"),
    ("GH_1063_3", "DS_r := DS_1;", "0-3-1-6"),
    ("GH_1063_4", "DS_r := DS_1;", "0-3-1-6"),
    ("GH_1063_5", "DS_r := DS_1;", "0-3-1-6"),
    ("GH_1063_6", "DS_r := DS_1;", "0-3-1-6"),
    ("GH_1063_7", "DS_r := DS_1;", "0-3-1-6"),
    ("GH_1063_8", "DS_r := DS_1;", "0-3-1-6"),
    ("GH_1063_9", "DS_r := DS_1;", "0-3-1-6"),
]


@pytest.mark.parametrize("code, expression", ds_param)
def test_case_ds(request, load_input, load_reference, code, expression):
    warnings.filterwarnings("ignore", category=FutureWarning)
    if _use_duckdb_backend():
        base_path = request.node.get_closest_marker("input_path").args[0]
        import os

        ds_dir = base_path / "DataStructure" / "input"
        prefix = f"{code}-"
        data_structures = sorted(ds_dir / f for f in os.listdir(ds_dir) if f.startswith(prefix))

        datapoints = {}
        import json

        for ds_file in data_structures:
            with open(ds_file) as f:
                structure = json.load(f)
            if "datasets" in structure:
                ds_name = structure["datasets"][0]["name"]
                csv_path = (
                    base_path / "DataSet" / "input" / f"{code}-{ds_file.stem.split('-')[-1]}.csv"
                )
                if csv_path.exists():
                    datapoints[ds_name] = csv_path

        result = run(
            script=expression,
            data_structures=data_structures,
            datapoints=datapoints,
            return_only_persistent=False,
            use_duckdb=True,
        )
    else:
        ast = create_ast(expression)
        interpreter = InterpreterAnalyzer(datasets=load_input[0], scalars=load_input[1])
        result = interpreter.visit(ast)
    reference = {**load_reference[0], **load_reference[1]}
    assert result == reference


def _duckdb_load_error(request, code, expression):
    """The code the DuckDb engine reports while loading the inputs of a test code.

    The pandas engine reports it as the inputs are read, which is what ``load_error``
    catches; the DuckDb engine reads them through ``run``, so an error test has to
    reach the engine under test to say anything about it.
    """
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

    try:
        run(
            script=expression,
            data_structures=data_structures,
            datapoints=datapoints,
            return_only_persistent=False,
            use_duckdb=True,
        )
    except (SemanticError, DataLoadError) as ex:
        return ex.args[1] if len(ex.args) > 1 else ex.args[0]
    return None


@pytest.mark.parametrize("code, expression, error_code", error_param)
def test_errors(request, load_error, code, expression, error_code):
    warnings.filterwarnings("ignore", category=FutureWarning)
    reported = (
        _duckdb_load_error(request, code, expression) if _use_duckdb_backend() else load_error
    )
    result = error_code == reported
    if result is False:
        print(f"\n{error_code} != {reported}")
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
    ast = create_ast(script)
    datasets, scalars, _ = load_datasets_with_data(data_structures, datapoints=None)
    InterpreterAnalyzer(datasets=datasets, scalars=scalars, only_semantic=True).visit(ast)


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


out_of_range_periods = [
    "2024M13",  # a month past December
    "2024W54",  # a week past the last of the year
    "2023D366",  # the 366th day of a year that holds 365
    "1900D366",  # a century year is a leap year only every four hundred
]


@pytest.mark.parametrize("value", out_of_range_periods)
def test_out_of_range_time_period_is_rejected(tmp_path, value):
    """A Time_Period whose period number is out of range never reaches the output."""
    csv_path = tmp_path / "DS_1.csv"
    csv_path.write_text(f"Id_1,Me_1\n1,{value}\n")
    output_folder = tmp_path / "out"
    output_folder.mkdir()

    with pytest.raises((DataLoadError, RT)):
        run(
            script="DS_r <- DS_1;",
            data_structures={
                "datasets": [
                    {
                        "name": "DS_1",
                        "DataStructure": [
                            {
                                "name": "Id_1",
                                "type": "Integer",
                                "role": "Identifier",
                                "nullable": False,
                            },
                            {
                                "name": "Me_1",
                                "type": "Time_Period",
                                "role": "Measure",
                                "nullable": True,
                            },
                        ],
                    }
                ]
            },
            datapoints={"DS_1": csv_path},
            output_folder=output_folder,
            use_duckdb=_use_duckdb_backend(),
        )

    assert not (output_folder / "DS_r.csv").exists()


def _check_shift_out_of_range(use_duckdb: bool, period: str, shift: int, year: int) -> None:
    """Check that timeshift of a Data Point whose Identifier is period raises 2-1-19-10."""
    structures = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {
                        "name": "Id_1",
                        "type": "Time_Period",
                        "role": "Identifier",
                        "nullable": False,
                    },
                    {"name": "Me_1", "type": "Integer", "role": "Measure", "nullable": True},
                ],
            }
        ]
    }
    with pytest.raises(RT) as context:
        run(
            script=f"DS_r <- timeshift(DS_1, {shift});",
            data_structures=structures,
            datapoints={"DS_1": pd.DataFrame({"Id_1": [period], "Me_1": [1]})},
            use_duckdb=use_duckdb,
        )
    assert context.value.args[1] == "2-1-19-10"
    assert context.value.args[0].startswith(f"Invalid year {year}, must be between 1000 and 9999.")


@pytest.mark.parametrize("use_duckdb", engine_params())
@pytest.mark.parametrize(
    "period, shift, year",
    [
        ("1000A", -1, 999),
        ("1000S1", -1, 999),
        ("1000Q1", -1, 999),
        ("1000M1", -1, 999),
        ("1000W1", -1, 999),
        ("1000D1", -1, 999),
        ("2020A", -2100, -80),
        ("2020S1", -4100, -30),
        ("2020Q1", -8100, -5),
        ("2020M1", -24300, -5),
        ("2020W1", -105650, -5),
        ("2020D1", -740000, -7),
        ("2020A", 300000000, 300002020),
        ("2020M1", -400000000, -33331314),
        ("2020W1", 400000000, 7668159),
        ("2020D1", 2147483648, 5881630),
    ],
)
def test_GH_1352_36j(use_duckdb: bool, period: str, shift: int, year: int) -> None:
    """Extra of case 36: timeshift of a Time_Period back to a year below 1000, also below 0,
    gives a year no Time_Period has, so it raises 2-1-19-10, checked for each period indicator
    on its own. The DuckDB engine wrote a null Identifier for S and Q and raised a raw
    ValueError for A, and the pandas engine raised a raw ValueError below the year 0 for S, Q, M,
    W and D. A shift of hundreds of millions of periods gives the same error and year: the DuckDB
    engine raised a raw OutOfRange, Overflow or Conversion exception (issue #1352)."""
    _check_shift_out_of_range(use_duckdb, period, shift, year)


@pytest.mark.parametrize("use_duckdb", engine_params())
@pytest.mark.parametrize("period", ["9999A", "9999S2", "9999Q4", "9999M12", "9999W52", "9999D365"])
def test_GH_1352_36k(use_duckdb: bool, period: str) -> None:
    """Extra of case 36: timeshift of the last Time_Period of the year 9999 forward by one
    period gives the year 10000, which no Time_Period has, so it raises 2-1-19-10, checked for
    each period indicator on its own. The DuckDB engine wrote a null Identifier (issue #1352)."""
    _check_shift_out_of_range(use_duckdb, period, 1, 10000)


def _check_dateadd_out_of_range(
    use_duckdb: bool, script: str, value: str, year: int, data_type: str = "Date"
) -> None:
    """Check that dateadd of the Measure Me_1 holding value raises 2-1-19-10 for the year."""
    structures = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                    {"name": "Me_1", "type": data_type, "role": "Measure", "nullable": True},
                ],
            }
        ]
    }
    with pytest.raises(RT) as context:
        run(
            script=script,
            data_structures=structures,
            datapoints={"DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [value]})},
            use_duckdb=use_duckdb,
        )
    assert context.value.args[1] == "2-1-19-10"
    assert context.value.args[0].startswith(f"Invalid year {year}, must be between 1000 and 9999.")


@pytest.mark.parametrize("use_duckdb", engine_params())
@pytest.mark.parametrize("period", ["A", "S", "Q", "M", "W", "D"])
def test_GH_1352_26b(use_duckdb: bool, period: str) -> None:
    """Extra of case 26: dateadd of the Date 9999-12-31 forward by one period gives the year
    10000, which no Date has, so it raises 2-1-19-10, checked for each period indicator on its
    own. The pandas engine raised a raw ValueError or OverflowError and the DuckDB engine wrote
    a Date of the year 10000 (issue #1352)."""
    script = f'DS_r <- DS_1 [ calc Me_2 := dateadd ( Me_1 , 1 , "{period}" ) ];'
    _check_dateadd_out_of_range(use_duckdb, script, "9999-12-31", 10000)


@pytest.mark.parametrize("use_duckdb", engine_params())
def test_GH_1352_26d(use_duckdb: bool) -> None:
    """Extra of case 26: dateadd of the Date 2020-01-31 forward by 300 million years gives the
    year 300002020, which no Date has, so it raises 2-1-19-10, as for one year past 9999. The
    DuckDB engine raised a raw OutOfRangeException (issue #1352)."""
    script = 'DS_r <- DS_1 [ calc Me_2 := dateadd ( Me_1 , 300000000 , "A" ) ];'
    _check_dateadd_out_of_range(use_duckdb, script, "2020-01-31", 300002020)


def _date_structures(scalar: bool = False) -> Dict[str, Any]:
    """DS_1 with an Integer Identifier and a String Measure Me_1, and a Date Scalar SC_1."""
    structures: Dict[str, Any] = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                    {"name": "Me_1", "type": "String", "role": "Measure", "nullable": True},
                ],
            }
        ]
    }
    if scalar:
        structures["scalars"] = [{"name": "SC_1", "type": "Date"}]
    return structures


@pytest.mark.parametrize("use_duckdb", engine_params())
@pytest.mark.parametrize(
    "path, script, me_type, scalar, code",
    [
        pytest.param("load", "DS_r <- DS_1;", "Date", False, "0-3-1-6", id="load"),
        pytest.param(
            "literal cast",
            'DS_r <- DS_1 [ calc Me_2 := cast ( "0999-12-31" , date ) ];',
            "String",
            False,
            "1-1-5-6",
            id="literal_cast",
        ),
        pytest.param(
            "data cast",
            "DS_r <- DS_1 [ calc Me_2 := cast ( Me_1 , date ) ];",
            "String",
            False,
            "2-1-19-8",
            id="data_cast",
        ),
        pytest.param(
            "scalar value",
            "DS_r <- DS_1 [ calc Me_2 := SC_1 ];",
            "String",
            True,
            "0-1-2-7",
            id="scalar_value",
        ),
        pytest.param(
            "result",
            'DS_r <- DS_1 [ calc Me_2 := dateadd ( Me_1 , -1 , "D" ) ];',
            "Date",
            False,
            "2-1-19-10",
            id="result",
        ),
    ],
)
def test_GH_1352_26g(
    use_duckdb: bool, path: str, script: str, me_type: str, scalar: bool, code: str
) -> None:
    """Extra of case 26: a Date has the years 1000 to 9999, so the year 999 raises the error of
    each path: 0999-12-31 loaded, cast from a literal or from data, given as a Scalar value,
    and the Date that dateadd gives one day before 1000-01-01 (issue #1352)."""
    structures = _date_structures(scalar)
    structures["datasets"][0]["DataStructure"][1]["type"] = me_type
    value = "1000-01-01" if path == "result" else "0999-12-31"
    with pytest.raises(VTLEngineException) as context:
        run(
            script=script,
            data_structures=structures,
            datapoints={"DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [value]})},
            scalar_values={"SC_1": "0999-12-31"} if scalar else None,
            use_duckdb=use_duckdb,
        )
    assert context.value.args[1] == code


@pytest.mark.parametrize("use_duckdb", engine_params())
@pytest.mark.parametrize(
    "value, period, year",
    [
        ("1000-01-01", "A", 999),
        ("1000-01-01", "S", 999),
        ("1000-01-01", "Q", 999),
        ("1000-01-01", "M", 999),
        ("1000-01-01", "W", 999),
        ("1000-01-01", "D", 999),
        ("1000Q1", "A", 999),
    ],
)
def test_GH_1352_26c(use_duckdb: bool, value: str, period: str, year: int) -> None:
    """Extra of case 26: dateadd back by one period of the Date 1000-01-01, or of the
    Time_Period 1000Q1, gives a year below 1000, which no Date has, so it raises 2-1-19-10,
    checked for each period indicator on its own. Both engines wrote the Date, 999-12-31 on the
    pandas engine and 0999-12-31 on the DuckDB engine (issue #1352)."""
    data_type = "Time_Period" if "Q" in value else "Date"
    script = f'DS_r <- DS_1 [ calc Me_2 := dateadd ( Me_1 , -1 , "{period}" ) ];'
    _check_dateadd_out_of_range(use_duckdb, script, value, year, data_type)
