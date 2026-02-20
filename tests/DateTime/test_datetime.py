import warnings

import pandas as pd
import pytest

from vtlengine import run
from vtlengine.API import create_ast
from vtlengine.DataTypes import Date, Integer
from vtlengine.DataTypes._time_checking import check_date
from vtlengine.DataTypes.TimeHandling import check_max_date
from vtlengine.Exceptions import InputValidationException, SemanticError
from vtlengine.Interpreter import InterpreterAnalyzer

check_date_valid_params = [
    pytest.param("2020-01-15", "2020-01-15", id="date_only"),
    pytest.param("2020-01-15T10:30:00", "2020-01-15 10:30:00", id="datetime_t_separator"),
    pytest.param("2020-01-15 10:30:00", "2020-01-15 10:30:00", id="datetime_space_separator"),
    pytest.param("2020-01-15T00:00:00", "2020-01-15 00:00:00", id="datetime_midnight"),
    pytest.param("2020-12-31T23:59:59", "2020-12-31 23:59:59", id="datetime_end_of_day"),
    pytest.param("2020-01-15T10:30", "2020-01-15 10:30:00", id="datetime_no_seconds_normalized"),
    pytest.param(
        "2020-01-15T10:30:00.123456",
        "2020-01-15 10:30:00.123456",
        id="datetime_microseconds",
    ),
    pytest.param(
        "2020-01-15 10:30:00.123456",
        "2020-01-15 10:30:00.123456",
        id="datetime_microseconds_space_separator",
    ),
    pytest.param(
        "2020-01-15T10:30:00.123456789",
        "2020-01-15 10:30:00.123456",
        id="datetime_nanoseconds_truncated",
    ),
]

check_date_invalid_params = [
    pytest.param("2020-01-15T25:00:00", InputValidationException, id="invalid_datetime_bad_hour"),
    pytest.param("1799-12-31", InputValidationException, id="invalid_year_below_range"),
]

check_max_date_valid_params = [
    pytest.param("2020-01-15", "2020-01-15", id="date_only"),
    pytest.param("2020-01-15T10:30:00", "2020-01-15 10:30:00", id="datetime_t_separator"),
    pytest.param("2020-01-15 10:30:00", "2020-01-15 10:30:00", id="datetime_space_separator"),
    pytest.param(
        "2020-01-15T10:30:00.123456",
        "2020-01-15 10:30:00.123456",
        id="datetime_microseconds",
    ),
    pytest.param(
        "2020-01-15T10:30:00.123456789",
        "2020-01-15 10:30:00.123456",
        id="datetime_nanoseconds_truncated",
    ),
]

check_max_date_invalid_params = [
    pytest.param("2020/01/15", SemanticError, id="invalid_format"),
]

scalar_time_params = [
    ('year(cast("2023-01-12T10:30:00", date))', 2023),
    ('year(cast("2023-01-12 10:30:00", date))', 2023),
    ('month(cast("2023-06-15T08:00:00", date))', 6),
    ('month(cast("2023-06-15 08:00:00", date))', 6),
    ('dayofmonth(cast("2023-01-12T15:45:00", date))', 12),
    ('dayofmonth(cast("2023-01-12 15:45:00", date))', 12),
    ('dayofyear(cast("2023-02-01T23:59:59", date))', 32),
    ('dayofyear(cast("2023-02-01 23:59:59", date))', 32),
]


datediff_params = [
    (
        'datediff(cast("2020-01-01T00:00:00", date), cast("2020-01-02T00:00:00", date))',
        1,
    ),
    (
        'datediff(cast("2020-01-01T10:00:00", date), cast("2020-01-01T23:59:59", date))',
        0,
    ),
    (
        'datediff(cast("2020-01-01", date), cast("2020-01-02T12:00:00", date))',
        1,
    ),
    (
        'datediff(cast("2020-01-01 10:00:00", date), cast("2020-01-01 23:59:59", date))',
        0,
    ),
]


dateadd_params = [
    ('dateadd(cast("2020-01-15T10:30:00", date), 1, "D")', "2020-01-16 10:30:00"),
    ('dateadd(cast("2020-01-15T10:30:00", date), 1, "M")', "2020-02-15 10:30:00"),
    ('dateadd(cast("2020-01-15T10:30:00", date), 1, "A")', "2021-01-15 10:30:00"),
    (
        'dateadd(cast("2020-01-15T10:30:00.123456", date), 1, "D")',
        "2020-01-16 10:30:00.123456",
    ),
    ('dateadd(cast("2020-01-15 10:30:00", date), 5, "D")', "2020-01-20 10:30:00"),
    ('dateadd(cast("2020-01-15 10:30:00", date), 3, "M")', "2020-04-15 10:30:00"),
]

dataload_params = [
    pytest.param(
        ["2020-01-15 10:30:00", "2020-06-01 00:00:00", "2020-12-31 23:59:59"],
        ["2020-01-15 10:30:00", "2020-06-01 00:00:00", "2020-12-31 23:59:59"],
        id="datetime_values",
    ),
    pytest.param(
        ["2020-01-15T10:30:00", "2020-06-01T00:00:00"],
        ["2020-01-15 10:30:00", "2020-06-01 00:00:00"],
        id="t_separator_normalized_to_space",
    ),
    pytest.param(
        ["2020-01-15 10:30:00", "2020-06-01 00:00:00.123456"],
        ["2020-01-15 10:30:00", "2020-06-01 00:00:00.123456"],
        id="space_separator_with_microseconds",
    ),
    pytest.param(
        ["2020-01-15", "2020-06-01 10:00:00"],
        ["2020-01-15", "2020-06-01 10:00:00"],
        id="mixed_date_and_datetime",
    ),
    pytest.param(
        ["2020-01-15T10:30:00.123456789"],
        ["2020-01-15 10:30:00.123456"],
        id="nanoseconds_truncated",
    ),
]

dataset_operator_params = [
    pytest.param(
        'DS_r <- DS_1[calc Me_1 := dateadd(Me_1, 1, "D")];',
        ["2020-01-15 10:30:00"],
        ["2020-01-16 10:30:00"],
        id="dateadd_day_preserves_time",
    ),
    pytest.param(
        'DS_r <- DS_1[calc Me_1 := dateadd(Me_1, 1, "M")];',
        ["2020-01-15 10:30:00"],
        ["2020-02-15 10:30:00"],
        id="dateadd_month_preserves_time",
    ),
    pytest.param(
        'DS_r <- DS_1[calc Me_1 := dateadd(Me_1, 1, "A")];',
        ["2020-01-15 10:30:00"],
        ["2021-01-15 10:30:00"],
        id="dateadd_year_preserves_time",
    ),
    pytest.param(
        'DS_r <- DS_1[calc Me_1 := dateadd(Me_1, 7, "D")];',
        ["2020-01-15 10:30:00.123456"],
        ["2020-01-22 10:30:00.123456"],
        id="dateadd_day_preserves_microseconds",
    ),
]


dataset_extraction_params = [
    pytest.param(
        "year",
        ["2023-01-12 10:30:00", "2024-06-15 08:00:00"],
        [2023, 2024],
        id="year_from_datetime",
    ),
    pytest.param(
        "month",
        ["2023-06-15 08:00:00", "2023-12-01 10:00:00"],
        [6, 12],
        id="month_from_datetime",
    ),
    pytest.param(
        "dayofmonth",
        ["2023-01-12 15:45:00", "2023-02-28 08:00:00"],
        [12, 28],
        id="dayofmonth_from_datetime",
    ),
    pytest.param(
        "dayofyear",
        ["2023-02-01 23:59:59", "2023-03-01 00:00:00"],
        [32, 60],
        id="dayofyear_from_datetime",
    ),
]


@pytest.mark.parametrize("input_value, expected", check_date_valid_params)
def test_check_date_valid(input_value, expected):
    assert check_date(input_value) == expected


@pytest.mark.parametrize("input_value, expected_exception", check_date_invalid_params)
def test_check_date_invalid(input_value, expected_exception):
    with pytest.raises(expected_exception):
        check_date(input_value)


@pytest.mark.parametrize("input_value, expected", check_max_date_valid_params)
def test_check_max_date_valid(input_value, expected):
    assert check_max_date(input_value) == expected


@pytest.mark.parametrize("input_value, expected_exception", check_max_date_invalid_params)
def test_check_max_date_invalid(input_value, expected_exception):
    with pytest.raises(expected_exception):
        check_max_date(input_value)


def test_check_max_date_none():
    assert check_max_date(None) is None


@pytest.mark.parametrize("text, reference", scalar_time_params)
def test_unary_time_scalar_datetime(text, reference):
    warnings.filterwarnings("ignore", category=FutureWarning)
    expression = f"DS_r := {text};"
    ast = create_ast(expression)
    interpreter = InterpreterAnalyzer({})
    result = interpreter.visit(ast)
    assert result["DS_r"].value == reference
    assert result["DS_r"].data_type == Integer


@pytest.mark.parametrize("text, reference", datediff_params)
def test_datediff_datetime(text, reference):
    warnings.filterwarnings("ignore", category=FutureWarning)
    expression = f"DS_r := {text};"
    ast = create_ast(expression)
    interpreter = InterpreterAnalyzer({})
    result = interpreter.visit(ast)
    assert result["DS_r"].value == reference
    assert result["DS_r"].data_type == Integer


@pytest.mark.parametrize("text, reference", dateadd_params)
def test_dateadd_datetime(text, reference):
    warnings.filterwarnings("ignore", category=FutureWarning)
    expression = f"DS_r := {text};"
    ast = create_ast(expression)
    interpreter = InterpreterAnalyzer({})
    result = interpreter.visit(ast)
    assert result["DS_r"].value == reference
    assert result["DS_r"].data_type == Date


_DS_1_STRUCTURE = {
    "datasets": [
        {
            "name": "DS_1",
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Date", "role": "Measure", "nullable": True},
            ],
        }
    ]
}


def _run_ds(script, input_values):
    data_df = pd.DataFrame({"Id_1": list(range(1, len(input_values) + 1)), "Me_1": input_values})
    result = run(script=script, data_structures=_DS_1_STRUCTURE, datapoints={"DS_1": data_df})
    return result["DS_r"].data["Me_1"].tolist()


# ---- Dataset-level dataload tests (parametrized) ----


@pytest.mark.parametrize("input_values, expected", dataload_params)
def test_dataset_dataload(input_values, expected):
    """Data loading normalizes datetime values: T→space, nanoseconds→microseconds."""
    result = _run_ds("DS_r <- DS_1;", input_values)
    assert result == expected


# ---- Dataset-level VTL operator tests (parametrized) ----


@pytest.mark.parametrize("script, input_values, expected", dataset_operator_params)
def test_dataset_operator(script, input_values, expected):
    """VTL operators preserve hours, minutes, seconds, and microseconds."""
    result = _run_ds(script, input_values)
    assert result == expected


# ---- Dataset-level extraction operator tests ----

_DS_1_INT_MEASURE = {
    "datasets": [
        {
            "name": "DS_1",
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Date", "role": "Measure", "nullable": True},
                {"name": "Me_2", "type": "Integer", "role": "Measure", "nullable": True},
            ],
        }
    ]
}


@pytest.mark.parametrize("op, input_values, expected", dataset_extraction_params)
def test_dataset_extraction_operator(op, input_values, expected):
    """Extraction operators (year, month, dayofmonth, dayofyear) work on datetime datasets."""
    script = f"DS_r <- DS_1[calc Me_2 := {op}(Me_1)];"
    data_df = pd.DataFrame(
        {
            "Id_1": list(range(1, len(input_values) + 1)),
            "Me_1": input_values,
            "Me_2": [0] * len(input_values),
        }
    )
    result = run(script=script, data_structures=_DS_1_INT_MEASURE, datapoints={"DS_1": data_df})
    assert result["DS_r"].data["Me_2"].tolist() == expected


# ---- Dataset-level datediff test ----


def test_dataset_datediff_with_datetime():
    """Datediff on datasets preserves date-only semantics (difference in days)."""
    script = "DS_r <- DS_1[calc Me_2 := datediff(Me_1, Me_2)];"
    data_structures = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                    {"name": "Me_1", "type": "Date", "role": "Measure", "nullable": True},
                    {"name": "Me_2", "type": "Date", "role": "Measure", "nullable": True},
                ],
            }
        ]
    }
    data_df = pd.DataFrame(
        {
            "Id_1": [1, 2],
            "Me_1": ["2020-01-01 00:00:00", "2020-06-15 12:00:00"],
            "Me_2": ["2020-01-10 23:59:59", "2020-06-15 23:59:59"],
        }
    )
    result = run(script=script, data_structures=data_structures, datapoints={"DS_1": data_df})
    assert result["DS_r"].data["Me_2"].tolist() == [9, 0]
