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

# ---- check_date tests ----


class TestCheckDate:
    def test_date_only(self):
        assert check_date("2020-01-15") == "2020-01-15"

    def test_datetime_t_separator(self):
        assert check_date("2020-01-15T10:30:00") == "2020-01-15T10:30:00"

    def test_datetime_space_separator(self):
        """Space separator input is accepted and normalized to T in output."""
        assert check_date("2020-01-15 10:30:00") == "2020-01-15T10:30:00"

    def test_datetime_midnight(self):
        assert check_date("2020-01-15T00:00:00") == "2020-01-15T00:00:00"

    def test_datetime_end_of_day(self):
        assert check_date("2020-12-31T23:59:59") == "2020-12-31T23:59:59"

    def test_datetime_no_seconds_normalized(self):
        """Partial time (HH:MM) is accepted and normalized to HH:MM:SS."""
        assert check_date("2020-01-15T10:30") == "2020-01-15T10:30:00"

    def test_datetime_microseconds(self):
        assert check_date("2020-01-15T10:30:00.123456") == "2020-01-15T10:30:00.123456"

    def test_datetime_microseconds_space_separator(self):
        assert check_date("2020-01-15 10:30:00.123456") == "2020-01-15T10:30:00.123456"

    def test_datetime_nanoseconds_truncated(self):
        """Nanosecond input is truncated to microsecond precision."""
        assert check_date("2020-01-15T10:30:00.123456789") == "2020-01-15T10:30:00.123456"

    def test_invalid_datetime_bad_hour(self):
        with pytest.raises(InputValidationException):
            check_date("2020-01-15T25:00:00")


# ---- check_max_date tests ----


class TestCheckMaxDate:
    def test_date_only(self):
        assert check_max_date("2020-01-15") == "2020-01-15"

    def test_datetime_t_separator(self):
        assert check_max_date("2020-01-15T10:30:00") == "2020-01-15T10:30:00"

    def test_datetime_space_separator(self):
        assert check_max_date("2020-01-15 10:30:00") == "2020-01-15T10:30:00"

    def test_datetime_microseconds(self):
        assert check_max_date("2020-01-15T10:30:00.123456") == "2020-01-15T10:30:00.123456"

    def test_datetime_nanoseconds_truncated(self):
        assert check_max_date("2020-01-15T10:30:00.123456789") == "2020-01-15T10:30:00.123456"

    def test_none(self):
        assert check_max_date(None) is None

    def test_invalid_format(self):
        with pytest.raises(SemanticError):
            check_max_date("2020/01/15")


# ---- Scalar operator tests with datetime ----


scalar_time_params = [
    ('year(cast("2023-01-12T10:30:00", date))', 2023),
    ('month(cast("2023-06-15T08:00:00", date))', 6),
    ('dayofmonth(cast("2023-01-12T15:45:00", date))', 12),
    ('dayofyear(cast("2023-02-01T23:59:59", date))', 32),
]


@pytest.mark.parametrize("text, reference", scalar_time_params)
def test_unary_time_scalar_datetime(text, reference):
    warnings.filterwarnings("ignore", category=FutureWarning)
    expression = f"DS_r := {text};"
    ast = create_ast(expression)
    interpreter = InterpreterAnalyzer({})
    result = interpreter.visit(ast)
    assert result["DS_r"].value == reference
    assert result["DS_r"].data_type == Integer


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
]


@pytest.mark.parametrize("text, reference", datediff_params)
def test_datediff_datetime(text, reference):
    warnings.filterwarnings("ignore", category=FutureWarning)
    expression = f"DS_r := {text};"
    ast = create_ast(expression)
    interpreter = InterpreterAnalyzer({})
    result = interpreter.visit(ast)
    assert result["DS_r"].value == reference
    assert result["DS_r"].data_type == Integer


dateadd_params = [
    ('dateadd(cast("2020-01-15T10:30:00", date), 1, "D")', "2020-01-16T10:30:00"),
    ('dateadd(cast("2020-01-15T10:30:00", date), 1, "M")', "2020-02-15T10:30:00"),
    ('dateadd(cast("2020-01-15T10:30:00", date), 1, "A")', "2021-01-15T10:30:00"),
    (
        'dateadd(cast("2020-01-15T10:30:00.123456", date), 1, "D")',
        "2020-01-16T10:30:00.123456",
    ),
]


@pytest.mark.parametrize("text, reference", dateadd_params)
def test_dateadd_datetime(text, reference):
    warnings.filterwarnings("ignore", category=FutureWarning)
    expression = f"DS_r := {text};"
    ast = create_ast(expression)
    interpreter = InterpreterAnalyzer({})
    result = interpreter.visit(ast)
    assert result["DS_r"].value == reference
    assert result["DS_r"].data_type == Date


# ---- Dataset-level test with run() API ----


def test_dataset_with_datetime_values():
    script = """DS_r <- DS_1;"""
    data_structures = {
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
    data_df = pd.DataFrame(
        {
            "Id_1": [1, 2, 3],
            "Me_1": ["2020-01-15T10:30:00", "2020-06-01T00:00:00", "2020-12-31T23:59:59"],
        }
    )
    datapoints = {"DS_1": data_df}
    result = run(script=script, data_structures=data_structures, datapoints=datapoints)
    assert result["DS_r"].data["Me_1"].tolist() == [
        "2020-01-15T10:30:00",
        "2020-06-01T00:00:00",
        "2020-12-31T23:59:59",
    ]


def test_dataset_space_separator_normalized():
    """Space separator input is normalized to T in output."""
    script = """DS_r <- DS_1;"""
    data_structures = {
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
    data_df = pd.DataFrame(
        {
            "Id_1": [1, 2],
            "Me_1": ["2020-01-15 10:30:00", "2020-06-01 00:00:00.123456"],
        }
    )
    datapoints = {"DS_1": data_df}
    result = run(script=script, data_structures=data_structures, datapoints=datapoints)
    assert result["DS_r"].data["Me_1"].tolist() == [
        "2020-01-15T10:30:00",
        "2020-06-01T00:00:00.123456",
    ]


def test_dataset_mixed_date_and_datetime():
    """Date-only and datetime values can coexist in the same column."""
    script = """DS_r <- DS_1;"""
    data_structures = {
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
    data_df = pd.DataFrame(
        {
            "Id_1": [1, 2],
            "Me_1": ["2020-01-15", "2020-06-01T10:00:00"],
        }
    )
    datapoints = {"DS_1": data_df}
    result = run(script=script, data_structures=data_structures, datapoints=datapoints)
    assert result["DS_r"].data["Me_1"].tolist() == [
        "2020-01-15",
        "2020-06-01T10:00:00",
    ]
