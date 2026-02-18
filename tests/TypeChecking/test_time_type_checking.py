"""Tests for implicit type promotion between Date, TimePeriod, and TimeInterval.

VTL 2.2 specifies these implicit casts for time types:
  - Date → TimeInterval
  - TimePeriod → TimeInterval

This means comparing a Date measure against a TimePeriod measure should
succeed by promoting both operands to TimeInterval.
"""

import pandas as pd
import pytest

from vtlengine import run
from vtlengine.DataTypes import (
    Boolean,
    Date,
    TimeInterval,
    TimePeriod,
    binary_implicit_promotion,
    check_binary_implicit_promotion,
)


class TestDateTimePeriodImplicitPromotion:
    """Date vs TimePeriod should promote to TimeInterval for comparison."""

    @pytest.mark.parametrize(
        "left, right, return_type, expected",
        [
            (Date, TimePeriod, Boolean, Boolean),
            (TimePeriod, Date, Boolean, Boolean),
            (Date, TimePeriod, None, TimeInterval),
            (TimePeriod, Date, None, TimeInterval),
        ],
        ids=[
            "Date_vs_TimePeriod_returns_Boolean",
            "TimePeriod_vs_Date_returns_Boolean",
            "Date_vs_TimePeriod_returns_TimeInterval",
            "TimePeriod_vs_Date_returns_TimeInterval",
        ],
    )
    def test_binary_implicit_promotion(self, left, right, return_type, expected):
        result = binary_implicit_promotion(left, right, return_type=return_type)
        assert result == expected

    @pytest.mark.parametrize(
        "left, right",
        [(Date, TimePeriod), (TimePeriod, Date)],
        ids=["Date_vs_TimePeriod", "TimePeriod_vs_Date"],
    )
    def test_check_binary_implicit_promotion(self, left, right):
        assert check_binary_implicit_promotion(left, right) is True


DATA_STRUCTURES = {
    "datasets": [
        {
            "name": "DS_date",
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Date", "role": "Measure", "nullable": True},
            ],
        },
        {
            "name": "DS_period",
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Time_Period", "role": "Measure", "nullable": True},
            ],
        },
    ]
}


class TestDateTimePeriodComparison:
    """End-to-end: comparing Date and TimePeriod datasets should work."""

    @pytest.mark.parametrize(
        "script, date_vals, period_vals, expected",
        [
            (
                "DS_r <- DS_date = DS_period;",
                ["2020-01-01"],
                ["2020D1"],
                [True],
            ),
            (
                "DS_r <- DS_date = DS_period;",
                ["2020-01-15"],
                ["2020Q1"],
                [False],
            ),
            (
                "DS_r <- DS_date <> DS_period;",
                ["2020-06-15", "2020-01-01"],
                ["2020Q1", "2020Q1"],
                [True, True],
            ),
            (
                "DS_r <- DS_period <> DS_date;",
                ["2020-06-15"],
                ["2020Q1"],
                [True],
            ),
        ],
        ids=[
            "date_eq_daily_period_true",
            "date_eq_quarterly_period_false",
            "date_neq_period",
            "period_neq_date",
        ],
    )
    def test_comparison(self, script, date_vals, period_vals, expected):
        ids = list(range(1, len(date_vals) + 1))
        datapoints = {
            "DS_date": pd.DataFrame({"Id_1": ids, "Me_1": date_vals}),
            "DS_period": pd.DataFrame({"Id_1": ids, "Me_1": period_vals}),
        }
        result = run(script=script, data_structures=DATA_STRUCTURES, datapoints=datapoints)
        assert "DS_r" in result
        assert list(result["DS_r"].data["bool_var"]) == expected
