"""Tests for implicit type promotion between Date, TimePeriod, and TimeInterval.

VTL 2.2 specifies these implicit casts for time types:
  - Date → TimeInterval
  - TimePeriod → TimeInterval

This means comparing a Date measure against a TimePeriod measure should
succeed by promoting both operands to TimeInterval.
"""

import pandas as pd
import pytest

from tests.Helper import _use_duckdb_backend
from vtlengine import run
from vtlengine.DataTypes import (
    Boolean,
    Date,
    TimeInterval,
    TimePeriod,
    binary_implicit_promotion,
    check_binary_implicit_promotion,
)
from vtlengine.Model import Dataset


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
        result = run(
            script=script,
            data_structures=DATA_STRUCTURES,
            datapoints=datapoints,
            use_duckdb=_use_duckdb_backend(),
        )
        assert "DS_r" in result
        assert list(result["DS_r"].data["bool_var"]) == expected


DURATION_DS = {
    "name": "DS_1",
    "DataStructure": [
        {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
        {"name": "Me_1", "type": "Duration", "role": "Measure", "nullable": True},
    ],
}

DURATION_SINGLE_DS = {"datasets": [DURATION_DS]}

DURATION_TWO_DS = {
    "datasets": [
        DURATION_DS,
        {
            "name": "DS_2",
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Duration", "role": "Measure", "nullable": True},
            ],
        },
    ]
}


class TestDurationComparison:
    """Duration comparisons must use magnitude order (A>S>Q>M>W>D), not alphabetical."""

    @pytest.mark.parametrize(
        "script, expected",
        [
            ('DS_r <- cast("A", duration) > cast("M", duration);', True),
            ('DS_r <- cast("A", duration) > cast("D", duration);', True),
            ('DS_r <- cast("D", duration) < cast("A", duration);', True),
            ('DS_r <- cast("S", duration) >= cast("Q", duration);', True),
            ('DS_r <- cast("W", duration) < cast("M", duration);', True),
            ('DS_r <- cast("A", duration) = cast("A", duration);', True),
            ('DS_r <- cast("D", duration) > cast("W", duration);', False),
            ('DS_r <- cast("M", duration) > cast("A", duration);', False),
        ],
        ids=[
            "annual_gt_month",
            "annual_gt_day",
            "day_lt_annual",
            "semester_gte_quarter",
            "week_lt_month",
            "annual_eq_annual",
            "day_not_gt_week",
            "month_not_gt_annual",
        ],
    )
    def test_scalar_comparison(self, script: str, expected: bool) -> None:
        result = run(
            script=script,
            data_structures={"datasets": []},
            datapoints={},
            use_duckdb=_use_duckdb_backend(),
        )
        scalar = result["DS_r"]
        assert not isinstance(scalar, Dataset)
        assert scalar.value == expected

    @pytest.mark.parametrize(
        "script, expected",
        [
            ("DS_r <- DS_1 > DS_2;", [True, False, False]),
            ("DS_r <- DS_1 < DS_2;", [False, True, True]),
            ("DS_r <- DS_1 >= DS_2;", [True, False, False]),
            ("DS_r <- DS_1 <= DS_2;", [False, True, True]),
            ("DS_r <- DS_1 = DS_2;", [False, False, False]),
            ("DS_r <- DS_1 <> DS_2;", [True, True, True]),
        ],
        ids=["ds_gt", "ds_lt", "ds_gte", "ds_lte", "ds_eq", "ds_neq"],
    )
    def test_dataset_comparison(self, script: str, expected: list[bool]) -> None:
        datapoints = {
            "DS_1": pd.DataFrame({"Id_1": [1, 2, 3], "Me_1": ["A", "M", "D"]}),
            "DS_2": pd.DataFrame({"Id_1": [1, 2, 3], "Me_1": ["M", "A", "W"]}),
        }
        result = run(
            script=script,
            data_structures=DURATION_TWO_DS,
            datapoints=datapoints,
            use_duckdb=_use_duckdb_backend(),
        )
        ds = result["DS_r"]
        assert isinstance(ds, Dataset)
        assert list(ds.data["bool_var"]) == expected

    @pytest.mark.parametrize(
        "script, expected",
        [
            ('DS_r <- DS_1 > cast("M", duration);', [True, True, False]),
            ('DS_r <- DS_1 < cast("M", duration);', [False, False, True]),
            ('DS_r <- DS_1 = cast("Q", duration);', [False, True, False]),
        ],
        ids=["ds_scalar_gt", "ds_scalar_lt", "ds_scalar_eq"],
    )
    def test_dataset_scalar_comparison(self, script: str, expected: list[bool]) -> None:
        datapoints = {
            "DS_1": pd.DataFrame({"Id_1": [1, 2, 3], "Me_1": ["A", "Q", "D"]}),
        }
        result = run(
            script=script,
            data_structures=DURATION_SINGLE_DS,
            datapoints=datapoints,
            use_duckdb=_use_duckdb_backend(),
        )
        ds = result["DS_r"]
        assert isinstance(ds, Dataset)
        assert list(ds.data["bool_var"]) == expected

    @pytest.mark.parametrize(
        "script, expected",
        [
            (
                'DS_r <- DS_1[calc Me_2 := Me_1 > cast("M", duration)];',
                [True, False, False],
            ),
            (
                'DS_r <- DS_1[calc Me_2 := Me_1 < cast("Q", duration)];',
                [False, True, True],
            ),
        ],
        ids=["comp_scalar_gt", "comp_scalar_lt"],
    )
    def test_component_scalar_comparison(self, script: str, expected: list[bool]) -> None:
        datapoints = {
            "DS_1": pd.DataFrame({"Id_1": [1, 2, 3], "Me_1": ["A", "M", "D"]}),
        }
        result = run(
            script=script,
            data_structures=DURATION_SINGLE_DS,
            datapoints=datapoints,
            use_duckdb=_use_duckdb_backend(),
        )
        ds = result["DS_r"]
        assert isinstance(ds, Dataset)
        assert list(ds.data["Me_2"]) == expected

    @pytest.mark.parametrize(
        "script, expected",
        [
            (
                "DS_r <- DS_1[calc Me_3 := Me_1 > Me_2];",
                [True, False, False],
            ),
            (
                "DS_r <- DS_1[calc Me_3 := Me_1 < Me_2];",
                [False, True, True],
            ),
            (
                "DS_r <- DS_1[calc Me_3 := Me_1 = Me_2];",
                [False, False, False],
            ),
        ],
        ids=["comp_comp_gt", "comp_comp_lt", "comp_comp_eq"],
    )
    def test_component_component_comparison(self, script: str, expected: list[bool]) -> None:
        data_structures = {
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
                            "type": "Duration",
                            "role": "Measure",
                            "nullable": True,
                        },
                        {
                            "name": "Me_2",
                            "type": "Duration",
                            "role": "Measure",
                            "nullable": True,
                        },
                    ],
                }
            ]
        }
        datapoints = {
            "DS_1": pd.DataFrame(
                {
                    "Id_1": [1, 2, 3],
                    "Me_1": ["A", "M", "D"],
                    "Me_2": ["M", "A", "W"],
                }
            ),
        }
        result = run(
            script=script,
            data_structures=data_structures,
            datapoints=datapoints,
            use_duckdb=_use_duckdb_backend(),
        )
        ds = result["DS_r"]
        assert isinstance(ds, Dataset)
        assert list(ds.data["Me_3"]) == expected


TIME_PERIOD_DS = {
    "name": "DS_1",
    "DataStructure": [
        {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
        {"name": "Me_1", "type": "Time_Period", "role": "Measure", "nullable": True},
    ],
}

TIME_PERIOD_SINGLE_DS = {"datasets": [TIME_PERIOD_DS]}

TIME_PERIOD_TWO_DS = {
    "datasets": [
        TIME_PERIOD_DS,
        {
            "name": "DS_2",
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Time_Period", "role": "Measure", "nullable": True},
            ],
        },
    ]
}


class TestTimePeriodComparison:
    """TimePeriod comparisons across all evaluation paths."""

    @pytest.mark.parametrize(
        "script, expected",
        [
            ('DS_r <- cast("2020Q3", time_period) > cast("2020Q1", time_period);', True),
            ('DS_r <- cast("2020Q1", time_period) > cast("2020Q3", time_period);', False),
            ('DS_r <- cast("2021M01", time_period) > cast("2020M12", time_period);', True),
            ('DS_r <- cast("2020Q1", time_period) = cast("2020Q1", time_period);', True),
            ('DS_r <- cast("2020Q1", time_period) <> cast("2020Q3", time_period);', True),
            ('DS_r <- cast("2020Q1", time_period) < cast("2021Q1", time_period);', True),
        ],
        ids=[
            "q3_gt_q1",
            "q1_not_gt_q3",
            "2021m01_gt_2020m12",
            "q1_eq_q1",
            "q1_neq_q3",
            "2020q1_lt_2021q1",
        ],
    )
    def test_scalar_comparison(self, script: str, expected: bool) -> None:
        result = run(
            script=script,
            data_structures={"datasets": []},
            datapoints={},
            use_duckdb=_use_duckdb_backend(),
        )
        scalar = result["DS_r"]
        assert not isinstance(scalar, Dataset)
        assert scalar.value == expected

    @pytest.mark.parametrize(
        "script, expected",
        [
            ("DS_r <- DS_1 > DS_2;", [False, True, False]),
            ("DS_r <- DS_1 < DS_2;", [True, False, True]),
            ("DS_r <- DS_1 = DS_2;", [False, False, False]),
            ("DS_r <- DS_1 <> DS_2;", [True, True, True]),
        ],
        ids=["ds_gt", "ds_lt", "ds_eq", "ds_neq"],
    )
    def test_dataset_comparison(self, script: str, expected: list[bool]) -> None:
        datapoints = {
            "DS_1": pd.DataFrame({"Id_1": [1, 2, 3], "Me_1": ["2020Q1", "2021M06", "2020-A1"]}),
            "DS_2": pd.DataFrame({"Id_1": [1, 2, 3], "Me_1": ["2020Q3", "2020M12", "2021-A1"]}),
        }
        result = run(
            script=script,
            data_structures=TIME_PERIOD_TWO_DS,
            datapoints=datapoints,
            use_duckdb=_use_duckdb_backend(),
        )
        ds = result["DS_r"]
        assert isinstance(ds, Dataset)
        assert list(ds.data["bool_var"]) == expected

    @pytest.mark.parametrize(
        "script, expected",
        [
            ('DS_r <- DS_1 > cast("2020Q2", time_period);', [False, True]),
            ('DS_r <- DS_1 < cast("2020Q2", time_period);', [True, False]),
            ('DS_r <- DS_1 = cast("2020Q1", time_period);', [True, False]),
        ],
        ids=["ds_scalar_gt", "ds_scalar_lt", "ds_scalar_eq"],
    )
    def test_dataset_scalar_comparison(self, script: str, expected: list[bool]) -> None:
        datapoints = {
            "DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": ["2020Q1", "2020Q3"]}),
        }
        result = run(
            script=script,
            data_structures=TIME_PERIOD_SINGLE_DS,
            datapoints=datapoints,
            use_duckdb=_use_duckdb_backend(),
        )
        ds = result["DS_r"]
        assert isinstance(ds, Dataset)
        assert list(ds.data["bool_var"]) == expected

    @pytest.mark.parametrize(
        "script, expected",
        [
            (
                'DS_r <- DS_1[calc Me_2 := Me_1 > cast("2020Q2", time_period)];',
                [False, True],
            ),
            (
                'DS_r <- DS_1[calc Me_2 := Me_1 < cast("2020Q2", time_period)];',
                [True, False],
            ),
        ],
        ids=["comp_scalar_gt", "comp_scalar_lt"],
    )
    def test_component_scalar_comparison(self, script: str, expected: list[bool]) -> None:
        datapoints = {
            "DS_1": pd.DataFrame({"Id_1": [1, 2], "Me_1": ["2020Q1", "2020Q3"]}),
        }
        result = run(
            script=script,
            data_structures=TIME_PERIOD_SINGLE_DS,
            datapoints=datapoints,
            use_duckdb=_use_duckdb_backend(),
        )
        ds = result["DS_r"]
        assert isinstance(ds, Dataset)
        assert list(ds.data["Me_2"]) == expected

    @pytest.mark.parametrize(
        "script, expected",
        [
            ("DS_r <- DS_1[calc Me_3 := Me_1 > Me_2];", [False, True, False]),
            ("DS_r <- DS_1[calc Me_3 := Me_1 < Me_2];", [True, False, True]),
            ("DS_r <- DS_1[calc Me_3 := Me_1 = Me_2];", [False, False, False]),
        ],
        ids=["comp_comp_gt", "comp_comp_lt", "comp_comp_eq"],
    )
    def test_component_component_comparison(self, script: str, expected: list[bool]) -> None:
        data_structures = {
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
                        {
                            "name": "Me_2",
                            "type": "Time_Period",
                            "role": "Measure",
                            "nullable": True,
                        },
                    ],
                }
            ]
        }
        datapoints = {
            "DS_1": pd.DataFrame(
                {
                    "Id_1": [1, 2, 3],
                    "Me_1": ["2020Q1", "2021M06", "2020-A1"],
                    "Me_2": ["2020Q3", "2020M12", "2021-A1"],
                }
            ),
        }
        result = run(
            script=script,
            data_structures=data_structures,
            datapoints=datapoints,
            use_duckdb=_use_duckdb_backend(),
        )
        ds = result["DS_r"]
        assert isinstance(ds, Dataset)
        assert list(ds.data["Me_3"]) == expected
