import re
from datetime import datetime, timedelta
from itertools import product
from typing import Optional

import pandas as pd
from dateutil.relativedelta import relativedelta

import DataTypes
import Operators
from Model import Dataset, DataComponent, Scalar, Component, Role


class Time(Operators.Operator):
    py_op = None
    TIME_DATA_TYPES = [DataTypes.Date, DataTypes.TimePeriod, DataTypes.TimeInterval]

    PERIODS = [
        (r"^\d{1,4}-(0?[1-9]|1[0-2])-\d{2}$", "D"),
        (r"^\d{1,4}-(M)?(0?[1-9]|1[0-2])$", "M"),
        (r"^\d{1,4}-Q[1-4]$", "Q"),
        (r"^\d{1,4}-S[1-2]$", "S"),
        (r"^\d{1,4}-M0?1/\d{1,4}-M12$", "P"),
        (r"^\d{1,4}$", "A"),
    ]

    PERIODS_PARSER = {
        "D": lambda x: datetime.strptime(x, "%Y-%m-%d"),
        "M": lambda x: (
            datetime.strptime(x, "%Y-%m")
            if re.match(r"^\d{1,4}-(0?[1-9]|1[0-2])$", x)
            else datetime.strptime(x, "%Y-M%m")
        ),
        "Q": lambda x: datetime.strptime(x, "%Y-Q%d"),
        "S": lambda x: datetime.strptime(x, "%Y-S%d"),
        "P": lambda x: datetime.strptime(x.split("/")[0], "%Y-M%m"),
        "A": lambda x: datetime.strptime(x, "%Y"),
    }

    PERIODS_FORMAT = {
        "D": lambda x, y, z: "{}-{}-{}".format(x, y, z),
        "M": lambda x, y: "{}-M{}".format(x, y),
        "Q": lambda x, y: "{}-Q{}".format(x, y),
        "S": lambda x, y: "{}-S{}".format(x, y),
        "P": lambda x: "{}-M1/{}-M12".format(x, x),
        "A": lambda x: "{}".format(x),
    }

    @classmethod
    def get_time_id(cls, operand: Dataset) -> Optional[str]:
        reference_id = None
        for id in operand.get_identifiers():
            if id.data_type in cls.TIME_DATA_TYPES:
                if reference_id is not None:
                    raise ValueError(
                        "FlowToStock can only be applied to a time dataset"
                    )
                reference_id = id.name
        return reference_id

    @classmethod
    def sort_by_time(cls, operand: Dataset) -> Optional[pd.DataFrame]:
        time_id = cls.get_time_id(operand)
        if time_id is None:
            return
        ids = [id.name for id in operand.get_identifiers() if id.name != time_id]
        ids.append(time_id)
        return operand.data.sort_values(by=ids).reset_index(drop=True)

    @classmethod
    def get_period_from_list(cls, series: pd.Series) -> pd.Series:
        result = series.copy()
        for i in range(0, len(result)):
            result[i] = cls.get_period(str(result[i]))
        return result

    @classmethod
    def get_period(cls, value) -> str | None:
        for format, period in cls.PERIODS:
            if re.match(format, str(value)):
                return period
        return None


class Unary(Time):
    is_from_stock = False

    @classmethod
    def evaluate(cls, operand: Dataset) -> Dataset:
        result = cls.validate(operand)
        if len(operand.data) < 2:
            return result
        reference_id = cls.get_time_id(operand)
        others_ids = [
            id.name for id in operand.get_identifiers() if id.name != reference_id
        ]
        last_from_period = {}
        previous_period = cls.get_period(str(operand.data[reference_id][0]))
        for i in range(1, len(operand.data[reference_id])):
            current_period = cls.get_period(str(operand.data[reference_id][i]))
            last_from_period[previous_period] = i - 1
            if current_period not in last_from_period.keys():
                last_from_period[current_period] = i
            else:
                j = last_from_period[current_period]
                for measure in operand.get_measures_names():
                    if all(
                        operand.data[id][i] == operand.data[id][j] for id in others_ids
                    ):
                        if cls.is_from_stock:
                            result.data[measure][i] = cls.py_op(
                                operand.data[measure][i], operand.data[measure][j]
                            )
                        else:
                            result.data[measure][i] = cls.py_op(
                                result.data[measure][i], result.data[measure][j]
                            )
            previous_period = current_period
        return result

    @classmethod
    def validate(cls, operand: Dataset) -> Dataset:
        if not isinstance(operand, Dataset):
            raise TypeError("FlowToStock can only be applied to a time dataset")
        if cls.get_time_id(operand) is None:
            raise ValueError("FlowToStock can only be applied to a time dataset")
        operand.data = cls.sort_by_time(operand)
        return Dataset(
            name="result",
            components=operand.components.copy(),
            data=operand.data.copy(),
        )


class Binary(Time):
    reference_id = None
    other_ids = None
    measures = None
    periods = None
    min_limit = True
    period_col, val_col = (
        "temp_period_indicator_col_var",
        "temp_significant_value_col_var",
    )

    PERIOD_ENUM = {"A": 0, "P": 1, "S": 2, "Q": 3, "M": 4, "D": 5}

    MAX_MIN_FROM_PERIOD = {}

    @classmethod
    def significant_from_period(cls, operand: str, period) -> int:
        if period in ["A", "P"]:
            return cls.get_year(operand)
        if period == "M":
            return cls.PERIODS_PARSER[period](operand).month
        return cls.PERIODS_PARSER[period](operand).day

    @classmethod
    def get_year(cls, operand: str) -> int:
        return cls.PERIODS_PARSER[cls.get_period(operand)](str(operand)).year

    @classmethod
    def sort_by_period(cls, operand: Dataset, fill_type: str) -> Dataset:
        data = operand.data.copy()
        data[cls.period_col] = data[cls.reference_id].apply(cls.get_period)
        cls.periods = data[cls.period_col].unique()
        data[cls.val_col] = data.apply(
            lambda row: cls.significant_from_period(
                row[cls.reference_id], row[cls.period_col]
            ),
            axis=1,
        )
        cls.get_max_min_from_period(data, fill_type)
        data[cls.period_col] = data[cls.period_col].map(cls.PERIOD_ENUM)
        data = data.sort_values(
            by=cls.other_ids + [cls.period_col, cls.reference_id]
        ).drop(columns=[cls.val_col, cls.period_col])
        return Dataset(name="result", components=operand.components.copy(), data=data)

    @classmethod
    def get_max_min_from_period(cls, data: pd.DataFrame, fill_type: str):
        if fill_type == "all":
            cls.MAX_MIN_FROM_PERIOD = {"max": {}, "min": {}}
            cls.MAX_MIN_FROM_PERIOD.update(
                data.groupby(cls.period_col)[cls.val_col].agg(["max", "min"]).to_dict()
            )
            years = data[cls.reference_id].map(cls.get_year).agg(["max", "min"])
            cls.MAX_MIN_FROM_PERIOD["max"].update({"A": years["max"]})
            cls.MAX_MIN_FROM_PERIOD["min"].update({"A": years["min"]})
            if "D" in cls.periods:
                days = data.loc[data[cls.period_col] == "D"]
                months = (
                    days[cls.reference_id]
                    .map(lambda x: cls.PERIODS_PARSER["D"](str(x)).month)
                    .agg(["max", "min"])
                )
                cls.MAX_MIN_FROM_PERIOD["max"].update({"M": months["max"]})
                cls.MAX_MIN_FROM_PERIOD["min"].update({"M": months["min"]})
        elif fill_type == "single":
            grouped_data = data.groupby(cls.other_ids)
            cls.MAX_MIN_FROM_PERIOD = {}
            for group_name, group_df in grouped_data:
                max_min = (
                    group_df.groupby(cls.period_col)[cls.val_col]
                    .agg(["max", "min"])
                    .to_dict()
                )
                cls.MAX_MIN_FROM_PERIOD[group_name] = {
                    "max": max_min["max"],
                    "min": max_min["min"],
                }
                years = group_df[cls.reference_id].map(cls.get_year).agg(["max", "min"])
                cls.MAX_MIN_FROM_PERIOD[group_name]["max"].update({"A": years["max"]})
                cls.MAX_MIN_FROM_PERIOD[group_name]["min"].update({"A": years["min"]})
                if "D" in cls.periods:
                    days = group_df.loc[group_df[cls.period_col] == "D"]
                    months = (
                        days[cls.reference_id]
                        .map(lambda x: cls.PERIODS_PARSER["D"](str(x)).month)
                        .agg(["max", "min"])
                    )
                    cls.MAX_MIN_FROM_PERIOD[group_name]["max"].update(
                        {"M": months["max"]}
                    )
                    cls.MAX_MIN_FROM_PERIOD[group_name]["min"].update(
                        {"M": months["min"]}
                    )


class Period_indicator(Unary):

    @classmethod
    def evaluate(
        cls, operand: Dataset | DataComponent | Scalar | str
    ) -> Dataset | DataComponent | Scalar | str:
        cls.validate(operand)
        if isinstance(operand, str):
            return cls.get_period(str(operand))
        if isinstance(operand, Scalar):
            return Scalar(
                name="result",
                data_type=DataTypes.Duration,
                value=cls.get_period(str(operand.value)),
            )
        if isinstance(operand, DataComponent):
            return DataComponent(
                name="result",
                data_type=DataTypes.Duration,
                data=cls.get_period_from_list(operand.data),
            )
        data = cls.get_period_from_list(operand.data[cls.get_time_id(operand)])
        operand.data = operand.data.drop(columns=operand.get_measures_names())
        operand.data["duration_var"] = data
        for measure in operand.get_measures_names():
            operand.components.pop(measure)
        operand.components["duration_var"] = Component(
            name="duration_var",
            data_type=DataTypes.Duration,
            role=Role.MEASURE,
            nullable=True,
        )
        return Dataset(name="result", components=operand.components, data=operand.data)

    @classmethod
    def validate(cls, operand: Dataset | DataComponent | Scalar | str) -> None:
        if isinstance(operand, str):
            return
        if isinstance(operand, Dataset):
            time_id = cls.get_time_id(operand)
            if (
                time_id is None
                or operand.components[time_id].data_type != DataTypes.TimePeriod
            ):
                raise ValueError(
                    "PeriodIndicator can only be applied to a time dataset"
                )
        else:
            if operand.data_type != DataTypes.TimePeriod:
                raise ValueError(
                    "PeriodIndicator can only be applied to a time dataset"
                )


class Flow_to_stock(Unary):
    py_op = lambda x, y: x + y  # noqa


class Stock_to_flow(Unary):
    is_from_stock = True
    py_op = lambda x, y: x - y  # noqa


class Fill_time_series(Binary):

    @classmethod
    def evaluate(cls, operand: Dataset, fill_type: str) -> Dataset:
        result = cls.validate(operand, fill_type)
        if len(result.data) < 2:
            return result
        result.data = cls.fill_period(result.data, fill_type)
        return result

    @classmethod
    def validate(cls, operand: Dataset, fill_type: str) -> Dataset:
        if not isinstance(operand, Dataset):
            raise TypeError("FillTimeSeries can only be applied to a time dataset")
        cls.reference_id = cls.get_time_id(operand)
        cls.other_ids = [
            id.name for id in operand.get_identifiers() if id.name != cls.reference_id
        ]
        cls.measures = operand.get_measures_names()
        if cls.reference_id is None:
            raise ValueError("FillTimeSeries can only be applied to a time dataset")
        if fill_type not in ["all", "single"]:
            fill_type = "all"
        return cls.sort_by_period(operand, fill_type)

    @classmethod
    def fill_period(cls, data: pd.DataFrame, fill_type: str) -> pd.DataFrame:
        result_data = cls.period_filler(
            data, cls.MAX_MIN_FROM_PERIOD, single=(fill_type != "all")
        )
        not_na = result_data[cls.measures].notna().any(axis=1)
        duplicated = result_data.duplicated(
            subset=(cls.other_ids + [cls.reference_id]), keep=False
        )
        return result_data[~duplicated | not_na]

    @classmethod
    def period_filler(
        cls, data: pd.DataFrame, max_min_period: dict, single=False
    ) -> pd.DataFrame:
        filled_data = []
        groups = data.groupby(cls.other_ids)

        for group, group_df in groups:
            period_limits = max_min_period if not single else max_min_period[group]
            years = range(period_limits["min"]["A"], period_limits["max"]["A"] + 1)
            for period in cls.periods:
                if period in ["A", "P"]:
                    filled_data.extend(cls.fill_rows(group_df, period, years))
                elif period == "D":
                    months = range(
                        period_limits["min"]["M"], period_limits["max"]["M"] + 1
                    )
                    vals = range(
                        period_limits["min"][period], period_limits["max"][period] + 1
                    )
                    filled_data.extend(
                        cls.fill_rows(group_df, period, years, months, vals)
                    )
                else:
                    vals = range(
                        period_limits["min"][period], period_limits["max"][period] + 1
                    )
                    filled_data.extend(
                        cls.fill_rows(group_df, period, years, vals=vals)
                    )

        filled_data = pd.concat(filled_data, ignore_index=True)
        combined_data = pd.concat([filled_data, data], ignore_index=True)
        combined_data[cls.reference_id] = combined_data[cls.reference_id].astype(
            int if len(cls.periods) == 1 and cls.periods[0] == "A" else str
        )
        return combined_data.sort_values(by=cls.other_ids + [cls.reference_id])

    @classmethod
    def fill_rows(cls, group_df, period, years, months=None, vals=None):
        rows = []
        for year in years:
            if period == "D":
                for month, val in product(months, vals):
                    rows.append(cls.create_row(group_df, period, year, month, val))
            elif period in ["A", "P"]:
                rows.append(cls.create_row(group_df, period, year))
            else:
                for val in vals:
                    rows.append(cls.create_row(group_df, period, year, val=val))
        return rows

    @classmethod
    def create_row(cls, group_df, period, year, month=None, val=None):
        row = group_df.iloc[0].copy()
        if period == "D":
            row[cls.reference_id] = str(cls.PERIODS_FORMAT[period](year, month, val))
        elif period in ["A", "P"]:
            row[cls.reference_id] = str(cls.PERIODS_FORMAT[period](year))
        else:
            row[cls.reference_id] = str(cls.PERIODS_FORMAT[period](year, val))
        row[cls.measures] = None
        return row.to_frame().T


class Time_Shift(Binary):

    @classmethod
    def evaluate(cls, operand, shift: Scalar) -> Dataset:
        result = cls.validate(operand, shift)
        result.data = operand.data.copy()
        if shift.value == 0:
            return result
        result = cls.shift(operand.data, int(shift.value))
        cls.periods = result[cls.reference_id].apply(cls.get_period).unique()
        if len(cls.periods) == 1 and cls.periods[0] == "A":
            result[cls.reference_id] = result[cls.reference_id].astype(int)
        return Dataset(name="result", components=operand.components.copy(), data=result)

    @classmethod
    def validate(cls, operand, shift) -> Dataset:
        if not isinstance(shift, Scalar):
            raise TypeError("TimeShift can only be applied with a scalar value")
        if not isinstance(operand, Dataset):
            raise TypeError("TimeShift can only be applied to a time dataset")
        cls.reference_id = cls.get_time_id(operand)
        if cls.reference_id is None:
            raise ValueError("TimeShift can only be applied to a time dataset")
        cls.sort_by_time(operand)
        return Dataset(name="result", components=operand.components.copy(), data=None)

    @classmethod
    def shift(cls, data, shift):
        def update_row(row):
            period = cls.get_period(row[cls.reference_id])
            val = cls.significant_from_period(row[cls.reference_id], period)
            year = cls.get_year(row[cls.reference_id])

            if period in ["A", "P"]:
                row[cls.reference_id] = cls.PERIODS_FORMAT[period](year + shift)
            elif period == "D":
                date = cls.PERIODS_PARSER[period](row[cls.reference_id]) + timedelta(
                    days=shift
                )
                row[cls.reference_id] = cls.PERIODS_FORMAT[period](
                    date.year, date.month, date.day
                )
            elif period == "M":
                delta = relativedelta(months=shift)
                row[cls.reference_id] = cls.PERIODS_FORMAT[period](
                    year, val + delta.months
                )
            elif period == "Q":
                delta = val + shift
                delta_years, quarters = divmod(delta, 4)
                if quarters == 0:
                    quarters = 4
                    delta_years -= 1
                row[cls.reference_id] = cls.PERIODS_FORMAT[period](
                    year + delta_years, quarters
                )
            elif period == "S":
                delta = val + shift
                delta_years, semesters = divmod(delta, 2)
                if semesters == 0:
                    semesters = 2
                    delta_years -= 1
                row[cls.reference_id] = cls.PERIODS_FORMAT[period](
                    year + delta_years, semesters
                )
            return row

        return data.apply(update_row, axis=1)
