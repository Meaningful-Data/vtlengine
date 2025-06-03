import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Type, Union

import pandas as pd

import vtlengine.Operators as Operators
from vtlengine.AST.Grammar.tokens import (
    DATE_ADD,
    DATEDIFF,
    DAYOFMONTH,
    DAYOFYEAR,
    DAYTOMONTH,
    DAYTOYEAR,
    FILL_TIME_SERIES,
    FLOW_TO_STOCK,
    MONTH,
    MONTHTODAY,
    PERIOD_INDICATOR,
    TIME_AGG,
    TIMESHIFT,
    YEAR,
    YEARTODAY,
)
from vtlengine.DataTypes import (
    Date,
    Duration,
    Integer,
    ScalarType,
    String,
    TimeInterval,
    TimePeriod,
    unary_implicit_promotion,
)
from vtlengine.DataTypes.TimeHandling import (
    PERIOD_IND_MAPPING,
    TimePeriodHandler,
    date_to_period,
    period_to_date,
)
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import Component, DataComponent, Dataset, Role, Scalar


class Time(Operators.Operator):
    periods: Any
    time_id: Any
    other_ids: Any
    measures: Any

    TIME_DATA_TYPES = [Date, TimePeriod, TimeInterval]

    FREQUENCY_MAP = {"Y": "years", "M": "months", "D": "days"}
    YEAR_TO_PERIOD = {"S": 2, "Q": 4, "M": 12, "W": 52, "D": 365}
    PERIOD_ORDER = {"A": 0, "S": 1, "Q": 2, "M": 3, "W": 4, "D": 5}

    op = FLOW_TO_STOCK

    @classmethod
    def _get_time_id(cls, operand: Dataset) -> str:
        reference_id = None
        identifiers = operand.get_identifiers()
        if len(identifiers) == 0:
            raise SemanticError("1-1-19-8", op=cls.op, comp_type="time dataset")
        for id in operand.get_identifiers():
            if id.data_type in cls.TIME_DATA_TYPES:
                if reference_id is not None:
                    raise SemanticError("1-1-19-8", op=cls.op, comp_type="time dataset")
                reference_id = id.name
        return str(reference_id)

    @classmethod
    def sort_by_time(cls, operand: Dataset) -> Optional[pd.DataFrame]:
        time_id = cls._get_time_id(operand)
        if time_id is None:
            return None
        ids = [id.name for id in operand.get_identifiers() if id.name != time_id]
        ids.append(time_id)
        if operand.data is None:
            return None
        return operand.data.sort_values(by=ids).reset_index(drop=True)

    @classmethod
    def _get_period(cls, value: str) -> str:
        tp_value = TimePeriodHandler(value)
        return tp_value.period_indicator

    @classmethod
    def parse_date(cls, date_str: str) -> date:
        return date.fromisoformat(date_str)

    @classmethod
    def get_frequencies(cls, dates: Any) -> Any:
        dates = pd.to_datetime(dates)
        dates = dates.sort_values()
        deltas = dates.diff().dropna()
        return deltas

    @classmethod
    def find_min_frequency(cls, differences: Any) -> str:
        months_deltas = differences.apply(lambda x: x.days // 30)
        days_deltas = differences.apply(lambda x: x.days)
        min_months = min(
            (diff for diff in months_deltas if diff > 0 and diff % 12 != 0),
            default=None,
        )
        min_days = min(
            (diff for diff in days_deltas if diff > 0 and diff % 365 != 0 and diff % 366 != 0),
            default=None,
        )
        return "D" if min_days else "M" if min_months else "Y"

    @classmethod
    def get_frequency_from_time(cls, interval: str) -> Any:
        start_date, end_date = interval.split("/")
        return date.fromisoformat(end_date) - date.fromisoformat(start_date)

    @classmethod
    def get_date_format(cls, date_str: Union[str, date]) -> str:
        date = cls.parse_date(date_str) if isinstance(date_str, str) else date_str
        return "%Y-%m-%d" if date.day >= 1 else "%Y-%m" if date.month >= 1 else "%Y"


class Unary(Time):
    @classmethod
    def validate(cls, operand: Any) -> Any:
        if not isinstance(operand, Dataset):
            raise SemanticError("1-1-19-8", op=cls.op, comp_type="time dataset")
        if cls._get_time_id(operand) is None:
            raise SemanticError("1-1-19-8", op=cls.op, comp_type="time dataset")
        operand.data = cls.sort_by_time(operand)
        return Dataset(name="result", components=operand.components.copy(), data=None)

    @classmethod
    def evaluate(cls, operand: Any) -> Any:
        result = cls.validate(operand)
        result.data = operand.data.copy() if operand.data is not None else pd.DataFrame()
        if len(operand.data) < 2:
            return result

        cls.time_id = cls._get_time_id(result)
        cls.other_ids = [id.name for id in result.get_identifiers() if id.name != cls.time_id]
        measure_names = result.get_measures_names()

        data_type = result.components[cls.time_id].data_type

        result.data = result.data.sort_values(by=cls.other_ids + [cls.time_id])
        if data_type == TimePeriod:
            result.data = cls._period_accumulation(result.data, measure_names)
        elif data_type in (Date, TimeInterval):
            result.data[measure_names] = (
                result.data.groupby(cls.other_ids)[measure_names]
                .apply(cls.py_op)
                .reset_index(drop=True)
            )
        else:
            raise SemanticError("1-1-19-8", op=cls.op, comp_type="dataset", param="date type")
        return result

    @classmethod
    def _period_accumulation(cls, data: pd.DataFrame, measure_names: List[str]) -> pd.DataFrame:
        data = data.copy()
        data["Period_group_col"] = (
            data[cls.time_id].apply(cls._get_period).apply(lambda x: cls.PERIOD_ORDER[x])
        )
        result = data.groupby(cls.other_ids + ["Period_group_col"], group_keys=False)[
            measure_names
        ].apply(cls.py_op)
        data[measure_names] = result.reset_index(drop=True)
        return data.drop(columns="Period_group_col")


class Binary(Time):
    pass


class Parameterized(Time):
    pass


class Period_indicator(Unary):
    op = PERIOD_INDICATOR

    @classmethod
    def validate(cls, operand: Any) -> Any:
        if isinstance(operand, Dataset):
            time_id = cls._get_time_id(operand)
            if operand.components[time_id].data_type != TimePeriod:
                raise SemanticError("1-1-19-8", op=cls.op, comp_type="time period dataset")
            result_components = {
                comp.name: comp
                for comp in operand.components.values()
                if comp.role == Role.IDENTIFIER
            }
            result_components["duration_var"] = Component(
                name="duration_var",
                data_type=Duration,
                role=Role.MEASURE,
                nullable=True,
            )
            return Dataset(name="result", components=result_components, data=None)
        # DataComponent and Scalar validation
        if operand.data_type != TimePeriod:
            raise SemanticError("1-1-19-8", op=cls.op, comp_type="time period component")
        if isinstance(operand, DataComponent):
            return DataComponent(name=operand.name, data_type=Duration, data=None)
        return Scalar(name=operand.name, data_type=Duration, value=None)

    @classmethod
    def evaluate(
        cls, operand: Union[Dataset, DataComponent, Scalar, str]
    ) -> Union[Dataset, DataComponent, Scalar, str]:
        result = cls.validate(operand)
        if isinstance(operand, str):
            return cls._get_period(str(operand))
        if isinstance(operand, Scalar):
            result.value = cls._get_period(str(operand.value))
            return result
        if isinstance(operand, DataComponent):
            if operand.data is not None:
                result.data = operand.data.map(cls._get_period, na_action="ignore")
            return result
        cls.time_id = cls._get_time_id(operand)
        result.data = (
            operand.data.copy()[result.get_identifiers_names()]
            if (operand.data is not None)
            else pd.Series()
        )
        period_series: Any = result.data[cls.time_id].map(cls._get_period)
        result.data["duration_var"] = period_series
        return result


class Parametrized(Time):
    @classmethod
    def validate(cls, operand: Any, param: Any) -> Any:
        pass

    @classmethod
    def evaluate(cls, operand: Any, param: Any) -> Any:
        pass


class Flow_to_stock(Unary):
    @classmethod
    def py_op(cls, x: Any) -> Any:
        return x.cumsum().fillna(x)


class Stock_to_flow(Unary):
    @classmethod
    def py_op(cls, x: Any) -> Any:
        return x.diff().fillna(x)


class Fill_time_series(Binary):
    op = FILL_TIME_SERIES

    @classmethod
    def evaluate(cls, operand: Dataset, fill_type: str) -> Dataset:
        result = cls.validate(operand, fill_type)
        if operand.data is None:
            operand.data = pd.DataFrame()
        result.data = operand.data.copy()
        result.data[cls.time_id] = result.data[cls.time_id].astype(str)
        if len(result.data) < 2:
            return result
        data_type = result.components[cls.time_id].data_type
        if data_type == TimePeriod:
            result.data = cls.fill_periods(result.data, fill_type)
        elif data_type == Date:
            frequencies = cls.get_frequencies(operand.data[cls.time_id].apply(cls.parse_date))
            result.data = cls.fill_dates(
                result.data, fill_type, cls.find_min_frequency(frequencies)
            )
        elif data_type == TimeInterval:
            frequencies = result.data[cls.time_id].apply(cls.get_frequency_from_time).unique()
            if len(frequencies) > 1:
                raise SemanticError(
                    "1-1-19-9",
                    op=cls.op,
                    comp_type="dataset",
                    param="single time interval frequency",
                )
            result.data = cls.fill_time_intervals(result.data, fill_type, frequencies[0])
        else:
            raise SemanticError("1-1-19-2", op=cls.op)
        return result

    @classmethod
    def validate(cls, operand: Dataset, fill_type: str) -> Dataset:
        if not isinstance(operand, Dataset):
            raise SemanticError("1-1-19-8", op=cls.op, comp_type="time dataset")
        cls.time_id = cls._get_time_id(operand)
        cls.other_ids = [id.name for id in operand.get_identifiers() if id.name != cls.time_id]
        cls.measures = operand.get_measures_names()
        if cls.time_id is None:
            raise SemanticError("1-1-19-8", op=cls.op, comp_type="time dataset")
        if fill_type not in ["all", "single"]:
            fill_type = "all"
        return Dataset(name="result", components=operand.components.copy(), data=None)

    @classmethod
    def max_min_from_period(cls, data: pd.DataFrame, mode: str = "all") -> Dict[str, Any]:
        result_dict: Dict[Any, Any] = {}
        data = data.assign(
            Periods_col=data[cls.time_id].apply(cls._get_period),
            Periods_values_col=data[cls.time_id].apply(
                lambda x: int(re.sub(r"[^\d]", "", x.split("-")[-1]))
            ),
            Year_values_col=data[cls.time_id].apply(lambda x: int(x.split("-")[0])),
        ).sort_values(by=["Year_values_col", "Periods_col", "Periods_values_col"])

        if mode == "all":
            min_year = data["Year_values_col"].min()
            max_year = data["Year_values_col"].max()
            result_dict = {"min": {"A": min_year}, "max": {"A": max_year}}
            for period, group in data.groupby("Periods_col"):
                if period != "A":
                    result_dict["min"][period] = group["Periods_values_col"].min()
                    result_dict["max"][period] = group["Periods_values_col"].max()

        elif mode == "single":
            for name, group in data.groupby(cls.other_ids + ["Periods_col"]):
                key = name[:-1] if len(name[:-1]) > 1 else name[0]
                period = name[-1]
                if key not in result_dict:
                    result_dict[key] = {
                        "min": {"A": group["Year_values_col"].min()},
                        "max": {"A": group["Year_values_col"].max()},
                    }
                if period != "A":
                    year_min = group["Year_values_col"].min()
                    year_max = group["Year_values_col"].max()

                    result_dict[key]["min"]["A"] = min(result_dict[key]["min"]["A"], year_min)
                    result_dict[key]["max"]["A"] = max(result_dict[key]["max"]["A"], year_max)
                    result_dict[key]["min"][period] = group["Periods_values_col"].min()
                    result_dict[key]["max"][period] = group["Periods_values_col"].max()

        else:
            raise ValueError("Mode must be either 'all' or 'single'")
        return result_dict

    @classmethod
    def fill_periods(cls, data: pd.DataFrame, fill_type: str) -> pd.DataFrame:
        result_data = cls.period_filler(data, single=(fill_type != "all"))
        not_na = result_data[cls.measures].notna().any(axis=1)
        duplicated = result_data.duplicated(subset=(cls.other_ids + [cls.time_id]), keep=False)
        return result_data[~duplicated | not_na]

    @classmethod
    def period_filler(cls, data: pd.DataFrame, single: bool = False) -> pd.DataFrame:
        filled_data = []
        MAX_MIN = cls.max_min_from_period(data, mode="single" if single else "all")
        cls.periods = (
            list(MAX_MIN[list(MAX_MIN.keys())[0]]["min"].keys())
            if single
            else list(MAX_MIN["min"].keys())
        )
        groups = data.groupby(cls.other_ids)

        for group, group_df in groups:
            period_limits = (
                MAX_MIN if not single else MAX_MIN[group if len(group) > 1 else group[0]]
            )
            years = list(range(period_limits["min"]["A"], period_limits["max"]["A"] + 1))
            for period in cls.periods:
                if period == "A":
                    filled_data.extend(cls.fill_periods_rows(group_df, period, years))
                else:
                    if period in period_limits["min"] and period in period_limits["max"]:
                        vals = list(
                            range(
                                period_limits["min"][period],
                                period_limits["max"][period] + 1,
                            )
                        )
                        filled_data.extend(
                            cls.fill_periods_rows(group_df, period, years, vals=vals)
                        )

        filled_data = pd.concat(filled_data, ignore_index=True)
        combined_data = pd.concat([filled_data, data], ignore_index=True)
        if len(cls.periods) == 1 and cls.periods[0] == "A":
            combined_data[cls.time_id] = combined_data[cls.time_id].astype(int)
        else:
            combined_data[cls.time_id] = combined_data[cls.time_id].astype(str)
        return combined_data.sort_values(by=cls.other_ids + [cls.time_id])

    @classmethod
    def fill_periods_rows(
        cls,
        group_df: Any,
        period: str,
        years: List[int],
        vals: Optional[List[int]] = None,
    ) -> List[Any]:
        rows = []
        for year in years:
            if period == "A":
                rows.append(cls.create_period_row(group_df, period, year))
            elif vals is not None:
                for val in vals:
                    rows.append(cls.create_period_row(group_df, period, year, val=val))
        return rows

    @classmethod
    def create_period_row(
        cls, group_df: Any, period: str, year: int, val: Optional[int] = None
    ) -> Any:
        row = group_df.iloc[0].copy()
        row[cls.time_id] = f"{year}" if period == "A" else f"{year}-{period}{val:d}"
        row[cls.measures] = None
        return row.to_frame().T

    @classmethod
    def max_min_from_date(cls, data: pd.DataFrame, fill_type: str = "all") -> Dict[str, Any]:
        def compute_min_max(group: Any) -> Dict[str, Any]:
            min_date = cls.parse_date(group.min())
            max_date = cls.parse_date(group.max())
            date_format = cls.get_date_format(max_date)
            return {"min": min_date, "max": max_date, "date_format": date_format}

        if fill_type == "all":
            return compute_min_max(data[cls.time_id])

        grouped = data.groupby(cls.other_ids)
        result_dict = {
            name if len(name) > 1 else name[0]: compute_min_max(group[cls.time_id])
            for name, group in grouped
        }
        return result_dict

    @classmethod
    def fill_dates(cls, data: pd.DataFrame, fill_type: str, min_frequency: str) -> pd.DataFrame:
        result_data = cls.date_filler(data, fill_type, min_frequency)
        not_na = result_data[cls.measures].notna().any(axis=1)
        duplicated = result_data.duplicated(subset=(cls.other_ids + [cls.time_id]), keep=False)
        return result_data[~duplicated | not_na]

    @classmethod
    def date_filler(cls, data: pd.DataFrame, fill_type: str, min_frequency: str) -> pd.DataFrame:
        MAX_MIN = cls.max_min_from_date(data, fill_type)
        date_format = None
        filled_data = []

        def create_filled_dates(group: Any, min_max: Dict[str, Any]) -> (pd.DataFrame, str):  # type: ignore[syntax]
            date_range = pd.date_range(start=min_max["min"], end=min_max["max"], freq=min_frequency)
            date_df = pd.DataFrame(date_range, columns=[cls.time_id])
            date_df[cls.other_ids] = group.iloc[0][cls.other_ids]
            date_df[cls.measures] = None
            return date_df, min_max["date_format"]

        for name, group in data.groupby(cls.other_ids):
            min_max = MAX_MIN if fill_type == "all" else MAX_MIN[name if len(name) > 1 else name[0]]
            filled_dates, date_format = create_filled_dates(group, min_max)
            filled_data.append(filled_dates)

        filled_data = pd.concat(filled_data, ignore_index=True)
        filled_data[cls.time_id] = filled_data[cls.time_id].dt.strftime(date_format)
        combined_data = pd.concat([filled_data, data], ignore_index=True)
        combined_data[cls.time_id] = combined_data[cls.time_id].astype(str)
        return combined_data.sort_values(by=cls.other_ids + [cls.time_id])

    @classmethod
    def max_min_from_time(cls, data: pd.DataFrame, fill_type: str = "all") -> Dict[str, Any]:
        data = data.applymap(str).sort_values(  # type: ignore[operator]
            by=cls.other_ids + [cls.time_id]
        )

        def extract_max_min(group: Any) -> Dict[str, Any]:
            start_dates = group.apply(lambda x: x.split("/")[0])
            end_dates = group.apply(lambda x: x.split("/")[1])
            return {
                "start": {"min": start_dates.min(), "max": start_dates.max()},
                "end": {"min": end_dates.min(), "max": end_dates.max()},
            }

        if fill_type == "all":
            return extract_max_min(data[cls.time_id])
        else:
            return {
                name: extract_max_min(group[cls.time_id])
                for name, group in data.groupby(cls.other_ids)
            }

    @classmethod
    def fill_time_intervals(
        cls, data: pd.DataFrame, fill_type: str, frequency: str
    ) -> pd.DataFrame:
        result_data = cls.time_filler(data, fill_type, frequency)
        not_na = result_data[cls.measures].notna().any(axis=1)
        duplicated = result_data.duplicated(subset=(cls.other_ids + [cls.time_id]), keep=False)
        return result_data[~duplicated | not_na]

    @classmethod
    def time_filler(cls, data: pd.DataFrame, fill_type: str, frequency: str) -> pd.DataFrame:
        MAX_MIN = cls.max_min_from_time(data, fill_type)

        def fill_group(group_df: pd.DataFrame) -> pd.DataFrame:
            group_key = group_df.iloc[0][cls.other_ids].values
            if fill_type != "all":
                group_key = group_key[0] if len(group_key) == 1 else tuple(group_key)
            group_dict = MAX_MIN if fill_type == "all" else MAX_MIN[group_key]

            intervals = [
                f"{group_dict['start']['min']}/{group_dict['end']['min']}",
                f"{group_dict['start']['max']}/{group_dict['end']['max']}",
            ]
            for interval in intervals:
                if interval not in group_df[cls.time_id].values:
                    empty_row = group_df.iloc[0].copy()
                    empty_row[cls.time_id] = interval
                    empty_row[cls.measures] = None
                    group_df = group_df.append(  # type: ignore[operator]
                        empty_row, ignore_index=True
                    )
            start_group_df = group_df.copy()
            start_group_df[cls.time_id] = start_group_df[cls.time_id].apply(
                lambda x: x.split("/")[0]
            )
            end_group_df = group_df.copy()
            end_group_df[cls.time_id] = end_group_df[cls.time_id].apply(lambda x: x.split("/")[1])
            start_filled = cls.date_filler(start_group_df, fill_type, frequency)
            end_filled = cls.date_filler(end_group_df, fill_type, frequency)
            start_filled[cls.time_id] = start_filled[cls.time_id].str.cat(
                end_filled[cls.time_id], sep="/"
            )
            return start_filled

        filled_data = [fill_group(group_df) for _, group_df in data.groupby(cls.other_ids)]
        return (
            pd.concat(filled_data, ignore_index=True)
            .sort_values(by=cls.other_ids + [cls.time_id])
            .drop_duplicates()
        )


class Time_Shift(Binary):
    op = TIMESHIFT

    @classmethod
    def evaluate(cls, operand: Dataset, shift_value: Any) -> Dataset:
        result = cls.validate(operand, shift_value)
        result.data = operand.data.copy() if operand.data is not None else pd.DataFrame()
        shift_value = int(shift_value.value)
        cls.time_id = cls._get_time_id(result)

        data_type: Any = result.components[cls.time_id].data_type

        if data_type == Date:
            freq = cls.find_min_frequency(
                cls.get_frequencies(
                    result.data[cls.time_id].map(cls.parse_date, na_action="ignore")
                )
            )
            result.data[cls.time_id] = cls.shift_dates(result.data[cls.time_id], shift_value, freq)
        elif data_type == Time:
            freq = cls.get_frequency_from_time(result.data[cls.time_id].iloc[0])
            result.data[cls.time_id] = result.data[cls.time_id].apply(
                lambda x: cls.shift_interval(x, shift_value, freq)
            )
        elif data_type == TimePeriod:
            periods = result.data[cls.time_id].apply(cls._get_period).unique()
            result.data[cls.time_id] = result.data[cls.time_id].apply(
                lambda x: cls.shift_period(x, shift_value)
            )
            if len(periods) == 1 and periods[0] == "A":
                result.data[cls.time_id] = result.data[cls.time_id].astype(int)
        else:
            raise SemanticError("1-1-19-2", op=cls.op)
        return result

    @classmethod
    def validate(cls, operand: Dataset, shift_value: str) -> Dataset:
        if cls._get_time_id(operand) is None:
            raise SemanticError("1-1-19-8", op=cls.op, comp_type="time dataset")
        return Dataset(name="result", components=operand.components.copy(), data=None)

    @classmethod
    def shift_dates(cls, dates: Any, shift_value: int, frequency: str) -> Any:
        dates = pd.to_datetime(dates)
        if frequency == "D":
            return dates + pd.to_timedelta(shift_value, unit="D")
        elif frequency == "W":
            return dates + pd.to_timedelta(shift_value, unit="W")
        elif frequency == "Y":
            return dates + pd.DateOffset(years=shift_value)
        elif frequency in ["M", "Q", "S"]:
            return dates + pd.DateOffset(months=shift_value)
        raise SemanticError("2-1-19-2", period=frequency)

    @classmethod
    def shift_period(
        cls, period_str: str, shift_value: int, frequency: Optional[int] = None
    ) -> str:
        period_type = cls._get_period(period_str)

        if period_type == "A":
            return str(int(period_str) + shift_value)

        if frequency:
            shift_value *= frequency

        tp_value = TimePeriodHandler(period_str)
        year, period, value = (
            tp_value.year,
            tp_value.period_indicator,
            tp_value.period_number + shift_value,
        )
        period_limit = cls.YEAR_TO_PERIOD[period]

        if value <= 0:
            year -= 1
            value += period_limit
        elif value > period_limit:
            year += (value - 1) // period_limit
            value = (value - 1) % period_limit + 1

        return f"{year}-{period}{value}"

    @classmethod
    def shift_interval(cls, interval: str, shift_value: Any, frequency: str) -> str:
        start_date, end_date = interval.split("/")
        start_date = cls.shift_dates(start_date, shift_value, frequency)
        end_date = cls.shift_dates(end_date, shift_value, frequency)
        return f"{start_date}/{end_date}"


class Time_Aggregation(Time):
    op = TIME_AGG

    @classmethod
    def _check_duration(cls, value: str) -> None:
        if value not in PERIOD_IND_MAPPING:
            raise SemanticError("1-1-19-3", op=cls.op, param="duration")

    @classmethod
    def _check_params(cls, period_from: Optional[str], period_to: str) -> None:
        cls._check_duration(period_to)
        if period_from is not None:
            cls._check_duration(period_from)
            if PERIOD_IND_MAPPING[period_to] <= PERIOD_IND_MAPPING[period_from]:
                # OPERATORS_TIMEOPERATORS.19
                raise SemanticError("1-1-19-4", op=cls.op, value_1=period_from, value_2=period_to)

    @classmethod
    def dataset_validation(
        cls, operand: Dataset, period_from: Optional[str], period_to: str, conf: str
    ) -> Dataset:
        # TODO: Review with VTL TF as this makes no sense

        count_time_types = 0
        for measure in operand.get_measures():
            if measure.data_type in cls.TIME_DATA_TYPES:
                count_time_types += 1
                if measure.data_type == TimePeriod and period_to == "D":
                    raise SemanticError("1-1-19-5", op=cls.op)
                if measure.data_type == TimeInterval:
                    raise SemanticError("1-1-19-6", op=cls.op, comp=measure.name)

        count_time_types = 0
        for id_ in operand.get_identifiers():
            if id_.data_type in cls.TIME_DATA_TYPES:
                count_time_types += 1
        if count_time_types != 1:
            raise SemanticError(
                "1-1-19-9",
                op=cls.op,
                comp_type="dataset",
                param="single time identifier",
            )

        if count_time_types != 1:
            raise SemanticError(
                "1-1-19-9", op=cls.op, comp_type="dataset", param="single time measure"
            )

        result_components = {
            comp.name: comp
            for comp in operand.components.values()
            if comp.role in [Role.IDENTIFIER, Role.MEASURE]
        }

        return Dataset(name=operand.name, components=result_components, data=None)

    @classmethod
    def component_validation(
        cls,
        operand: DataComponent,
        period_from: Optional[str],
        period_to: str,
        conf: str,
    ) -> DataComponent:
        if operand.data_type not in cls.TIME_DATA_TYPES:
            raise SemanticError("1-1-19-8", op=cls.op, comp_type="time component")
        if operand.data_type == TimePeriod and period_to == "D":
            raise SemanticError("1-1-19-5", op=cls.op)
        if operand.data_type == TimeInterval:
            raise SemanticError("1-1-19-6", op=cls.op, comp=operand.name)

        return DataComponent(name=operand.name, data_type=operand.data_type, data=None)

    @classmethod
    def scalar_validation(
        cls, operand: Scalar, period_from: Optional[str], period_to: str, conf: str
    ) -> Scalar:
        if operand.data_type not in cls.TIME_DATA_TYPES:
            raise SemanticError("1-1-19-8", op=cls.op, comp_type="time scalar")

        return Scalar(name=operand.name, data_type=operand.data_type, value=None)

    @classmethod
    def _execute_time_aggregation(
        cls,
        value: str,
        data_type: Type[ScalarType],
        period_from: Optional[str],
        period_to: str,
        conf: str,
    ) -> str:
        if data_type == TimePeriod:  # Time period
            return _time_period_access(value, period_to)

        elif data_type == Date:
            start = conf == "first"
            # Date
            if period_to == "D":
                return value
            return _date_access(value, period_to, start)
        else:
            raise NotImplementedError

    @classmethod
    def dataset_evaluation(
        cls, operand: Dataset, period_from: Optional[str], period_to: str, conf: str
    ) -> Dataset:
        result = cls.dataset_validation(operand, period_from, period_to, conf)
        result.data = operand.data.copy() if operand.data is not None else pd.DataFrame()
        time_measure = [m for m in operand.get_measures() if m.data_type in cls.TIME_DATA_TYPES][0]
        result.data[time_measure.name] = result.data[time_measure.name].map(
            lambda x: cls._execute_time_aggregation(
                x, time_measure.data_type, period_from, period_to, conf
            ),
            na_action="ignore",
        )

        return result

    @classmethod
    def component_evaluation(
        cls,
        operand: DataComponent,
        period_from: Optional[str],
        period_to: str,
        conf: str,
    ) -> DataComponent:
        result = cls.component_validation(operand, period_from, period_to, conf)
        if operand.data is not None:
            result.data = operand.data.map(
                lambda x: cls._execute_time_aggregation(
                    x, operand.data_type, period_from, period_to, conf
                ),
                na_action="ignore",
            )
        return result

    @classmethod
    def scalar_evaluation(
        cls, operand: Scalar, period_from: Optional[str], period_to: str, conf: str
    ) -> Scalar:
        result = cls.scalar_validation(operand, period_from, period_to, conf)
        result.value = cls._execute_time_aggregation(
            operand.value, operand.data_type, period_from, period_to, conf
        )
        return result

    @classmethod
    def validate(
        cls,
        operand: Union[Dataset, DataComponent, Scalar],
        period_from: Optional[str],
        period_to: str,
        conf: str,
    ) -> Union[Dataset, DataComponent, Scalar]:
        cls._check_params(period_from, period_to)
        if isinstance(operand, Dataset):
            return cls.dataset_validation(operand, period_from, period_to, conf)
        elif isinstance(operand, DataComponent):
            return cls.component_validation(operand, period_from, period_to, conf)
        else:
            return cls.scalar_validation(operand, period_from, period_to, conf)

    @classmethod
    def evaluate(
        cls,
        operand: Union[Dataset, DataComponent, Scalar],
        period_from: Optional[str],
        period_to: str,
        conf: str,
    ) -> Union[Dataset, DataComponent, Scalar]:
        cls._check_params(period_from, period_to)
        if isinstance(operand, Dataset):
            return cls.dataset_evaluation(operand, period_from, period_to, conf)
        elif isinstance(operand, DataComponent):
            return cls.component_evaluation(operand, period_from, period_to, conf)
        else:
            return cls.scalar_evaluation(operand, period_from, period_to, conf)


def _time_period_access(v: Any, to_param: str) -> Any:
    v = TimePeriodHandler(v)
    if v.period_indicator == to_param:
        return str(v)
    v.change_indicator(to_param)
    return str(v)


def _date_access(v: str, to_param: str, start: bool) -> Any:
    period_value = date_to_period(date.fromisoformat(v), to_param)
    if start:
        return period_value.start_date()
    return period_value.end_date()


class Current_Date(Time):
    @classmethod
    def validate(cls) -> Scalar:
        return Scalar(name="current_date", data_type=Date, value=None)

    @classmethod
    def evaluate(cls) -> Scalar:
        result = cls.validate()
        result.value = date.today().isoformat()
        return result


class SimpleBinaryTime(Operators.Binary):
    @classmethod
    def validate_type_compatibility(cls, left: Any, right: Any) -> bool:
        if left == Date and right == TimePeriod:
            return False

        if left == TimePeriod and right == Date:
            return False

        return not (left == TimePeriod and right == Date)

    @classmethod
    def validate(
        cls,
        left_operand: Union[Dataset, DataComponent, Scalar],
        right_operand: Union[Dataset, DataComponent, Scalar],
    ) -> Union[Dataset, DataComponent, Scalar]:
        if isinstance(left_operand, Dataset) or isinstance(right_operand, Dataset):
            raise SemanticError("1-1-19-8", op=cls.op, comp_type="time dataset")
        if not cls.validate_type_compatibility(left_operand.data_type, right_operand.data_type):
            raise SemanticError(
                "1-1-1-2",
                type_1=left_operand.data_type,
                type_2=right_operand.data_type,
                type_check=cls.type_to_check,
            )
        return super().validate(left_operand, right_operand)

    @classmethod
    def evaluate(
        cls,
        left_operand: Union[Dataset, DataComponent, Scalar],
        right_operand: Union[Dataset, DataComponent, Scalar],
    ) -> Union[Dataset, DataComponent, Scalar]:
        if isinstance(left_operand, Dataset) or isinstance(right_operand, Dataset):
            raise SemanticError("1-1-19-8", op=cls.op, comp_type="time dataset")
        else:
            cls.validate(left_operand, right_operand)
            return super().evaluate(left_operand, right_operand)


class Date_Diff(SimpleBinaryTime):
    op = DATEDIFF
    type_to_check = TimeInterval
    return_type = Integer

    @classmethod
    def py_op(cls, x: Any, y: Any) -> int:
        if (x.count("/") >= 1) or (y.count("/") >= 1):
            raise SemanticError("1-1-19-8", op=cls.op, comp_type="time dataset")

        if x.count("-") == 2:
            fecha1 = datetime.strptime(x, "%Y-%m-%d").date()
        else:
            fecha1 = TimePeriodHandler(x).end_date(as_date=True)  # type: ignore[assignment]

        if y.count("-") == 2:
            fecha2 = datetime.strptime(y, "%Y-%m-%d").date()
        else:
            fecha2 = TimePeriodHandler(y).end_date(as_date=True)  # type: ignore[assignment]

        return abs((fecha2 - fecha1).days)


class Date_Add(Parametrized):
    op = DATE_ADD

    @classmethod
    def validate(
        cls, operand: Union[Scalar, DataComponent, Dataset], param_list: List[Scalar]
    ) -> Union[Scalar, DataComponent, Dataset]:
        expected_types = [Integer, String]
        for i, param in enumerate(param_list):
            error = (
                12
                if not isinstance(param, Scalar)  # type: ignore[redundant-expr]
                else 13
                if (param.data_type != expected_types[i])
                else None
            )
            if error is not None:
                raise SemanticError(
                    f"2-1-19-{error}",
                    op=cls.op,
                    type=(param.__class__.__name__ if error == 12 else param.data_type.__name__),
                    name="shiftNumber" if error == 12 else "periodInd",
                    expected="Scalar" if error == 12 else expected_types[i].__name__,
                )

        if isinstance(operand, (Scalar, DataComponent)) and operand.data_type not in [
            Date,
            TimePeriod,
        ]:
            unary_implicit_promotion(operand.data_type, Date)

        if isinstance(operand, Scalar):
            return Scalar(name=operand.name, data_type=operand.data_type, value=None)
        if isinstance(operand, DataComponent):
            return DataComponent(name=operand.name, data_type=operand.data_type, data=None)

        if all(comp.data_type not in [Date, TimePeriod] for comp in operand.components.values()):
            raise SemanticError("2-1-19-14", op=cls.op, name=operand.name)
        return Dataset(name="result", components=operand.components.copy(), data=None)

    @classmethod
    def evaluate(
        cls, operand: Union[Scalar, DataComponent, Dataset], param_list: List[Scalar]
    ) -> Union[Scalar, DataComponent, Dataset]:
        result = cls.validate(operand, param_list)
        shift, period = param_list[0].value, param_list[1].value
        is_tp = isinstance(operand, (Scalar, DataComponent)) and operand.data_type == TimePeriod

        if isinstance(result, Scalar) and isinstance(operand, Scalar) and operand.value is not None:
            result.value = cls.py_op(operand.value, shift, period, is_tp)
        elif (
            isinstance(result, DataComponent)
            and isinstance(operand, DataComponent)
            and operand.data is not None
        ):
            result.data = operand.data.map(
                lambda x: cls.py_op(x, shift, period, is_tp), na_action="ignore"
            )
        elif (
            isinstance(result, Dataset)
            and isinstance(operand, Dataset)
            and operand.data is not None
        ):
            result.data = operand.data.copy()
            for measure in operand.get_measures():
                if measure.data_type in [Date, TimePeriod]:
                    result.data[measure.name] = result.data[measure.name].map(
                        lambda x: cls.py_op(str(x), shift, period, measure.data_type == TimePeriod),
                        na_action="ignore",
                    )
                    measure.data_type = Date

        if isinstance(result, (Scalar, DataComponent)):
            result.data_type = Date
        return result

    @classmethod
    def py_op(cls, date_str: str, shift: int, period: str, is_tp: bool = False) -> str:
        if is_tp:
            tp_value = TimePeriodHandler(date_str)
            date = period_to_date(tp_value.year, tp_value.period_indicator, tp_value.period_number)
        else:
            date = datetime.strptime(date_str, "%Y-%m-%d")

        if period in ["D", "W"]:
            days_shift = shift * (7 if period == "W" else 1)
            return (date + timedelta(days=days_shift)).strftime("%Y-%m-%d")

        month_shift = {"M": 1, "Q": 3, "S": 6, "A": 12}[period] * shift
        new_year = date.year + (date.month - 1 + month_shift) // 12
        new_month = (date.month - 1 + month_shift) % 12 + 1
        last_day = (datetime(new_year, new_month % 12 + 1, 1) - timedelta(days=1)).day
        return date.replace(year=new_year, month=new_month, day=min(date.day, last_day)).strftime(
            "%Y-%m-%d"
        )


class SimpleUnaryTime(Operators.Unary):
    @classmethod
    def validate(
        cls, operand: Union[Dataset, DataComponent, Scalar]
    ) -> Union[Dataset, DataComponent, Scalar]:
        if isinstance(operand, Dataset):
            raise SemanticError("1-1-19-8", op=cls.op, comp_type="time dataset")

        # Limit the operand to Date and TimePeriod (cannot be implemented with type_to_check)
        if operand.data_type == TimeInterval or operand.data_type not in (
            Date,
            TimePeriod,
            Duration,
        ):
            raise SemanticError("1-1-19-10", op=cls.op)

        return super().validate(operand)

    @classmethod
    def evaluate(
        cls, operand: Union[Dataset, DataComponent, Scalar]
    ) -> Union[Dataset, DataComponent, Scalar]:
        cls.validate(operand)
        return super().evaluate(operand)


class Year(SimpleUnaryTime):
    op = YEAR

    @classmethod
    def py_op(cls, value: str) -> int:
        return int(value[:4])

    return_type = Integer


class Month(SimpleUnaryTime):
    op = MONTH
    return_type = Integer

    @classmethod
    def py_op(cls, value: str) -> int:
        if value.count("-") == 2:
            return date.fromisoformat(value).month

        result = TimePeriodHandler(value).start_date(as_date=True)
        return result.month  # type: ignore[union-attr]


class Day_of_Month(SimpleUnaryTime):
    op = DAYOFMONTH
    return_type = Integer

    @classmethod
    def py_op(cls, value: str) -> int:
        if value.count("-") == 2:
            return date.fromisoformat(value).day

        result = TimePeriodHandler(value).end_date(as_date=True)
        return result.day  # type: ignore[union-attr]


class Day_of_Year(SimpleUnaryTime):
    op = DAYOFYEAR
    return_type = Integer

    @classmethod
    def py_op(cls, value: str) -> int:
        if value.count("-") == 2:
            day_y = datetime.strptime(value, "%Y-%m-%d")
            return day_y.timetuple().tm_yday

        result = TimePeriodHandler(value).end_date(as_date=True)
        datetime_value = datetime(
            year=result.year,  # type: ignore[union-attr]
            month=result.month,  # type: ignore[union-attr]
            day=result.day,  # type: ignore[union-attr]
        )
        return datetime_value.timetuple().tm_yday


class Day_to_Year(Operators.Unary):
    op = DAYTOYEAR
    return_type = Duration

    @classmethod
    def py_op(cls, value: int) -> str:
        if value < 0:
            raise SemanticError("2-1-19-16", op=cls.op)
        years = 0
        days_remaining = value
        if value >= 365:
            years = value // 365
            days_remaining = value % 365
        return f"P{int(years)}Y{int(days_remaining)}D"


class Day_to_Month(Operators.Unary):
    op = DAYTOMONTH
    return_type = Duration

    @classmethod
    def py_op(cls, value: int) -> str:
        if value < 0:
            raise SemanticError("2-1-19-16", op=cls.op)
        months = 0
        days_remaining = value
        if value >= 30:
            months = value // 30
            days_remaining = value % 30
        return f"P{int(months)}M{int(days_remaining)}D"


class Year_to_Day(Operators.Unary):
    op = YEARTODAY
    return_type = Integer

    @classmethod
    def py_op(cls, value: str) -> int:
        days = Duration.to_days(value)
        return days


class Month_to_Day(Operators.Unary):
    op = MONTHTODAY
    return_type = Integer

    @classmethod
    def py_op(cls, value: str) -> int:
        days = Duration.to_days(value)
        return days
