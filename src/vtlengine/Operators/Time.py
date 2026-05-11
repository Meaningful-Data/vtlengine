import re
from datetime import date
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
from vtlengine.DataTypes._time_checking import parse_date_value
from vtlengine.DataTypes.TimeHandling import (
    PERIOD_IND_MAPPING,
    TimePeriodHandler,
    date_to_period,
    generate_period_range,
    max_periods_in_year,
)
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import Component, DataComponent, Dataset, Role, Scalar
from vtlengine.Utils.__Virtual_Assets import VirtualCounter


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
        if reference_id is None:
            raise SemanticError("1-1-19-1", op=cls.op, data_type="Time_Period", comp="identifier")
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
        return parse_date_value(date_str)


class Unary(Time):
    @classmethod
    def validate(cls, operand: Any) -> Any:
        dataset_name = VirtualCounter._new_ds_name()
        if not isinstance(operand, Dataset):
            raise SemanticError("1-1-19-8", op=cls.op, comp_type="time dataset")
        if cls._get_time_id(operand) is None:
            raise SemanticError("1-1-19-8", op=cls.op, comp_type="time dataset")
        operand.data = cls.sort_by_time(operand)
        return Dataset(name=dataset_name, components=operand.components.copy(), data=None)


class Binary(Time):
    pass


class Parameterized(Time):
    pass


class Period_indicator(Unary):
    op = PERIOD_INDICATOR

    @classmethod
    def validate(cls, operand: Any) -> Any:
        dataset_name = VirtualCounter._new_ds_name()
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
            return Dataset(name=dataset_name, components=result_components, data=None)
        # DataComponent and Scalar validation
        if operand.data_type != TimePeriod:
            raise SemanticError("1-1-19-8", op=cls.op, comp_type="time period component")
        if isinstance(operand, DataComponent):
            return DataComponent(
                name=operand.name, data_type=Duration, data=None, nullable=operand.nullable
            )
        return Scalar(name=operand.name, data_type=Duration, value=None)


class Parametrized(Time):
    @classmethod
    def validate(cls, operand: Any, param: Any) -> Any:
        pass


class Flow_to_stock(Unary):
    pass


class Stock_to_flow(Unary):
    pass


class Fill_time_series(Binary):
    op = FILL_TIME_SERIES

    @classmethod
    def validate(cls, operand: Dataset, fill_type: str) -> Dataset:
        dataset_name = VirtualCounter._new_ds_name()
        if not isinstance(operand, Dataset):
            raise SemanticError("1-1-19-8", op=cls.op, comp_type="time dataset")
        cls.time_id = cls._get_time_id(operand)
        cls.other_ids = [id.name for id in operand.get_identifiers() if id.name != cls.time_id]
        cls.measures = operand.get_measures_names()
        if cls.time_id is None:
            raise SemanticError("1-1-19-8", op=cls.op, comp_type="time dataset")
        if fill_type not in ["all", "single"]:
            fill_type = "all"
        return Dataset(name=dataset_name, components=operand.components.copy(), data=None)

    @classmethod
    def fill_periods(cls, data: pd.DataFrame, fill_type: str) -> pd.DataFrame:
        # Parse each time_id value once and reuse throughout
        data = data.copy()
        tp_parsed = data[cls.time_id].map(lambda x: TimePeriodHandler(x), na_action="ignore")
        data[cls.time_id] = tp_parsed.map(str, na_action="ignore")
        data = data.assign(_freq=tp_parsed.map(lambda x: x.period_indicator, na_action="ignore"))

        # Determine global year range (for "all" mode)
        if fill_type == "all":
            global_min_year: int = tp_parsed.map(lambda x: x.year).min()
            global_max_year: int = tp_parsed.map(lambda x: x.year).max()

        # Group by other_ids + frequency and fill missing periods
        filled_rows: List[Dict[str, Any]] = []
        non_id_cols = [
            c for c in data.columns if c not in cls.other_ids and c != cls.time_id and c != "_freq"
        ]

        for group_key, group_df in data.groupby(cls.other_ids + ["_freq"]):
            if isinstance(group_key, tuple):
                freq = group_key[-1]
                other_id_values = group_key[:-1]
            else:
                freq = group_key
                other_id_values = ()

            group_tp = tp_parsed.loc[group_df.index]

            # Determine range start/end
            if fill_type == "all":
                if freq == "A":
                    start = TimePeriodHandler(f"{global_min_year}A")
                    end = TimePeriodHandler(f"{global_max_year}A")
                else:
                    max_p = max_periods_in_year(freq, global_max_year)
                    start = TimePeriodHandler(f"{global_min_year}-{freq}1")
                    end = TimePeriodHandler(f"{global_max_year}-{freq}{max_p}")
            else:  # single
                sorted_tp = sorted(group_tp.tolist(), key=lambda x: (x.year, x.period_number))
                start, end = sorted_tp[0], sorted_tp[-1]

            # Generate all expected periods and find missing ones
            expected = generate_period_range(start, end)
            existing = set(group_df[cls.time_id].tolist())

            # Build other_ids dict for fill rows
            other_vals: Dict[str, Any] = {}
            if cls.other_ids:
                for i, col in enumerate(cls.other_ids):
                    other_vals[col] = other_id_values[i]

            for tp in expected:
                tp_str = str(tp)
                if tp_str not in existing:
                    row: Dict[str, Any] = {**other_vals, cls.time_id: tp_str}
                    for col in non_id_cols:
                        row[col] = None
                    filled_rows.append(row)

        # Combine and return
        data = data.drop(columns=["_freq"])
        if filled_rows:
            fill_df = pd.DataFrame(filled_rows)
            result = pd.concat([data, fill_df], ignore_index=True)
        else:
            result = data
        result[cls.time_id] = result[cls.time_id].astype("string[pyarrow]")
        return result.sort_values(by=cls.other_ids + [cls.time_id]).reset_index(drop=True)


class Time_Shift(Binary):
    op = TIMESHIFT

    @classmethod
    def validate(cls, operand: Dataset, shift_value: str) -> Dataset:
        dataset_name = VirtualCounter._new_ds_name()
        if cls._get_time_id(operand) is None:
            raise SemanticError("1-1-19-8", op=cls.op, comp_type="time dataset")
        return Dataset(name=dataset_name, components=operand.components.copy(), data=None)

    @classmethod
    def shift_period(
        cls, period_str: str, shift_value: int, frequency: Optional[int] = None
    ) -> str:
        period_type = cls._get_period(period_str)

        if period_type == "A":
            tp = TimePeriodHandler(period_str)
            tp.year += shift_value
            return str(tp)

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

        return str(TimePeriodHandler(f"{year}-{period}{value}"))


class Time_Aggregation(Time):
    op = TIME_AGG

    @classmethod
    def _execute_without_operand(
        cls,
        aggregation_dataset: Dataset,
        period_from: Optional[str],
        period_to: str,
        conf: Optional[str],
    ) -> Any:
        time_id_name = cls._get_time_id(aggregation_dataset)
        time_id = aggregation_dataset.components[time_id_name]
        if time_id.data_type == TimePeriod and period_to == "D":
            raise SemanticError("1-1-19-5", op=cls.op)
        if time_id.data_type == TimeInterval:
            raise SemanticError("1-1-19-6", op=cls.op, comp=time_id.name)
        if time_id.data_type == Date and conf is None:
            raise SemanticError("1-1-19-11")

        if aggregation_dataset.data is not None:
            series_data = aggregation_dataset.data[time_id_name].map(
                lambda x: cls._execute_time_aggregation(
                    x, time_id.data_type, period_from, period_to, conf
                ),
                na_action="ignore",
            )
        else:
            series_data = None  # For semantic
        result = DataComponent(
            name=time_id_name,
            data=series_data,
            data_type=TimePeriod,
            role=Role.IDENTIFIER,
            nullable=False,
        )

        return result

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
        cls, operand: Dataset, period_from: Optional[str], period_to: str, conf: Optional[str]
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
                if measure.data_type == Date and conf is None:
                    raise SemanticError("1-1-19-11")

        if count_time_types != 1:
            raise SemanticError(
                "1-1-19-9", op=cls.op, comp_type="dataset", param="single time measure"
            )

        result_components = {
            comp.name: comp
            for comp in operand.components.values()
            if comp.role in [Role.IDENTIFIER, Role.MEASURE, Role.VIRAL_ATTRIBUTE]
        }

        return Dataset(name=operand.name, components=result_components, data=None)

    @classmethod
    def component_validation(
        cls,
        operand: DataComponent,
        period_from: Optional[str],
        period_to: str,
        conf: Optional[str],
    ) -> DataComponent:
        if operand.data_type not in cls.TIME_DATA_TYPES:
            raise SemanticError("1-1-19-8", op=cls.op, comp_type="time component")
        if operand.data_type == TimePeriod and period_to == "D":
            raise SemanticError("1-1-19-5", op=cls.op)
        if operand.data_type == TimeInterval:
            raise SemanticError("1-1-19-6", op=cls.op, comp=operand.name)
        if operand.data_type == Date and conf is None:
            raise SemanticError("1-1-19-11")

        return DataComponent(
            name=operand.name, data_type=operand.data_type, data=None, nullable=operand.nullable
        )

    @classmethod
    def scalar_validation(
        cls, operand: Scalar, period_from: Optional[str], period_to: str, conf: Optional[str]
    ) -> Scalar:
        if operand.data_type not in cls.TIME_DATA_TYPES:
            raise SemanticError("1-1-19-8", op=cls.op, comp_type="time scalar")
        if operand.data_type == TimePeriod and period_to == "D":
            raise SemanticError("1-1-19-5", op=cls.op)
        if operand.data_type == TimeInterval:
            raise SemanticError("1-1-19-6", op=cls.op, comp=operand.name)
        if operand.data_type == Date and conf is None:
            raise SemanticError("1-1-19-11")

        return Scalar(name=operand.name, data_type=operand.data_type, value=None)

    @classmethod
    def _execute_time_aggregation(
        cls,
        value: str,
        data_type: Type[ScalarType],
        period_from: Optional[str],
        period_to: str,
        conf: Optional[str],
    ) -> str:
        if data_type == TimePeriod:  # Time period
            return _time_period_access(value, period_to)

        elif data_type == Date:
            if conf is None:
                raise SemanticError("1-1-19-11")
            start = conf == "first"
            # Date
            if period_to == "D":
                return value
            return _date_access(value, period_to, start)
        else:
            raise NotImplementedError

    @classmethod
    def validate(
        cls,
        operand: Union[Dataset, DataComponent, Scalar],
        period_from: Optional[str],
        period_to: str,
        conf: Optional[str],
    ) -> Union[Dataset, DataComponent, Scalar]:
        cls._check_params(period_from, period_to)
        if isinstance(operand, Dataset):
            return cls.dataset_validation(operand, period_from, period_to, conf)
        elif isinstance(operand, DataComponent):
            return cls.component_validation(operand, period_from, period_to, conf)
        else:
            return cls.scalar_validation(operand, period_from, period_to, conf)


def _time_period_access(v: Any, to_param: str) -> Any:
    v = TimePeriodHandler(v)
    if v.period_indicator == to_param:
        return str(v)
    v.change_indicator(to_param)
    return str(v)


def _date_access(v: str, to_param: str, start: bool) -> Any:
    period_value = date_to_period(parse_date_value(v), to_param)
    if start:
        return period_value.start_date()
    return period_value.end_date()


class Current_Date(Time):
    @classmethod
    def validate(cls) -> Scalar:
        return Scalar(name="current_date", data_type=Date, value=None)


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


class Date_Diff(SimpleBinaryTime):
    op = DATEDIFF
    type_to_check = TimeInterval
    return_type = Integer


class Date_Add(Parametrized):
    op = DATE_ADD

    @classmethod
    def validate(
        cls, operand: Union[Scalar, DataComponent, Dataset], param_list: List[Scalar]
    ) -> Union[Scalar, DataComponent, Dataset]:
        dataset_name = VirtualCounter._new_ds_name()
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
            return DataComponent(
                name=operand.name, data_type=operand.data_type, data=None, nullable=operand.nullable
            )

        if all(comp.data_type not in [Date, TimePeriod] for comp in operand.components.values()):
            raise SemanticError("2-1-19-14", op=cls.op, name=operand.name)
        return Dataset(name=dataset_name, components=operand.components.copy(), data=None)


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


class Year(SimpleUnaryTime):
    op = YEAR
    return_type = Integer


class Month(SimpleUnaryTime):
    op = MONTH
    return_type = Integer


class Day_of_Month(SimpleUnaryTime):
    op = DAYOFMONTH
    return_type = Integer


class Day_of_Year(SimpleUnaryTime):
    op = DAYOFYEAR
    return_type = Integer


class Day_to_Year(Operators.Unary):
    op = DAYTOYEAR
    return_type = String


class Day_to_Month(Operators.Unary):
    op = DAYTOMONTH
    return_type = String


class Year_to_Day(Operators.Unary):
    op = YEARTODAY
    return_type = Integer
    _duration_pattern = re.compile(r"^P(?=\d)(\d+Y)?(\d+D)?$")


class Month_to_Day(Operators.Unary):
    op = MONTHTODAY
    return_type = Integer
    _duration_pattern = re.compile(r"^P(?=\d)(\d+M)?(\d+D)?$")
