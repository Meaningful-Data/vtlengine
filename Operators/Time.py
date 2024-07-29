from datetime import datetime

import pandas as pd
import re

import DataTypes
import Operators
from Model import Dataset, DataComponent, Scalar, Component, Role


class Time(Operators.Operator):
    TIME_DATA_TYPES = [DataTypes.Time, DataTypes.Date, DataTypes.Time_Period, DataTypes.Time_Interval]

    PERIODS = {
        'D': lambda x: datetime.strptime(x, "%Y-%m-%d"),
        'M': lambda x: datetime.strptime(x, "%Y-%d") or datetime.strptime(x, "%Y-M%d"),
        'Q': lambda x: datetime.strptime(x, "%Y-Q%d"),
        'S': lambda x: datetime.strptime(x, "%Y-S%d"),
        'A': lambda x: datetime.strptime(x, "%Y")
    }

    py_op = None

    @classmethod
    def get_time_id(cls, operand: Dataset) -> str:
        reference_id = None
        for id in operand.get_identifiers():
            if id.data_type in cls.TIME_DATA_TYPES:
                if reference_id is not None:
                    raise ValueError("FlowToStock can only be applied to a time dataset")
                reference_id = id.name
        return reference_id

    @classmethod
    def order_by_time(cls, operand: Dataset) -> Dataset:
        time_id = cls.get_time_id(operand)
        if time_id is None:
            return operand
        ids = [id.name for id in operand.get_identifiers() if id.name != time_id]
        ids.append(time_id)
        return operand.data.sort_values(by=ids).reset_index(drop=True)

    @classmethod
    def get_period_from_list(cls, series: pd.Series) -> pd.Series:
        result = series.copy()
        for i in range(0, len(result)):
            result[i] = cls.get_period(result[i])
        return result

    @classmethod
    def get_period(cls, value) -> str | None:
        for period, func in cls.PERIODS.items():
            try:
                dt = func(f'{value}')
                return period
            except ValueError:
                continue
        return None


class Unary(Time):

    @classmethod
    def evaluate(cls, operand: Dataset) -> Dataset:
        cls.validate(operand)
        if len(operand.data) < 2:
            return operand
        reference_id = cls.get_time_id(operand)
        others_ids = [id.name for id in operand.get_identifiers() if id.name != reference_id]
        operand.data = cls.order_by_time(operand)
        last_from_period = {}
        for i in range(1, len(operand.data[reference_id])):
            current_period = cls.get_period(operand.data[reference_id][i])
            previous_period = cls.get_period(operand.data[reference_id][i - 1])
            last_from_period[previous_period] = i - 1
            if current_period not in last_from_period.keys():
                last_from_period[current_period] = i
            else:
                j = last_from_period[current_period]
                for measure in operand.get_measures_names():
                    if all(operand.data[id][i] == operand.data[id][j] for id in others_ids):
                        operand.data[measure][i] = cls.py_op(operand.data[measure][i], operand.data[measure][j])
        return operand

    @classmethod
    def validate(cls, operand: Dataset) -> None:
        if not isinstance(operand, Dataset):
            raise TypeError("FlowToStock can only be applied to a time dataset")
        if cls.get_time_id(operand) is None:
            raise ValueError("FlowToStock can only be applied to a time dataset")


class Binary(Time):
    pass


class Period_indicator(Unary):

    @classmethod
    def evaluate(cls, operand: Dataset | DataComponent | Scalar | str) -> Dataset | DataComponent | Scalar | str:
        cls.validate(operand)
        if isinstance(operand, str):
            return cls.get_period(operand)
        if isinstance(operand, Scalar):
            return Scalar(name='result', data_type=DataTypes.Duration, value=cls.get_period(operand.value))
        if isinstance(operand, DataComponent):
            return DataComponent(name='result',
                                 data_type=DataTypes.Duration,
                                 data=cls.get_period_from_list(operand.data))
        data = cls.get_period_from_list(operand.data[cls.get_time_id(operand)])
        operand.data = operand.data.drop(columns=operand.get_measures_names())
        operand.data['duration_var'] = data
        for measure in operand.get_measures_names():
            operand.components.pop(measure)
        operand.components['duration_var'] = Component(name='duration_var', data_type=DataTypes.Duration,
                                                       role=Role.MEASURE, nullable=True)
        return Dataset(name='result',
                       components=operand.components,
                       data=operand.data)

    @classmethod
    def validate(cls, operand: Dataset | DataComponent | Scalar | str) -> None:
        if isinstance(operand, str):
            return
        if isinstance(operand, Dataset):
            time_id = cls.get_time_id(operand)
            if time_id is None or operand.components[time_id].data_type != DataTypes.Time_Period:
                raise ValueError("PeriodIndicator can only be applied to a time dataset")
        else:
            if operand.data_type != DataTypes.Time_Period:
                raise ValueError("PeriodIndicator can only be applied to a time dataset")


class Flow_to_stock(Unary):

    py_op = lambda x, y: x + y

class Stock_to_flow(Unary):

    py_op = lambda x, y: x - y
