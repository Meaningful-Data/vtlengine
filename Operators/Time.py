import DataTypes
import Operators
from Model import Dataset, DataComponent, Scalar, Component, Role

import re
import pandas as pd
from datetime import datetime
from itertools import combinations
from dateutil.relativedelta import relativedelta
from typing import Optional


class Time(Operators.Operator):

    periods = None
    time_id = None
    other_ids = None
    measures = None

    TIME_DATA_TYPES = [DataTypes.Date, DataTypes.TimePeriod, DataTypes.TimeInterval]

    PERIOD_PATTERN = r'^(\d{1,4})(?:-([SQMD])(\d{1,3}))?$'
    TIME_PATTERN = r'^(.+?)/(.+?)$'

    FREQUENCY_MAP = {'Y': 'years', 'M': 'months', 'D': 'days'}
    YEAR_TO_PERIOD = {'S': 2, 'Q': 4, 'M': 12, 'D': 365}
    PERIOD_ORDER = {'A': 0, 'S': 1, 'Q': 2, 'M': 3, 'D': 4}
    DATE_TO_PERIOD_PARSER = {
        'D': lambda x: x.strftime('%Y-D%d'),
        'M': lambda x: x.strftime('%Y-M%m'),
        'Q': lambda x: x.strftime('%Y-Q%d'),
        'S': lambda x: x.strftime('%Y-S%d'),
        'A': lambda x: x.strftime('%Y')
    }
    PERIODS_TO_DATE_PARSER = {
        'D': lambda x, y, z: "{}-{}-{}".format(x, y, z),
        'M': lambda x, y: "{}-M{}".format(x, y),
        'Q': lambda x, y: "{}-Q{}".format(x, y),
        'S': lambda x, y: "{}-S{}".format(x, y),
        'A': lambda y: "{}".format(y)
    }

    @classmethod
    def get_time_id(cls, operand: Dataset) -> Optional[str]:
        reference_id = None
        for id in operand.get_identifiers():
            if id.data_type in cls.TIME_DATA_TYPES:
                if reference_id is not None:
                    raise ValueError("FlowToStock can only be applied to a time dataset")
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
    def get_period(cls, value) -> str | None:
        value = str(value)
        _, period, _ = re.match(cls.PERIOD_PATTERN, value).groups()
        return period if period else 'A'

    @classmethod
    def sort_by_period(cls, operand: Dataset) -> Dataset:
        data = operand.data.copy()
        data['Periods_col'] = data[cls.time_id].apply(cls.get_period).apply(lambda x: cls.PERIOD_ORDER[x])
        data['Year_values_col'] = data[cls.time_id].apply(lambda x: int(x.split('-')[0]))
        data['Period_value_col'] = data[cls.time_id].apply(lambda x: int(re.sub(r'[^\d]', '', x.split('-')[-1])))
        data = data.sort_values(by=cls.other_ids + ['Year_values_col', 'Periods_col', 'Period_value_col'])
        data = data.drop(columns=['Periods_col', 'Year_values_col', 'Period_value_col'])
        return Dataset(name='result', components=operand.components.copy(), data=data)

    @classmethod
    def is_period(cls, date_str):
        return bool(re.match(cls.PERIOD_PATTERN, date_str))

    @classmethod
    def is_date(cls, date_str):
        date_patterns = [
            r'^\d{1,4}$',  # YYYY
            r'^\d{1,4}-\d{1,2}$',  # YYYY-MM
            r'^\d{1,4}-\d{1,2}-\d{1,2}$'  # YYYY-MM-DD
        ]
        return any(bool(re.match(pattern, date_str)) for pattern in date_patterns)

    @classmethod
    def is_time(cls, date_str):
        parts = re.match(cls.TIME_PATTERN, date_str)
        if parts:
            part1, part2 = parts.groups()
            return (cls.is_period(part1) or cls.is_date(part1)) and (cls.is_period(part2) or cls.is_date(part2))
        return False

    @classmethod
    def identify_date_type(cls, date_str):
        if cls.is_time(date_str):
            return 'Time'
        elif cls.is_period(date_str):
            return 'Period'
        elif cls.is_date(date_str):
            return 'Date'
        else:
            return 'Unknown'

    @classmethod
    def classify_dates(cls, series):
        series = series.apply(str)
        return series.apply(cls.identify_date_type)

    @classmethod
    def parse_date(cls, date_str):
        parts = date_str.split("-")
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 else 1
        day = int(parts[2]) if len(parts) > 2 else 1
        return datetime(year, month, day)

    @classmethod
    def value_from_period(cls, date_str: str) -> int:
        match = re.match(cls.PERIOD_PATTERN, date_str)
        return int(match.group(3)) if match else int(date_str)

    @classmethod
    def get_year(cls, value: str) -> int:
        return int(re.match(r'^(\d{1,4})', value).group(1))

    @classmethod
    def get_frequencies(cls, dates):
        return [relativedelta(d2, d1) for d1, d2 in combinations(dates, 2)]

    @classmethod
    def find_min_frequency(cls, differences):
        min_months = min((diff.months for diff in differences if diff.months > 0), default=None)
        min_days = min((diff.days for diff in differences if diff.days > 0), default=None)
        return 'D' if min_days else 'M' if min_months else 'Y'

    @classmethod
    def get_frequency_from_time(cls, interval, time_format, time_type):
        start_date, end_date = interval.split('/')
        if time_type == 'Period':
            years = cls.get_year(end_date) - cls.get_year(start_date)
            if time_format == 'A':
                return years
            return cls.value_from_period(end_date) - cls.value_from_period(start_date) + 1 + years * \
                cls.YEAR_TO_PERIOD[time_format]
        return cls.parse_date(end_date) - cls.parse_date(start_date)

    @classmethod
    def get_format_from_time(cls, interval):
        start_date, end_date = interval.split('/')
        if cls.is_period(start_date) and cls.is_period(end_date):
            period = cls.get_period(start_date)
            if period != cls.get_period(end_date):
                raise ValueError("Start and end dates must have the same period")
            return period
        elif cls.is_date(start_date) and cls.is_date(end_date):
            init_format = cls.get_date_format(start_date)
            if init_format != cls.get_date_format(end_date):
                raise ValueError("Start and end dates must have the same format")
            return init_format

    @classmethod
    def get_date_format(cls, date_str):
        date = cls.parse_date(date_str) if isinstance(date_str, str) else date_str
        return '%Y-%m-%d' if date.day > 1 else '%Y-%m' if date.month > 1 else '%Y'


class Unary(Time):

    @classmethod
    def py_op(cls, x):
        raise NotImplementedError

    @classmethod
    def evaluate(cls, operand: Dataset) -> Dataset:
        result = cls.validate(operand)
        result.data = operand.data.copy()
        if len(operand.data) < 2:
            return result

        cls.time_id = cls.get_time_id(result)
        cls.other_ids = [id.name for id in result.get_identifiers() if id.name != cls.time_id]
        cls.measures = result.get_measures_names()
        date_type = cls.classify_dates(operand.data[cls.time_id]).unique()
        if len(date_type) > 1:
            raise ValueError("FlowToStock can only be applied to a dataset with a single date type")
        result.data = result.data.sort_values(by=cls.other_ids + [cls.time_id])
        if date_type[0] == 'Period':
            result.data = cls.period_accumulation(result.data)
        elif date_type[0] in ['Date', 'Time']:
            result.data[cls.measures] = result.data.groupby(cls.other_ids)[cls.measures].apply(cls.py_op)
        else:
            raise ValueError("FlowToStock can only be applied to a dataset with a date type")
        return result

    @classmethod
    def validate(cls, operand: Dataset) -> Dataset:
        if not isinstance(operand, Dataset):
            raise TypeError("FlowToStock can only be applied to a time dataset")
        if cls.get_time_id(operand) is None:
            raise ValueError("FlowToStock can only be applied to a time dataset")
        operand.data = cls.sort_by_time(operand)
        return Dataset(name='result', components=operand.components.copy(), data=None)

    @classmethod
    def period_accumulation(cls, data):
        data = data.copy()
        data['Period_group_col'] = data[cls.time_id].apply(cls.get_period).apply(lambda x: cls.PERIOD_ORDER[x])
        data[cls.measures] = data.groupby(cls.other_ids + ['Period_group_col'])[cls.measures].apply(cls.py_op).reset_index(
            drop=True)
        return data.drop(columns='Period_group_col')


class Binary(Time):
    pass


class Period_indicator(Unary):

    @classmethod
    def evaluate(cls, operand: Dataset | DataComponent | Scalar | str) -> Dataset | DataComponent | Scalar | str:
        cls.validate(operand)
        if isinstance(operand, str):
            return cls.get_period(str(operand))
        if isinstance(operand, Scalar):
            return Scalar(name='result', data_type=DataTypes.Duration, value=cls.get_period(str(operand.value)))
        if isinstance(operand, DataComponent):
            return DataComponent(name='result',
                                 data_type=DataTypes.Duration,
                                 data=operand.data.apply(cls.get_period))
        data = operand.data[cls.time_id].apply(cls.get_period)
        operand.data = operand.data.drop(columns=operand.get_measures_names())
        operand.data['duration_var'] = data
        for measure in operand.get_measures_names():
            operand.components.pop(measure)
        operand.components['duration_var'] = Component(name='duration_var', data_type=DataTypes.Duration,
                                                       role=Role.MEASURE, nullable=True)
        return Dataset(name='result', components=operand.components, data=operand.data)

    @classmethod
    def validate(cls, operand: Dataset | DataComponent | Scalar | str) -> None:
        if isinstance(operand, str):
            return
        if isinstance(operand, Dataset):
            time_id = cls.get_time_id(operand)
            if time_id is None or operand.components[time_id].data_type != DataTypes.TimePeriod:
                raise ValueError("PeriodIndicator can only be applied to a time dataset")
        else:
            if operand.data_type != DataTypes.TimePeriod:
                raise ValueError("PeriodIndicator can only be applied to a time dataset")


class Flow_to_stock(Unary):

    py_op = lambda x: x.cumsum()


class Stock_to_flow(Unary):

    py_op = lambda x: x.diff().fillna(x)


class Fill_time_series(Binary):

    @classmethod
    def evaluate(cls, operand: Dataset, fill_type: str) -> Dataset:
        result = cls.validate(operand, fill_type)
        result.data = operand.data.copy()
        result.data[cls.time_id] = result.data[cls.time_id].astype(str)
        if len(result.data) < 2:
            return result
        date_type = cls.classify_dates(operand.data[cls.time_id]).unique()
        if len(date_type) > 1:
            raise ValueError("FillTimeSeries can only be applied to a dataset with a single date type")
        if date_type[0] == 'Period':
            result.data = cls.fill_periods(result.data, fill_type)
        elif date_type[0] == 'Date':
            frequencies = cls.get_frequencies(operand.data[cls.time_id].apply(cls.parse_date))
            result.data = cls.fill_dates(result.data, fill_type, cls.find_min_frequency(frequencies))
        elif date_type[0] == 'Time':
            interval_format = result.data[cls.time_id].apply(cls.get_format_from_time).unique()
            time_type = 'Period' if interval_format[0] in ['A', 'S', 'Q', 'M', 'D'] else 'Date'
            if len(interval_format) > 1:
                raise ValueError("FillTimeSeries can only be applied to a dataset with a single time interval format")
            frequencies = result.data[cls.time_id].apply(cls.get_frequency_from_time,
                                                         args=(interval_format[0], time_type)).unique()
            if len(frequencies) > 1:
                raise ValueError(
                    "FillTimeSeries can only be applied to a dataset with a single time interval frequency")
            result.data = cls.fill_time_intervals(result.data, fill_type, time_type, interval_format[0], frequencies[0])
        else:
            raise ValueError("FillTimeSeries can only be applied to a dataset with a date type")
        return result

    @classmethod
    def validate(cls, operand: Dataset, fill_type: str) -> Dataset:
        if not isinstance(operand, Dataset):
            raise TypeError("FillTimeSeries can only be applied to a time dataset")
        cls.time_id = cls.get_time_id(operand)
        cls.other_ids = [id.name for id in operand.get_identifiers() if id.name != cls.time_id]
        cls.measures = operand.get_measures_names()
        if cls.time_id is None:
            raise ValueError("FillTimeSeries can only be applied to a time dataset")
        if fill_type not in ['all', 'single']:
            fill_type = 'all'
        return Dataset(name='result', components=operand.components.copy(), data=None)

    @classmethod
    def max_min_from_period(cls, data, mode='all'):
        data = data.assign(
            Periods_col=data[cls.time_id].apply(cls.get_period),
            Periods_values_col=data[cls.time_id].apply(lambda x: int(re.sub(r'[^\d]', '', x.split('-')[-1]))),
            Year_values_col=data[cls.time_id].apply(lambda x: int(x.split('-')[0]))
        ).sort_values(by=['Year_values_col', 'Periods_col', 'Periods_values_col'])

        if mode == 'all':
            min_year = data['Year_values_col'].min()
            max_year = data['Year_values_col'].max()
            result_dict = {
                'min': {'A': min_year},
                'max': {'A': max_year}
            }
            for period, group in data.groupby('Periods_col'):
                if period != 'A':
                    result_dict['min'][period] = group['Periods_values_col'].min()
                    result_dict['max'][period] = group['Periods_values_col'].max()

        elif mode == 'single':
            result_dict = {}
            for name, group in data.groupby(cls.other_ids + ['Periods_col']):
                key = name[:-1] if len(name[:-1]) > 1 else name[0]
                period = name[-1]
                if key not in result_dict:
                    result_dict[key] = {
                        'min': {'A': group['Year_values_col'].min()},
                        'max': {'A': group['Year_values_col'].max()}
                    }
                if period != 'A':
                    year_min = group['Year_values_col'].min()
                    year_max = group['Year_values_col'].max()

                    result_dict[key]['min']['A'] = min(result_dict[key]['min']['A'], year_min)
                    result_dict[key]['max']['A'] = max(result_dict[key]['max']['A'], year_max)
                    result_dict[key]['min'][period] = group['Periods_values_col'].min()
                    result_dict[key]['max'][period] = group['Periods_values_col'].max()

        else:
            raise ValueError("Mode must be either 'all' or 'single'")
        return result_dict

    @classmethod
    def fill_periods(cls, data: pd.DataFrame, fill_type: str) -> pd.DataFrame:
        result_data = cls.period_filler(data, single=(fill_type != 'all'))
        not_na = result_data[cls.measures].notna().any(axis=1)
        duplicated = result_data.duplicated(subset=(cls.other_ids + [cls.time_id]), keep=False)
        return result_data[~duplicated | not_na]

    @classmethod
    def period_filler(cls, data: pd.DataFrame, single=False) -> pd.DataFrame:
        filled_data = []
        MAX_MIN = cls.max_min_from_period(data, mode='single' if single else 'all')
        cls.periods = list(MAX_MIN[list(MAX_MIN.keys())[0]]['min'].keys()) if single else list(MAX_MIN['min'].keys())
        groups = data.groupby(cls.other_ids)

        for group, group_df in groups:
            period_limits = MAX_MIN if not single else MAX_MIN[group]
            years = range(period_limits['min']['A'], period_limits['max']['A'] + 1)
            for period in cls.periods:
                if period == 'A':
                    filled_data.extend(cls.fill_periods_rows(group_df, period, years))
                elif period == 'D':
                    vals = range(period_limits['min'][period], period_limits['max'][period] + 1)
                    filled_data.extend(cls.fill_periods_rows(group_df, period, years, vals))
                else:
                    vals = range(period_limits['min'][period], period_limits['max'][period] + 1)
                    filled_data.extend(cls.fill_periods_rows(group_df, period, years, vals=vals))

        filled_data = pd.concat(filled_data, ignore_index=True)
        combined_data = pd.concat([filled_data, data], ignore_index=True)
        if len(cls.periods) == 1 and cls.periods[0] == 'A':
            combined_data[cls.time_id] = combined_data[cls.time_id].astype(int)
        else:
            combined_data[cls.time_id] = combined_data[cls.time_id].astype(str)
        return combined_data.sort_values(by=cls.other_ids + [cls.time_id])

    @classmethod
    def fill_periods_rows(cls, group_df, period, years, vals=None):
        rows = []
        for year in years:
            if period == 'A':
                rows.append(cls.create_period_row(group_df, period, year))
            else:
                for val in vals:
                    rows.append(cls.create_period_row(group_df, period, year, val=val))
        return rows

    @classmethod
    def create_period_row(cls, group_df, period, year, val=None):
        row = group_df.iloc[0].copy()
        row[cls.time_id] = f"{year}" if period == 'A' else f"{year}-{period}{val:d}"
        row[cls.measures] = None
        return row.to_frame().T

    @classmethod
    def max_min_from_date(cls, data, fill_type='all'):
        def compute_min_max(group):
            min_date = cls.parse_date(group.min())
            max_date = cls.parse_date(group.max())
            date_format = cls.get_date_format(max_date)
            return {'min': min_date, 'max': max_date, 'date_format': date_format}

        if fill_type == 'all':
            return compute_min_max(data[cls.time_id])

        grouped = data.groupby(cls.other_ids)
        result_dict = {name if len(name) > 1 else name[0]: compute_min_max(group[cls.time_id]) for name, group in
                       grouped}
        return result_dict

    @classmethod
    def fill_dates(cls, data: pd.DataFrame, fill_type, min_frequency) -> pd.DataFrame:
        result_data = cls.date_filler(data, fill_type, min_frequency)
        not_na = result_data[cls.measures].notna().any(axis=1)
        duplicated = result_data.duplicated(subset=(cls.other_ids + [cls.time_id]), keep=False)
        return result_data[~duplicated | not_na]

    @classmethod
    def date_filler(cls, data: pd.DataFrame, fill_type, min_frequency) -> pd.DataFrame:
        MAX_MIN = cls.max_min_from_date(data, fill_type)
        date_format = None
        filled_data = []

        def create_filled_dates(group, min_max):
            date_range = pd.date_range(start=min_max['min'], end=min_max['max'], freq=min_frequency)
            date_df = pd.DataFrame(date_range, columns=[cls.time_id])
            date_df[cls.other_ids] = group.iloc[0][cls.other_ids]
            date_df[cls.measures] = None
            return date_df, min_max['date_format']

        for name, group in data.groupby(cls.other_ids):
            min_max = MAX_MIN if fill_type == 'all' else MAX_MIN[name if len(name) > 1 else name[0]]
            filled_dates, date_format = create_filled_dates(group, min_max)
            filled_data.append(filled_dates)

        filled_data = pd.concat(filled_data, ignore_index=True)
        filled_data[cls.time_id] = filled_data[cls.time_id].dt.strftime(date_format)
        combined_data = pd.concat([filled_data, data], ignore_index=True)
        combined_data[cls.time_id] = combined_data[cls.time_id].astype(str)
        return combined_data.sort_values(by=cls.other_ids + [cls.time_id])

    @classmethod
    def max_min_from_time(cls, data, fill_type='all'):
        data = data.applymap(str).sort_values(by=cls.other_ids + [cls.time_id])

        def extract_max_min(group):
            start_dates = group.apply(lambda x: x.split('/')[0])
            end_dates = group.apply(lambda x: x.split('/')[1])
            return {'start': {'min': start_dates.min(), 'max': start_dates.max()},
                    'end': {'min': end_dates.min(), 'max': end_dates.max()}}

        if fill_type == 'all':
            return extract_max_min(data[cls.time_id])
        else:
            return {name: extract_max_min(group[cls.time_id]) for name, group in data.groupby(cls.other_ids)}

    @classmethod
    def fill_time_intervals(cls, data: pd.DataFrame, fill_type, time_type, time_format, frequency) -> pd.DataFrame:
        result_data = cls.time_filler(data, fill_type, time_type, time_format, frequency)
        not_na = result_data[cls.measures].notna().any(axis=1)
        duplicated = result_data.duplicated(subset=(cls.other_ids + [cls.time_id]), keep=False)
        return result_data[~duplicated | not_na]

    @classmethod
    def time_filler(cls, data: pd.DataFrame, fill_type, time_type, time_format, frequency) -> pd.DataFrame:
        MAX_MIN = cls.max_min_from_time(data, fill_type)

        def fill_group(group_df):
            group_key = group_df.iloc[0][cls.other_ids].values
            if fill_type != 'all':
                group_key = group_key[0] if len(group_key) == 1 else tuple(group_key)
            group_dict = MAX_MIN if fill_type == 'all' else MAX_MIN[group_key]

            intervals = [f"{group_dict['start']['min']}/{group_dict['end']['min']}",
                         f"{group_dict['start']['max']}/{group_dict['end']['max']}"]
            for interval in intervals:
                if interval not in group_df[cls.time_id].values:
                    empty_row = group_df.iloc[0].copy()
                    empty_row[cls.time_id] = interval
                    empty_row[cls.measures] = None
                    group_df = group_df.append(empty_row, ignore_index=True)
            start_group_df = group_df.copy()
            start_group_df[cls.time_id] = start_group_df[cls.time_id].apply(lambda x: x.split('/')[0])
            end_group_df = group_df.copy()
            end_group_df[cls.time_id] = end_group_df[cls.time_id].apply(lambda x: x.split('/')[1])
            if time_type == 'Period':
                start_filled = cls.period_filler(start_group_df, single=(fill_type == 'single'))
                end_filled = cls.period_filler(end_group_df, single=(fill_type == 'single'))
            else:
                start_filled = cls.date_filler(start_group_df, fill_type, frequency)
                end_filled = cls.date_filler(end_group_df, fill_type, frequency)
            start_filled[cls.time_id] = start_filled[cls.time_id].str.cat(end_filled[cls.time_id],
                                                                          sep='/')
            if time_type == 'Period':
                return start_filled[
                    start_filled[cls.time_id].apply(lambda x: cls.get_period(x.split('/')[0])) == time_format]
            return start_filled

        filled_data = [fill_group(group_df) for _, group_df in data.groupby(cls.other_ids)]
        return pd.concat(filled_data, ignore_index=True).sort_values(
            by=cls.other_ids + [cls.time_id]).drop_duplicates()


class Time_Shift(Time):

    @classmethod
    def evaluate(cls, operand: Dataset, shift_value: Scalar) -> Dataset:
        result = cls.validate(operand)
        result.data = operand.data.copy()
        shift_value = int(shift_value.value)
        cls.time_id = cls.get_time_id(result)
        date_type = cls.classify_dates(result.data[cls.time_id]).unique()[0]

        if date_type == 'Date':
            freq = cls.find_min_frequency(cls.get_frequencies(result.data[cls.time_id].apply(cls.parse_date)))
            result.data[cls.time_id] = result.data[cls.time_id].apply(
                lambda x: cls.shift_date(x, shift_value, freq)).astype(str)
        elif date_type == 'Time':
            interval_format = cls.get_format_from_time(result.data[cls.time_id].iloc[0])
            time_type = 'Period' if interval_format in ['A', 'S', 'Q', 'M', 'D'] else 'Date'
            freq = cls.get_frequency_from_time(result.data[cls.time_id].iloc[0], interval_format, time_type)
            result.data[cls.time_id] = result.data[cls.time_id].apply(
                lambda x: cls.shift_interval(x, shift_value, interval_format, freq))
        elif date_type == 'Period':
            periods = result.data[cls.time_id].apply(cls.get_period).unique()
            result.data[cls.time_id] = result.data[cls.time_id].apply(lambda x: cls.shift_period(x, shift_value))
            if len(periods) == 1 and periods[0] == 'A':
                result.data[cls.time_id] = result.data[cls.time_id].astype(int)
        else:
            raise ValueError("Unknown date type for Timeshift")
        return result

    @classmethod
    def validate(cls, operand: Dataset) -> Dataset:
        if not isinstance(operand, Dataset) or cls.get_time_id(operand) is None:
            raise TypeError("Timeshift can only be applied to a time dataset")
        return Dataset(name='result', components=operand.components.copy(), data=None)

    @classmethod
    def shift_date(cls, date, shift_value, frequency):
        return pd.to_datetime(date) + relativedelta(**{cls.FREQUENCY_MAP[frequency]: shift_value})

    @classmethod
    def shift_period(cls, period_str, shift_value, frequency=None):
        period_type = cls.get_period(period_str)

        if period_type == 'A':
            return str(int(period_str) + shift_value)

        if frequency:
            shift_value *= frequency

        match = re.match(cls.PERIOD_PATTERN, period_str)
        year, period, value = int(match.group(1)), match.group(2), int(match.group(3)) + shift_value
        period_limit = cls.YEAR_TO_PERIOD[period]

        if value <= 0:
            year -= 1
            value += period_limit
        elif value > period_limit:
            year += (value - 1) // period_limit
            value = (value - 1) % period_limit + 1

        return f"{year}-{period}{value}"

    @classmethod
    def shift_interval(cls, interval, shift_value, time_format, frequency):
        start_date, end_date = interval.split('/')
        shift_func = cls.shift_period if time_format in ['A', 'S', 'Q', 'M', 'D'] else cls.shift_date
        start_date = shift_func(start_date, shift_value, frequency)
        end_date = shift_func(end_date, shift_value, frequency)
        return f'{start_date}/{end_date}'
