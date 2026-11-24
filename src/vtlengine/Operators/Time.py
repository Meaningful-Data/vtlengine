import calendar
import re
from copy import copy
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Type, Union, cast

import pandas as pd
from dateutil.relativedelta import relativedelta  # type: ignore[import-untyped]

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
    Number,
    ScalarType,
    String,
    TimeInterval,
    TimePeriod,
    check_unary_implicit_promotion,
    unary_implicit_promotion,
)
from vtlengine.DataTypes._time_checking import (
    _has_time_component,
    parse_date_value,
    parse_datetime_value,
)
from vtlengine.DataTypes.TimeHandling import (
    MAX_YEAR,
    PERIOD_IND_MAPPING,
    TimePeriodHandler,
    check_year,
    date_of_ordinal,
    date_to_period,
    generate_period_range,
    max_periods_in_year,
    shifted_year,
)
from vtlengine.Exceptions import RunTimeError, SemanticError
from vtlengine.Model import Component, DataComponent, Dataset, Role, Scalar, value_of
from vtlengine.Utils.__Virtual_Assets import VirtualCounter


class Time(Operators.Operator):
    periods: Any
    time_id: Any
    other_ids: Any
    measures: Any

    TIME_DATA_TYPES = [Date, TimePeriod, TimeInterval]

    FREQUENCY_MAP = {"Y": "years", "M": "months", "D": "days"}
    YEAR_TO_PERIOD = {"S": 2, "Q": 4, "M": 12}
    PERIOD_ORDER = {"A": 0, "S": 1, "Q": 2, "M": 3, "W": 4, "D": 5}
    # The scalar that gives the operator a parameter, as 2-1-19-24 names it when the scalar
    # holds no value
    SCALAR_PARAM = "scalar {name}"

    op = FLOW_TO_STOCK

    @classmethod
    def no_value_error(cls, name: str) -> RunTimeError:
        """The error of a scalar that gives the operator a parameter and holds no value."""
        return RunTimeError("2-1-19-24", op=cls.op, scalar=cls.SCALAR_PARAM.format(name=name))

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
    def _iter_groups(cls, data: pd.DataFrame) -> Any:
        """Iterate over the series the operand holds, one per combination of the
        non-time identifiers.

        A Data Set may have the time identifier as its only identifier, in which
        case it holds a single series and Pandas cannot group it.
        """
        if not cls.other_ids:
            return [((), data)]
        return data.groupby(cls.other_ids)

    @staticmethod
    def _group_key(values: Any) -> Any:
        """The dictionary key a group of non-time identifier values maps to.

        Pandas hands over a one-element tuple when grouping by a single column, so
        the values have to be unwrapped the same way on both sides of a lookup.
        """
        values = tuple(values)
        return values[0] if len(values) == 1 else values

    @classmethod
    def parse_date(cls, date_str: str) -> date:
        return parse_date_value(date_str)

    @staticmethod
    def _is_whole_month_gap(start: date, end: date, months: int) -> bool:
        """Whether the gap spans a whole number of calendar months."""
        if months <= 0:
            return False
        if start + relativedelta(months=months) == end:
            return True
        # Month ends stay month ends even though their day numbers differ.
        return all(day.day == calendar.monthrange(day.year, day.month)[1] for day in (start, end))

    @classmethod
    def find_min_frequency(cls, values: Any) -> str:
        """Infer the period of a Date time series from the gaps between the days of its points.

        The shortest period of a Date is the day, so a series with two points on one day, at
        different times of it, has the period of a day (issue #1352).
        """
        values = pd.Series(list(values)).dropna().drop_duplicates()
        days = values.map(lambda value: cls.parse_date(str(value)))
        if days.duplicated().any():
            return "D"
        # Python dates, which hold every year of a Date on any pandas version (issue #1352)
        ordered = sorted(days)
        if len(ordered) < 2:
            return "Y"

        gaps = list(zip(ordered[:-1], ordered[1:]))
        day_diffs = [(end - start).days for start, end in gaps]
        month_diffs = [
            (end.year - start.year) * 12 + (end.month - start.month) for start, end in gaps
        ]

        if not all(
            cls._is_whole_month_gap(start, end, months)
            for (start, end), months in zip(gaps, month_diffs)
        ):
            return "W" if all(diff % 7 == 0 for diff in day_diffs) else "D"
        if all(months % 12 == 0 for months in month_diffs):
            return "Y"
        if all(months % 6 == 0 for months in month_diffs):
            return "S"
        if all(months % 3 == 0 for months in month_diffs):
            return "Q"
        return "M"

    @classmethod
    def _series_frequency(cls, data: pd.DataFrame) -> str:
        """The period of the reference time Identifier of timeshift and fill_time_series on
        Dates, read from each time series.

        The reference time Identifier has to be periodical, so every series that shows
        a gap must agree on one period; the gaps are taken within a series rather than
        over the whole column, where two series of the same period interleave into
        shorter ones. A series of a single Data Point shows no gap and
        takes the period the rest of the Data Set agrees on.
        """
        frequencies = set()
        for _, group in cls._iter_groups(data):
            # The values of the group itself, which an index label of the operand may repeat
            series = group[cls.time_id].dropna()
            if series.nunique() > 1:
                frequencies.add(cls.find_min_frequency(series))
        if len(frequencies) > 1:
            raise RunTimeError(
                "2-1-19-23",
                op=cls.op,
                comp_type="dataset",
                param="single time series frequency",
            )
        return frequencies.pop() if frequencies else "Y"

    _PERIOD_BY_RELATIVEDELTA = {
        (1, 0, 0): "Y",
        (0, 6, 0): "S",
        (0, 3, 0): "Q",
        (0, 1, 0): "M",
        (0, 0, 7): "W",
        (0, 0, 1): "D",
    }

    @classmethod
    def _interval_endpoints(cls, interval: str) -> Any:
        """The two endpoint dates of a TimeInterval.

        The endpoints may carry a time component, which the input format allows,
        so only the date part is read.
        """
        start_str, end_str = interval.split("/")
        return date.fromisoformat(start_str[:10]), date.fromisoformat(end_str[:10])

    @classmethod
    def get_frequency_from_time(cls, interval: str) -> Any:
        start, end = cls._interval_endpoints(interval)
        return end - start

    @classmethod
    def _classify_interval_period(cls, interval: str) -> str:
        start, end = cls._interval_endpoints(interval)
        if end.year == MAX_YEAR:
            # The day after an end in 9999 has no date: the calendar repeats every 400 years
            start, end = start.replace(year=start.year - 400), end.replace(year=end.year - 400)
        candidates = [relativedelta(endpoint, start) for endpoint in (end, end + timedelta(days=1))]
        candidates = [
            rd
            for rd in candidates
            if rd.years >= 0
            and rd.months >= 0
            and rd.days >= 0
            and (rd.years or rd.months or rd.days)
        ]
        if not candidates:
            return f"P{(end - start).days}D"
        for rd in candidates:
            canonical = cls._PERIOD_BY_RELATIVEDELTA.get((rd.years, rd.months, rd.days))
            if canonical is not None:
                return canonical
        chosen = min(
            candidates, key=lambda r: sum(1 for x in (r.years, r.months, r.days) if x != 0)
        )
        parts = []
        if chosen.years:
            parts.append(f"{chosen.years}Y")
        if chosen.months:
            parts.append(f"{chosen.months}M")
        if chosen.days:
            parts.append(f"{chosen.days}D")
        return "P" + "".join(parts)

    _PERIOD_DURATION_RE = re.compile(r"^P(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)D)?$")

    _OFFSET_ARGS_BY_PERIOD = {
        "A": {"years": 1},
        "Y": {"years": 1},
        "S": {"months": 6},
        "Q": {"months": 3},
        "M": {"months": 1},
        "W": {"days": 7},
        "D": {"days": 1},
    }

    @classmethod
    def _period_offset(cls, code: str) -> Any:
        """One period as a date offset, for stepping a series from its lower limit."""
        args = cls._OFFSET_ARGS_BY_PERIOD.get(code)
        if args is None and code.startswith("P"):
            m = cls._PERIOD_DURATION_RE.match(code)
            if m is not None:
                args = {
                    "years": int(m.group(1) or 0),
                    "months": int(m.group(2) or 0),
                    "days": int(m.group(3) or 0),
                }
        args = args or {"days": 1}
        return pd.DateOffset(
            years=args.get("years", 0),
            months=args.get("months", 0),
            days=args.get("days", 0),
        )

    @classmethod
    def _period_range(cls, start: Any, end: Any, code: str) -> List[datetime]:
        """Dates from *start* to *end*, one period at a time.

        Every date is measured from the lower limit rather than from its predecessor.
        A month-based step out of a period end clamps in the shorter month, and
        carrying that clamped day forward drifts the grid off the period ends:
        stepping one month at a time from 2020-01-31 reaches 2020-02-29 and then
        2020-03-29, whereas the reference manual identifies a date-typed period by its
        last day and so expects 2020-03-31 (issue #961). Measuring from the lower
        limit keeps a period-end series on period ends and leaves any other series
        exactly where it was.
        """
        # Python datetimes, which hold every year of a Date on any pandas version (issue #1352)
        offset = relativedelta(**cls._period_offset(code).kwds)
        start_ts = start if isinstance(start, datetime) else parse_datetime_value(str(start))
        limit = end if isinstance(end, datetime) else parse_datetime_value(str(end))
        if start_ts - offset >= start_ts:  # guard against a zero-length period
            return [start_ts] if start_ts <= limit else []
        dates: List[datetime] = []
        step = 0
        while True:
            try:
                current = start_ts + step * offset if step else start_ts
            except (ValueError, OverflowError):
                # A date past the year 9999, which no datetime holds, is past the limit too
                # (issue #1352)
                return dates
            if current > limit:
                return dates
            dates.append(current)
            step += 1


class Unary(Time):
    @classmethod
    def validate(cls, operand: Any) -> Any:
        dataset_name = VirtualCounter._new_ds_name()
        if not isinstance(operand, Dataset):
            raise SemanticError("1-1-19-8", op=cls.op, comp_type="time dataset")
        if cls._get_time_id(operand) is None:
            raise SemanticError("1-1-19-8", op=cls.op, comp_type="time dataset")
        operand.data = cls.sort_by_time(operand)
        return Dataset(name=dataset_name, components=cls.components(operand), data=None)

    @staticmethod
    def components(operand: Dataset) -> Dict[str, Component]:
        """The Components of the result, copied: only the number Measures accumulate, so
        only they leave their value domain, and the other Measures keep it (issue #1348)."""
        components = {name: copy(comp) for name, comp in operand.components.items()}
        for comp in components.values():
            if comp.role == Role.MEASURE and comp.data_type in (Integer, Number):
                comp.value_domain = None
        return components

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
            if cls.other_ids:
                accumulated = (
                    result.data.groupby(cls.other_ids)[measure_names]
                    .apply(cls.py_op)
                    .reset_index(drop=True)
                )
            else:
                accumulated = cls.py_op(result.data[measure_names].copy()).reset_index(drop=True)
            result.data[measure_names] = accumulated
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
        dataset_name = VirtualCounter._new_ds_name()
        if isinstance(operand, Dataset):
            time_id = cls._get_time_id(operand)
            if operand.components[time_id].data_type != TimePeriod:
                raise SemanticError("1-1-19-8", op=cls.op, comp_type="time period dataset")
            result_components = {
                comp.name: comp
                for comp in operand.components.values()
                if comp.role in (Role.IDENTIFIER, Role.VIRAL_ATTRIBUTE)
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
        return Scalar(name=operand.name, data_type=Duration, value=None, nullable=operand.nullable)

    @classmethod
    def evaluate(
        cls, operand: Union[Dataset, DataComponent, Scalar, str]
    ) -> Union[Dataset, DataComponent, Scalar, str]:
        result = cls.validate(operand)
        if isinstance(operand, str):
            return cls._get_period(str(operand))
        if isinstance(operand, Scalar):
            value = operand.value
            result.value = None if value is None else cls._get_period(str(value))
            return result
        if isinstance(operand, DataComponent):
            if operand.data is not None:
                result.data = operand.data.map(cls._get_period, na_action="ignore")
            return result
        cls.time_id = cls._get_time_id(operand)
        result.data = (
            operand.data.copy()[
                result.get_identifiers_names() + result.get_viral_attributes_names()
            ]
            if (operand.data is not None)
            else pd.Series()
        )
        period_series: Any = result.data[cls.time_id].map(cls._get_period)
        result.data["duration_var"] = period_series
        # Row-preserving operator: viral attributes are copied through unchanged (issue #906).
        return result


class Parametrized(Time):
    @classmethod
    def validate(cls, operand: Any, param: Any) -> Any:
        pass

    @classmethod
    def evaluate(cls, operand: Any, param: Any) -> Any:
        pass


class Flow_to_stock(Unary):
    """Only number measures accumulate; the reference manual types the operand as
    ``measure<number>``, so Boolean and the other types pass through untouched
    (issue #931)."""

    @classmethod
    def py_op(cls, x: Any) -> Any:
        if isinstance(x, pd.DataFrame):
            numeric = x.select_dtypes(include="number")
            x[numeric.columns] = numeric.cumsum().fillna(numeric)
            return x
        return x.cumsum().fillna(x)


class Stock_to_flow(Unary):
    """Counterpart of :class:`Flow_to_stock`; see its note on measure types."""

    @classmethod
    def py_op(cls, x: Any) -> Any:
        if isinstance(x, pd.DataFrame):
            numeric = x.select_dtypes(include="number")
            x[numeric.columns] = numeric.diff().fillna(numeric)
            return x
        return x.diff().fillna(x)


class Fill_time_series(Binary):
    op = FILL_TIME_SERIES

    @classmethod
    def evaluate(cls, operand: Dataset, fill_type: str) -> Dataset:
        result = cls.validate(operand, fill_type)
        if operand.data is None:
            operand.data = pd.DataFrame()
        result.data = operand.data.copy()
        result.data[cls.time_id] = result.data[cls.time_id].astype("string[pyarrow]")
        if result.data.empty:
            # A single Data Point is filled as well: a Time_Period in all mode to the whole of
            # its year, while a Date or a Time shows no period and stays (issue #1352)
            return result
        if not cls.other_ids:
            # The operand holds a single series, so both limits span the same range.
            fill_type = "all"
        data_type = result.components[cls.time_id].data_type
        if data_type == TimePeriod:
            result.data = cls.fill_periods(result.data, fill_type)
        elif data_type == Date:
            frequency = cls._series_frequency(result.data)
            result.data = cls.fill_dates(result.data, fill_type, frequency)
        elif data_type == TimeInterval:
            categories = result.data[cls.time_id].apply(cls._classify_interval_period).unique()
            if len(categories) > 1:
                raise RunTimeError(
                    "2-1-19-23",
                    op=cls.op,
                    comp_type="dataset",
                    param="single time interval frequency",
                )
            result.data = cls.fill_time_intervals(result.data, fill_type, categories[0])
        else:
            raise SemanticError("1-1-19-2", op=cls.op)
        return result

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
        # The filled periods are new values of the time Identifier
        components = Operators.on_basic_types(operand.components, [cls.time_id])
        return Dataset(name=dataset_name, components=components, data=None)

    @classmethod
    def fill_periods(cls, data: pd.DataFrame, fill_type: str) -> pd.DataFrame:
        # Parse each time_id value once and reuse throughout
        data = data.copy()
        tp_parsed = data[cls.time_id].map(TimePeriodHandler, na_action="ignore")
        data[cls.time_id] = tp_parsed.map(str, na_action="ignore")
        # The frequency stays out of the frame, where a Component of any name may be (#1352)
        freq_series = tp_parsed.map(lambda x: x.period_indicator, na_action="ignore")

        # Determine global year range (for "all" mode)
        global_min_year = global_max_year = 0
        if fill_type == "all":
            global_min_year = tp_parsed.map(lambda x: x.year).min()
            global_max_year = tp_parsed.map(lambda x: x.year).max()

        # Group by other_ids + frequency and fill missing periods
        filled_rows: List[Dict[str, Any]] = []
        non_id_cols = [c for c in data.columns if c not in cls.other_ids and c != cls.time_id]

        groupers = [data[other_id] for other_id in cls.other_ids] + [freq_series]
        for group_key, group_df in data.groupby(groupers):
            if isinstance(group_key, tuple):
                freq = str(group_key[-1])
                other_id_values = group_key[:-1]
            else:
                freq = str(group_key)
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
        if filled_rows:
            fill_df = pd.DataFrame(filled_rows)
            result = pd.concat([data, fill_df], ignore_index=True)
        else:
            result = data
        result[cls.time_id] = result[cls.time_id].astype("string[pyarrow]")
        return result.sort_values(by=cls.other_ids + [cls.time_id]).reset_index(drop=True)

    @classmethod
    def max_min_from_date(cls, data: pd.DataFrame, fill_type: str = "all") -> Dict[str, Any]:
        def compute_min_max(group: Any) -> Dict[str, Any]:
            # The limits keep their time of the day, from which each period is counted, as the
            # manual counts it from the lower limit (issue #1352)
            return {
                "min": parse_datetime_value(str(group.min())),
                "max": parse_datetime_value(str(group.max())),
            }

        if fill_type == "all":
            return compute_min_max(data[cls.time_id])

        return {
            cls._group_key(name): compute_min_max(group[cls.time_id])
            for name, group in cls._iter_groups(data)
        }

    @classmethod
    def fill_dates(cls, data: pd.DataFrame, fill_type: str, min_frequency: str) -> pd.DataFrame:
        return cls.date_filler(data, fill_type, min_frequency)

    @classmethod
    def date_filler(cls, data: pd.DataFrame, fill_type: str, min_frequency: str) -> pd.DataFrame:
        MAX_MIN = cls.max_min_from_date(data, fill_type)
        filled_data: List[pd.DataFrame] = []

        def create_filled_dates(group: Any, min_max: Dict[str, Any]) -> pd.DataFrame:
            date_range = cls._period_range(min_max["min"], min_max["max"], min_frequency)
            # Each Date written as the loaders write it, with its time of the day if it has one
            date_df = pd.DataFrame(
                {cls.time_id: [value_of(Date, moment.isoformat()) for moment in date_range]}
            )
            if cls.other_ids:
                # One column at a time: a multi-column assignment from the row Series
                # takes its values positionally, which pandas 3 no longer allows.
                first_row = group.iloc[0]
                for other_id in cls.other_ids:
                    date_df[other_id] = first_row[other_id]
            date_df[cls.measures] = None
            return date_df

        for name, group in cls._iter_groups(data):
            min_max = MAX_MIN if fill_type == "all" else MAX_MIN[cls._group_key(name)]
            filled_data.append(create_filled_dates(group, min_max))

        combined_data = pd.concat([*filled_data, data], ignore_index=True)
        combined_data[cls.time_id] = combined_data[cls.time_id].astype("string[pyarrow]")
        # The added Data Point of a date the operand has goes, and the operand's own one, which
        # comes last, stays, also when all its Measures are null. No marker column is added to
        # the frame, where a Component of any name may be (issue #1352)
        duplicated = combined_data.duplicated(subset=(cls.other_ids + [cls.time_id]), keep="last")
        return combined_data[~duplicated].sort_values(by=cls.other_ids + [cls.time_id])

    @classmethod
    def max_min_from_time(cls, data: pd.DataFrame, fill_type: str = "all") -> Dict[str, Any]:
        data = data.sort_values(by=cls.other_ids + [cls.time_id])

        def extract_max_min(group: Any) -> Dict[str, Any]:
            intervals = group.astype(str)
            start_dates = intervals.str.split("/").str[0]
            end_dates = intervals.str.split("/").str[1]
            return {
                "start": {"min": start_dates.min(), "max": start_dates.max()},
                "end": {"min": end_dates.min(), "max": end_dates.max()},
            }

        if fill_type == "all":
            return extract_max_min(data[cls.time_id])
        return {
            cls._group_key(name): extract_max_min(group[cls.time_id])
            for name, group in cls._iter_groups(data)
        }

    @classmethod
    def fill_time_intervals(
        cls, data: pd.DataFrame, fill_type: str, frequency: str
    ) -> pd.DataFrame:
        """Add the Data Points the interval grid expects and keep the operand's own.

        Both endpoints are stepped by one frequency from their own lower limit, and
        the k-th start is then paired with the k-th end. Only Data Points whose key
        is missing are added, so the operand's own intervals always survive, even
        when they overlap and the two endpoint grids come out different lengths.
        """
        MAX_MIN = cls.max_min_from_time(data, fill_type)
        non_ids = [c for c in data.columns if c not in cls.other_ids and c != cls.time_id]

        def grid_intervals(limits: Dict[str, Any], sample: str) -> List[str]:
            # The limits stay ISO strings so that a time of day survives the range.
            starts = cls._period_range(limits["start"]["min"], limits["start"]["max"], frequency)
            ends = cls._period_range(limits["end"]["min"], limits["end"]["max"], frequency)
            fmt = "%Y-%m-%dT%H:%M:%S" if _has_time_component(sample.split("/")[0]) else "%Y-%m-%d"
            return [f"{s.strftime(fmt)}/{e.strftime(fmt)}" for s, e in zip(starts, ends)]

        filled_rows: List[Dict[str, Any]] = []
        for name, group_df in cls._iter_groups(data):
            limits = MAX_MIN if fill_type == "all" else MAX_MIN[cls._group_key(name)]
            existing = set(group_df[cls.time_id].astype(str))
            other_vals = dict(zip(cls.other_ids, tuple(name)))
            for interval in grid_intervals(limits, str(group_df[cls.time_id].iloc[0])):
                if interval not in existing:
                    filled_rows.append(
                        {**other_vals, cls.time_id: interval, **dict.fromkeys(non_ids)}
                    )

        result = data
        if filled_rows:
            result = pd.concat([data, pd.DataFrame(filled_rows)], ignore_index=True)
        result[cls.time_id] = result[cls.time_id].astype("string[pyarrow]")
        return result.sort_values(by=cls.other_ids + [cls.time_id]).reset_index(drop=True)


class Time_Shift(Binary):
    op = TIMESHIFT
    SCALAR_PARAM = "scalar {name} that gives the shift number"

    @classmethod
    def evaluate(cls, operand: Dataset, shift_value: Any) -> Dataset:
        result = cls.validate(operand, shift_value)
        if shift_value.value is None:
            # A shift of no periods has no result: the time Identifier cannot be null (#1352)
            raise cls.no_value_error(shift_value.name)
        result.data = operand.data.copy() if operand.data is not None else pd.DataFrame()
        shift_value = int(shift_value.value)
        cls.time_id = cls._get_time_id(result)

        data_type: Any = result.components[cls.time_id].data_type
        cls.other_ids = [i.name for i in result.get_identifiers() if i.name != cls.time_id]

        if data_type == Date:
            freq = cls._series_frequency(result.data)
            # The Dates move as dateadd moves them, which writes a Date as the loaders do and
            # keeps it in the years a Date has (issue #1352)
            period = "A" if freq == "Y" else freq
            result.data[cls.time_id] = result.data[cls.time_id].map(
                lambda x: Date_Add.py_op(x, shift_value, period), na_action="ignore"
            )
        elif data_type == TimeInterval:
            # An empty operand carries no interval to read a frequency from and
            # nothing to shift (issue #1034).
            if not result.data.empty:
                categories = result.data[cls.time_id].apply(cls._classify_interval_period).unique()
                if len(categories) > 1:
                    raise RunTimeError(
                        "2-1-19-23",
                        op=cls.op,
                        comp_type="dataset",
                        param="single time series frequency",
                    )
                interval_freq = categories[0]
                result.data[cls.time_id] = result.data[cls.time_id].map(
                    lambda x: cls.shift_interval(x, shift_value, interval_freq),
                    na_action="ignore",
                )
        elif data_type == TimePeriod:
            result.data[cls.time_id] = result.data[cls.time_id].apply(
                lambda x: cls.shift_period(x, shift_value)
            )
        else:
            raise SemanticError("1-1-19-2", op=cls.op)
        return result

    @classmethod
    def validate(cls, operand: Dataset, shift_value: Any) -> Dataset:
        dataset_name = VirtualCounter._new_ds_name()
        if not isinstance(operand, Dataset):
            raise SemanticError("1-1-19-7", op=cls.op)
        if not isinstance(shift_value, Scalar) or not check_unary_implicit_promotion(
            shift_value.data_type, Integer
        ):
            raise SemanticError(
                "1-1-18-4", op=cls.op, param_type="Shift number", correct_type="an Integer Scalar"
            )
        if shift_value.data_type == Number:
            # A shift is a whole number of periods, as the one of dateadd (issue #1352)
            raise SemanticError(
                "1-1-19-15", op=cls.op, type="Number", name="shiftNumber", expected="Integer"
            )
        if cls._get_time_id(operand) is None:
            raise SemanticError("1-1-19-8", op=cls.op, comp_type="time dataset")
        # The shifted periods are new values of the time Identifier
        components = Operators.on_basic_types(operand.components, [cls._get_time_id(operand)])
        return Dataset(name=dataset_name, components=components, data=None)

    @classmethod
    def _interval_step(cls, frequency: str) -> tuple[int, int]:
        """The months and the days of one step of the frequency of a Time series."""
        steps = {"Y": (12, 0), "S": (6, 0), "Q": (3, 0), "M": (1, 0), "W": (0, 7), "D": (0, 1)}
        if frequency in steps:
            return steps[frequency]
        m = cls._PERIOD_DURATION_RE.match(frequency)
        if m is None:
            raise RunTimeError("2-1-19-2", period=frequency)
        return int(m.group(1) or 0) * 12 + int(m.group(2) or 0), int(m.group(3) or 0)

    @classmethod
    def shift_dates(cls, dates: Any, shift_value: int, frequency: str) -> datetime:
        # A Python datetime, which holds every year of a Time on any pandas version (#1352)
        months, days = cls._interval_step(frequency)
        return parse_datetime_value(dates) + relativedelta(
            months=months * shift_value, days=days * shift_value
        )

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
        if period in ("W", "D"):
            # A year has 52 or 53 weeks and 365 or 366 days, so the period moves on the
            # calendar (issue #1352)
            days = shift_value * (7 if period == "W" else 1)
            start = cast(date, tp_value.start_date(as_date=True))
            day, years = date_of_ordinal(start.toordinal() + days)
            if period == "W":
                year, value = day.isocalendar()[:2]
            else:
                year, value = day.year, day.timetuple().tm_yday
            year += years
        else:
            # The year moves by as many years as the periods cover, back or forward (issue #1352)
            period_limit = cls.YEAR_TO_PERIOD[period]
            year += (value - 1) // period_limit
            value = (value - 1) % period_limit + 1
        # The year is checked before the period is built from its text, which cannot write a
        # year below 0 (issue #1352)
        check_year(year)

        return str(TimePeriodHandler(f"{year}-{period}{value}"))

    @classmethod
    def shift_interval(cls, interval: str, shift_value: Any, frequency: str) -> str:
        start_date, end_date = interval.split("/")
        # A Time has the years 1000 to 9999: the year of each shifted end is checked before the
        # end is computed, which a shift that leaves those years by far could not be (#1352)
        months, days = cls._interval_step(frequency)
        for end in (start_date, end_date):
            day = date.fromisoformat(end[:10])
            check_year(shifted_year(day, months * shift_value, days * shift_value))
        fmt = "%Y-%m-%dT%H:%M:%S" if _has_time_component(start_date) else "%Y-%m-%d"
        start_shifted = cls.shift_dates(start_date, shift_value, frequency)
        end_shifted = cls.shift_dates(end_date, shift_value, frequency)
        return f"{start_shifted.strftime(fmt)}/{end_shifted.strftime(fmt)}"


class Time_Aggregation(Time):
    op = TIME_AGG
    SCALAR_PARAM = "period indicator {name} to aggregate to"

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
    def _check_duration(cls, value: Optional[str]) -> None:
        if value is not None and value not in PERIOD_IND_MAPPING:
            raise SemanticError("1-1-19-3", op=cls.op, param="duration")

    @classmethod
    def check_period_value(cls, period_from: Optional[str], period_to: str) -> None:
        """Check a period to aggregate to read from a scalar, which is data (issue #1192)."""
        if period_to not in PERIOD_IND_MAPPING:
            raise RunTimeError("2-1-19-2", period=period_to)
        if (
            period_from is not None
            and period_from in PERIOD_IND_MAPPING
            and PERIOD_IND_MAPPING[period_to] <= PERIOD_IND_MAPPING[period_from]
        ):
            raise RunTimeError("2-1-19-26", value_1=period_from, value_2=period_to)

    @classmethod
    def _check_params(cls, period_from: Optional[str], period_to: Optional[str]) -> None:
        cls._check_duration(period_to)
        if period_from is not None:
            cls._check_duration(period_from)
            # A period read from a scalar is only known at run time (issue #1192)
            if (
                period_to is not None
                and PERIOD_IND_MAPPING[period_to] <= PERIOD_IND_MAPPING[period_from]
            ):
                # OPERATORS_TIMEOPERATORS.19
                raise SemanticError("1-1-19-4", op=cls.op, value_1=period_from, value_2=period_to)

    @classmethod
    def dataset_validation(
        cls,
        operand: Dataset,
        period_from: Optional[str],
        period_to: Optional[str],
        conf: Optional[str],
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
        # The aggregated periods are new values of the time Measure
        time_measure = [
            m.name for m in operand.get_measures() if m.data_type in cls.TIME_DATA_TYPES
        ]
        result_components = Operators.on_basic_types(result_components, time_measure)
        return Dataset(name=operand.name, components=result_components, data=None)

    @classmethod
    def component_validation(
        cls,
        operand: DataComponent,
        period_from: Optional[str],
        period_to: Optional[str],
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
        cls,
        operand: Scalar,
        period_from: Optional[str],
        period_to: Optional[str],
        conf: Optional[str],
    ) -> Scalar:
        if operand.data_type not in cls.TIME_DATA_TYPES:
            raise SemanticError("1-1-19-8", op=cls.op, comp_type="time scalar")
        if operand.data_type == TimePeriod and period_to == "D":
            raise SemanticError("1-1-19-5", op=cls.op)
        if operand.data_type == TimeInterval:
            raise SemanticError("1-1-19-6", op=cls.op, comp=operand.name)
        if operand.data_type == Date and conf is None:
            raise SemanticError("1-1-19-11")

        return Scalar(
            name=operand.name, data_type=operand.data_type, value=None, nullable=operand.nullable
        )

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

        # Date, the only other type the validation lets through
        if conf is None:
            raise SemanticError("1-1-19-11")
        start = conf == "first"
        if period_to == "D":
            return value
        return _date_access(value, period_to, start)

    @classmethod
    def dataset_evaluation(
        cls, operand: Dataset, period_from: Optional[str], period_to: str, conf: Optional[str]
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
        result.data = result.data[result.get_components_names()]
        return result

    @classmethod
    def component_evaluation(
        cls,
        operand: DataComponent,
        period_from: Optional[str],
        period_to: str,
        conf: Optional[str],
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
        cls, operand: Scalar, period_from: Optional[str], period_to: str, conf: Optional[str]
    ) -> Scalar:
        result = cls.scalar_validation(operand, period_from, period_to, conf)
        if operand.value is not None:
            result.value = cls._execute_time_aggregation(
                operand.value, operand.data_type, period_from, period_to, conf
            )
        return result

    @classmethod
    def validate(
        cls,
        operand: Union[Dataset, DataComponent, Scalar],
        period_from: Optional[str],
        period_to: Optional[str],
        conf: Optional[str],
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
        conf: Optional[str],
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
    period_value = date_to_period(parse_date_value(v), to_param)
    if start:
        return period_value.start_date()
    return period_value.end_date()


class Current_Date(Time):
    @classmethod
    def validate(cls) -> Scalar:
        return Scalar(name="current_date", data_type=Date, value=None, nullable=False)

    @classmethod
    def evaluate(cls) -> Scalar:
        result = cls.validate()
        result.value = date.today().isoformat()
        return result


class SimpleBinaryTime(Operators.Binary):
    """Binary time operators taking a scalar or component on both sides.

    The two operands do not have to share a time type. datediff is typed on a plain
    time operand and states no additional constraints, and the reference manual mixes a
    Date with a Time_Period both in its valid syntaxes and in its example, so the
    ordinary implicit promotion applies here rather than a stricter rule.
    """

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
        # datediff counts the days between dates or time periods: a Time value holds two
        # dates. The reference manual types the operands as time, so the value is checked,
        # not the type (issue #1192)
        if (x.count("/") >= 1) or (y.count("/") >= 1):
            raise RunTimeError("2-1-19-25", op=cls.op, comp_type="Date or Time_Period value")

        if x.count("-") == 2:
            date_1 = parse_date_value(x)
        else:
            date_1 = TimePeriodHandler(x).end_date(as_date=True)  # type: ignore[assignment]

        if y.count("-") == 2:
            date_2 = parse_date_value(y)
        else:
            date_2 = TimePeriodHandler(y).end_date(as_date=True)  # type: ignore[assignment]

        return abs((date_2 - date_1).days)


class Date_Add(Parametrized):
    op = DATE_ADD

    @classmethod
    def _dateadd_targets(cls, operand: Dataset) -> List[Component]:
        """The Components a Data Set operand shifts.

        The reference manual types the operand as ``dataset {identifier<time> _,
        identifier _*}``, so the reference time Identifier is what moves. A Data Set
        whose only time Components are Measures is accepted as well, and those Measures
        move instead, which is what the operator did before it reached an Identifier at
        all. Components of any other type are left alone either way, since
        there is no date to add to.
        """
        time_ids = [c for c in operand.get_identifiers() if c.data_type in (Date, TimePeriod)]
        if time_ids:
            return time_ids
        return [m for m in operand.get_measures() if m.data_type in (Date, TimePeriod)]

    @classmethod
    def validate(
        cls, operand: Union[Scalar, DataComponent, Dataset], param_list: List[Scalar]
    ) -> Union[Scalar, DataComponent, Dataset]:
        dataset_name = VirtualCounter._new_ds_name()
        expected_types = [Integer, String]
        for i, param in enumerate(param_list):
            if not isinstance(param, Scalar):
                raise SemanticError(
                    "1-1-19-14",
                    op=cls.op,
                    type=param.__class__.__name__,
                    name="shiftNumber",
                    expected="Scalar",
                )
            if param.data_type != expected_types[i]:
                raise SemanticError(
                    "1-1-19-15",
                    op=cls.op,
                    type=param.data_type.__name__,
                    name="periodInd",
                    expected=expected_types[i].__name__,
                )

        period_ind = param_list[1].value if len(param_list) > 1 else None
        if isinstance(period_ind, str) and period_ind not in PERIOD_IND_MAPPING:
            # A literal is part of the script, a scalar value is data (issue #1192)
            if param_list[1].literal:
                raise SemanticError("1-1-19-17", period=period_ind)
            raise RunTimeError("2-1-19-2", period=period_ind)

        if isinstance(operand, (Scalar, DataComponent)) and operand.data_type not in [
            Date,
            TimePeriod,
        ]:
            unary_implicit_promotion(operand.data_type, Date)

        # A null shift or period gives null, so a result shifted by one is nullable (issue #1354)
        null_param = any(Operators.may_be_null(param) for param in param_list)
        if isinstance(operand, Scalar):
            return Scalar(
                name=operand.name,
                data_type=Date,
                value=None,
                nullable=operand.nullable or null_param,
            )
        if isinstance(operand, DataComponent):
            return DataComponent(
                name=operand.name,
                data_type=Date,
                data=None,
                nullable=operand.nullable or null_param,
            )

        if all(comp.data_type not in [Date, TimePeriod] for comp in operand.components.values()):
            raise SemanticError("1-1-19-16", op=cls.op, name=operand.name)
        # Fresh Component copies: the retype below must not reach the operand's
        # (shared) Component objects (#1032)
        result_components = {name: copy(comp) for name, comp in operand.components.items()}
        for target in cls._dateadd_targets(operand):
            # The shifted Dates are new values, of the basic type Date (issue #1348)
            result_components[target.name].data_type = Date
            result_components[target.name].value_domain = None
            if target.role == Role.MEASURE:
                result_components[target.name].nullable = target.nullable or null_param
        return Dataset(name=dataset_name, components=result_components, data=None)

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
            for comp in cls._dateadd_targets(operand):
                is_period = comp.data_type == TimePeriod
                result.data[comp.name] = result.data[comp.name].map(
                    lambda x: cls.py_op(str(x), shift, period, is_period),  # noqa: B023
                    na_action="ignore",
                )

        return result

    @classmethod
    def py_op(
        cls, date_str: str, shift: Optional[int], period: Optional[str], is_tp: bool = False
    ) -> Optional[str]:
        if shift is None or period is None:
            # A null shift or period gives null, as any null operand (issue #1192)
            return None
        clock = None
        if is_tp:
            tp_value = TimePeriodHandler(date_str)
            if tp_value.period_indicator == "W":
                # The last day of a week of 9999 may be in the year 10000, which no date holds
                ordinal = cast(date, tp_value.start_date(as_date=True)).toordinal() + 6
            else:
                ordinal = cast(date, tp_value.end_date(as_date=True)).toordinal()
        else:
            dt_val = datetime.fromisoformat(date_str)
            ordinal = dt_val.toordinal()
            clock = dt_val.timetz() if _has_time_component(date_str) else None

        # The result is worked out on the calendar, also out of the years a date holds, and is
        # a Date only in the years a Date has (issue #1352)
        if period in ["D", "W"]:
            day, years = date_of_ordinal(ordinal + shift * (7 if period == "W" else 1))
            year, month, day_number = day.year + years, day.month, day.day
            check_year(year)
        else:
            day, years = date_of_ordinal(ordinal)
            months = (day.year + years) * 12 + day.month - 1
            months += {"M": 1, "Q": 3, "S": 6, "A": 12}[period] * shift
            year, month = months // 12, months % 12 + 1
            check_year(year)
            # The day is kept, or the last day of a shorter month taken
            day_number = min(day.day, calendar.monthrange(year, month)[1])
        result = date(year, month, day_number)
        if clock is not None:
            return datetime.combine(result, clock).isoformat(sep=" ")
        return result.isoformat()


class SimpleUnaryTime(Operators.Unary):
    @classmethod
    def validate(
        cls, operand: Union[Dataset, DataComponent, Scalar]
    ) -> Union[Dataset, DataComponent, Scalar]:
        if isinstance(operand, Dataset):
            raise SemanticError("1-1-19-8", op=cls.op, comp_type="time dataset")

        # Limit the operand to Date and TimePeriod (cannot be implemented with type_to_check). A
        # Duration is no time value (issue #1356)
        if operand.data_type not in (Date, TimePeriod):
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
            return parse_date_value(value).month

        result = TimePeriodHandler(value).start_date(as_date=True)
        return result.month  # type: ignore[union-attr]


class Day_of_Month(SimpleUnaryTime):
    op = DAYOFMONTH
    return_type = Integer

    @classmethod
    def py_op(cls, value: str) -> int:
        if value.count("-") == 2:
            return parse_date_value(value).day

        result = TimePeriodHandler(value).end_date(as_date=True)
        return result.day  # type: ignore[union-attr]


class Day_of_Year(SimpleUnaryTime):
    op = DAYOFYEAR
    return_type = Integer

    @classmethod
    def py_op(cls, value: str) -> int:
        if value.count("-") == 2:
            d = parse_date_value(value)
            return d.timetuple().tm_yday

        result = TimePeriodHandler(value).end_date(as_date=True)
        datetime_value = datetime(
            year=result.year,  # type: ignore[union-attr]
            month=result.month,  # type: ignore[union-attr]
            day=result.day,  # type: ignore[union-attr]
        )
        return datetime_value.timetuple().tm_yday


class DaysToDuration(Operators.Unary):
    """daytoyear and daytomonth take an Integer number of days. A Number, which this engine
    promotes to an Integer elsewhere, is refused, as the VTL 2.2 manual types the operand
    integer (issue #1356)."""

    type_to_check = Integer
    return_type = String

    @classmethod
    def validate_type_compatibility(cls, operand: Any) -> bool:
        return operand != Number and super().validate_type_compatibility(operand)

    @classmethod
    def type_validation(cls, operand: Any) -> Any:
        if operand == Number:
            raise SemanticError("1-1-1-1", type_1="Number", type_2="Integer")
        return super().type_validation(operand)


class Day_to_Year(DaysToDuration):
    op = DAYTOYEAR

    @classmethod
    def py_op(cls, value: int) -> str:
        if value < 0:
            raise RunTimeError("2-1-19-16", op=cls.op)
        years = 0
        days_remaining = value
        if value >= 365:
            years = value // 365
            days_remaining = value % 365
        return f"P{int(years)}Y{int(days_remaining)}D"


class Day_to_Month(DaysToDuration):
    op = DAYTOMONTH

    @classmethod
    def py_op(cls, value: int) -> str:
        if value < 0:
            raise RunTimeError("2-1-19-16", op=cls.op)
        months = 0
        days_remaining = value
        if value >= 30:
            months = value // 30
            days_remaining = value % 30
        return f"P{int(months)}M{int(days_remaining)}D"


class Year_to_Day(Operators.Unary):
    op = YEARTODAY
    return_type = Integer
    # The duration PnYnD or PnMnD, a String in this engine (issue #1356)
    type_to_check = String
    _duration_pattern = re.compile(r"^P(?=\d)(\d+Y)?(\d+D)?$")

    @classmethod
    def py_op(cls, value: str) -> int:
        if not cls._duration_pattern.match(value):
            raise RunTimeError("2-1-19-22", op=cls.op, value=value, expected="PnYnD")
        return cls.to_days(value)


class Month_to_Day(Operators.Unary):
    op = MONTHTODAY
    return_type = Integer
    # The duration PnYnD or PnMnD, a String in this engine (issue #1356)
    type_to_check = String
    _duration_pattern = re.compile(r"^P(?=\d)(\d+M)?(\d+D)?$")

    @classmethod
    def py_op(cls, value: str) -> int:
        if not cls._duration_pattern.match(value):
            raise RunTimeError("2-1-19-22", op=cls.op, value=value, expected="PnMnD")
        return cls.to_days(value)
