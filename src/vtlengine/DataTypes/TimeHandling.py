import calendar
import copy
import operator
from datetime import date
from datetime import datetime as dt
from typing import Any, Dict, Optional, Union

import pandas as pd

from vtlengine.Exceptions import SemanticError

PERIOD_IND_MAPPING = {"A": 6, "S": 5, "Q": 4, "M": 3, "W": 2, "D": 1}

PERIOD_IND_MAPPING_REVERSE = {6: "A", 5: "S", 4: "Q", 3: "M", 2: "W", 1: "D"}

PERIOD_INDICATORS = ["A", "S", "Q", "M", "W", "D"]


def date_to_period(date_value: date, period_indicator: str) -> Any:
    if period_indicator == "A":
        return TimePeriodHandler(f"{date_value.year}A")
    elif period_indicator == "S":
        return TimePeriodHandler(f"{date_value.year}S{((date_value.month - 1) // 6) + 1}")
    elif period_indicator == "Q":
        return TimePeriodHandler(f"{date_value.year}Q{((date_value.month - 1) // 3) + 1}")
    elif period_indicator == "M":
        return TimePeriodHandler(f"{date_value.year}M{date_value.month}")
    elif period_indicator == "W":
        cal = date_value.isocalendar()
        return TimePeriodHandler(f"{cal[0]}W{cal[1]}")
    elif period_indicator == "D":  # Extract day of the year
        return TimePeriodHandler(f"{date_value.year}D{date_value.timetuple().tm_yday}")


def period_to_date(
    year: int, period_indicator: str, period_number: int, start: bool = False
) -> date:
    if period_indicator == "A":
        return date(year, 1, 1) if start else date(year, 12, 31)
    periods = {
        "S": [
            (date(year, 1, 1), date(year, 6, 30)),
            (date(year, 7, 1), date(year, 12, 31)),
        ],
        "Q": [
            (date(year, 1, 1), date(year, 3, 31)),
            (date(year, 4, 1), date(year, 6, 30)),
            (date(year, 7, 1), date(year, 9, 30)),
            (date(year, 10, 1), date(year, 12, 31)),
        ],
    }
    if period_indicator in periods:
        start_date, end_date = periods[period_indicator][period_number - 1]
        return start_date if start else end_date
    if period_indicator == "M":
        day = 1 if start else calendar.monthrange(year, period_number)[1]
        return date(year, period_number, day)
    if period_indicator == "W":
        week_day = 1 if start else 0
        return dt.strptime(f"{year}-W{period_number}-{week_day}", "%G-W%V-%w").date()
    if period_indicator == "D":
        return dt.strptime(f"{year}-D{period_number}", "%Y-D%j").date()
    raise SemanticError("2-1-19-2", period=period_indicator)


def day_of_year(date: str) -> int:
    """
    Returns the day of the year for a given date string
    2020-01-01 -> 1
    """
    # Convert the date string to a datetime object
    date_object = dt.strptime(date, "%Y-%m-%d")
    # Get the day number in the year
    day_number = date_object.timetuple().tm_yday
    return day_number


def from_input_customer_support_to_internal(period: str) -> tuple[int, str, int]:
    """
    Converts a period string from the input customer support format to the internal format
    2020-01-01 -> (2020, 'D', 1)
    2020-01 -> (2020, 'M', 1)
    2020-Q1 -> (2020, 'Q', 1)
    2020-S1 -> (2020, 'S', 1)
    2020-M01 -> (2020, 'M', 1)
    2020-W01 -> (2020, 'W', 1)
    """
    parts = period.split("-")
    year = int(parts[0])
    if len(parts) == 3:  # 'YYYY-MM-DD' case
        return year, "D", int(day_of_year(period))
    second_term = parts[1]
    length = len(second_term)
    if length == 4:  # 'YYYY-Dxxx' case
        return year, "D", int(second_term[1:])
    if length == 3:  # 'YYYY-Wxx' or 'YYYY-Mxx' case
        return year, second_term[0], int(second_term[1:])
    if length == 2:  # 'YYYY-Qx', 'YYYY-Sx', 'YYYY-Ax', or 'YYYY-MM' case
        indicator = second_term[0]
        return (
            (year, indicator, int(second_term[1:]))
            if indicator in PERIOD_INDICATORS
            else (year, "M", int(second_term))
        )
    raise SemanticError("2-1-19-6", period_format=period)
    # raise ValueError


class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances: Dict[Any, Any] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class PeriodDuration(metaclass=SingletonMeta):
    periods = {"D": 366, "W": 53, "M": 12, "Q": 4, "S": 2, "A": 1}

    def __contains__(self, item: Any) -> bool:
        return item in self.periods

    @property
    def member_names(self) -> list[str]:
        return list(self.periods.keys())

    @classmethod
    def check_period_range(cls, letter: str, value: Any) -> bool:
        if letter == "A":
            return True
        return value in range(1, cls.periods[letter] + 1)


class TimePeriodHandler:
    _year: int
    _period_indicator: str
    _period_number: int

    def __init__(self, period: str) -> None:
        if isinstance(period, int):
            period = str(period)
        if "-" in period:
            self.year, self.period_indicator, self.period_number = (
                from_input_customer_support_to_internal(period)
            )
        else:
            self.year = int(period[:4])
            if len(period) > 4:
                self.period_indicator = period[4]
            else:
                self.period_indicator = "A"
            if len(period) > 5:
                self.period_number = int(period[5:])
            else:
                self.period_number = 1

    def __str__(self) -> str:
        if self.period_indicator == "A":
            # return f"{self.year}{self.period_indicator}"
            return f"{self.year}"  # Drop A from exit time period year
        if self.period_indicator in ["W", "M"]:
            period_number_str = f"{self.period_number:02}"
        elif self.period_indicator == "D":
            period_number_str = f"{self.period_number:03}"
        else:
            period_number_str = str(self.period_number)
        return f"{self.year}-{self.period_indicator}{period_number_str}"

    @staticmethod
    def _check_year(year: int) -> None:
        if year < 1900 or year > 9999:
            raise SemanticError("2-1-19-10", year=year)
            # raise ValueError(f'Invalid year {year}, must be between 1900 and 9999.')

    @property
    def year(self) -> int:
        return self._year

    @year.setter
    def year(self, value: int) -> None:
        self._check_year(value)
        self._year = value

    @property
    def period_indicator(self) -> str:
        return self._period_indicator

    @period_indicator.setter
    def period_indicator(self, value: str) -> None:
        if value not in PeriodDuration():
            raise SemanticError("2-1-19-2", period=value)
        self._period_indicator = value

    @property
    def period_magnitude(self) -> int:
        return PERIOD_IND_MAPPING[self.period_indicator]

    @property
    def period_number(self) -> int:
        return self._period_number

    @period_number.setter
    def period_number(self, value: int) -> None:
        if not PeriodDuration.check_period_range(self.period_indicator, value):
            raise SemanticError(
                "2-1-19-7",
                periods=PeriodDuration.periods[self.period_indicator],
                period_indicator=self.period_indicator,
            )
            # raise ValueError(f'Period Number must be between 1 and '
            #                  f'{PeriodDuration.periods[self.period_indicator]} '
            #                  f'for period indicator {self.period_indicator}.')
        # check day is correct for year
        if self.period_indicator == "D":
            if calendar.isleap(self.year):
                if value > 366:
                    raise SemanticError("2-1-19-9", day=value, year=self.year)
                    # raise ValueError(f'Invalid day {value} for year {self.year}.')
            else:
                if value > 365:
                    raise SemanticError("2-1-19-9", day=value, year=self.year)
                    # raise ValueError(f'Invalid day {value} for year {self.year}.')
        self._period_number = value

    @property
    def period_dates(self) -> tuple[date, date]:
        return (
            period_to_date(self.year, self.period_indicator, self.period_number, start=True),
            period_to_date(self.year, self.period_indicator, self.period_number, start=False),
        )

    def _meta_comparison(self, other: Any, py_op: Any) -> Optional[bool]:
        if pd.isnull(other):
            return None

        if py_op in (operator.eq, operator.ne):
            return py_op(str(self), str(other))

        if py_op in (operator.ge, operator.le) and str(self) == str(other):
            return True

        if isinstance(other, str):
            other = TimePeriodHandler(other)

        self_lapse, other_lapse = self.period_dates, other.period_dates
        is_lt_or_le = py_op in [operator.lt, operator.le]
        is_gt_or_ge = py_op in [operator.gt, operator.ge]

        if is_lt_or_le or is_gt_or_ge:
            idx = 0 if is_lt_or_le else 1
            if self_lapse[idx] != other_lapse[idx]:
                return (
                    self_lapse[idx] < other_lapse[idx]
                    if is_lt_or_le
                    else self_lapse[idx] > other_lapse[idx]
                )
            if self.period_magnitude != other.period_magnitude:
                return (
                    self.period_magnitude < other.period_magnitude
                    if is_lt_or_le
                    else self.period_magnitude > other.period_magnitude
                )

        return False

    def start_date(self, as_date: bool = False) -> Union[date, str]:
        """
        Gets the starting date of the Period
        """
        date_value = period_to_date(
            year=self.year,
            period_indicator=self.period_indicator,
            period_number=self.period_number,
            start=True,
        )
        return date_value if as_date else date_value.isoformat()

    def end_date(self, as_date: bool = False) -> Union[date, str]:
        """
        Gets the ending date of the Period
        """
        date_value = period_to_date(
            year=self.year,
            period_indicator=self.period_indicator,
            period_number=self.period_number,
            start=False,
        )
        return date_value if as_date else date_value.isoformat()

    def __eq__(self, other: Any) -> Optional[bool]:  # type: ignore[override]
        return self._meta_comparison(other, operator.eq)

    def __ne__(self, other: Any) -> Optional[bool]:  # type: ignore[override]
        return not self._meta_comparison(other, operator.eq)

    def __lt__(self, other: Any) -> Optional[bool]:
        return self._meta_comparison(other, operator.lt)

    def __le__(self, other: Any) -> Optional[bool]:
        return self._meta_comparison(other, operator.le)

    def __gt__(self, other: Any) -> Optional[bool]:
        return self._meta_comparison(other, operator.gt)

    def __ge__(self, other: Any) -> Optional[bool]:
        return self._meta_comparison(other, operator.ge)

    def change_indicator(self, new_indicator: str) -> None:
        if self.period_indicator == new_indicator:
            return
        date_value = period_to_date(self.year, self.period_indicator, self.period_number)
        self.period_indicator = new_indicator
        self.period_number = date_to_period(
            date_value, period_indicator=new_indicator
        ).period_number

    def vtl_representation(self) -> str:
        if self.period_indicator == "A":
            return f"{self.year}"  # Drop A from exit time period year
        if self.period_indicator in ["W", "M"]:
            period_number_str = f"{self.period_number:02}"
        elif self.period_indicator == "D":
            period_number_str = f"{self.period_number:03}"
        else:
            period_number_str = str(self.period_number)
        return f"{self.year}{self.period_indicator}{period_number_str}"

    def sdmx_gregorian_representation(self) -> None:
        raise NotImplementedError


class TimeIntervalHandler:
    _date1: str = "0"
    _date2: str = "Z"

    def __init__(self, date1: str, date2: str) -> None:
        self.set_date1(date1)
        self.set_date2(date2)
        # if date1 > date2:
        #     raise ValueError(f'Invalid Time with duration less than 0 ({self.length} days)')

    @classmethod
    def from_dates(cls, date1: date, date2: date) -> "TimeIntervalHandler":
        return cls(date1.isoformat(), date2.isoformat())

    @classmethod
    def from_iso_format(cls, dates: str) -> "TimeIntervalHandler":
        return cls(*dates.split("/", maxsplit=1))

    @property
    def date1(self, as_date: bool = False) -> Union[date, str]:
        return date.fromisoformat(self._date1) if as_date else self._date1

    @property
    def date2(self, as_date: bool = False) -> Union[date, str]:
        return date.fromisoformat(self._date2) if as_date else self._date2

    # @date1.setter
    def set_date1(self, value: str) -> None:
        date.fromisoformat(value)
        if value > self.date2.__str__():
            raise SemanticError("2-1-19-4", date=self.date2, value=value)
            # raise ValueError(f"({value} > {self.date2}).
            # Cannot set date1 with a value greater than date2.")
        self._date1 = value

    def set_date2(self, value: str) -> None:
        date.fromisoformat(value)
        if value < self.date1.__str__():
            raise SemanticError("2-1-19-5", date=self.date1, value=value)
            # raise ValueError(f"({value} < {self.date1}).
            # Cannot set date2 with a value lower than date1.")
        self._date2 = value

    @property
    def length(self) -> int:
        date_left = date.fromisoformat(self.date1.__str__())
        date_right = date.fromisoformat(self.date2.__str__())
        return (date_right - date_left).days

    __len__ = length

    def __str__(self) -> str:
        return f"{self.date1}/{self.date2}"

    __repr__ = __str__

    def _meta_comparison(self, other: Any, py_op: Any) -> Optional[bool]:
        if pd.isnull(other):
            return None
        if isinstance(other, str):
            if len(other) == 0:
                return False
            other = TimeIntervalHandler(*other.split("/", maxsplit=1))
        return py_op(self.length, other.length)

    def __eq__(self, other: Any) -> Optional[bool]:  # type: ignore[override]
        return self._meta_comparison(other, operator.eq)

    def __ne__(self, other: Any) -> Optional[bool]:  # type: ignore[override]
        return self._meta_comparison(other, operator.ne)

    def __lt__(self, other: Any) -> Optional[bool]:
        return self._meta_comparison(other, operator.lt)

    def __le__(self, other: Any) -> Optional[bool]:
        return self._meta_comparison(other, operator.le)

    def __gt__(self, other: Any) -> Optional[bool]:
        return self._meta_comparison(other, operator.gt)

    def __ge__(self, other: Any) -> Optional[bool]:
        return self._meta_comparison(other, operator.ge)

    @classmethod
    def from_time_period(cls, value: TimePeriodHandler) -> "TimeIntervalHandler":
        date1 = period_to_date(value.year, value.period_indicator, value.period_number, start=True)
        date2 = period_to_date(value.year, value.period_indicator, value.period_number, start=False)
        return cls.from_dates(date1, date2)


def sort_dataframe_by_period_column(
    data: pd.DataFrame, name: str, identifiers_names: list[str]
) -> pd.DataFrame:
    """
    Sorts dataframe by TimePeriod period_indicator and period_number.
    Assuming all values are present (only for identifiers)
    """
    new_component_name = "@period_number"

    # New auxiliary component with pandas type datetime for sorting
    data["duration_var"] = data[name].map(lambda x: x.period_indicator)
    data[new_component_name] = data[name].map(lambda x: x.period_number)
    identifiers_names.append("duration_var")
    identifiers_names.append(new_component_name)
    # Sort the rows by identifiers
    data = data.sort_values(by=identifiers_names)
    # Drop the new auxiliary component
    del data["duration_var"]
    del data[new_component_name]

    identifiers_names.remove("duration_var")
    identifiers_names.remove(new_component_name)
    return data


def next_period(x: TimePeriodHandler) -> TimePeriodHandler:
    y = copy.copy(x)
    if y.period_number == PeriodDuration.periods[x.period_indicator]:
        y.year += 1
        y.period_number = 1
    else:
        y.period_number += 1
    return y


def previous_period(x: TimePeriodHandler) -> TimePeriodHandler:
    y = copy.copy(x)
    if x.period_number == 1:
        y.year -= 1
        y.period_number = PeriodDuration.periods[x.period_indicator]
    else:
        y.period_number -= 1
    return y


def shift_period(x: TimePeriodHandler, shift_param: int) -> TimePeriodHandler:
    if x.period_indicator == "A":
        x.year += shift_param
        return x
    for _ in range(abs(shift_param)):
        x = next_period(x) if shift_param >= 0 else previous_period(x)
    return x


def sort_time_period(series: Any) -> Any:
    values_sorted = sorted(
        series.to_list(),
        key=lambda s: (s.year, PERIOD_IND_MAPPING[s.period_indicator], s.period_number),
    )
    return pd.Series(values_sorted, name=series.name)


def generate_period_range(
    start: TimePeriodHandler, end: TimePeriodHandler
) -> list[TimePeriodHandler]:
    period_range = [start]
    if start.period_indicator != end.period_indicator:
        raise SemanticError(
            "2-1-19-3", period1=start.period_indicator, period2=end.period_indicator
        )
        # raise Exception("Only same period indicator allowed")
    if start.period_indicator == "A":
        for _ in range(end.year - start.year):
            period_range.append(next_period(period_range[-1]))
        return period_range
    while str(end) != str(period_range[-1]):
        period_range.append(next_period(period_range[-1]))

    return period_range


def check_max_date(str_: Optional[str]) -> Optional[str]:
    if pd.isnull(str_) or str_ == "nan" or str_ == "NaT" or str_ is None:
        return None

    if len(str_) == 9 and str_[7] == "-":
        str_ = str_[:-1] + "0" + str_[-1]

    # Format 2010-01-01. Prevent passthrough of other ISO 8601 formats.
    if len(str_) != 10 or str_[7] != "-":
        raise SemanticError("2-1-19-8", date=str_)
        # raise ValueError(f"Invalid date format, must be YYYY-MM-DD: {str_}")

    result = date.fromisoformat(str_)
    return result.isoformat()


def str_period_to_date(value: str, start: bool = False) -> Any:
    if len(value) < 6:
        return date(int(value[:4]), 1, 1) if start else date(int(value[:4]), 12, 31)
    return (
        TimePeriodHandler(value).start_date(as_date=False)
        if start
        else (TimePeriodHandler(value).end_date(as_date=False))
    )


def date_to_period_str(date_value: date, period_indicator: str) -> Any:
    if isinstance(date_value, str):
        date_value = check_max_date(date_value)
        date_value = date.fromisoformat(date_value)
    if period_indicator == "A":
        return f"{date_value.year}A"
    elif period_indicator == "S":
        return f"{date_value.year}S{((date_value.month - 1) // 6) + 1}"
    elif period_indicator == "Q":
        return f"{date_value.year}Q{((date_value.month - 1) // 3) + 1}"
    elif period_indicator == "M":
        return f"{date_value.year}M{date_value.month}"
    elif period_indicator == "W":
        cal = date_value.isocalendar()
        return f"{cal[0]}W{cal[1]}"
    elif period_indicator == "D":  # Extract day of the year
        return f"{date_value.year}D{date_value.timetuple().tm_yday}"
