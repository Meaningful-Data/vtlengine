from datetime import date, datetime, timedelta
from typing import Any, Optional, Union

from vtlengine.DataTypes import Duration
from vtlengine.DataTypes.TimeHandling import TimePeriodHandler, date_to_period, period_to_date
from vtlengine.Exceptions import SemanticError


def year_duck(value: Optional[Union[date, str]]) -> Optional[int]:
    """
    Extracts the year from a date or time value.
    If the input value is None or NaN, it returns None.
    If the input value is not a valid date or time, it raises a SemanticError.

    Args:
        value (str): The date or time value in string format.

    Returns:
        int | None: The extracted year as an integer, or None if the input is None or NaN.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return int(value[:4])
    return value.year


def month_duck(value: Optional[Union[date, str]]) -> Optional[int]:
    """
    Extracts the month from a date or time value.
    Args:
        value (Optional[Union[date, str]]): The date or time value,
        which can be a date object or a string in ISO format.

    Returns:
        int | None: The extracted month as an integer, or None if the input is None or NaN.

    """
    if value is None:
        return None

    if isinstance(value, date):
        return value.month

    if isinstance(value, str):
        if value.count("-") == 2:
            return date.fromisoformat(value).month
        else:
            result = TimePeriodHandler(value).start_date(as_date=True)
            return result.month  # type: ignore[union-attr]


def day_of_month_duck(value: Optional[Union[date, str]]) -> Optional[int]:
    """Extracts the day of the month from a date or time value.
    Args:
        value (Optional[Union[date, str]]): The date or time value,
        which can be a date object or a string in ISO format.

    Returns:
        int | None: The extracted day of the month as an integer,
        or None if the input is None or NaN.
    """
    if value is None:
        return None

    if isinstance(value, date):
        return value.day

    if isinstance(value, str):
        if value.count("-") == 2:
            return date.fromisoformat(value).day
        else:
            result = TimePeriodHandler(value).end_date(as_date=True)
            return result.day  # type: ignore[union-attr]


def day_of_year_duck(value: Optional[Union[date, str]]) -> Optional[int]:
    """Extracts the day of the year from a date or time value.
    Args:
        value (Optional[Union[date, str]]): The date or time value,
        which can be a date object or a string in ISO format.

    Returns:
        int | None: The extracted day of the year as an integer,
        or None if the input is None or NaN.
    """
    if value is None:
        return None

    if isinstance(value, date):
        return value.timetuple().tm_yday

    if isinstance(value, str):
        if value.count("-") == 2:
            return date.fromisoformat(value).timetuple().tm_yday
        else:
            result = TimePeriodHandler(value).end_date(as_date=True)
            return result.timetuple().tm_yday  # type: ignore[union-attr]


def day_to_year_duck(value: int) -> Optional[str]:
    """
    Converts a number of days into a string representation of years and remaining days.
    Args:
        value(int): The number of days to convert.

    Returns:
        str | None: A string in the format "P{years}Y{days}D" representing the duration,
        or None if the input is None.

    """
    if value is None:
        return None
    if isinstance(value, int):
        if value < 0:
            raise SemanticError("2-1-19-16", op="day_to_year_duck")
        years = 0
        days_remaining = value
        if value >= 365:
            years = value // 365
            days_remaining = value % 365
        return f"P{int(years)}Y{int(days_remaining)}D"


def day_to_month_duck(value: int) -> Optional[str]:
    """
    Converts a number of days into a string representation of months and remaining days.
    Args:
        value(int): The number of days to convert.

    Returns:
        str | None: A string in the format "P{months}M{days}D" representing the duration,
        or None if the input is None.

    """
    if value is None:
        return None
    if isinstance(value, int):
        if value < 0:
            raise SemanticError("2-1-19-16", op="day_to_month_duck")
        months = 0
        days_remaining = value
        if value >= 30:
            months = value // 30
            days_remaining = value % 30
        return f"P{int(months)}M{int(days_remaining)}D"


def year_to_day_duck(value: str) -> Optional[int]:
    """
    Converts a duration string representing years into the equivalent number of days.
    Args:
        value(str): The duration string in the format "P{years}Y{days}D".

    Returns:
        The equivalent number of days as an integer, or None if the input is None.
    """
    if value is None:
        return None
    if isinstance(value, str):
        days = Duration.to_days(value)
        return days


def month_to_day_duck(value: str) -> Optional[int]:
    """
    Converts a duration string representing months into the equivalent number of days.
    Args:
        value(str): The duration string in the format "P{months}M{days}D".

    Returns:
        The equivalent number of days as an integer, or None if the input is None.

    """
    if value is None:
        return None
    if isinstance(value, str):
        days = Duration.to_days(value)
        return days


def date_diff_duck(x: Union[str, date], y: Union[str, date]) -> Optional[int]:
    """
    Calculates the absolute difference in days between two date.
    Args:
        x: Left date value, can be a string in ISO format or a date object.
        y: Right date value, can be a string in ISO format or a date object.

    Returns:
        int | None: The absolute difference in days between the two dates, or None if either
        date is None.
    """
    if x is None:
        return None
    if y is None:
        return None
    if isinstance(x, str):
        if x.count("-") == 2:
            x = datetime.strptime(x, "%Y-%m-%d").date()
        else:
            x = TimePeriodHandler(x).end_date(as_date=True)

    if isinstance(y, str):
        if y.count("-") == 2:
            y = datetime.strptime(y, "%Y-%m-%d").date()
        else:
            y = TimePeriodHandler(y).end_date(as_date=True)

    return abs((y - x).days)  # type: ignore[operator]


def date_add_duck(value: Union[date, str], period: str, shift: int) -> date:
    """
    Adds a specified period and shift to a date or time value.
    Args:
        value(Union[date, str]): The date or time value to which the period will be added.
        period(str): The period to add, which can be "D" (day), "W" (week), "M" (month),
            "Q" (quarter), "S" (semester), or "A" (year).
        shift(int): The number of periods to add. Positive values shift forward in time,
            while negative values shift backward.

    Returns:
        date: The new date after adding the specified period and shift.
    """
    if isinstance(value, str):
        tp_value = TimePeriodHandler(value)
        date_value = period_to_date(
            tp_value.year, tp_value.period_indicator, tp_value.period_number
        )
    else:
        date_value = value

    if period in ["D", "W"]:
        days_shift = shift * (7 if period == "W" else 1)
        new_date = date_value + timedelta(days=days_shift)
    else:
        month_shift = {"M": 1, "Q": 3, "S": 6, "A": 12}[period] * shift
        new_year = date_value.year + (date_value.month - 1 + month_shift) // 12
        new_month = (date_value.month - 1 + month_shift) % 12 + 1
        last_day = (datetime(new_year, new_month % 12 + 1, 1) - timedelta(days=1)).day
        new_date = date_value.replace(
            year=new_year, month=new_month, day=min(date_value.day, last_day)
        )

    return new_date


def time_agg_duck(
    value: Optional[Union[str, date]], period_from: Optional[str], period_to: str, conf: str
) -> Optional[str]:
    """
    Aggregates a date or time value to a specified time period.
    Args:
        value (Optional[Union[str, date]]): The date or time value to aggregate.
        period_from (Optional[str]): The time period to aggregate from, not used in this function.
        period_to (str): The time period to aggregate to, such as "D", "W", "M", "Q", "S", or "A".
        conf (str): Configuration parameter, where "first" indicates the start of the period
            and any other value indicates the end of the period.

    Returns:
        str | None: The aggregated date or time value as a string in ISO format,
        or None if the input value is None.
    """

    def time_period_access(v: Any, to_param: str) -> str:
        v = TimePeriodHandler(v)
        if v.period_indicator == to_param:
            return str(v)
        v.change_indicator(to_param)
        return str(v)

    def date_access(value: date, to_param: str, start: bool) -> date:
        period_value = date_to_period(value, to_param)
        if start:
            return period_value.start_date(as_date=True)
        return period_value.end_date(as_date=True)

    if value is None:
        return None
    if isinstance(value, date):
        start = conf == "first"
        return date_access(value, period_to, start).isoformat()
    else:
        return time_period_access(value, period_to)


def period_ind_duck(value: str) -> Optional[str]:
    """
    Returns the period indicator for a given time period string.

    Args:
        value (str): The time period string.

    Returns:
        str: The period indicator, or None if the input is None.
    """
    if value is None:
        return None
    return TimePeriodHandler(value).period_indicator
