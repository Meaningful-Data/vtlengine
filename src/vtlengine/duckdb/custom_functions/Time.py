from datetime import date, datetime
from typing import Union, Optional

from vtlengine.DataTypes import Duration
from vtlengine.DataTypes.TimeHandling import TimePeriodHandler
from vtlengine.Exceptions import SemanticError


def year_duck(value: Optional[Union[date, str]])-> Optional[int]:
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

def month_duck(value: Optional[Union[date, str]])-> Optional[int]:
    if value is None:
        return None

    if isinstance(value, date):
        return value.month

    if isinstance(value, str):
        if value.count("-") == 2:
            return date.fromisoformat(value).month
        else:
            result = TimePeriodHandler(value).start_date(as_date=True)
            return result.month

def day_of_month_duck(value: Optional[Union[date, str]])-> Optional[int]:
    if value is None:
        return None

    if isinstance(value, date):
        return value.day

    if isinstance(value, str):
        if value.count("-") == 2:
            return date.fromisoformat(value).day
        else:
            result = TimePeriodHandler(value).end_date(as_date=True)
            return result.day

def day_of_year_duck(value: Optional[Union[date, str]])-> Optional[int]:
    if value is None:
        return None

    if isinstance(value, date):
        return value.timetuple().tm_yday

    if isinstance(value, str):
        if value.count("-") == 2:
            return date.fromisoformat(value).timetuple().tm_yday
        else:
            result = TimePeriodHandler(value).end_date(as_date=True)
            return result.timetuple().tm_yday

def day_to_year_duck(value: int) -> Optional[str]:
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
    if value is None:
        return None
    if isinstance(value, str):
        days = Duration.to_days(value)
        return days

def month_to_day_duck(value: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str):
        days = Duration.to_days(value)
        return days