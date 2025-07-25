from datetime import date, datetime, timedelta
from typing import Union, Optional

from vtlengine.DataTypes import Duration
from vtlengine.DataTypes.TimeHandling import TimePeriodHandler, period_to_date
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

def date_diff_duck(x: Union[str, date], y: Union[str, date]) -> Optional[int]:
    if x is None or y is None:
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

    return abs((y - x).days)


def date_add_duck(value: Union[date, str], period: str, shift: int) -> date:
    if isinstance(value, str):
        tp_value = TimePeriodHandler(value)
        date_value = period_to_date(tp_value.year, tp_value.period_indicator, tp_value.period_number)
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
        new_date = date_value.replace(year=new_year, month=new_month, day=min(date_value.day, last_day))

    return new_date