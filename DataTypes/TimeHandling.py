import calendar
from datetime import date, datetime as dt
import re
import pandas as pd

def period_to_date(year, period_indicator, period_number, start=False):
    if period_indicator == 'A':
        if start:
            return date(year, 1, 1)
        else:
            return date(year, 12, 31)
    if period_indicator == 'S':
        if period_number == 1:
            if start:
                return date(year, 1, 1)
            else:
                return date(year, 6, 30)
        else:
            if start:
                return date(year, 7, 1)
            else:
                return date(year, 12, 31)
    if period_indicator == 'Q':
        if period_number == 1:
            if start:
                return date(year, 1, 1)
            else:
                return date(year, 3, 31)
        elif period_number == 2:
            if start:
                return date(year, 4, 1)
            else:
                return date(year, 6, 30)
        elif period_number == 3:
            if start:
                return date(year, 7, 1)
            else:
                return date(year, 9, 30)
        else:
            if start:
                return date(year, 10, 1)
            else:
                return date(year, 12, 31)
    if period_indicator == "M":
        if start:
            return date(year, period_number, 1)
        else:
            day = int(calendar.monthrange(year, period_number)[1])
            return date(year, period_number, day)
    if period_indicator == "W":  # 0 for Sunday, 1 for Monday in %w
        if start:
            return dt.strptime(f"{year}-W{period_number}-1", "%G-W%V-%w").date()
        else:
            return dt.strptime(f"{year}-W{period_number}-0", "%G-W%V-%w").date()
    if period_indicator == "D":
        return dt.strptime(f"{year}-D{period_number}", "%Y-D%j").date()

    raise ValueError(f'Invalid Period Indicator {period_indicator}')

def check_max_date(str_: str):
    if pd.isnull(str_) or str_ == 'nan' or str_ == 'NaT':
        return pd.NA

    if len(str_) == 9 and str_[7] == '-':
        str_ = str_[:-1] + '0' + str_[-1]

    if len(str_) != 10 or str_[7] != '-':  # Format 2010-01-01. Prevent passthrough of other ISO 8601 formats.
        raise ValueError

    date.fromisoformat(str_)
    return str_

def str_period_to_date(value: str, start=False) -> date:
    if len(value) < 6:
        if start:
            return date(int(value[:4]), 1, 1)
        else:
            return date(int(value[:4]), 12, 31)

    year = int(value[:4])
    period_indicator = value[4]
    period_number = int(value[5:])

    return period_to_date(year, period_indicator, period_number, start)

def date_to_period_str(date_value: date, period_indicator):
    if isinstance(date_value, str):
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
