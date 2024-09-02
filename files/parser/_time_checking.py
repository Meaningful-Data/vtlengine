import calendar
from datetime import datetime, date
import re

from DataTypes.TimeHandling import TimePeriodHandler
from Exceptions import InputValidationException


def check_date(value: str):
    """
    Check if the date is in the correct format.
    """
    try:
        date.fromisoformat(value)
    except ValueError:
        raise InputValidationException(f"Date {value} is not in the correct format. "
                                       f"Use YYYY-MM-DD.")

    # Check date is between 1900 and 9999
    if not 1900 <= int(value[:4]) <= 9999:
        raise InputValidationException(f"Date {value} is not in the correct format. "
                                       f"Year must be between 1900 and 9999.")

    return value


def dates_to_string(date1, date2):
    date1_str = date1.strftime('%Y-%m-%d')
    date2_str = date2.strftime('%Y-%m-%d')
    return f"{date1_str}/{date2_str}"


date_pattern = r'\d{4}[-][0-1]?\d[-][0-3]?\d'
year_pattern = r'\d{4}'
month_pattern = r'\d{4}[-][0-1]?\d'
time_pattern = r'^' + date_pattern + r'/' + date_pattern + r'$'


def check_time(value: str):
    year_result = re.fullmatch(year_pattern, value)
    if year_result is not None:
        date1_time = datetime.strptime(value, '%Y')
        date2_time = date1_time.replace(day=31, month=12)
        return dates_to_string(date1_time, date2_time)
    month_result = re.fullmatch(month_pattern, value)
    if month_result is not None:
        date1_time = datetime.strptime(value, '%Y-%m')
        last_month_day = calendar.monthrange(date1_time.year, date1_time.month)[1]
        date2_time = date1_time.replace(day=last_month_day)
        return dates_to_string(date1_time, date2_time)
    time_result = re.fullmatch(time_pattern, value)
    if time_result is not None:
        time_list = value.split('/')
        if time_list[0] > time_list[1]:
            raise ValueError("Start date is greater than end date.")
        return value
    raise ValueError("Time is not in the correct format. "
                     "Use YYYY-MM-DD/YYYY-MM-DD or YYYY or YYYY-MM.")

day_period_pattern = r'^\d{4}[-][0-1]?\d[-][0-3]?\d$'
month_period_pattern = r'^\d{4}[-][0-1]?\d$'
year_period_pattern = r'^\d{4}$'
period_pattern = r'^\d{4}[A]$|^\d{4}[S][1-2]$|^\d{4}[Q][1-4]$|^\d{4}[M][0-1]?\d$|^\d{4}[W][0-5]?\d$|^\d{4}[D][0-3]?[0-9]?\d$'

## Related with gitlab issue #440, we can say that period pattern matches with our internal representation (or vtl user manual)
## and further_options_period_pattern matches with other kinds of inputs that we have to accept for the period.
further_options_period_pattern = r'\d{4}-\d{2}-\d{2}|^\d{4}-D[0-3]\d\d$|^\d{4}-W([0-4]\d|5[0-3])|^\d{4}-(0[1-9]|1[0-2]|M(0[1-9]|1[0-2]))$|^\d{4}-Q[1-4]$|^\d{4}-S[1-2]$|^\d{4}-A1$'

def check_time_period(value: str):
    period_result = re.fullmatch(period_pattern, value)
    if period_result is not None:
        TimePeriodHandler(value)
        return value

    # We allow the user to input the time period in different formats.
    # See gl-440 or documentation in time period tests.
    further_options_period_result = re.fullmatch(further_options_period_pattern, value)
    if further_options_period_result is not None:
        TimePeriodHandler(value)
        return value

    year_result = re.fullmatch(year_period_pattern, value)
    if year_result is not None:
        year = datetime.datetime.strptime(value, '%Y')
        year_period = year.strftime('%YA')
        year_period_wo_A = str(year.year)
        return year_period_wo_A
        # return year_period

    month_result = re.fullmatch(month_period_pattern, value)
    if month_result is not None:
        month = datetime.datetime.strptime(value, '%Y-%m')
        month_period = month.strftime('%YM%m')
        return month_period

    # are we use that? is covered by further option period_pattern
    day_result = re.fullmatch(day_period_pattern, value)
    if day_result is not None:
        day = datetime.datetime.strptime(value, '%Y-%m-%d')
        day_period = day.strftime('%YD%-j')
        return day_period
    raise ValueError
