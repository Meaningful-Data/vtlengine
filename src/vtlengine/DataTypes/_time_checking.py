import calendar
import re
from datetime import date, datetime

from vtlengine.DataTypes.TimeHandling import TimePeriodHandler
from vtlengine.Exceptions import InputValidationException


def _has_time_component(value: str) -> bool:
    """Check if a date string includes a time component (T or space separator at position 10)."""
    return len(value) > 10 and value[10] in ("T", " ")


def _truncate_nanoseconds(value: str) -> str:
    """Truncate sub-second precision beyond 6 digits (microseconds) for Python compatibility."""
    dot_idx = value.find(".")
    if dot_idx == -1:
        return value
    # Keep at most 6 decimal digits after the dot
    frac_end = dot_idx + 1
    while frac_end < len(value) and value[frac_end].isdigit():
        frac_end += 1
    frac_digits = value[dot_idx + 1 : frac_end]
    if len(frac_digits) > 6:
        return value[:dot_idx] + "." + frac_digits[:6] + value[frac_end:]
    return value


def parse_date_value(value: str) -> date:
    """Parse a date or datetime string into a date object (time part is discarded)."""
    return date.fromisoformat(value[:10])


def check_date(value: str) -> str:
    """
    Check if the date is in the correct format.
    Accepts YYYY-MM-DD and YYYY-MM-DD[T| ]HH:MM:SS[.fffffffff] (ISO 8601).
    Output always uses T separator for datetime values, no timezone.
    Nanosecond input is truncated to microsecond precision.
    """
    value = value.strip()
    has_time = _has_time_component(value)
    try:
        if has_time:
            iso_result = datetime.fromisoformat(_truncate_nanoseconds(value)).isoformat(sep=" ")
        else:
            if len(value) == 9 and value[7] == "-":
                value = value[:-1] + "0" + value[-1]
            iso_result = date.fromisoformat(value).isoformat()
    except ValueError as e:
        if "is out of range" in str(e):
            raise InputValidationException(f"Date {value} is out of range for the month.")
        if "month must be in 1..12" in str(e):
            raise InputValidationException(
                f"Date {value} is invalid. Month must be between 1 and 12."
            )
        raise InputValidationException(
            f"Date {value} is not in the correct format. Use YYYY-MM-DD or YYYY-MM-DD HH:MM:SS."
        )

    # Check date is between 1800 and 9999
    year = int(value[:4])
    if not 1800 <= year <= 9999:
        raise InputValidationException(
            f"Date {value} is invalid. Year must be between 1900 and 9999."
        )

    return iso_result


def dates_to_string(date1: date, date2: date) -> str:
    date1_str = date1.strftime("%Y-%m-%d")
    date2_str = date2.strftime("%Y-%m-%d")
    return f"{date1_str}/{date2_str}"


date_pattern = r"\d{4}[-][0-1]?\d[-][0-3]?\d([T ]\d{2}:\d{2}:\d{2}(\.\d+)?)?"
year_pattern = r"\d{4}"
month_pattern = r"\d{4}[-][0-1]?\d"
time_pattern = r"^" + date_pattern + r"/" + date_pattern + r"$"


def check_time(value: str) -> str:
    value = value.strip()
    year_result = re.fullmatch(year_pattern, value)
    if year_result is not None:
        date1_time = datetime.strptime(value, "%Y")
        date2_time = date1_time.replace(day=31, month=12)
        return dates_to_string(date1_time, date2_time)
    month_result = re.fullmatch(month_pattern, value)
    if month_result is not None:
        date1_time = datetime.strptime(value, "%Y-%m")
        last_month_day = calendar.monthrange(date1_time.year, date1_time.month)[1]
        date2_time = date1_time.replace(day=last_month_day)
        return dates_to_string(date1_time, date2_time)
    time_result = re.fullmatch(time_pattern, value)
    if time_result is not None:
        time_list = value.split("/")
        if time_list[0] > time_list[1]:
            raise ValueError("Start date is greater than end date.")
        return value
    raise ValueError(
        "Time is not in the correct format. Use YYYY-MM-DD/YYYY-MM-DD or YYYY or YYYY-MM."
    )


day_period_pattern = r"^\d{4}[-][0-1]?\d[-][0-3]?\d$"
month_period_pattern = r"^\d{4}[-][0-1]?\d$"
year_period_pattern = r"^\d{4}$"
period_pattern = (
    r"^\d{4}[A]$|^\d{4}[S][1-2]$|^\d{4}[Q][1-4]$|^\d{4}[M]"
    r"[0-1]?\d$|^\d{4}[W][0-5]?\d$|^\d{4}[D][0-3]?[0-9]?\d$"
)

# Related with gitlab issue #440, we can say that period pattern
# matches with our internal representation (or vtl user manual)
# and further_options_period_pattern matches
# with other kinds of inputs that we have to accept for the period.
further_options_period_pattern = (
    r"\d{4}-\d{2}-\d{2}|^\d{4}-D[0-3]\d\d$|^\d{4}-W([0-4]"
    r"\d|5[0-3])|^\d{4}-(0[1-9]|1[0-2]|M(0[1-9]|1[0-2]|[1-9]))$|^"
    r"\d{4}-Q[1-4]$|^\d{4}-S[1-2]$|^\d{4}-A1$"
)


def check_time_period(value: str) -> str:
    if isinstance(value, int):
        value = str(value)
    value = value.strip()

    match = re.fullmatch(r"^(\d{4})-(\d{2})$", value)
    if match:
        value = f"{match.group(1)}-M{match.group(2)}"

    period_result = re.fullmatch(period_pattern, value)
    if period_result is not None:
        result = TimePeriodHandler(value)
        return str(result)

    # We allow the user to input the time period in different formats.
    # See gl-440 or documentation in time period tests.
    further_options_period_result = re.fullmatch(further_options_period_pattern, value)
    if further_options_period_result is not None:
        result = TimePeriodHandler(value)
        return str(result)

    year_result = re.fullmatch(year_period_pattern, value)
    if year_result is not None:
        result = TimePeriodHandler(f"{value}A")
        return str(result)

    month_result = re.fullmatch(month_period_pattern, value)
    if month_result is not None:
        month = datetime.strptime(value, "%Y-%m")
        month_period = month.strftime("%YM%m")
        result = TimePeriodHandler(month_period)
        return str(result)

    # TODO: Do we use this?
    day_result = re.fullmatch(day_period_pattern, value)
    if day_result is not None:
        day = datetime.strptime(value, "%Y-%m-%d")
        day_period = day.strftime("%YD%-j")
        return day_period
    raise ValueError
