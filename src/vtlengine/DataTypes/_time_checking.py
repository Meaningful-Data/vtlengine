import calendar
import re
from datetime import date, datetime
from functools import lru_cache

from vtlengine.DataTypes.TimeHandling import TimePeriodHandler
from vtlengine.Exceptions import InputValidationException


def check_date(value: str) -> str:
    """
    Check if the date is in the correct format.
    """
    # Remove all whitespaces
    value = value.strip()
    try:
        if len(value) == 9 and value[7] == "-":
            value = value[:-1] + "0" + value[-1]
        date_value = date.fromisoformat(value)
    except ValueError as e:
        if "is out of range" in str(e):
            raise InputValidationException(f"Date {value} is out of range for the month.")
        if "month must be in 1..12" in str(e):
            raise InputValidationException(
                f"Date {value} is invalid. Month must be between 1 and 12."
            )
        raise InputValidationException(
            f"Date {value} is not in the correct format. Use YYYY-MM-DD."
        )

    # Check date is between 1900 and 9999
    if not 1800 <= date_value.year <= 9999:
        raise InputValidationException(
            f"Date {value} is invalid. Year must be between 1900 and 9999."
        )

    return date_value.isoformat()


def dates_to_string(date1: date, date2: date) -> str:
    date1_str = date1.strftime("%Y-%m-%d")
    date2_str = date2.strftime("%Y-%m-%d")
    return f"{date1_str}/{date2_str}"


date_pattern = r"\d{4}[-][0-1]?\d[-][0-3]?\d"
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


# Comprehensive time period pattern covering all accepted input formats.
# Compact formats (no hyphen): YYYY, YYYYA, YYYYSN, YYYYQN, YYYYM[M], YYYYW[W], YYYYD[DD]
_vtl_period_re = re.compile(
    r"^\d{4}$"  # YYYY (year only)
    r"|^\d{4}A$"  # YYYYA (annual with indicator)
    r"|^\d{4}S[1-2]$"  # YYYYSN (semester)
    r"|^\d{4}Q[1-4]$"  # YYYYQN (quarter)
    r"|^\d{4}M[0-1]?\d$"  # YYYYM[M] (month, 1 or 2 digits)
    r"|^\d{4}W[0-5]?\d$"  # YYYYW[W] (week, 1 or 2 digits)
    r"|^\d{4}D[0-3]?[0-9]?\d$"  # YYYYD[DD] (day of year, 1 to 3 digits)
)

# Hyphenated formats: YYYY-MM, YYYY-M, YYYY-MM-DD, YYYY-MXX, YYYY-QX, YYYY-SX, YYYY-WXX,
# YYYY-DXXX, YYYY-A1
_sdmx_period_re = re.compile(
    r"^\d{4}-\d{1,2}$"  # YYYY-MM or YYYY-M (ISO month, 1 or 2 digits)
    r"|^\d{4}-\d{2}-\d{2}$"  # YYYY-MM-DD (ISO date)
    r"|^\d{4}-M(0[1-9]|1[0-2]|[1-9])$"  # YYYY-MXX (hyphenated month)
    r"|^\d{4}-Q[1-4]$"  # YYYY-QX (hyphenated quarter)
    r"|^\d{4}-S[1-2]$"  # YYYY-SX (hyphenated semester)
    r"|^\d{4}-W([0-4]\d|5[0-3]|[1-9])$"  # YYYY-WXX (hyphenated week)
    r"|^\d{4}-D[0-3]\d\d$"  # YYYY-DXXX (hyphenated day)
    r"|^\d{4}-A1$"  # YYYY-A1 (SDMX reporting annual)
)

_iso_date_re = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
_iso_month_re = re.compile(r"^(\d{4})-(\d{1,2})$")


def check_time_period(value: str) -> str:
    if isinstance(value, int):
        value = str(value)
    value = value.strip()
    return _check_time_period_cached(value)


@lru_cache(maxsize=4096)
def _check_time_period_cached(value: str) -> str:
    # Try vtl formats first
    if _vtl_period_re.fullmatch(value) is not None:
        result = TimePeriodHandler(f"{value}A") if len(value) == 4 else TimePeriodHandler(value)
        return str(result)

    # Normalize YYYY-M-D, YYYY-M-DD, YYYY-MM-D to zero-padded YYYY-MM-DD
    match_iso_date = _iso_date_re.fullmatch(value)
    if match_iso_date:
        year, month, day = match_iso_date.groups()
        value = f"{year}-{int(month):02d}-{int(day):02d}"

    # Convert YYYY-MM or YYYY-M (ISO month) to hyphenated month format for TimePeriodHandler
    match_iso_month = _iso_month_re.fullmatch(value)
    if match_iso_month:
        value = f"{match_iso_month.group(1)}-M{match_iso_month.group(2)}"

    # Try sdmx formats
    if _sdmx_period_re.fullmatch(value) is not None:
        result = TimePeriodHandler(value)
        return str(result)

    raise ValueError(
        f"Time period '{value}' is not in a valid format. "
        f"Accepted formats: YYYY, YYYYA, YYYYSn, YYYYQn, YYYYMm, YYYYWw, YYYYDd, "
        f"YYYY-MM, YYYY-MM-DD, YYYY-Mxx, YYYY-Qx, YYYY-Sx, YYYY-Wxx, YYYY-Dxxx, YYYY-A1."
    )
