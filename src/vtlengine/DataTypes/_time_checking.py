import calendar
import re
from datetime import date, datetime
from functools import lru_cache

from vtlengine.DataTypes.TimeHandling import TimePeriodHandler
from vtlengine.Exceptions import InputValidationException

_DATE_PART_RE = re.compile(r"^(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})")


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


# A full date followed by a COMPLETE HH:MM:SS time. Hours, minutes AND seconds
# are required when a time component is present; fractional seconds and an optional
# timezone (+HH:MM, -HH:MM or Z) are allowed. Partial times ("HH" / "HH:MM") are
# rejected here; value ranges (hour <= 23, real calendar day) are validated
# afterwards by datetime.fromisoformat.
_STRICT_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?([+-]\d{2}:\d{2}|Z)?$"
)


def normalize_datetime(value: str) -> str:
    """Validate a datetime string carrying a time component and normalize it.

    Requires a full ``YYYY-MM-DD[T| ]HH:MM:SS`` value (hours, minutes and seconds);
    fractional seconds and a timezone suffix are accepted. Sub-second precision beyond
    microseconds is truncated. Output uses a single space separator. Raises
    ``ValueError`` if the time component is missing, truncated, malformed or out of range.
    """
    if _STRICT_DATETIME_RE.match(value) is None:
        raise ValueError(f"Invalid or incomplete time component in {value!r}")
    return datetime.fromisoformat(_truncate_nanoseconds(value)).isoformat(sep=" ")


def parse_date_value(value: str) -> date:
    """Parse a date or datetime string into a date object (time part is discarded)."""
    return date.fromisoformat(value[:10])


def _build_date_error(value: str) -> InputValidationException:
    """Build a precise error for a value that failed date parsing.

    Distinguishes an unusable format, a month outside 1..12, a real-but-out-of-range
    calendar day (e.g. 2020-02-31), and a valid date part carrying an invalid or
    incomplete time (e.g. 2020-01-01T25:00:00, 2020-01-01T12:30, 2020-01-01X12:30:45).
    """
    date_part_match = _DATE_PART_RE.match(value)
    if not date_part_match:
        return InputValidationException(
            f"Date {value} is not in the correct format. Use YYYY-MM-DD or YYYY-MM-DD HH:MM:SS."
        )
    month = int(date_part_match.group("month"))
    if not 1 <= month <= 12:
        return InputValidationException(f"Date {value} is invalid. Month must be between 1 and 12.")
    try:
        date.fromisoformat(value[:10])
    except ValueError:
        return InputValidationException(f"Date {value} is out of range for the month.")
    return InputValidationException(
        f"Date {value} has an invalid or incomplete time; expected YYYY-MM-DD HH:MM:SS."
    )


def check_date(value: str) -> str:
    """Check a date is in the correct format.

    Accepts ``YYYY-MM-DD`` and ``YYYY-MM-DD[T| ]HH:MM:SS[.ffffff][+HH:MM|Z]`` (ISO 8601).
    A time component, when present, must be a COMPLETE ``HH:MM:SS``; partial times such
    as ``HH`` or ``HH:MM`` are rejected. Datetime output uses a single space separator;
    nanosecond input is truncated to microsecond precision.
    """
    value = value.strip()
    has_time = _has_time_component(value)
    try:
        if has_time:
            iso_result = normalize_datetime(value)
        else:
            if len(value) == 9 and value[7] == "-":
                value = value[:-1] + "0" + value[-1]
            iso_result = date.fromisoformat(value).isoformat()
    except ValueError:
        raise _build_date_error(value) from None

    # Check date is between 1800 and 9999
    year = int(value[:4])
    if not 1800 <= year <= 9999:
        raise InputValidationException(
            f"Date {value} is invalid. Year must be between 1800 and 9999."
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


# Comprehensive time period pattern covering all accepted input formats.
# Compact formats (no hyphen): YYYY, YYYYA, YYYYSN, YYYYQN, YYYYM[M], YYYYW[W], YYYYD[DD]
_vtl_period_re = re.compile(
    r"^\d{4}$"  # YYYY (year only)
    r"|^\d{4}A$"  # YYYYA (annual with indicator)
    r"|^\d{4}S[1-2]$"  # YYYYSN (semester)
    r"|^\d{4}Q[1-4]$"  # YYYYQN (quarter)
    r"|^\d{4}M[0-1]?\d$"  # YYYYM[M] (month, 1 or 2 digits)
    r"|^\d{4}W[0-5]?\d$"  # YYYYW[W] (week, 1 or 2 digits)
    r"|^\d{4}D[0-3]?\d{1,2}$"  # YYYYD[DD] (day of year, 1 to 3 digits)
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
    r"|^\d{4}-D[0-3]?\d{1,2}$"  # YYYY-D[XX]X (hyphenated day)
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
