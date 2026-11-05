"""The formatting masks of cast between strings and the other scalar types.

VTL 2.2, Typical behaviours, "Type Conversion and Formatting Mask". A cast with a mask turns
a string into a number, an integer, a date, a time, a time period or a duration, or one of
them into a string. Both engines call these functions, the DuckDB engine through a UDF, so
they give the same results.

Number masks are made of D (one digit), * (any number of digits), + (at least one digit)
and a . or , separator between the integer and the decimal parts.

Time masks are made of the time units, a digit for each letter: C century, Y year,
S semester, Q quarter, M month, W week, D day, h hour, m minute, s second, d decimal of
second, P period indicator and p number of periods. Braces enclose the digits that may be
left out, as in {M}M, a reverse slash writes the next character as it is, as in \\Q, and
any other character appears as it is. A time is two date masks separated by a /.

Not supported (SemanticError 1-3-13): the exponent E of number masks, the textual words
MONTH, DAY and AM/PM, the ISO composite durations, which a duration of vtlengine, one
period indicator, cannot hold, and the times written with a duration.
"""

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

from vtlengine.DataTypes import (
    SCALAR_TYPES_CLASS_REVERSE,
    Date,
    Duration,
    Integer,
    Number,
    ScalarType,
    String,
    TimeInterval,
    TimePeriod,
)
from vtlengine.DataTypes.TimeHandling import TimePeriodHandler
from vtlengine.Exceptions import RunTimeError, SemanticError

MASK_TYPES = (Date, TimeInterval, TimePeriod, Duration, Number, Integer)

_TIME_UNITS = frozenset("CYSQMWDhmsdPp")
_TEXTUAL_WORDS = ("MONTH", "DAY", "AM/PM")
_PERIOD_UNITS = ("S", "Q", "M", "W", "D")
_TIME_OF_DAY = ("h", "m", "s", "d")
_DURATIONS = "ASQMWD"


@dataclass(frozen=True)
class _Unit:
    """A time unit of a mask, written with between min_digits and max_digits digits."""

    letter: str
    min_digits: int
    max_digits: int


_Token = Union[str, _Unit]


class _MaskMismatch(ValueError):
    """A value that its mask does not describe."""


def _invalid(from_type: Type[ScalarType], to_type: Type[ScalarType]) -> SemanticError:
    return SemanticError(
        "1-1-5-4",
        type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
        type_2=SCALAR_TYPES_CLASS_REVERSE[to_type],
    )


def _unsupported(construct: str, mask: str) -> SemanticError:
    return SemanticError("1-3-13", construct=f'{construct} in the mask "{mask}"')


# Number masks


@dataclass(frozen=True)
class _NumberMask:
    """The digits of the integer and decimal parts, None for no maximum, and the separator."""

    int_digits: Tuple[int, Optional[int]]
    separator: str
    dec_digits: Tuple[int, Optional[int]]


def _digit_range(part: str) -> Tuple[int, Optional[int]]:
    fixed = part.count("D") + part.count("+")
    return fixed, None if "*" in part or "+" in part else fixed


@lru_cache(maxsize=256)
def _number_mask(mask: str, integer: bool) -> Optional[_NumberMask]:
    """The number mask, or None when it is not one."""
    if "E" in mask:
        raise _unsupported("the exponent E", mask)
    separators = [char for char in mask if char in ".,"]
    if any(char not in "D*+.," for char in mask) or len(separators) > 1:
        return None
    if integer and separators:
        return None
    int_part, separator, dec_part = mask.partition(separators[0]) if separators else (mask, "", "")
    if not int_part or (separator and not dec_part):
        return None
    return _NumberMask(_digit_range(int_part), separator, _digit_range(dec_part))


def _digits_pattern(digits: Tuple[int, Optional[int]]) -> str:
    low, high = digits
    return f"\\d{{{low},{'' if high is None else high}}}"


def _parse_number(text: str, mask: _NumberMask, integer: bool) -> Union[int, float]:
    pattern = f"([+-]?)({_digits_pattern(mask.int_digits)})"
    if mask.separator:
        pattern += f"{re.escape(mask.separator)}({_digits_pattern(mask.dec_digits)})"
    match = re.fullmatch(pattern, text)
    if match is None:
        raise _MaskMismatch(text)
    sign, int_part = match.group(1), match.group(2)
    if integer:
        return int(f"{sign}{int_part}")
    dec_part = match.group(3) if mask.separator else ""
    return float(f"{sign}{int_part or '0'}.{dec_part or '0'}")


def _format_number(value: Union[int, float], mask: _NumberMask) -> str:
    sign = "-" if value < 0 else ""
    if isinstance(value, int):
        int_part, dec_part = str(abs(value)), ""
    else:
        decimals = mask.dec_digits[1]
        if decimals is not None:
            int_part, _, dec_part = f"{abs(value):.{decimals}f}".partition(".")
        else:
            # The shortest decimal text of the value, as many decimals as it has
            int_part, _, dec_part = format(Decimal(repr(abs(value))), "f").partition(".")
            dec_part = dec_part.rstrip("0").ljust(mask.dec_digits[0], "0")
    low, high = mask.int_digits
    int_part = (
        int_part.lstrip("0").rjust(max(low, 1), "0") if int_part != "0" else "0" * max(low, 1)
    )
    if high is not None and len(int_part) > high:
        raise _MaskMismatch(str(value))
    return f"{sign}{int_part}{mask.separator + dec_part if mask.separator else ''}"


# Time masks


def _time_tokens(mask: str) -> Optional[List[_Token]]:
    """The literal characters and time units of a time mask, or None when it is not one."""
    tokens: List[_Token] = []
    i = 0
    while i < len(mask):
        char = mask[i]
        if char == "\\":
            if i + 1 == len(mask):
                return None
            tokens.append(mask[i + 1])
            i += 2
            continue
        word = next((w for w in _TEXTUAL_WORDS if mask[i : i + len(w)].upper() == w), None)
        if word is not None:
            raise _unsupported(f"the textual word {mask[i : i + len(word)]}", mask)
        optional = 0
        if char == "{":
            end = mask.find("}", i)
            inner = mask[i + 1 : end]
            if end < 0 or not inner or len(set(inner)) != 1 or inner[0] not in _TIME_UNITS:
                return None
            char, optional, i = inner[0], len(inner), end + 1
        elif char not in _TIME_UNITS:
            tokens.append(char)
            i += 1
            continue
        else:
            i += 1
            optional = -1  # the letter itself is one written digit
        run = 0 if optional >= 0 else 1
        while i < len(mask) and mask[i] == char:
            run += 1
            i += 1
        optional = max(optional, 0)
        tokens.append(_Unit(char, max(run, 1), run + optional))
    return tokens


def _units(tokens: List[_Token]) -> Dict[str, _Unit]:
    """The units of a mask by letter, None when a letter appears twice."""
    units: Dict[str, _Unit] = {}
    for token in tokens:
        if isinstance(token, _Unit):
            if token.letter in units:
                raise KeyError(token.letter)
            units[token.letter] = token
    return units


def _split_unescaped(mask: str, separator: str) -> List[str]:
    parts, current, i = [], "", 0
    while i < len(mask):
        if mask[i] == "\\" and i + 1 < len(mask):
            current += mask[i : i + 2]
            i += 2
            continue
        if mask[i] == separator:
            parts.append(current)
            current = ""
        else:
            current += mask[i]
        i += 1
    parts.append(current)
    return parts


def _is_date_mask(units: Dict[str, _Unit]) -> bool:
    if "Y" not in units or any(letter in units for letter in ("S", "Q", "P", "p")):
        return False
    if "D" not in units or ("M" in units and "W" in units):
        return False
    return "C" not in units or units["Y"].max_digits <= 2


def _is_period_mask(units: Dict[str, _Unit]) -> bool:
    if "Y" not in units or "C" in units or any(letter in units for letter in _TIME_OF_DAY):
        return False
    periods = [letter for letter in _PERIOD_UNITS if letter in units]
    if "P" in units:
        return not periods and units["P"].max_digits == 1
    return "p" not in units and len(periods) <= 1


@dataclass(frozen=True)
class _TimeMask:
    """A mask of one of the time types, with the tokens of each date it writes."""

    kind: Type[ScalarType]
    parts: Tuple[Tuple[_Token, ...], ...]


@lru_cache(maxsize=256)
def _time_mask(mask: str, kind: Type[ScalarType]) -> Optional[_TimeMask]:
    """The mask of a date, time, time period or duration, or None when it is not one."""
    texts = _split_unescaped(mask, "/") if kind is TimeInterval else [mask]
    if kind is TimeInterval and len(texts) != 2:
        return None
    parts = []
    for text in texts:
        tokens = _time_tokens(text)
        if tokens is None:
            return None
        try:
            units = _units(tokens)
        except KeyError:
            return None
        if kind is Duration:
            if set(units) - {"P", "p"}:
                raise _unsupported("an ISO composite duration", mask)
            if "P" not in units or units["P"].max_digits != 1:
                return None
        elif kind is TimeInterval and ("P" in units or "p" in units):
            raise _unsupported("a time written with a duration", mask)
        elif not (_is_period_mask(units) if kind is TimePeriod else _is_date_mask(units)):
            # A date, or each date of a time, or a time period
            return None
        parts.append(tuple(tokens))
    return _TimeMask(kind, tuple(parts))


def _pattern(tokens: Tuple[_Token, ...]) -> "re.Pattern[str]":
    parts = []
    for token in tokens:
        if isinstance(token, str):
            parts.append(re.escape(token))
        elif token.letter == "P":
            parts.append(f"([{_DURATIONS}])")
        else:
            parts.append(f"(\\d{{{token.min_digits},{token.max_digits}}})")
    return re.compile("".join(parts))


def _fields(text: str, tokens: Tuple[_Token, ...]) -> Dict[str, str]:
    match = _pattern(tokens).fullmatch(text)
    if match is None:
        raise _MaskMismatch(text)
    units = [token for token in tokens if isinstance(token, _Unit)]
    return {unit.letter: value for unit, value in zip(units, match.groups())}


def _write(tokens: Tuple[_Token, ...], values: Dict[str, Union[int, str]]) -> str:
    written = []
    for token in tokens:
        if isinstance(token, str):
            written.append(token)
            continue
        value = str(values[token.letter])
        if token.letter != "P":
            value = value.rjust(token.min_digits, "0")
        if len(value) > token.max_digits:
            raise _MaskMismatch(value)
        written.append(value)
    return "".join(written)


def _year(fields: Dict[str, str]) -> int:
    year = int(fields["Y"])
    return int(fields["C"]) * 100 + year if "C" in fields else year


def _parse_date(text: str, tokens: Tuple[_Token, ...]) -> str:
    fields = _fields(text, tokens)
    year, day = _year(fields), int(fields["D"])
    if "M" in fields:
        day_date = date(year, int(fields["M"]), day)
    elif "W" in fields:
        day_date = date.fromisocalendar(year, int(fields["W"]), day)
    else:
        day_date = date(year, 1, 1) + timedelta(days=day - 1)
        if day < 1 or day_date.year != year:
            raise _MaskMismatch(text)
    if not any(letter in fields for letter in _TIME_OF_DAY):
        return day_date.isoformat()
    fraction = fields.get("d", "")
    moment = datetime(
        day_date.year,
        day_date.month,
        day_date.day,
        int(fields.get("h", 0)),
        int(fields.get("m", 0)),
        int(fields.get("s", 0)),
        int(fraction.ljust(6, "0")[:6]) if fraction else 0,
    )
    written = moment.strftime("%Y-%m-%d %H:%M:%S")
    return f"{written}.{fraction}" if fraction else written


def _format_date(value: str, tokens: Tuple[_Token, ...]) -> str:
    moment = datetime.fromisoformat(value.strip().replace(" ", "T"))
    units = _units(list(tokens))
    iso_year, iso_week, iso_day = moment.isocalendar()
    year = iso_year if "W" in units else moment.year
    values: Dict[str, Union[int, str]] = {
        "C": year // 100,
        "Y": year % 100 if "C" in units else year,
        "M": moment.month,
        "W": iso_week,
        "D": moment.day
        if "M" in units
        else iso_day
        if "W" in units
        else moment.timetuple().tm_yday,
        "h": moment.hour,
        "m": moment.minute,
        "s": moment.second,
    }
    if "d" in units:
        values["d"] = f"{moment.microsecond:06d}"[: units["d"].max_digits]
    return _write(tokens, values)


def _parse_period(text: str, tokens: Tuple[_Token, ...]) -> str:
    fields = _fields(text, tokens)
    if "P" in fields:
        indicator, number = fields["P"], int(fields.get("p", 1))
    else:
        unit = next((letter for letter in _PERIOD_UNITS if letter in fields), None)
        indicator, number = (unit, int(fields[unit])) if unit else ("A", 1)
    if indicator == "A":
        if number != 1:
            raise _MaskMismatch(text)
        return str(TimePeriodHandler(f"{_year(fields)}A"))
    return str(TimePeriodHandler(f"{_year(fields)}{indicator}{number}"))


def _format_period(value: str, tokens: Tuple[_Token, ...]) -> str:
    period = TimePeriodHandler(value)
    units = _units(list(tokens))
    values: Dict[str, Union[int, str]] = {"Y": period.year}
    if "P" in units:
        values["P"], values["p"] = period.period_indicator, period.period_number
    else:
        unit = next((letter for letter in _PERIOD_UNITS if letter in units), "A")
        if unit != period.period_indicator:
            raise _MaskMismatch(value)
        values[unit] = period.period_number
    return _write(tokens, values)


def _parse_duration(text: str, tokens: Tuple[_Token, ...]) -> str:
    fields = _fields(text, tokens)
    # A duration of vtlengine is one period indicator, so one period of it
    if int(fields.get("p", 1)) != 1:
        raise _MaskMismatch(text)
    return fields["P"]


def _parse(text: str, to_type: Type[ScalarType], mask: str) -> Any:
    if to_type in (Number, Integer):
        # check_mask has accepted the mask
        number_mask = cast(_NumberMask, _number_mask(mask, to_type is Integer))
        return _parse_number(text, number_mask, to_type is Integer)
    time_mask = cast(_TimeMask, _time_mask(mask, to_type))
    if to_type is TimeInterval:
        texts = _split_unescaped(text, "/")
        if len(texts) != 2:
            raise _MaskMismatch(text)
        start, end = (_parse_date(t, p) for t, p in zip(texts, time_mask.parts))
        if not TimeInterval.check(f"{start}/{end}"):
            raise _MaskMismatch(text)
        return f"{start}/{end}"
    parser = {Date: _parse_date, TimePeriod: _parse_period, Duration: _parse_duration}[to_type]
    return parser(text, time_mask.parts[0])


def _format(value: Any, from_type: Type[ScalarType], mask: str) -> str:
    if from_type in (Number, Integer):
        number_mask = cast(_NumberMask, _number_mask(mask, from_type is Integer))
        number = int(value) if from_type is Integer else float(value)
        return _format_number(number, number_mask)
    time_mask = cast(_TimeMask, _time_mask(mask, from_type))
    tokens = time_mask.parts[0]
    if from_type is TimeInterval:
        dates = str(value).split("/")
        return "/".join(_format_date(d, p) for d, p in zip(dates, time_mask.parts))
    if from_type is Date:
        return _format_date(str(value), tokens)
    if from_type is TimePeriod:
        return _format_period(str(value), tokens)
    return _write(tokens, {"P": str(value).strip(), "p": 1})


def check_mask(from_type: Type[ScalarType], to_type: Type[ScalarType], mask: str) -> None:
    """Raise when a cast between the types does not take a mask, or the mask is not one of the
    type that is not a string."""
    if from_type is String and to_type in MASK_TYPES:
        other = to_type
    elif to_type is String and from_type in MASK_TYPES:
        other = from_type
    else:
        raise SemanticError(
            "1-1-5-5",
            type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
            type_2=SCALAR_TYPES_CLASS_REVERSE[to_type],
            mask_value=mask,
        )
    if other in (Number, Integer):
        valid = _number_mask(mask, other is Integer) is not None
    else:
        valid = _time_mask(mask, other) is not None
    if not valid:
        raise _invalid(from_type, to_type)


def cast_with_mask(
    value: Any, from_type: Type[ScalarType], to_type: Type[ScalarType], mask: str
) -> Any:
    """Cast a value that is not null with a mask that check_mask accepts. A value that the mask
    does not describe raises RunTimeError 2-1-5-3."""
    try:
        if from_type is String:
            return _parse(str(value).strip(), to_type, mask)
        return _format(value, from_type, mask)
    except (ValueError, OverflowError, RunTimeError) as error:
        # A period number out of range raises a RunTimeError of the time handling
        raise RunTimeError(
            "2-1-5-3",
            value=value,
            type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
            type_2=SCALAR_TYPES_CLASS_REVERSE[to_type],
            mask=mask,
        ) from error
