"""Comparison of the outputs of an example: with the expected ones, and between the engines.

``compare_dataset`` and ``compare_scalar`` judge an engine's output against the expected one and
return a short message about what differs, or None. ``outputs_equal`` judges whether the outputs
of two engines agree with each other. Both rest on the same normalisation of values, written
here and independent of vtlengine on purpose: the map measures that code, so it must not rely on
it. Different spellings of one value, such as ``2013-Q1`` and ``2013Q1``, are the same value.

They differ in how a Number is matched. The manual prints shortened results, rounded
(``1.632993``) or truncated (``2980.95``), so an expected Number takes its tolerance from the
digits written in it. Two engines are expected to agree to the last digits, so they only get a
tiny relative or absolute tolerance.

The order of the data points of a dataset does not matter: a data point is found by the canonical
value of its identifiers, and a key that occurs twice on either side is a problem of its own,
never silently merged. Nothing here raises for odd values: a text that is not a number is
compared as text.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from itertools import islice
from typing import Final, Union

from tool.examples import ExpectedDataset, ExpectedScalar

_MAX_PROBLEMS: Final = 3
_IDENTIFIER: Final = "Identifier"

# The VTL types by their lowercase name, to their canonical name in vtlengine's structure files.
# The last two lines are the engine's class names, where they differ from the VTL names.
_TYPE_NAMES: Final = {
    "integer": "Integer",
    "number": "Number",
    "string": "String",
    "boolean": "Boolean",
    "date": "Date",
    "time": "Time",
    "time_period": "Time_Period",
    "duration": "Duration",
    "timeperiod": "Time_Period",
    "timeinterval": "Time",
}

# A finite decimal number in ASCII digits. ``Decimal`` alone would also take ``1_0``, ``NaN``,
# ``inf`` and the digits of other scripts. The sizes are bounded so that arithmetic on a match
# cannot overflow the exponent range of ``Decimal``: a longer text is not a number to compare,
# it is compared as text.
_NUMBER: Final = re.compile(
    r"[+-]?(?:\d{1,400}(?:\.\d{0,400})?|\.\d{1,400})(?:[eE][+-]?\d{1,3})?", re.ASCII
)

# The spellings of a time period: ``2010``, ``2010Q1`` / ``2010-Q1`` / ``2010-Q-1`` (the number
# may be missing) and ``2010-01`` for a month.
_YEAR: Final = re.compile(r"\d{4}", re.ASCII)
_PERIOD: Final = re.compile(r"(\d{4})-?([ASQMWD])-?(\d{1,3})?", re.ASCII)
_YEAR_MONTH: Final = re.compile(r"(\d{4})-(\d{2})", re.ASCII)

_ONE: Final = Decimal(1)
# Against an expected Number, relative to it: with no decimals written, or with an exponent.
_NO_DECIMALS_TOLERANCE: Final = Decimal("1e-9")
_EXPONENT_TOLERANCE: Final = Decimal("1e-6")
# Between two engines: relative, and absolute for numbers close to zero.
_RELATIVE: Final = Decimal("1e-9")
_ABSOLUTE: Final = Decimal("1e-12")


@dataclass(frozen=True)
class ActualComponent:
    """A component of a dataset that an engine returned."""

    role: str  # ``Identifier``, ``Measure``, ``Attribute`` or ``Viral Attribute``
    type: str  # the VTL type; ``normalize_type`` maps the engine's spelling to the canonical one
    nullable: bool


@dataclass(frozen=True)
class ActualDataset:
    """A result dataset that an engine returned."""

    name: str
    components: dict[str, ActualComponent]
    rows: list[dict[str, str | None]]  # one dict per data point; values stringified, null None


@dataclass(frozen=True)
class ActualScalar:
    """A result scalar that an engine returned."""

    name: str
    type: str
    value: str | None  # stringified; None for null


# ``Union`` rather than ``|``: the alias is evaluated when the module is imported, on Python 3.9.
Output = Union[ActualDataset, ActualScalar]


def normalize_role(role: str) -> str:
    """The canonical spelling of a role: ``ViralAttribute`` is written ``Viral Attribute``."""
    return "Viral Attribute" if role == "ViralAttribute" else role


def normalize_type(name: str) -> str:
    """The canonical VTL name of a type, whether it is a VTL name or an engine's class name.

    ``TimePeriod`` is ``Time_Period`` and ``TimeInterval`` is ``Time``. The case of the name does
    not matter: ``integer`` is ``Integer``. A name that is none of the VTL types is returned
    unchanged.
    """
    return _TYPE_NAMES.get(name.lower(), name)


def values_equal(vtype: str, expected: str | None, actual: str | None) -> bool:
    """Whether ``actual`` is the expected value of the VTL type ``vtype``.

    The whitespace around a value is ignored, except for a String. None and the empty text are
    null (so is a blank text, except for a String), and a null equals only a null. Then, by type:

    * **Number:** with ``d`` digits after the ``.`` of the expected value, ``d >= 1`` matches
      when ``|actual - expected| < 10**-d``, one unit of the last digit written. Without
      decimals it matches when ``|actual - expected| <= 1e-9 * max(1, |expected|)``. An expected
      value written with an exponent matches within a relative 1e-6. A text that is not a finite
      number is compared as text;
    * **Integer:** the same whole number, so ``10`` and ``10.0`` agree but ``10.5`` is never ten;
    * **Boolean** and **Duration:** ignoring case;
    * **Time_Period:** by ``(year, indicator, number)``, so ``2013-Q1`` is ``2013Q1`` and ``2010``
      is ``2010A``; any other text is compared as written;
    * **Time:** ignoring the whitespace around the parts split on ``/``;
    * every other type, Date and String included: exactly.
    """
    return _same_value(vtype, expected, actual, _matches_printed_precision)


def compare_dataset(expected: ExpectedDataset, actual: ActualDataset) -> str | None:
    """What differs between an expected dataset and an actual one; None when they agree.

    The structure is compared first: the component names, then the roles (after
    ``normalize_role``), then the types (the actual type must be one of the expected ones, after
    ``normalize_type``), then the nullability (only where the expected structure states it).
    The data points are compared only when the structure agrees. They are found by the identifier
    values, and every other column is compared with ``values_equal``, as the type that the engine
    returned. At most three problems are reported, joined with ``"; "``. The names of the
    datasets are not compared: the caller pairs the outputs by name.
    """
    return "; ".join(islice(_dataset_problems(expected, actual), _MAX_PROBLEMS)) or None


def compare_scalar(expected: ExpectedScalar, actual: ActualScalar) -> str | None:
    """What differs between an expected scalar and an actual one; None when they agree.

    The type is compared first, after ``normalize_type``, and the value only when the type
    agrees, with ``values_equal``. The names are not compared: the caller pairs the outputs by
    name.
    """
    expected_type, actual_type = normalize_type(expected.type), normalize_type(actual.type)
    if expected_type != actual_type:
        return f"type expected {expected_type} got {actual_type}"
    if not values_equal(actual_type, expected.value, actual.value):
        return f"value expected {_show(expected.value)} got {_show(actual.value)}"
    return None


def outputs_equal(a: Sequence[Output], b: Sequence[Output]) -> bool:
    """Whether the outputs of two engines agree with each other.

    The outputs are matched by name, in any order. Both sides need the same names, each used
    once. A scalar agrees with a scalar of the same type and value. A dataset agrees with a
    dataset that has the same components with the same roles and types (nullability is not
    compared) and the same data points, found by the identifiers of ``a``. A Number agrees within
    a relative 1e-9, or an absolute 1e-12 near zero; every other type is compared as in
    ``values_equal``.
    """
    left = {output.name: output for output in a}
    right = {output.name: output for output in b}
    if len(left) != len(a) or len(right) != len(b) or left.keys() != right.keys():
        return False
    return all(_outputs_agree(left[name], right[name]) for name in left)


# --- Outputs ---


def _outputs_agree(left: Output, right: Output) -> bool:
    if isinstance(left, ActualDataset) and isinstance(right, ActualDataset):
        return _datasets_agree(left, right)
    if isinstance(left, ActualScalar) and isinstance(right, ActualScalar):
        vtype = normalize_type(left.type)
        return vtype == normalize_type(right.type) and _same_value(
            vtype, left.value, right.value, _matches_relative
        )
    return False


def _datasets_agree(left: ActualDataset, right: ActualDataset) -> bool:
    structure = _structure(left)
    if structure != _structure(right):
        return False
    ids = [name for name, (role, _) in structure.items() if role == _IDENTIFIER]
    types = {name: type_ for name, (_, type_) in structure.items()}
    problems = _row_problems(ids, types, left.rows, right.rows, _matches_relative)
    return next(problems, None) is None


def _structure(dataset: ActualDataset) -> dict[str, tuple[str, str]]:
    """The canonical role and type of every component."""
    return {
        name: (normalize_role(component.role), normalize_type(component.type))
        for name, component in dataset.components.items()
    }


# --- Datasets ---


def _dataset_problems(expected: ExpectedDataset, actual: ActualDataset) -> Iterator[str]:
    structure = list(_structure_problems(expected, actual))
    if structure:
        yield from structure
        return
    ids = [
        name
        for name, component in expected.components.items()
        if normalize_role(component.role) == _IDENTIFIER
    ]
    types = {name: normalize_type(actual.components[name].type) for name in expected.components}
    yield from _row_problems(ids, types, expected.rows, actual.rows, _matches_printed_precision)


def _structure_problems(expected: ExpectedDataset, actual: ActualDataset) -> Iterator[str]:
    missing = [name for name in expected.components if name not in actual.components]
    if missing:
        yield f"missing components: {', '.join(missing)}"
    unexpected = [name for name in actual.components if name not in expected.components]
    if unexpected:
        yield f"unexpected components: {', '.join(unexpected)}"

    shared = [name for name in expected.components if name in actual.components]
    for name in shared:
        want = normalize_role(expected.components[name].role)
        got = normalize_role(actual.components[name].role)
        if want != got:
            yield f"role of {name} expected {want} got {got}"
    for name in shared:
        allowed = list(dict.fromkeys(normalize_type(t) for t in expected.components[name].types))
        got = normalize_type(actual.components[name].type)
        if got not in allowed:
            yield f"type of {name} expected {' or '.join(allowed)} got {got}"
    for name in shared:
        nullable = expected.components[name].nullable
        if nullable is not None and nullable != actual.components[name].nullable:
            yield (
                f"nullable of {name} expected {str(nullable).lower()} "
                f"got {str(actual.components[name].nullable).lower()}"
            )


def _row_problems(
    ids: Sequence[str],
    types: Mapping[str, str],
    expected_rows: Sequence[Mapping[str, str | None]],
    actual_rows: Sequence[Mapping[str, str | None]],
    numbers: Callable[[str, str], bool],
) -> Iterator[str]:
    """How ``actual_rows`` differ from ``expected_rows``, found by their ``ids`` columns.

    ``types`` is the canonical type of every column and ``numbers`` the rule that matches two
    Number values, the expected one first. When a key occurs twice on either side, only those
    duplicates are reported: the data points cannot be matched.
    """
    wanted, wanted_repeated = _index_rows(expected_rows, ids, types)
    got, got_repeated = _index_rows(actual_rows, ids, types)
    if wanted_repeated or got_repeated:
        for row in wanted_repeated:
            yield f"duplicate identifier key {_key_text(ids, row) or '()'} in expected"
        for row in got_repeated:
            yield f"duplicate identifier key {_key_text(ids, row) or '()'} in actual"
        return

    others = [name for name in types if name not in ids]
    for key, want in wanted.items():
        have = got.get(key)
        if have is None:
            yield f"missing {_label(ids, want)}"
            continue
        for name in others:
            if not _same_value(types[name], want.get(name), have.get(name), numbers):
                yield (
                    f"{_label(ids, want)}: {name} expected {_show(want.get(name))} "
                    f"got {_show(have.get(name))}"
                )
    for key, have in got.items():
        if key not in wanted:
            yield f"unexpected {_label(ids, have)}"


def _index_rows(
    rows: Sequence[Mapping[str, str | None]], ids: Sequence[str], types: Mapping[str, str]
) -> tuple[dict[tuple[Hashable, ...], Mapping[str, str | None]], list[Mapping[str, str | None]]]:
    """The data points by the canonical value of their identifiers, and the repeated keys.

    A key keeps its first data point in the index. The next ones are not indexed; the list has the
    first data point of each key that occurs again, once per key, in the order of the keys.
    """
    index: dict[tuple[Hashable, ...], Mapping[str, str | None]] = {}
    repeated: dict[tuple[Hashable, ...], Mapping[str, str | None]] = {}
    for row in rows:
        key = tuple(_canonical(types[name], row.get(name)) for name in ids)
        if key in index:
            repeated.setdefault(key, index[key])
        else:
            index[key] = row
    return index, list(repeated.values())


def _key_text(ids: Sequence[str], row: Mapping[str, str | None]) -> str:
    """``Id_1=1, Id_2=A``: the identifier values of a data point as they are written."""
    return ", ".join(f"{name}={_show(row.get(name))}" for name in ids)


def _label(ids: Sequence[str], row: Mapping[str, str | None]) -> str:
    """``row Id_1=1``, or just ``row`` for a dataset without identifiers."""
    key = _key_text(ids, row)
    return f"row {key}" if key else "row"


def _show(value: str | None) -> str:
    return value or "null"


# --- Values ---


def _same_value(
    vtype: str, left: str | None, right: str | None, numbers: Callable[[str, str], bool]
) -> bool:
    """Whether two values of ``vtype`` agree, the Numbers by the rule ``numbers``."""
    vtype = normalize_type(vtype)
    if vtype == "Number":
        left_text, right_text = _clean(vtype, left), _clean(vtype, right)
        if left_text is not None and right_text is not None:
            return numbers(left_text, right_text)
    return _canonical(vtype, left) == _canonical(vtype, right)


def _clean(vtype: str, value: str | None) -> str | None:
    """``value`` without the whitespace around it, except for a String; None for a null."""
    if value is not None and vtype != "String":
        value = value.strip()
    return value or None


def _canonical(vtype: str, value: str | None) -> Hashable:
    """A value that is the same for every spelling of one value of ``vtype``; None for a null.

    A Number is only canonical to be an identifier: its comparison has a tolerance, which has no
    canonical form.
    """
    text = _clean(vtype, value)
    if text is None:
        return None
    if vtype in ("Number", "Integer"):
        number = _decimal(text)
        # A text that is not a number stays as written, and so does a fraction as an Integer: it
        # is never truncated into a whole number.
        if number is None or (vtype == "Integer" and number != number.to_integral_value()):
            return text
        # Not ``normalize()``: it rounds to the precision of the context, 28 digits.
        return number
    if vtype in ("Boolean", "Duration"):
        return text.lower()
    if vtype == "Time_Period":
        return _period(text)
    if vtype == "Time":
        return "/".join(part.strip() for part in text.split("/"))
    return text


def _decimal(text: str) -> Decimal | None:
    """The number written in ``text``, or None when it is not a finite decimal number."""
    return Decimal(text) if _NUMBER.fullmatch(text) else None


def _period(text: str) -> tuple[int, str, int] | str:
    """``(year, indicator, number)`` of a time period in a usual spelling; else the text."""
    if _YEAR.fullmatch(text):
        return int(text), "A", 1
    period = _PERIOD.fullmatch(text)
    if period:
        return int(period[1]), period[2], int(period[3] or 1)
    month = _YEAR_MONTH.fullmatch(text)
    if month:
        return int(month[1]), "M", int(month[2])
    return text


def _matches_printed_precision(expected: str, actual: str) -> bool:
    """Whether the Number ``actual`` is the Number ``expected``, as precisely as it is written."""
    want, have = _decimal(expected), _decimal(actual)
    if want is None or have is None:
        return expected == actual
    distance = abs(have - want)
    if "e" in expected.lower():
        return distance <= abs(want) * _EXPONENT_TOLERANCE
    decimals = len(expected.partition(".")[2])
    if decimals:
        # Less than one unit of the last digit written, so that a rounded print matches as well
        # as a truncated one. ``Decimal`` keeps this exact: 0.7 and 0.6 are 0.1 apart, no less.
        return distance < _ONE.scaleb(-decimals)
    return distance <= _NO_DECIMALS_TOLERANCE * max(_ONE, abs(want))


def _matches_relative(left: str, right: str) -> bool:
    """Whether two Numbers of two engines agree, with no regard to how they are written."""
    first, second = _decimal(left), _decimal(right)
    if first is None or second is None:
        return left == right
    return abs(first - second) <= max(_RELATIVE * max(abs(first), abs(second)), _ABSOLUTE)
