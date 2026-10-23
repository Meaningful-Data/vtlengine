# ruff: noqa: S101
"""Checks for the comparison of the outputs of an example."""

from __future__ import annotations

import dataclasses
from dataclasses import replace

import pytest

from tool.compare import (
    ActualComponent,
    ActualDataset,
    ActualScalar,
    Output,
    compare_dataset,
    compare_scalar,
    normalize_role,
    normalize_type,
    outputs_equal,
    values_equal,
)
from tool.examples import ExpectedComponent, ExpectedDataset, ExpectedScalar

VALUE_TYPES = ("Number", "Integer", "String", "Boolean", "Date", "Time", "Time_Period", "Duration")

# The expected structure allows Integer or Number for Me_1 and does not state its nullability.
EXPECTED_COMPONENTS = {
    "Id_1": ExpectedComponent("Identifier", ("Integer",), False),
    "Me_1": ExpectedComponent("Measure", ("Integer", "Number"), None),
    "Me_2": ExpectedComponent("Measure", ("String",), True),
}
ACTUAL_COMPONENTS = {
    "Id_1": ActualComponent("Identifier", "Integer", False),
    "Me_1": ActualComponent("Measure", "Integer", True),
    "Me_2": ActualComponent("Measure", "String", True),
}


def row(id_1: str, me_1: str | None, me_2: str | None = None) -> dict[str, str | None]:
    return {"Id_1": id_1, "Me_1": me_1, "Me_2": me_2}


def base_rows() -> list[dict[str, str | None]]:
    return [row("1", "3", "a"), row("2", "5", None)]


def expected_dataset(
    rows: list[dict[str, str | None]] | None = None,
    components: dict[str, ExpectedComponent] | None = None,
) -> ExpectedDataset:
    return ExpectedDataset(
        "DS_r",
        dict(EXPECTED_COMPONENTS) if components is None else components,
        base_rows() if rows is None else rows,
    )


def actual_dataset(
    rows: list[dict[str, str | None]] | None = None,
    components: dict[str, ActualComponent] | None = None,
) -> ActualDataset:
    return ActualDataset(
        "DS_r",
        dict(ACTUAL_COMPONENTS) if components is None else components,
        base_rows() if rows is None else rows,
    )


def scalar(value: str | None, type_: str = "Number", name: str = "sc_r") -> ActualScalar:
    return ActualScalar(name, type_, value)


# --- Values ---


def test_number_tolerance_follows_printed_precision() -> None:
    assert values_equal("Number", "1.632993", "1.632993161855452")
    assert values_equal("Number", "2980.95", "2980.9579870417283")  # truncated print
    assert values_equal("Number", "5.0", "4.999998928")
    assert not values_equal("Number", "1.63", "1.65")
    assert values_equal("Number", "10", "10.0")
    assert not values_equal("Number", "10", "10.4")


def test_booleans_time_periods_and_nulls() -> None:
    assert values_equal("Boolean", "true", "True")
    assert values_equal("Time_Period", "2013-Q1", "2013Q1")
    assert values_equal("Time_Period", "2010", "2010A")
    assert values_equal("Time_Period", "2010M01", "2010-M1")
    assert values_equal("String", "", None)
    assert not values_equal("String", None, "x")


def test_number_without_decimals_matches_almost_exactly() -> None:
    assert values_equal("Number", "1000000", "1000000.0000001")  # within 1e-9 relative
    assert not values_equal("Number", "1000000", "1000000.01")
    assert values_equal("Number", "0", "0.0000000001")  # 1e-10, inside the 1e-9 floor
    assert not values_equal("Number", "0", "0.00000001")
    assert values_equal("Number", "-3", "-3.0")
    assert not values_equal("Number", "-3", "3")


def test_number_precision_is_less_than_one_unit_of_the_last_printed_digit() -> None:
    # 0.7 and 0.6 are exactly 0.1 apart, which is not less than 0.1. In floating point
    # 0.7 - 0.6 is 0.09999999999999998, so the comparison must be done on the printed digits.
    assert not values_equal("Number", "0.7", "0.6")
    assert values_equal("Number", "0.7", "0.69")
    assert values_equal("Number", "0.7", "0.7999")
    assert not values_equal("Number", "0.7", "0.8")
    assert values_equal("Number", "-1.50", "-1.5049")
    assert not values_equal("Number", "-1.50", "-1.51")
    assert values_equal("Number", "+2.5", "2.55")
    assert values_equal("Number", ".5", "0.54")


def test_number_written_with_an_exponent_uses_a_relative_tolerance() -> None:
    assert values_equal("Number", "1E-7", "1.0000005E-7")  # 5e-14 apart, tolerance 1e-13
    assert values_equal("Number", "1E-7", "0.0000001")
    assert not values_equal("Number", "1E-7", "1.00002E-7")  # 2e-12 apart
    assert not values_equal("Number", "1.5E-7", "0.05")  # the decimals rule would accept this
    assert values_equal("Number", "2.5e+3", "2500.001")  # 1e-3 apart, tolerance 2.5e-3
    assert not values_equal("Number", "2.5e+3", "2500.01")
    assert values_equal("Number", "0E+0", "0")
    assert not values_equal("Number", "0E+0", "0.000001")


def test_integers_are_exact_and_never_truncated() -> None:
    assert values_equal("Integer", "10", "10")
    assert values_equal("Integer", "10", "10.0")
    assert values_equal("Integer", "10", "1E+1")
    assert values_equal("Integer", "-0", "0")
    assert not values_equal("Integer", "10", "11")
    assert not values_equal("Integer", "10", "10.5")  # int(Decimal("10.5")) would be 10
    assert not values_equal("Integer", "10.5", "10")
    big = "12345678901234567890123456789012345"
    assert values_equal("Integer", big, big + ".0")
    assert not values_equal("Integer", big, "12345678901234567890123456789012399")


@pytest.mark.parametrize("vtype", ["Number", "Integer"])
def test_unparsable_numbers_are_compared_as_text_and_never_raise(vtype: str) -> None:
    assert values_equal(vtype, "abc", "abc")
    assert not values_equal(vtype, "abc", "abd")
    assert not values_equal(vtype, "1", "x")
    assert not values_equal(vtype, "x", "1")
    assert not values_equal(vtype, "1,5", "1.5")
    assert not values_equal(vtype, "1_0", "10")  # Decimal() would accept the underscore
    arabic_indic = chr(0x661) + chr(0x662)  # Decimal() would read these digits as 12
    assert not values_equal(vtype, arabic_indic, "12")
    assert values_equal(vtype, "NaN", "NaN")
    assert not values_equal(vtype, "NaN", "1")
    assert values_equal(vtype, "inf", "inf")
    assert not values_equal(vtype, "inf", "1")
    assert not values_equal(vtype, "-inf", "inf")
    assert not values_equal(vtype, "1e", "1")
    assert not values_equal(vtype, ".", "0")


@pytest.mark.parametrize("vtype", ["Number", "Integer"])
def test_numbers_of_unrepresentable_size_are_compared_as_text(vtype: str) -> None:
    # Arithmetic on these would overflow the exponent range of Decimal.
    for huge in ("1e99999999999999999999", "1e1000000", "1" + "0" * 1000):
        assert values_equal(vtype, huge, huge)
        assert not values_equal(vtype, huge, "1")
        assert not values_equal(vtype, "1", huge)
        assert not values_equal(vtype, huge, huge + "1")


def test_whitespace_around_a_value_is_ignored_except_for_strings() -> None:
    assert values_equal("Number", "2.1     ", "2.1")  # a manual value with its padding
    assert values_equal("Number", "2.1", "\t2.1 ")
    assert values_equal("Integer", " 7 ", "7")
    assert values_equal("Boolean", " true", "TRUE ")
    assert values_equal("Date", "2020-01-01 ", "2020-01-01")
    assert values_equal("Time_Period", " 2013-Q1", "2013Q1\t")
    assert values_equal("Duration", "Q ", " q")
    assert not values_equal("String", "a ", "a")
    assert not values_equal("String", " a", "a")
    assert values_equal("String", " a ", " a ")


def test_a_blank_value_is_null_except_for_strings() -> None:
    assert values_equal("Number", "  ", None)
    assert values_equal("Integer", None, " ")
    assert values_equal("Date", "\t", "")
    assert not values_equal("Number", " ", "0")
    assert not values_equal("String", "  ", None)  # a String is compared exactly
    assert values_equal("String", "  ", "  ")


@pytest.mark.parametrize("vtype", VALUE_TYPES)
def test_null_equals_only_null(vtype: str) -> None:
    assert values_equal(vtype, None, None)
    assert values_equal(vtype, "", None)
    assert values_equal(vtype, None, "")
    assert values_equal(vtype, "", "")
    assert not values_equal(vtype, None, "0")
    assert not values_equal(vtype, "0", None)
    assert not values_equal(vtype, "", "0")


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("2013-Q1", "2013Q1"),
        ("2010", "2010A"),
        ("2010", "2010-A1"),
        ("2010-A", "2010A1"),
        ("2010M01", "2010-M1"),
        ("2010-01", "2010M01"),
        ("2010-12", "2010-M12"),
        ("2010-S2", "2010S2"),
        ("2010-D001", "2010D1"),
        ("2010-W07", "2010W7"),
        ("2010-Q01", "2010-Q1"),
        ("foo", "foo"),
    ],
)
def test_time_periods_with_the_same_canonical_form_are_equal(left: str, right: str) -> None:
    assert values_equal("Time_Period", left, right)
    assert values_equal("Time_Period", right, left)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("2013Q1", "2013Q2"),
        ("2013Q1", "2014Q1"),
        ("2010", "2010Q1"),
        ("2010", "2011"),
        ("2010M1", "2010Q1"),
        ("2010-12", "2010Q4"),  # no conversion between indicators
        ("2010-1", "2010M1"),  # not one of the three patterns: compared exactly
        ("2010q1", "2010Q1"),
        ("foo", "FOO"),
    ],
)
def test_time_periods_with_another_canonical_form_differ(left: str, right: str) -> None:
    assert not values_equal("Time_Period", left, right)
    assert not values_equal("Time_Period", right, left)


def test_times_ignore_whitespace_around_their_parts() -> None:
    assert values_equal("Time", "2010-01-01/2010-12-31", " 2010-01-01 / 2010-12-31 ")
    assert values_equal("Time", "2010-01-01 / 2010-12-31", "2010-01-01/2010-12-31")
    assert not values_equal("Time", "2010-01-01/2010-12-31", "2010-01-01/2010-12-30")
    assert not values_equal("Time", "2010-01-01/2010-12-31", "2010-01-01")


def test_durations_and_booleans_ignore_case_but_dates_and_strings_do_not() -> None:
    assert values_equal("Duration", "a", "A")
    assert values_equal("Boolean", "FALSE", "false")
    assert not values_equal("Duration", "A", "Q")
    assert not values_equal("Boolean", "true", "false")
    assert not values_equal("Date", "2010-01-01", "2010-01-02")
    assert not values_equal("String", "abc", "ABC")
    assert not values_equal("Mystery", "a", "A")  # any other type is compared exactly
    assert values_equal("Mystery", "a", "a")


@pytest.mark.parametrize(
    ("vtype", "expected", "actual"),
    [
        ("TimePeriod", "2013-Q1", "2013Q1"),
        ("time_period", "2013-Q1", "2013Q1"),
        ("TIME_PERIOD", "2013-Q1", "2013Q1"),
        ("TimeInterval", "2010-01-01 / 2010-12-31", "2010-01-01/2010-12-31"),
        ("time", "2010-01-01 / 2010-12-31", "2010-01-01/2010-12-31"),
        ("number", "5.0", "4.999998928"),
        ("integer", "10", "10.0"),
        ("boolean", "true", "TRUE"),
    ],
)
def test_the_value_type_is_normalised_before_comparing(
    vtype: str, expected: str, actual: str
) -> None:
    assert values_equal(vtype, expected, actual)


# --- Roles and types ---


@pytest.mark.parametrize(
    ("name", "canonical"),
    [
        # The engine's class names.
        ("TimePeriod", "Time_Period"),
        ("TimeInterval", "Time"),
        # The lowercase VTL names.
        ("integer", "Integer"),
        ("number", "Number"),
        ("string", "String"),
        ("boolean", "Boolean"),
        ("date", "Date"),
        ("time", "Time"),
        ("time_period", "Time_Period"),
        ("duration", "Duration"),
        # Case does not matter.
        ("INTEGER", "Integer"),
        ("time_PERIOD", "Time_Period"),
        ("TIMEPERIOD", "Time_Period"),
        # The canonical names pass through.
        ("Integer", "Integer"),
        ("Number", "Number"),
        ("String", "String"),
        ("Boolean", "Boolean"),
        ("Date", "Date"),
        ("Time", "Time"),
        ("Time_Period", "Time_Period"),
        ("Duration", "Duration"),
        # Unknown names are returned unchanged.
        ("Null", "Null"),
        ("Scalar", "Scalar"),
        ("Foo_Bar", "Foo_Bar"),
        ("", ""),
    ],
)
def test_normalize_type(name: str, canonical: str) -> None:
    assert normalize_type(name) == canonical


@pytest.mark.parametrize(
    ("role", "canonical"),
    [
        ("ViralAttribute", "Viral Attribute"),
        ("Viral Attribute", "Viral Attribute"),
        ("Identifier", "Identifier"),
        ("Measure", "Measure"),
        ("Attribute", "Attribute"),
    ],
)
def test_normalize_role(role: str, canonical: str) -> None:
    assert normalize_role(role) == canonical


# --- Datasets ---


def test_compare_dataset_reports_structure_then_rows() -> None:
    assert compare_dataset(expected_dataset(), actual_dataset()) is None

    # A missing component.
    without_me_2 = {name: ACTUAL_COMPONENTS[name] for name in ("Id_1", "Me_1")}
    message = compare_dataset(expected_dataset(), actual_dataset(components=without_me_2))
    assert message is not None
    assert "missing components: Me_2" in message
    assert message == "missing components: Me_2"

    # A role that differs only in its spelling is the same role.
    expected_components = {
        **EXPECTED_COMPONENTS,
        "Me_2": ExpectedComponent("ViralAttribute", ("String",), True),
    }
    actual_components = {
        **ACTUAL_COMPONENTS,
        "Me_2": ActualComponent("Viral Attribute", "String", True),
    }
    assert (
        compare_dataset(
            expected_dataset(components=expected_components),
            actual_dataset(components=actual_components),
        )
        is None
    )

    # The actual type only has to be one of the expected types.
    assert ACTUAL_COMPONENTS["Me_1"].type == "Integer"
    assert EXPECTED_COMPONENTS["Me_1"].types == ("Integer", "Number")
    number_components = {**ACTUAL_COMPONENTS, "Me_1": ActualComponent("Measure", "Number", True)}
    assert compare_dataset(expected_dataset(), actual_dataset(components=number_components)) is None

    # A value that differs.
    rows = [row("1", "4", "a"), row("2", "5", None)]
    message = compare_dataset(expected_dataset(), actual_dataset(rows=rows))
    assert message is not None
    assert "Me_1 expected 3 got 4" in message
    assert message == "row Id_1=1: Me_1 expected 3 got 4"


def test_row_order_does_not_matter() -> None:
    reordered = list(reversed(base_rows()))
    assert compare_dataset(expected_dataset(), actual_dataset(rows=reordered)) is None


def test_unexpected_components_are_reported_after_the_missing_ones() -> None:
    extra = {**ACTUAL_COMPONENTS, "Me_3": ActualComponent("Measure", "Integer", True)}
    message = compare_dataset(expected_dataset(), actual_dataset(components=extra))
    assert message == "unexpected components: Me_3"

    renamed = {
        "Id_1": ACTUAL_COMPONENTS["Id_1"],
        "Me_3": ActualComponent("Measure", "Integer", True),
        "Me_4": ActualComponent("Measure", "Integer", True),
    }
    message = compare_dataset(expected_dataset(), actual_dataset(components=renamed))
    assert message == "missing components: Me_1, Me_2; unexpected components: Me_3, Me_4"


def test_roles_types_and_nullability_mismatches_are_named() -> None:
    attribute = {**ACTUAL_COMPONENTS, "Me_1": replace(ACTUAL_COMPONENTS["Me_1"], role="Attribute")}
    message = compare_dataset(expected_dataset(), actual_dataset(components=attribute))
    assert message == "role of Me_1 expected Measure got Attribute"

    text = {**ACTUAL_COMPONENTS, "Me_1": replace(ACTUAL_COMPONENTS["Me_1"], type="String")}
    message = compare_dataset(expected_dataset(), actual_dataset(components=text))
    assert message == "type of Me_1 expected Integer or Number got String"

    number = {**ACTUAL_COMPONENTS, "Me_2": replace(ACTUAL_COMPONENTS["Me_2"], type="Number")}
    message = compare_dataset(expected_dataset(), actual_dataset(components=number))
    assert message == "type of Me_2 expected String got Number"

    nullable = {**ACTUAL_COMPONENTS, "Id_1": replace(ACTUAL_COMPONENTS["Id_1"], nullable=True)}
    message = compare_dataset(expected_dataset(), actual_dataset(components=nullable))
    assert message == "nullable of Id_1 expected false got true"

    # Me_1 does not state its nullability, so nothing is checked there, whatever the engine says.
    for flag in (True, False):
        free = {**ACTUAL_COMPONENTS, "Me_1": replace(ACTUAL_COMPONENTS["Me_1"], nullable=flag)}
        assert compare_dataset(expected_dataset(), actual_dataset(components=free)) is None


def test_structure_problems_come_in_a_fixed_order_and_skip_the_rows() -> None:
    actual_components = {
        "Id_1": replace(ACTUAL_COMPONENTS["Id_1"], nullable=True),
        "Me_1": replace(ACTUAL_COMPONENTS["Me_1"], role="Attribute"),
        "Me_2": replace(ACTUAL_COMPONENTS["Me_2"], type="Integer"),
    }
    rows = [row("1", "4", "a"), row("2", "5", None)]  # Me_1 differs too
    message = compare_dataset(
        expected_dataset(), actual_dataset(rows=rows, components=actual_components)
    )
    # Roles, then types, then nullability; the row difference is not reported.
    assert message == (
        "role of Me_1 expected Measure got Attribute; "
        "type of Me_2 expected String got Integer; "
        "nullable of Id_1 expected false got true"
    )


def test_rows_are_not_compared_when_the_structure_differs() -> None:
    # One structure problem, so the cap does not hide whether the rows were compared as well.
    nullable = {**ACTUAL_COMPONENTS, "Id_1": replace(ACTUAL_COMPONENTS["Id_1"], nullable=True)}
    rows = [row("1", "4", "a"), row("2", "5", None)]  # Me_1 differs
    message = compare_dataset(expected_dataset(), actual_dataset(rows=rows, components=nullable))
    assert message == "nullable of Id_1 expected false got true"

    # Rows of a dataset that lacks a component cannot be compared at all.
    without_me_2 = {name: ACTUAL_COMPONENTS[name] for name in ("Id_1", "Me_1")}
    message = compare_dataset(
        expected_dataset(), actual_dataset(rows=rows, components=without_me_2)
    )
    assert message == "missing components: Me_2"


def test_at_most_three_problems_are_reported() -> None:
    # Four structure problems: only the first three are kept, in the order names, roles, types.
    actual_components = {
        "Id_1": replace(ACTUAL_COMPONENTS["Id_1"], type="String"),
        "Me_1": replace(ACTUAL_COMPONENTS["Me_1"], role="Attribute"),
        "Me_3": ActualComponent("Measure", "Integer", True),
    }
    message = compare_dataset(expected_dataset(), actual_dataset(components=actual_components))
    assert message == (
        "missing components: Me_2; unexpected components: Me_3; "
        "role of Me_1 expected Measure got Attribute"
    )

    # Five rows with a wrong value: three messages.
    wanted = [row(str(i), "1") for i in range(1, 6)]
    got = [row(str(i), "2") for i in range(1, 6)]
    message = compare_dataset(expected_dataset(rows=wanted), actual_dataset(rows=got))
    assert message == "; ".join(f"row Id_1={i}: Me_1 expected 1 got 2" for i in (1, 2, 3))


def test_rows_are_matched_by_identifier_not_by_position() -> None:
    reordered = [row("2", "6", None), row("1", "3", "a")]
    message = compare_dataset(expected_dataset(), actual_dataset(rows=reordered))
    assert message == "row Id_1=2: Me_1 expected 5 got 6"


def test_missing_and_unexpected_rows_are_named_by_their_key() -> None:
    rows = [row("1", "3", "a"), row("3", "7", None)]
    message = compare_dataset(expected_dataset(), actual_dataset(rows=rows))
    assert message == "missing row Id_1=2; unexpected row Id_1=3"


def test_empty_datasets() -> None:
    assert compare_dataset(expected_dataset(rows=[]), actual_dataset(rows=[])) is None
    assert (
        compare_dataset(expected_dataset(rows=[]), actual_dataset(rows=[row("1", "3")]))
        == "unexpected row Id_1=1"
    )
    assert (
        compare_dataset(expected_dataset(rows=[row("1", "3")]), actual_dataset(rows=[]))
        == "missing row Id_1=1"
    )


def test_duplicate_identifier_keys_are_reported_not_collapsed() -> None:
    twice = [row("1", "3", "a"), row("1", "3", "a"), row("2", "5", None)]
    message = compare_dataset(expected_dataset(rows=twice), actual_dataset())
    assert message == "duplicate identifier key Id_1=1 in expected"
    message = compare_dataset(expected_dataset(), actual_dataset(rows=twice))
    assert message == "duplicate identifier key Id_1=1 in actual"

    # Both sides, several keys, a key repeated three times: one problem per key.
    thrice = [row("1", "3"), row("1", "3"), row("1", "3"), row("2", "5"), row("2", "5")]
    message = compare_dataset(expected_dataset(rows=twice), actual_dataset(rows=thrice))
    assert message == (
        "duplicate identifier key Id_1=1 in expected; "
        "duplicate identifier key Id_1=1 in actual; "
        "duplicate identifier key Id_1=2 in actual"
    )

    # Equal keys that are spelled differently are still the same key.
    spelled = [row("1", "3", "a"), row("1.0", "3", "a")]
    assert compare_dataset(expected_dataset(), actual_dataset(rows=spelled)) == (
        "duplicate identifier key Id_1=1 in actual"
    )


def test_a_dataset_without_identifiers_is_one_row() -> None:
    wanted = {"Me_1": ExpectedComponent("Measure", ("Integer",), None)}
    got = {"Me_1": ActualComponent("Measure", "Integer", True)}

    def compare(
        expected: list[dict[str, str | None]], actual: list[dict[str, str | None]]
    ) -> str | None:
        return compare_dataset(
            ExpectedDataset("DS_r", wanted, expected), ActualDataset("DS_r", got, actual)
        )

    assert compare([{"Me_1": "3"}], [{"Me_1": "3"}]) is None
    assert compare([{"Me_1": "3"}], [{"Me_1": "4"}]) == "row: Me_1 expected 3 got 4"
    assert compare([{"Me_1": "3"}], []) == "missing row"
    assert compare([], [{"Me_1": "3"}]) == "unexpected row"
    assert compare([], []) is None
    twice: list[dict[str, str | None]] = [{"Me_1": "3"}, {"Me_1": "3"}]
    assert compare([{"Me_1": "3"}], twice) == "duplicate identifier key () in actual"
    assert compare(twice, [{"Me_1": "3"}]) == "duplicate identifier key () in expected"


def test_identifiers_are_matched_in_canonical_form() -> None:
    wanted = {
        "Id_1": ExpectedComponent("Identifier", ("Time_Period",), False),
        "Id_2": ExpectedComponent("Identifier", ("String",), False),
        "Me_1": ExpectedComponent("Measure", ("Integer",), None),
    }
    got = {
        "Id_1": ActualComponent("Identifier", "Time_Period", False),
        "Id_2": ActualComponent("Identifier", "String", False),
        "Me_1": ActualComponent("Measure", "Integer", True),
    }

    def compare(period: str, code: str) -> str | None:
        return compare_dataset(
            ExpectedDataset("DS_r", wanted, [{"Id_1": "2013-Q1", "Id_2": "a", "Me_1": "1"}]),
            ActualDataset("DS_r", got, [{"Id_1": period, "Id_2": code, "Me_1": "1"}]),
        )

    assert compare("2013-Q1", "a") is None
    assert compare("2013Q1", "a") is None  # a Time_Period is compared in canonical form
    # A String is compared exactly, so the key differs and both rows are named.
    assert compare("2013Q1", "A") == (
        "missing row Id_1=2013-Q1, Id_2=a; unexpected row Id_1=2013Q1, Id_2=A"
    )
    assert compare("2013Q2", "a") == (
        "missing row Id_1=2013-Q1, Id_2=a; unexpected row Id_1=2013Q2, Id_2=a"
    )


def test_numeric_identifiers_are_matched_by_value_not_by_spelling() -> None:
    def compare(vtype: str, expected_id: str, actual_id: str) -> str | None:
        return compare_dataset(
            ExpectedDataset(
                "DS_r",
                {"Id_1": ExpectedComponent("Identifier", (vtype,), False)},
                [{"Id_1": expected_id}],
            ),
            ActualDataset(
                "DS_r", {"Id_1": ActualComponent("Identifier", vtype, False)}, [{"Id_1": actual_id}]
            ),
        )

    assert compare("Number", "1.50", "1.5") is None
    assert compare("Number", "1.5", "1.5E+0") is None
    assert compare("Integer", "7", "7.0") is None
    # An identifier is matched exactly, with no tolerance: a different key is a different row.
    assert compare("Number", "1.5", "1.5000000001") == (
        "missing row Id_1=1.5; unexpected row Id_1=1.5000000001"
    )
    assert compare("Integer", "7", "8") == "missing row Id_1=7; unexpected row Id_1=8"


def test_values_are_compared_with_the_type_the_engine_returned() -> None:
    wanted = expected_dataset(rows=[row("1", "20")])
    as_number = {**ACTUAL_COMPONENTS, "Me_1": ActualComponent("Measure", "Number", True)}
    # As a Number, 20.0000000001 is within the tolerance of the printed 20...
    assert (
        compare_dataset(
            wanted, actual_dataset(rows=[row("1", "20.0000000001")], components=as_number)
        )
        is None
    )
    # ...but as an Integer, it is not even a whole number.
    assert compare_dataset(wanted, actual_dataset(rows=[row("1", "20.0000000001")])) == (
        "row Id_1=1: Me_1 expected 20 got 20.0000000001"
    )


def test_numbers_follow_the_printed_precision_of_the_expected_value() -> None:
    wanted_components = {
        "Id_1": EXPECTED_COMPONENTS["Id_1"],
        "Me_1": ExpectedComponent("Measure", ("Number",), None),
        "Me_2": EXPECTED_COMPONENTS["Me_2"],
    }
    got_components = {**ACTUAL_COMPONENTS, "Me_1": ActualComponent("Measure", "Number", True)}
    wanted = expected_dataset(
        rows=[row("1", "1.632993"), row("2", "2980.95")], components=wanted_components
    )
    close = actual_dataset(
        rows=[row("1", "1.632993161855452"), row("2", "2980.9579870417283")],
        components=got_components,
    )
    assert compare_dataset(wanted, close) is None
    far = actual_dataset(
        rows=[row("1", "1.65"), row("2", "2980.9579870417283")], components=got_components
    )
    assert compare_dataset(wanted, far) == "row Id_1=1: Me_1 expected 1.632993 got 1.65"


def test_nulls_in_rows_are_shown_as_null() -> None:
    assert (
        compare_dataset(
            expected_dataset(), actual_dataset(rows=[row("1", "3", "a"), row("2", "5", "")])
        )
        is None
    )
    message = compare_dataset(
        expected_dataset(), actual_dataset(rows=[row("1", "3", "a"), row("2", "5", "0")])
    )
    assert message == "row Id_1=2: Me_2 expected null got 0"
    message = compare_dataset(
        expected_dataset(), actual_dataset(rows=[row("1", "3", None), row("2", "5")])
    )
    assert message == "row Id_1=1: Me_2 expected a got null"


def test_manual_values_with_trailing_spaces_match() -> None:
    wanted_components = {
        **EXPECTED_COMPONENTS,
        "Me_1": ExpectedComponent("Measure", ("Number",), None),
    }
    got_components = {**ACTUAL_COMPONENTS, "Me_1": ActualComponent("Measure", "Number", True)}
    wanted = expected_dataset(rows=[row("1", "2.1     ", "a")], components=wanted_components)
    assert (
        compare_dataset(
            wanted, actual_dataset(rows=[row("1", "2.1", "a")], components=got_components)
        )
        is None
    )
    # A String measure keeps its padding.
    padded = expected_dataset(rows=[row("1", "2", "a  ")])
    assert compare_dataset(padded, actual_dataset(rows=[row("1", "2", "a")])) == (
        "row Id_1=1: Me_2 expected a   got a"
    )


def test_engine_type_names_are_normalised_in_datasets() -> None:
    wanted = {
        "Id_1": ExpectedComponent("Identifier", ("Time_Period",), False),
        "Me_1": ExpectedComponent("Measure", ("Time",), None),
    }
    got = {
        "Id_1": ActualComponent("Identifier", "TimePeriod", False),
        "Me_1": ActualComponent("Measure", "TimeInterval", True),
    }
    rows: list[dict[str, str | None]] = [{"Id_1": "2013Q1", "Me_1": "2010-01-01/2010-12-31"}]
    assert (
        compare_dataset(ExpectedDataset("DS_r", wanted, rows), ActualDataset("DS_r", got, rows))
        is None
    )
    # The expected types are normalised too.
    lower = {name: replace(component, types=("time_period",)) for name, component in wanted.items()}
    assert compare_dataset(
        ExpectedDataset("DS_r", lower, rows), ActualDataset("DS_r", got, rows)
    ) == ("type of Me_1 expected Time_Period got Time")


# --- Scalars ---


def test_compare_scalar() -> None:
    wanted = ExpectedScalar("sc_r", "Integer", "8")
    assert compare_scalar(wanted, scalar("8", "Integer")) is None
    assert compare_scalar(wanted, scalar("8.0", "Integer")) is None
    assert compare_scalar(wanted, scalar("9", "Integer")) == "value expected 8 got 9"
    assert compare_scalar(wanted, scalar("8.0", "Number")) == "type expected Integer got Number"
    # A wrong type is reported alone, whatever the value.
    assert compare_scalar(wanted, scalar("9", "Number")) == "type expected Integer got Number"


def test_compare_scalar_nulls_strings_and_precision() -> None:
    null = ExpectedScalar("sc_r", "Integer", None)
    assert compare_scalar(null, scalar(None, "Integer")) is None
    assert compare_scalar(null, scalar("", "Integer")) is None
    assert compare_scalar(null, scalar("0", "Integer")) == "value expected null got 0"
    assert compare_scalar(ExpectedScalar("sc_r", "Integer", "0"), scalar(None, "Integer")) == (
        "value expected 0 got null"
    )
    empty = ExpectedScalar("sc_r", "String", "")
    assert compare_scalar(empty, scalar(None, "String")) is None
    assert compare_scalar(ExpectedScalar("sc_r", "String", "ab"), scalar("AB", "String")) == (
        "value expected ab got AB"
    )
    printed = ExpectedScalar("sc_r", "Number", "1.63")
    assert compare_scalar(printed, scalar("1.6329931618554521", "Number")) is None
    assert compare_scalar(printed, scalar("1.65", "Number")) == "value expected 1.63 got 1.65"


def test_compare_scalar_normalises_engine_type_names() -> None:
    period = ExpectedScalar("sc_r", "Time_Period", "2013-Q1")
    assert compare_scalar(period, scalar("2013Q1", "TimePeriod")) is None
    interval = ExpectedScalar("sc_r", "time", "2010-01-01 / 2010-12-31")
    assert compare_scalar(interval, scalar("2010-01-01/2010-12-31", "TimeInterval")) is None
    assert compare_scalar(interval, scalar("2013Q1", "TimePeriod")) == (
        "type expected Time got Time_Period"
    )


# --- Cross-engine equality ---


def dataset(
    name: str = "DS_r",
    me_1: str | None = "1.5",
    components: dict[str, ActualComponent] | None = None,
    rows: list[dict[str, str | None]] | None = None,
) -> ActualDataset:
    if components is None:
        components = {
            "Id_1": ActualComponent("Identifier", "Integer", False),
            "Me_1": ActualComponent("Measure", "Number", True),
        }
    if rows is None:
        rows = [{"Id_1": "1", "Me_1": me_1}, {"Id_1": "2", "Me_1": "7"}]
    return ActualDataset(name, components, rows)


def point(id_1: str, me_1: str | None) -> dict[str, str | None]:
    """A data point of the default ``dataset()``."""
    return {"Id_1": id_1, "Me_1": me_1}


def test_outputs_equal_cross_engine() -> None:
    assert not outputs_equal([scalar("0.708194")], [scalar("0.195471")])
    assert outputs_equal([scalar("1.6329931618554521")], [scalar("1.632993161855452")])
    assert not outputs_equal([dataset(me_1="0.708194")], [dataset(me_1="0.195471")])
    assert outputs_equal([dataset(me_1="1.6329931618554521")], [dataset(me_1="1.632993161855452")])


def test_outputs_are_matched_by_name_in_any_order() -> None:
    first, second = dataset(), scalar("1", "Integer", name="sc_1")
    assert outputs_equal([first, second], [second, first])
    assert outputs_equal([], [])
    assert not outputs_equal([first], [])
    assert not outputs_equal([first], [first, second])
    assert not outputs_equal([first], [dataset(name="DS_s")])
    # A dataset and a scalar with the same name are different outputs.
    assert not outputs_equal([scalar("1", "Integer", name="DS_r")], [first])
    assert not outputs_equal([first], [scalar("1", "Integer", name="DS_r")])
    # Two outputs with one name cannot be compared by name.
    assert not outputs_equal([first, first], [first, first])
    assert not outputs_equal([first, first], [first])


def test_outputs_must_agree_on_the_structure() -> None:
    base = dataset()
    assert outputs_equal([base], [dataset()])

    extra = {**base.components, "Me_2": ActualComponent("Measure", "Number", True)}
    assert not outputs_equal([base], [dataset(components=extra)])
    assert not outputs_equal([dataset(components=extra)], [base])

    attribute = {**base.components, "Me_1": replace(base.components["Me_1"], role="Attribute")}
    assert not outputs_equal([base], [dataset(components=attribute)])

    integer = {**base.components, "Me_1": replace(base.components["Me_1"], type="Integer")}
    assert not outputs_equal([base], [dataset(components=integer)])

    # Spellings of the same role and type agree; nullability and component order are ignored.
    viral = {**base.components, "Me_1": ActualComponent("ViralAttribute", "Number", True)}
    spelled = {**base.components, "Me_1": ActualComponent("Viral Attribute", "number", False)}
    assert outputs_equal([dataset(components=viral)], [dataset(components=spelled)])
    period = {"Id_1": ActualComponent("Identifier", "TimePeriod", False)}
    other = {"Id_1": ActualComponent("Identifier", "Time_Period", True)}
    rows: list[dict[str, str | None]] = [{"Id_1": "2013Q1"}]
    assert outputs_equal(
        [dataset(components=period, rows=rows)], [dataset(components=other, rows=rows)]
    )
    reordered = dict(reversed(list(base.components.items())))
    assert outputs_equal([base], [dataset(components=reordered)])


def test_outputs_rows_are_matched_by_key() -> None:
    base = dataset()
    reordered = dataset(rows=list(reversed(base.rows)))
    assert outputs_equal([base], [reordered])
    assert outputs_equal([reordered], [base])

    different = dataset(rows=[{"Id_1": "1", "Me_1": "1.5"}, {"Id_1": "3", "Me_1": "7"}])
    assert not outputs_equal([base], [different])
    shorter = dataset(rows=[{"Id_1": "1", "Me_1": "1.5"}])
    assert not outputs_equal([base], [shorter])
    assert not outputs_equal([shorter], [base])
    assert outputs_equal(
        [dataset(rows=[{"Id_1": "1", "Me_1": None}])], [dataset(rows=[{"Id_1": "1", "Me_1": ""}])]
    )
    assert not outputs_equal(
        [dataset(rows=[{"Id_1": "1", "Me_1": None}])], [dataset(rows=[{"Id_1": "1", "Me_1": "0"}])]
    )


def test_outputs_identifiers_are_matched_exactly() -> None:
    components = {
        "Id_1": ActualComponent("Identifier", "Number", False),
        "Me_1": ActualComponent("Measure", "Integer", True),
    }

    def equal(left: str, right: str) -> bool:
        first = dataset(components=components, rows=[{"Id_1": left, "Me_1": "1"}])
        second = dataset(components=components, rows=[{"Id_1": right, "Me_1": "1"}])
        return outputs_equal([first], [second])

    assert equal("1.5", "1.50")  # the same number, written twice
    assert equal("1.5", "1.5E+0")
    # A key has no tolerance: a Number that differs by noise is another identifier, so another
    # data point, while the same difference in a measure would be within the tolerance.
    assert not equal("1.5", "1.5000000001")


def test_outputs_with_the_same_duplicate_identifier_keys_agree() -> None:
    # Two engines that return the same invalid dataset agree with each other. Against the
    # expected output compare_dataset still reports the duplicate.
    rows = [point("1", "1.5"), point("1", "2.5"), point("2", "7")]
    assert outputs_equal([dataset(rows=rows)], [dataset(rows=rows)])
    wanted = ExpectedDataset(
        "DS_r",
        {
            "Id_1": ExpectedComponent("Identifier", ("Integer",), False),
            "Me_1": ExpectedComponent("Measure", ("Number",), None),
        },
        [point("1", "1.5"), point("2", "7")],
    )
    assert compare_dataset(wanted, dataset(rows=rows)) == (
        "duplicate identifier key Id_1=1 in actual"
    )

    # In any order, and with numbers that differ within the tolerance between engines.
    other = [point("2", "7"), point("1", "2.5000000000001"), point("1", "1.5")]
    assert outputs_equal([dataset(rows=rows)], [dataset(rows=other)])
    assert outputs_equal([dataset(rows=other)], [dataset(rows=rows)])

    # Nulls and text where a number is expected sort with the rest, and a data point may repeat.
    odd = [point("1", None), point("1", "abc"), point("1", "abc"), point("1", "2")]
    assert outputs_equal([dataset(rows=odd)], [dataset(rows=list(reversed(odd)))])

    # Without identifiers there is a single key, so a second data point repeats it.
    measure = {"Me_1": ActualComponent("Measure", "Number", True)}
    pair: list[dict[str, str | None]] = [{"Me_1": "1.5"}, {"Me_1": "1.5"}]
    assert outputs_equal(
        [dataset(components=measure, rows=pair)], [dataset(components=measure, rows=list(pair))]
    )


def test_duplicate_key_data_points_are_sorted_so_that_number_noise_cannot_reorder_them() -> None:
    components = {
        "Id_1": ActualComponent("Identifier", "Integer", False),
        "Me_1": ActualComponent("Measure", "Number", True),
        "Me_2": ActualComponent("Measure", "String", True),
    }
    # The two engines differ by 1.5e-10 on the first data point, within the tolerance. By Me_1
    # alone the data points would come in another order on each side (1.0000000001 sorts before
    # 1.0000000002, but 1.00000000025 sorts after it), so the Numbers must sort last.
    first = [row("1", "1.0000000001", "a"), row("1", "1.0000000002", "b")]
    second = [row("1", "1.00000000025", "a"), row("1", "1.0000000002", "b")]
    left = dataset(components=components, rows=first)
    assert outputs_equal([left], [dataset(components=components, rows=second)])
    assert outputs_equal([dataset(components=components, rows=second)], [left])

    # Still a disagreement when a data point really differs.
    third = [row("1", "1.00000000025", "a"), row("1", "1.0000000002", "c")]
    assert not outputs_equal([left], [dataset(components=components, rows=third)])


def test_outputs_with_different_duplicate_data_points_disagree() -> None:
    one, two = point("1", "1.5"), point("1", "2.5")
    # The same keys, and a duplicate data point that differs.
    assert not outputs_equal([dataset(rows=[one, two])], [dataset(rows=[one, point("1", "2.6")])])
    # A duplicate on one side only, which is a different number of data points as well.
    assert not outputs_equal([dataset(rows=[one, two])], [dataset(rows=[one])])
    assert not outputs_equal([dataset(rows=[one])], [dataset(rows=[one, two])])
    # The same data points a different number of times.
    assert not outputs_equal([dataset(rows=[one, one, two])], [dataset(rows=[one, two, two])])
    assert not outputs_equal([dataset(rows=[one, one])], [dataset(rows=[one, one, one])])
    # A duplicate on one side against two different keys on the other.
    other_key = point("2", "1.5")
    assert not outputs_equal([dataset(rows=[one, one])], [dataset(rows=[one, other_key])])
    assert not outputs_equal([dataset(rows=[one, other_key])], [dataset(rows=[one, one])])
    # An empty dataset is not a dataset of duplicates.
    assert not outputs_equal([dataset(rows=[one, one])], [dataset(rows=[])])
    assert outputs_equal([dataset(rows=[])], [dataset(rows=[])])


def test_outputs_numbers_are_equal_at_relative_1e9_or_absolute_1e12() -> None:
    def equal(left: str, right: str) -> bool:
        forward = outputs_equal([scalar(left)], [scalar(right)])
        assert forward == outputs_equal([scalar(right)], [scalar(left)])
        return forward

    assert equal("1000000", "1000000.0001")  # 1e-4 apart, tolerance 1e-3
    assert not equal("1000000", "1000000.01")
    assert equal("0", "1e-13")  # inside the absolute 1e-12
    assert not equal("0", "1e-11")
    assert equal("1.5", "1.5000000000001")
    assert not equal("1.63", "1.65")  # the printed precision does not apply here
    assert not equal("5.0", "4.99")
    assert equal("-2.5", "-2.5000000001")
    assert not equal("-2.5", "2.5")


def test_outputs_other_types_are_canonical_equal() -> None:
    def equal(vtype: str, left: str | None, right: str | None) -> bool:
        return outputs_equal([scalar(left, vtype)], [scalar(right, vtype)])

    assert equal("Boolean", "true", "True")
    assert equal("Time_Period", "2013-Q1", "2013Q1")
    assert equal("Integer", "10", "10.0")
    assert equal("Time", "2010-01-01/2010-12-31", "2010-01-01 / 2010-12-31")
    assert equal("Duration", "a", "A")
    assert equal("String", None, "")
    assert equal("Date", "2020-01-01 ", "2020-01-01")
    assert not equal("Integer", "10", "11")
    assert not equal("Integer", "10", "10.5")
    assert not equal("String", "a", "A")
    assert not equal("String", "a ", "a")
    assert not equal("Time_Period", "2013Q1", "2013Q2")
    assert not equal("Date", "2020-01-01", "2020-01-02")
    assert not equal("Number", None, "0")
    assert not equal("Number", "x", "1")


def test_outputs_scalars_must_agree_on_the_type() -> None:
    assert not outputs_equal([scalar("1", "Integer")], [scalar("1", "Number")])
    assert outputs_equal([scalar("1", "TimePeriod")], [scalar("1", "Time_Period")])


def test_outputs_union_alias_accepts_both_kinds() -> None:
    outputs: list[Output] = [dataset(), scalar("1")]
    assert outputs_equal(outputs, list(outputs))


def test_actual_values_are_frozen_dataclasses() -> None:
    for value, field in (
        (ActualComponent("Measure", "Integer", True), "role"),
        (dataset(), "name"),
        (scalar("1"), "value"),
    ):
        assert dataclasses.is_dataclass(value)
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(value, field, "other")
