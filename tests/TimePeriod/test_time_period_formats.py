"""
Tests for TimePeriod input parsing, internal representation, and external representations.

Covers:
- All accepted input formats -> internal representation
- VTL representation output
- SDMX Reporting representation output
- SDMX Gregorian representation output (including error for unsupported indicators)
- format_time_period_external_representation on Datasets
"""

from typing import List, Optional

import pandas as pd
import pytest

from vtlengine.DataTypes import TimePeriod
from vtlengine.DataTypes._time_checking import check_time_period
from vtlengine.DataTypes.TimeHandling import TimePeriodHandler
from vtlengine.Exceptions import SemanticError
from vtlengine.files.output._time_period_representation import (
    TimePeriodRepresentation,
    format_time_period_external_representation,
)
from vtlengine.Model import Component, Dataset, Role, Scalar

# Methods individual tests

vtl_repr_params = [
    # (input, expected_internal, description)
    # Annual
    ("2020", "2020A", "YYYY year only"),
    ("2020A", "2020A", "YYYYA annual with indicator"),
    # Semester
    ("2020S1", "2020-S1", "YYYYSN semester 1"),
    ("2020S2", "2020-S2", "YYYYSN semester 2"),
    # Quarter
    ("2020Q1", "2020-Q1", "YYYYQN quarter 1"),
    ("2020Q4", "2020-Q4", "YYYYQN quarter 4"),
    # Month
    ("2020M1", "2020-M01", "YYYYM single digit month"),
    ("2020M9", "2020-M09", "YYYYM single digit month 9"),
    ("2020M10", "2020-M10", "YYYYMM double digit month 10"),
    ("2020M12", "2020-M12", "YYYYMM double digit month 12"),
    # Week
    ("2020W1", "2020-W01", "YYYYW single digit week"),
    ("2020W9", "2020-W09", "YYYYW single digit week 9"),
    ("2020W10", "2020-W10", "YYYYWW double digit week 10"),
    ("2020W53", "2020-W53", "YYYYWW double digit week 53"),
    # Day
    ("2020D1", "2020-D001", "YYYYD single digit day"),
    ("2020D10", "2020-D010", "YYYYDD double digit day"),
    ("2020D99", "2020-D099", "YYYYDD double digit day 99"),
    ("2020D100", "2020-D100", "YYYYDDD triple digit day 100"),
    ("2020D366", "2020-D366", "YYYYDDD triple digit day 366"),
]

sdmx_params = [
    # SDMX gregorian day
    ("2020-11-1", "2020-D306", "YYYY-MM-D ISO day 306"),
    ("2020-2-15", "2020-D046", "YYYY-MM-DD ISO day 46"),
    ("2020-3-3", "2020-D063", "YYYY-M-D ISO day 63"),
    ("2020-12-31", "2020-D366", "YYYY-MM-DD ISO date end of leap year"),
    ("2019-12-31", "2019-D365", "YYYY-MM-DD ISO date end of non-leap year"),
    # SDMX gregorian month
    ("2020-01", "2020-M01", "YYYY-MM ISO month 01"),
    ("2020-12", "2020-M12", "YYYY-MM ISO month 12"),
    ("2020-1", "2020-M01", "YYYY-M single digit ISO month"),
    ("2020-9", "2020-M09", "YYYY-M single digit ISO month 9"),
    # SDMX gregorian year
    ("2020", "2020A", "YYYY year only"),
    # SDMX reporting day
    ("2020-D001", "2020-D001", "YYYY-DXXX hyphenated day 001"),
    ("2020-D366", "2020-D366", "YYYY-DXXX hyphenated day 366"),
    # SDMX reporting week
    ("2020-W01", "2020-W01", "YYYY-WXX hyphenated week 01"),
    ("2020-W53", "2020-W53", "YYYY-WXX hyphenated week 53"),
    # SDMX reporting month
    ("2020-M01", "2020-M01", "YYYY-MXX hyphenated month 01"),
    ("2020-M1", "2020-M01", "YYYY-MX hyphenated month single digit"),
    ("2020-M12", "2020-M12", "YYYY-MXX hyphenated month 12"),
    # SDMX reporting quarter
    ("2020-Q1", "2020-Q1", "YYYY-QX hyphenated quarter 1"),
    ("2020-Q4", "2020-Q4", "YYYY-QX hyphenated quarter 4"),
    # SDMX reporting semester
    ("2020-S1", "2020-S1", "YYYY-SX hyphenated semester 1"),
    ("2020-S2", "2020-S2", "YYYY-SX hyphenated semester 2"),
    # SDMX reporting annual
    ("2020-A1", "2020A", "YYYY-A1 SDMX reporting annual"),
]


@pytest.mark.parametrize(
    "input_val, expected",
    [(c[0], c[1]) for c in vtl_repr_params],
    ids=[c[2] for c in vtl_repr_params],
)
def test_check_time_period_vtl(input_val: str, expected: str) -> None:
    assert check_time_period(input_val) == expected


@pytest.mark.parametrize(
    "input_val, expected",
    [(c[0], c[1]) for c in sdmx_params],
    ids=[c[2] for c in sdmx_params],
)
def test_check_time_period_sdmx(input_val: str, expected: str) -> None:
    assert check_time_period(input_val) == expected


whitespace_params = [
    ("  2020  ", "2020A", "strips whitespace year"),
    (" 2020M1 ", "2020-M01", "strips whitespace month"),
]


@pytest.mark.parametrize(
    "input_val, expected",
    [(c[0], c[1]) for c in whitespace_params],
    ids=[c[2] for c in whitespace_params],
)
def test_check_time_period_whitespace(input_val: str, expected: str) -> None:
    assert check_time_period(input_val) == expected


def test_check_time_period_integer_as_input() -> None:
    assert check_time_period(2020) == "2020A"  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "input_val",
    ["invalid", "20-01", "2020XY1"],
    ids=["garbage", "short year", "unknown indicator"],
)
def test_check_time_period_invalid(input_val: str) -> None:
    with pytest.raises(ValueError, match="not in a valid format"):
        check_time_period(input_val)


consistency_params = [
    ("2020A", ["2020", "2020A", "2020-A1"], "annual"),
    (
        "2020-M01",
        ["2020M1", "2020M01", "2020-01", "2020-1", "2020-M01", "2020-M1"],
        "month",
    ),
    ("2020-Q1", ["2020Q1", "2020-Q1"], "quarter"),
    ("2020-S1", ["2020S1", "2020-S1"], "semester"),
    ("2020-W01", ["2020W1", "2020W01", "2020-W01"], "week"),
    (
        "2020-D001",
        [
            "2020D1",
            "2020D01",
            "2020D001",
            "2020-D001",
            "2020-01-01",
            "2020-1-01",
            "2020-01-1",
            "2020-1-1",
        ],
        "day",
    ),
]


@pytest.mark.parametrize(
    "expected, inputs",
    [(c[0], c[1]) for c in consistency_params],
    ids=[c[2] for c in consistency_params],
)
def test_check_time_period_all_formats_consistent(expected: str, inputs: list) -> None:
    for input_val in inputs:
        assert check_time_period(input_val) == expected, (
            f"Input '{input_val}' produced '{check_time_period(input_val)}', expected '{expected}'"
        )


vtl_repr_params = [
    ("2020A", "2020", "annual"),
    ("2020S1", "2020S1", "semester 1"),
    ("2020S2", "2020S2", "semester 2"),
    ("2020Q1", "2020Q1", "quarter 1"),
    ("2020Q4", "2020Q4", "quarter 4"),
    ("2020M1", "2020M1", "month 1"),
    ("2020M12", "2020M12", "month 12"),
    ("2020W1", "2020W1", "week 1"),
    ("2020W53", "2020W53", "week 53"),
    ("2020D1", "2020D1", "day 1"),
    ("2020D100", "2020D100", "day 100"),
    ("2020D366", "2020D366", "day 366"),
]


@pytest.mark.parametrize(
    "internal, expected",
    [(c[0], c[1]) for c in vtl_repr_params],
    ids=[c[2] for c in vtl_repr_params],
)
def test_vtl_representation(internal: str, expected: str) -> None:
    assert TimePeriodHandler(internal).vtl_representation() == expected


sdmx_reporting_params = [
    ("2020A", "2020-A1", "annual"),
    ("2020S1", "2020-S1", "semester 1"),
    ("2020S2", "2020-S2", "semester 2"),
    ("2020Q1", "2020-Q1", "quarter 1"),
    ("2020Q4", "2020-Q4", "quarter 4"),
    ("2020M1", "2020-M01", "month 1 zero-padded"),
    ("2020M12", "2020-M12", "month 12"),
    ("2020W1", "2020-W01", "week 1 zero-padded"),
    ("2020W53", "2020-W53", "week 53"),
    ("2020D1", "2020-D001", "day 1 zero-padded"),
    ("2020D100", "2020-D100", "day 100"),
    ("2020D366", "2020-D366", "day 366"),
]


@pytest.mark.parametrize(
    "internal, expected",
    [(c[0], c[1]) for c in sdmx_reporting_params],
    ids=[c[2] for c in sdmx_reporting_params],
)
def test_sdmx_reporting_representation(internal: str, expected: str) -> None:
    assert TimePeriodHandler(internal).sdmx_reporting_representation() == expected


sdmx_gregorian_params = [
    ("2020A", "2020", "annual"),
    ("2020M1", "2020-01", "month 1"),
    ("2020M12", "2020-12", "month 12"),
    ("2020D1", "2020-01-01", "day 1"),
    ("2020D59", "2020-02-28", "day 59"),
    ("2020D366", "2020-12-31", "day 366 leap year"),
]


sdmx_gregorian_error_params = [
    ("2020S1", "semester"),
    ("2020Q1", "quarter"),
    ("2020W1", "week"),
]


@pytest.mark.parametrize(
    "internal, expected",
    [(c[0], c[1]) for c in sdmx_gregorian_params],
    ids=[c[2] for c in sdmx_gregorian_params],
)
def test_sdmx_gregorian_representation(internal: str, expected: str) -> None:
    assert TimePeriodHandler(internal).sdmx_gregorian_representation() == expected


@pytest.mark.parametrize(
    "internal",
    [c[0] for c in sdmx_gregorian_error_params],
    ids=[c[1] for c in sdmx_gregorian_error_params],
)
def test_sdmx_gregorian_representation_unsupported(internal: str) -> None:
    with pytest.raises(SemanticError, match="2-1-19-21"):
        TimePeriodHandler(internal).sdmx_gregorian_representation()


# VTL Data Types to external representations tests


def get_tp_dataset(values: Optional[List[str]]) -> Dataset:
    """Helper to create a Dataset with a TimePeriod column."""
    components = {
        "Id_1": Component(name="Id_1", data_type=TimePeriod, role=Role.IDENTIFIER, nullable=False),
    }
    data = pd.DataFrame({"Id_1": values}) if values is not None else None
    return Dataset(name="test", components=components, data=data)


def get_tp_scalar(value: Optional[str]) -> Scalar:
    """Helper to create a Scalar with a TimePeriod value."""
    return Scalar(name="test", data_type=TimePeriod, value=value)


format_dataset_params = [
    # (internal_value, mode, expected_output, id)
    # VTL
    ("2020A", TimePeriodRepresentation.VTL, "2020", "vtl annual"),
    ("2020-M01", TimePeriodRepresentation.VTL, "2020M1", "vtl month"),
    ("2020-Q3", TimePeriodRepresentation.VTL, "2020Q3", "vtl quarter"),
    ("2020-S2", TimePeriodRepresentation.VTL, "2020S2", "vtl semester"),
    ("2020-W01", TimePeriodRepresentation.VTL, "2020W1", "vtl week"),
    ("2020-D001", TimePeriodRepresentation.VTL, "2020D1", "vtl day"),
    # SDMX Reporting
    ("2020A", TimePeriodRepresentation.SDMX_REPORTING, "2020-A1", "reporting annual"),
    ("2020-M01", TimePeriodRepresentation.SDMX_REPORTING, "2020-M01", "reporting month"),
    ("2020-Q3", TimePeriodRepresentation.SDMX_REPORTING, "2020-Q3", "reporting quarter"),
    ("2020-W01", TimePeriodRepresentation.SDMX_REPORTING, "2020-W01", "reporting week"),
    ("2020-D001", TimePeriodRepresentation.SDMX_REPORTING, "2020-D001", "reporting day"),
    # SDMX Gregorian (only A, M, D)
    ("2020A", TimePeriodRepresentation.SDMX_GREGORIAN, "2020", "gregorian annual"),
    ("2020-M01", TimePeriodRepresentation.SDMX_GREGORIAN, "2020-01", "gregorian month"),
    ("2020-D001", TimePeriodRepresentation.SDMX_GREGORIAN, "2020-01-01", "gregorian day"),
]


@pytest.mark.parametrize(
    "internal, mode, expected",
    [(c[0], c[1], c[2]) for c in format_dataset_params],
    ids=[c[3] for c in format_dataset_params],
)
def test_format_external_representation(
    internal: str, mode: TimePeriodRepresentation, expected: str
) -> None:
    ds = get_tp_dataset([internal])
    format_time_period_external_representation(ds, mode)
    assert ds.data["Id_1"].iloc[0] == expected


@pytest.mark.parametrize(
    "internal, mode, expected",
    [(c[0], c[1], c[2]) for c in format_dataset_params],
    ids=[c[3] for c in format_dataset_params],
)
def test_format_external_representation_scalars(
    internal: str, mode: TimePeriodRepresentation, expected: str
) -> None:
    sc = get_tp_scalar(internal)
    format_time_period_external_representation(sc, mode)
    assert sc.value == expected


gregorian_error_params = [
    ("2020-S1", "gregorian semester error"),
    ("2020-Q1", "gregorian quarter error"),
    ("2020-W1", "gregorian week error"),
]


@pytest.mark.parametrize(
    "internal",
    [c[0] for c in gregorian_error_params],
    ids=[c[1] for c in gregorian_error_params],
)
def test_format_external_representation_gregorian_error(internal: str) -> None:
    ds = get_tp_dataset([internal])
    with pytest.raises(SemanticError, match="2-1-19-21"):
        format_time_period_external_representation(ds, TimePeriodRepresentation.SDMX_GREGORIAN)


def test_format_external_empty_dataset() -> None:
    components = {
        "Id_1": Component(name="Id_1", data_type=TimePeriod, role=Role.IDENTIFIER, nullable=False),
    }
    ds = Dataset(name="test", components=components, data=pd.DataFrame({"Id_1": []}))
    format_time_period_external_representation(ds, TimePeriodRepresentation.VTL)
    assert ds.data["Id_1"].empty


def test_format_external_none() -> None:
    components = {
        "Id_1": Component(name="Id_1", data_type=TimePeriod, role=Role.IDENTIFIER, nullable=False),
    }
    ds = Dataset(name="test", components=components, data=None)
    format_time_period_external_representation(ds, TimePeriodRepresentation.VTL)
    assert ds.data is None


def test_format_external_none_scalar() -> None:
    sc = get_tp_scalar(None)
    format_time_period_external_representation(sc, TimePeriodRepresentation.VTL)
    assert sc.value is None


def test_format_external_multiple_values() -> None:
    ds = get_tp_dataset(["2020A", "2020-M06", "2020-Q2"])
    format_time_period_external_representation(ds, TimePeriodRepresentation.VTL)
    assert len(ds.data["Id_1"]) == 3
    assert ds.data["Id_1"].equals(
        pd.Series(["2020", "2020M6", "2020Q2"], name="Id_1", dtype="string[pyarrow]")
    )
