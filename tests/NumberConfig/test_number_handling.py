"""
Tests for Number type handling: environment variables, comparisons, and output formatting.
"""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pandas as pd
import pytest

from vtlengine.API import run
from vtlengine.Operators.Comparison import (
    _get_number_tolerance,
    _numbers_equal,
)
from vtlengine.Utils._number_config import (
    DEFAULT_SIGNIFICANT_DIGITS,
    DISABLED_VALUE,
    ENV_COMPARISON_THRESHOLD,
    ENV_OUTPUT_SIGNIFICANT_DIGITS,
    MAX_SIGNIFICANT_DIGITS,
    MIN_SIGNIFICANT_DIGITS,
    _parse_env_value,
    get_effective_comparison_digits,
    get_effective_output_digits,
    get_float_format,
    numbers_are_equal,
)

# --- Environment Variable Parsing ---


@pytest.mark.parametrize(
    "env_value, expected",
    [
        pytest.param(None, None, id="not_set"),
        pytest.param("", None, id="empty_string"),
        pytest.param("   ", None, id="whitespace"),
        pytest.param("-1", DISABLED_VALUE, id="disabled"),
        pytest.param(str(MIN_SIGNIFICANT_DIGITS), MIN_SIGNIFICANT_DIGITS, id="min_value"),
        pytest.param(str(MAX_SIGNIFICANT_DIGITS), MAX_SIGNIFICANT_DIGITS, id="max_value"),
        pytest.param("10", 10, id="middle_value"),
    ],
)
def test_parse_env_value_valid(env_value: str, expected: int) -> None:
    env = {ENV_COMPARISON_THRESHOLD: env_value} if env_value is not None else {}
    with mock.patch.dict(os.environ, env, clear=True):
        result = _parse_env_value(ENV_COMPARISON_THRESHOLD)
        assert result == expected


@pytest.mark.parametrize(
    "env_value",
    [
        pytest.param("5", id="too_low"),
        pytest.param("16", id="too_high"),
        pytest.param("abc", id="non_integer"),
        pytest.param("10.5", id="float"),
    ],
)
def test_parse_env_value_invalid(env_value: str) -> None:
    with (
        mock.patch.dict(os.environ, {ENV_COMPARISON_THRESHOLD: env_value}),
        pytest.raises(ValueError, match="Invalid value"),
    ):
        _parse_env_value(ENV_COMPARISON_THRESHOLD)


# --- Effective Digits ---


@pytest.mark.parametrize(
    "env_var, env_value, func, expected",
    [
        pytest.param(
            ENV_COMPARISON_THRESHOLD,
            None,
            get_effective_comparison_digits,
            DEFAULT_SIGNIFICANT_DIGITS,
            id="comparison_default",
        ),
        pytest.param(
            ENV_COMPARISON_THRESHOLD,
            "8",
            get_effective_comparison_digits,
            8,
            id="comparison_custom",
        ),
        pytest.param(
            ENV_COMPARISON_THRESHOLD,
            "-1",
            get_effective_comparison_digits,
            None,
            id="comparison_disabled",
        ),
        pytest.param(
            ENV_OUTPUT_SIGNIFICANT_DIGITS,
            None,
            get_effective_output_digits,
            DEFAULT_SIGNIFICANT_DIGITS,
            id="output_default",
        ),
        pytest.param(
            ENV_OUTPUT_SIGNIFICANT_DIGITS, "12", get_effective_output_digits, 12, id="output_custom"
        ),
        pytest.param(
            ENV_OUTPUT_SIGNIFICANT_DIGITS,
            "-1",
            get_effective_output_digits,
            None,
            id="output_disabled",
        ),
    ],
)
def test_effective_digits(env_var: str, env_value: str, func, expected) -> None:
    env = {env_var: env_value} if env_value is not None else {}
    with mock.patch.dict(os.environ, env, clear=True):
        if env_value is None:
            os.environ.pop(env_var, None)
        assert func() == expected


# --- Float Format ---


@pytest.mark.parametrize(
    "env_value, expected",
    [
        pytest.param(None, f"%.{DEFAULT_SIGNIFICANT_DIGITS}g", id="default"),
        pytest.param("8", "%.8g", id="custom"),
        pytest.param("-1", None, id="disabled"),
    ],
)
def test_get_float_format(env_value: str, expected: str) -> None:
    env = {ENV_OUTPUT_SIGNIFICANT_DIGITS: env_value} if env_value is not None else {}
    with mock.patch.dict(os.environ, env, clear=True):
        if env_value is None:
            os.environ.pop(ENV_OUTPUT_SIGNIFICANT_DIGITS, None)
        assert get_float_format() == expected


# --- Number Tolerance ---


@pytest.mark.parametrize(
    "sig_digits, expected",
    [
        pytest.param(None, None, id="disabled"),
        pytest.param(10, 0.5e-9, id="10_digits"),
        pytest.param(6, 0.5e-5, id="6_digits"),
    ],
)
def test_get_number_tolerance(sig_digits: int, expected: float) -> None:
    assert _get_number_tolerance(sig_digits) == expected


# --- Numbers Equal ---


@pytest.mark.parametrize(
    "a, b, sig_digits, expected",
    [
        pytest.param(1.0, 1.0, 10, True, id="exact_equality"),
        pytest.param(1.0, 1.0 + 1e-11, 10, True, id="within_tolerance"),
        pytest.param(1.0, 1.001, 10, False, id="outside_tolerance"),
        pytest.param(1.0, 1.0, None, True, id="disabled_equal"),
        pytest.param(1.0, 1.0 + 1e-15, None, False, id="disabled_not_equal"),
        pytest.param(0.0, 0.0, 10, True, id="both_zero"),
        pytest.param(1e10, 1e10 + 1, 10, True, id="large_within_tolerance"),
        pytest.param(1e10, 1e10 + 100, 10, False, id="large_outside_tolerance"),
        pytest.param(1e-10, 1e-10 + 1e-21, 10, True, id="small_within_tolerance"),
    ],
)
def test_numbers_equal(a: float, b: float, sig_digits: int, expected: bool) -> None:
    rel_tol = _get_number_tolerance(sig_digits)
    assert _numbers_equal(a, b, rel_tol) == expected


# --- Numbers Are Equal (wrapper) ---


def test_numbers_are_equal_default() -> None:
    with mock.patch.dict(os.environ, {}, clear=True):
        os.environ.pop(ENV_COMPARISON_THRESHOLD, None)
        assert numbers_are_equal(1.0, 1.0 + 1e-15)


@pytest.mark.parametrize(
    "a, b, sig_digits, expected",
    [
        pytest.param(1.0, 1.0 + 1e-7, 6, True, id="within_tolerance"),
        pytest.param(1.0, 1.001, 6, False, id="outside_tolerance"),
    ],
)
def test_numbers_are_equal_custom(a: float, b: float, sig_digits: int, expected: bool) -> None:
    assert numbers_are_equal(a, b, significant_digits=sig_digits) == expected


# --- VTL Comparison Operators (Integration) ---


@pytest.fixture
def ds_structure():
    return {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {
                        "name": "Id_1",
                        "type": "Integer",
                        "role": "Identifier",
                        "nullable": False,
                    },
                    {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                ],
            }
        ]
    }


@pytest.mark.parametrize(
    "script, me_values, expected",
    [
        pytest.param(
            "DS_r <- DS_1 = 1.0;",
            [1.0, 1.0 + 1e-11, 1.001],
            [True, True, False],
            id="equal_with_tolerance",
        ),
        pytest.param(
            "DS_r <- DS_1 >= 1.0;",
            [1.0 - 1e-11, 0.999, 1.001],
            [True, False, True],
            id="greater_equal_with_tolerance",
        ),
        pytest.param(
            "DS_r <- DS_1 <= 1.0;",
            [1.0 + 1e-11, 1.001, 0.999],
            [True, False, True],
            id="less_equal_with_tolerance",
        ),
    ],
)
def test_vtl_comparison_with_tolerance(
    ds_structure, script: str, me_values: list, expected: list
) -> None:
    with mock.patch.dict(os.environ, {ENV_COMPARISON_THRESHOLD: "10"}):
        datapoints = pd.DataFrame({"Id_1": list(range(1, len(me_values) + 1)), "Me_1": me_values})
        result = run(script=script, data_structures=ds_structure, datapoints={"DS_1": datapoints})
        assert result["DS_r"].data["bool_var"].tolist() == expected


def test_vtl_equal_disabled(ds_structure) -> None:
    with mock.patch.dict(os.environ, {ENV_COMPARISON_THRESHOLD: "-1"}):
        datapoints = pd.DataFrame({"Id_1": [1, 2], "Me_1": [1.0, 1.0 + 1e-15]})
        result = run(
            script="DS_r <- DS_1 = 1.0;",
            data_structures=ds_structure,
            datapoints={"DS_1": datapoints},
        )
        assert result["DS_r"].data["bool_var"].tolist()[0]


def test_vtl_between_with_tolerance(ds_structure) -> None:
    with mock.patch.dict(os.environ, {ENV_COMPARISON_THRESHOLD: "10"}):
        datapoints = pd.DataFrame(
            {
                "Id_1": [1, 2, 3, 4, 5],
                "Me_1": [1.0 - 1e-11, 2.0 + 1e-11, 1.5, 0.5, 2.5],
            }
        )
        result = run(
            script="DS_r <- between(DS_1, 1.0, 2.0);",
            data_structures=ds_structure,
            datapoints={"DS_1": datapoints},
        )
        assert result["DS_r"].data["bool_var"].tolist() == [True, True, True, False, False]


# --- Output Formatting ---


@pytest.mark.parametrize(
    "env_value, expected_substring",
    [
        pytest.param(None, "1.23456789", id="default"),
        pytest.param("-1", "1.234567890123", id="disabled"),
    ],
)
def test_output_formatting(env_value: str, expected_substring: str) -> None:
    ds_structure = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {
                        "name": "Id_1",
                        "type": "Integer",
                        "role": "Identifier",
                        "nullable": False,
                    },
                    {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                ],
            }
        ]
    }
    datapoints = pd.DataFrame({"Id_1": [1], "Me_1": [1.23456789012345]})

    env = {ENV_OUTPUT_SIGNIFICANT_DIGITS: env_value} if env_value is not None else {}
    with mock.patch.dict(os.environ, env, clear=True):
        if env_value is None:
            os.environ.pop(ENV_OUTPUT_SIGNIFICANT_DIGITS, None)
        with TemporaryDirectory() as tmpdir:
            run(
                script="DS_r <- DS_1;",
                data_structures=ds_structure,
                datapoints={"DS_1": datapoints},
                output_folder=Path(tmpdir),
            )
            content = (Path(tmpdir) / "DS_r.csv").read_text()
            assert expected_substring in content
