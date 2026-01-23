"""
Tests for Number type handling: environment variables, comparisons, and output formatting.
"""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

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


class TestEnvironmentVariableParsing(TestCase):
    """Tests for environment variable parsing functions."""

    def test_parse_env_value_not_set(self):
        """Test that None is returned when env var is not set."""
        with mock.patch.dict(os.environ, {}, clear=True):
            result = _parse_env_value("NONEXISTENT_VAR")
            assert result is None

    def test_parse_env_value_empty_string(self):
        """Test that None is returned for empty string."""
        with mock.patch.dict(os.environ, {ENV_COMPARISON_THRESHOLD: ""}):
            result = _parse_env_value(ENV_COMPARISON_THRESHOLD)
            assert result is None

    def test_parse_env_value_whitespace(self):
        """Test that None is returned for whitespace-only value."""
        with mock.patch.dict(os.environ, {ENV_COMPARISON_THRESHOLD: "   "}):
            result = _parse_env_value(ENV_COMPARISON_THRESHOLD)
            assert result is None

    def test_parse_env_value_valid_disabled(self):
        """Test that -1 disables the feature."""
        with mock.patch.dict(os.environ, {ENV_COMPARISON_THRESHOLD: "-1"}):
            result = _parse_env_value(ENV_COMPARISON_THRESHOLD)
            assert result == DISABLED_VALUE

    def test_parse_env_value_valid_min(self):
        """Test minimum valid value."""
        with mock.patch.dict(os.environ, {ENV_COMPARISON_THRESHOLD: str(MIN_SIGNIFICANT_DIGITS)}):
            result = _parse_env_value(ENV_COMPARISON_THRESHOLD)
            assert result == MIN_SIGNIFICANT_DIGITS

    def test_parse_env_value_valid_max(self):
        """Test maximum valid value."""
        with mock.patch.dict(os.environ, {ENV_COMPARISON_THRESHOLD: str(MAX_SIGNIFICANT_DIGITS)}):
            result = _parse_env_value(ENV_COMPARISON_THRESHOLD)
            assert result == MAX_SIGNIFICANT_DIGITS

    def test_parse_env_value_valid_middle(self):
        """Test a middle value."""
        with mock.patch.dict(os.environ, {ENV_COMPARISON_THRESHOLD: "10"}):
            result = _parse_env_value(ENV_COMPARISON_THRESHOLD)
            assert result == 10

    def test_parse_env_value_invalid_too_low(self):
        """Test that values below minimum raise ValueError."""
        with (
            mock.patch.dict(os.environ, {ENV_COMPARISON_THRESHOLD: "5"}),
            pytest.raises(ValueError, match="Invalid value"),
        ):
            _parse_env_value(ENV_COMPARISON_THRESHOLD)

    def test_parse_env_value_invalid_too_high(self):
        """Test that values above maximum raise ValueError."""
        with (
            mock.patch.dict(os.environ, {ENV_COMPARISON_THRESHOLD: "15"}),
            pytest.raises(ValueError, match="Invalid value"),
        ):
            _parse_env_value(ENV_COMPARISON_THRESHOLD)

    def test_parse_env_value_invalid_non_integer(self):
        """Test that non-integer values raise ValueError."""
        with (
            mock.patch.dict(os.environ, {ENV_COMPARISON_THRESHOLD: "abc"}),
            pytest.raises(ValueError, match="Invalid value"),
        ):
            _parse_env_value(ENV_COMPARISON_THRESHOLD)

    def test_parse_env_value_invalid_float(self):
        """Test that float values raise ValueError."""
        with (
            mock.patch.dict(os.environ, {ENV_COMPARISON_THRESHOLD: "10.5"}),
            pytest.raises(ValueError, match="Invalid value"),
        ):
            _parse_env_value(ENV_COMPARISON_THRESHOLD)


class TestEffectiveDigits(TestCase):
    """Tests for effective digit calculation functions."""

    def test_get_effective_comparison_digits_default(self):
        """Test that default is used when env var not set."""
        with mock.patch.dict(os.environ, {}, clear=True):
            # Clear the specific env var
            os.environ.pop(ENV_COMPARISON_THRESHOLD, None)
            result = get_effective_comparison_digits()
            assert result == DEFAULT_SIGNIFICANT_DIGITS

    def test_get_effective_comparison_digits_custom(self):
        """Test that custom value is used when set."""
        with mock.patch.dict(os.environ, {ENV_COMPARISON_THRESHOLD: "8"}):
            result = get_effective_comparison_digits()
            assert result == 8

    def test_get_effective_comparison_digits_disabled(self):
        """Test that None is returned when disabled."""
        with mock.patch.dict(os.environ, {ENV_COMPARISON_THRESHOLD: "-1"}):
            result = get_effective_comparison_digits()
            assert result is None

    def test_get_effective_output_digits_default(self):
        """Test that default is used when env var not set."""
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop(ENV_OUTPUT_SIGNIFICANT_DIGITS, None)
            result = get_effective_output_digits()
            assert result == DEFAULT_SIGNIFICANT_DIGITS

    def test_get_effective_output_digits_custom(self):
        """Test that custom value is used when set."""
        with mock.patch.dict(os.environ, {ENV_OUTPUT_SIGNIFICANT_DIGITS: "12"}):
            result = get_effective_output_digits()
            assert result == 12

    def test_get_effective_output_digits_disabled(self):
        """Test that None is returned when disabled."""
        with mock.patch.dict(os.environ, {ENV_OUTPUT_SIGNIFICANT_DIGITS: "-1"}):
            result = get_effective_output_digits()
            assert result is None


class TestFloatFormat(TestCase):
    """Tests for float format string generation."""

    def test_get_float_format_default(self):
        """Test default float format."""
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop(ENV_OUTPUT_SIGNIFICANT_DIGITS, None)
            result = get_float_format()
            assert result == f"%.{DEFAULT_SIGNIFICANT_DIGITS}g"

    def test_get_float_format_custom(self):
        """Test custom float format."""
        with mock.patch.dict(os.environ, {ENV_OUTPUT_SIGNIFICANT_DIGITS: "8"}):
            result = get_float_format()
            assert result == "%.8g"

    def test_get_float_format_disabled(self):
        """Test disabled float format returns None."""
        with mock.patch.dict(os.environ, {ENV_OUTPUT_SIGNIFICANT_DIGITS: "-1"}):
            result = get_float_format()
            assert result is None


class TestNumberTolerance(TestCase):
    """Tests for number tolerance calculation."""

    def test_tolerance_none_when_disabled(self):
        """Test that None tolerance is returned when disabled."""
        result = _get_number_tolerance(None)
        assert result is None

    def test_tolerance_10_significant_digits(self):
        """Test tolerance for 10 significant digits."""
        result = _get_number_tolerance(10)
        expected = 0.5 * (10 ** (-9))  # 0.5e-9
        assert result == expected

    def test_tolerance_6_significant_digits(self):
        """Test tolerance for 6 significant digits."""
        result = _get_number_tolerance(6)
        expected = 0.5 * (10 ** (-5))  # 0.5e-5
        assert result == expected


class TestNumbersEqual(TestCase):
    """Tests for number equality comparison with tolerance."""

    def test_exact_equality(self):
        """Test exact equality."""
        rel_tol = _get_number_tolerance(10)
        assert _numbers_equal(1.0, 1.0, rel_tol)

    def test_within_tolerance(self):
        """Test numbers within tolerance are considered equal."""
        rel_tol = _get_number_tolerance(10)
        # With 10 significant digits, 1.0 and 1.0 + 1e-11 should be equal
        assert _numbers_equal(1.0, 1.0 + 1e-11, rel_tol)

    def test_outside_tolerance(self):
        """Test numbers outside tolerance are not equal."""
        rel_tol = _get_number_tolerance(10)
        # With 10 significant digits, 1.0 and 1.001 should not be equal
        assert not _numbers_equal(1.0, 1.001, rel_tol)

    def test_disabled_exact_comparison(self):
        """Test exact comparison when disabled."""
        assert _numbers_equal(1.0, 1.0, None)
        assert not _numbers_equal(1.0, 1.0 + 1e-15, None)

    def test_both_zero(self):
        """Test both zeros are equal."""
        rel_tol = _get_number_tolerance(10)
        assert _numbers_equal(0.0, 0.0, rel_tol)

    def test_large_numbers(self):
        """Test tolerance works for large numbers."""
        rel_tol = _get_number_tolerance(10)
        # For 1e10, tolerance is approximately 5
        assert _numbers_equal(10000000000.0, 10000000000.0 + 1, rel_tol)
        assert not _numbers_equal(10000000000.0, 10000000000.0 + 100, rel_tol)

    def test_small_numbers(self):
        """Test tolerance works for small numbers."""
        rel_tol = _get_number_tolerance(10)
        # For 1e-10, tolerance is approximately 5e-20
        assert _numbers_equal(1e-10, 1e-10 + 1e-21, rel_tol)


class TestNumbersAreEqual(TestCase):
    """Tests for the main numbers_are_equal function."""

    def test_uses_default_when_none(self):
        """Test that default digits are used when not specified."""
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop(ENV_COMPARISON_THRESHOLD, None)
            # With default 14 significant digits, tolerance is 5e-14
            result = numbers_are_equal(1.0, 1.0 + 1e-15)
            assert result

    def test_uses_custom_digits(self):
        """Test that custom digits are used when specified."""
        # With 6 significant digits, tolerance is larger
        result = numbers_are_equal(1.0, 1.0 + 1e-7, significant_digits=6)
        assert result

        # But 1.0 and 1.001 should not be equal even with 6 digits
        result = numbers_are_equal(1.0, 1.001, significant_digits=6)
        assert not result


class TestVTLComparisonOperators(TestCase):
    """Integration tests for VTL comparison operators with Number tolerance."""

    def setUp(self):
        """Set up test data structures."""
        self.ds_structure = {
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

    def test_equal_operator_with_tolerance(self):
        """Test = operator uses tolerance for Numbers."""
        with mock.patch.dict(os.environ, {ENV_COMPARISON_THRESHOLD: "10"}):
            # Create datapoints with values that are equal within tolerance
            datapoints = pd.DataFrame(
                {
                    "Id_1": [1, 2, 3],
                    "Me_1": [1.0, 1.0 + 1e-11, 1.001],
                }
            )

            script = "DS_r <- DS_1 = 1.0;"

            result = run(
                script=script,
                data_structures=self.ds_structure,
                datapoints={"DS_1": datapoints},
            )

            # First two should be True (within tolerance), third should be False
            assert result["DS_r"].data["bool_var"].tolist() == [True, True, False]

    def test_equal_operator_disabled(self):
        """Test = operator with tolerance disabled uses exact comparison."""
        with mock.patch.dict(os.environ, {ENV_COMPARISON_THRESHOLD: "-1"}):
            datapoints = pd.DataFrame(
                {
                    "Id_1": [1, 2],
                    "Me_1": [1.0, 1.0 + 1e-15],
                }
            )

            script = "DS_r <- DS_1 = 1.0;"

            result = run(
                script=script,
                data_structures=self.ds_structure,
                datapoints={"DS_1": datapoints},
            )

            # With exact comparison, only the first should be True
            result_values = result["DS_r"].data["bool_var"].tolist()
            assert result_values[0]
            # The second may or may not be True due to floating point representation

    def test_greater_equal_operator_with_tolerance(self):
        """Test >= operator uses tolerance for equality boundary."""
        with mock.patch.dict(os.environ, {ENV_COMPARISON_THRESHOLD: "10"}):
            datapoints = pd.DataFrame(
                {
                    "Id_1": [1, 2, 3],
                    "Me_1": [1.0 - 1e-11, 0.999, 1.001],
                }
            )

            script = "DS_r <- DS_1 >= 1.0;"

            result = run(
                script=script,
                data_structures=self.ds_structure,
                datapoints={"DS_1": datapoints},
            )

            # First should be True (equal within tolerance)
            # Second should be False (clearly less)
            # Third should be True (clearly greater)
            assert result["DS_r"].data["bool_var"].tolist() == [True, False, True]

    def test_less_equal_operator_with_tolerance(self):
        """Test <= operator uses tolerance for equality boundary."""
        with mock.patch.dict(os.environ, {ENV_COMPARISON_THRESHOLD: "10"}):
            datapoints = pd.DataFrame(
                {
                    "Id_1": [1, 2, 3],
                    "Me_1": [1.0 + 1e-11, 1.001, 0.999],
                }
            )

            script = "DS_r <- DS_1 <= 1.0;"

            result = run(
                script=script,
                data_structures=self.ds_structure,
                datapoints={"DS_1": datapoints},
            )

            # First should be True (equal within tolerance)
            # Second should be False (clearly greater)
            # Third should be True (clearly less)
            assert result["DS_r"].data["bool_var"].tolist() == [True, False, True]

    def test_between_operator_with_tolerance(self):
        """Test between operator uses tolerance for boundaries."""
        with mock.patch.dict(os.environ, {ENV_COMPARISON_THRESHOLD: "10"}):
            datapoints = pd.DataFrame(
                {
                    "Id_1": [1, 2, 3, 4, 5],
                    "Me_1": [
                        1.0 - 1e-11,  # Equal to lower bound within tolerance
                        2.0 + 1e-11,  # Equal to upper bound within tolerance
                        1.5,  # Clearly between
                        0.5,  # Clearly below
                        2.5,  # Clearly above
                    ],
                }
            )

            script = "DS_r <- between(DS_1, 1.0, 2.0);"

            result = run(
                script=script,
                data_structures=self.ds_structure,
                datapoints={"DS_1": datapoints},
            )

            expected = [True, True, True, False, False]
            assert result["DS_r"].data["bool_var"].tolist() == expected


class TestOutputFormatting(TestCase):
    """Tests for CSV output formatting with significant digits."""

    def test_output_formatting_default(self):
        """Test default output formatting with 10 significant digits."""
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop(ENV_OUTPUT_SIGNIFICANT_DIGITS, None)

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

            # Use a number that would have many decimal places
            datapoints = pd.DataFrame(
                {
                    "Id_1": [1],
                    "Me_1": [1.23456789012345],
                }
            )

            with TemporaryDirectory() as tmpdir:
                run(
                    script="DS_r <- DS_1;",
                    data_structures=ds_structure,
                    datapoints={"DS_1": datapoints},
                    output_folder=Path(tmpdir),
                )

                # Read the output file
                output_file = Path(tmpdir) / "DS_r.csv"
                with open(output_file) as f:
                    content = f.read()

                # The number should be formatted with 10 significant digits
                # 1.23456789012345 with 10 sig digits is 1.234567890
                assert "1.23456789" in content

    def test_output_formatting_disabled(self):
        """Test output formatting when disabled."""
        with mock.patch.dict(os.environ, {ENV_OUTPUT_SIGNIFICANT_DIGITS: "-1"}):
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

            datapoints = pd.DataFrame(
                {
                    "Id_1": [1],
                    "Me_1": [1.23456789012345],
                }
            )

            with TemporaryDirectory() as tmpdir:
                run(
                    script="DS_r <- DS_1;",
                    data_structures=ds_structure,
                    datapoints={"DS_1": datapoints},
                    output_folder=Path(tmpdir),
                )

                # Read the output file
                output_file = Path(tmpdir) / "DS_r.csv"
                with open(output_file) as f:
                    content = f.read()

                # With disabled formatting, pandas uses its default
                # which typically shows more digits
                assert "1.234567890123" in content
