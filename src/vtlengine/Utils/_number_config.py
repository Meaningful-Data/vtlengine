"""
Configuration utilities for VTL Number type handling.

This module provides functions to read and validate environment variables
that control Number type behavior in numeric operations, comparisons, and output formatting.
"""

import os
from typing import Optional

# Environment variable names
ENV_COMPARISON_THRESHOLD = "COMPARISON_ABSOLUTE_THRESHOLD"
ENV_OUTPUT_SIGNIFICANT_DIGITS = "OUTPUT_NUMBER_SIGNIFICANT_DIGITS"

# Default value for significant digits
DEFAULT_SIGNIFICANT_DIGITS = 15

# Valid range for significant digits
MIN_SIGNIFICANT_DIGITS = 6
MAX_SIGNIFICANT_DIGITS = 15

# Value to disable the feature
DISABLED_VALUE = -1


def _parse_env_value(env_var: str) -> Optional[int]:
    """
    Parse an environment variable value for significant digits configuration.

    Args:
        env_var: Name of the environment variable to read.

    Returns:
        - None if the environment variable is not set (use default)
        - The integer value if valid
        - Raises ValueError for invalid values

    Raises:
        ValueError: If the value is not a valid integer or out of range.
    """
    value = os.environ.get(env_var)

    if value is None or value.strip() == "":
        return None

    try:
        int_value = int(value)
    except ValueError:
        raise ValueError(
            f"Invalid value for {env_var}: '{value}'. "
            f"Expected an integer between {MIN_SIGNIFICANT_DIGITS} and {MAX_SIGNIFICANT_DIGITS}, "
            f"or {DISABLED_VALUE} to disable."
        ) from None

    if int_value == DISABLED_VALUE:
        return DISABLED_VALUE

    if int_value < MIN_SIGNIFICANT_DIGITS or int_value > MAX_SIGNIFICANT_DIGITS:
        raise ValueError(
            f"Invalid value for {env_var}: {int_value}. "
            f"Expected an integer between {MIN_SIGNIFICANT_DIGITS} and {MAX_SIGNIFICANT_DIGITS}, "
            f"or {DISABLED_VALUE} to disable."
        )

    return int_value


def get_comparison_significant_digits() -> Optional[int]:
    """
    Get the number of significant digits for Number comparison operations.

    This affects equality-based comparison operators: =, >=, <=, between.

    Returns:
        - DISABLED_VALUE (-1): Feature is disabled, use Python's default comparison
        - None or positive int: Number of significant digits for tolerance calculation
          (None means use DEFAULT_SIGNIFICANT_DIGITS)
    """
    return _parse_env_value(ENV_COMPARISON_THRESHOLD)


def get_output_significant_digits() -> Optional[int]:
    """
    Get the number of significant digits for Number output formatting.

    This affects how Number values are formatted when writing to CSV.

    Returns:
        - DISABLED_VALUE (-1): Feature is disabled, use pandas default formatting
        - None or positive int: Number of significant digits for float_format
          (None means use DEFAULT_SIGNIFICANT_DIGITS)
    """
    return _parse_env_value(ENV_OUTPUT_SIGNIFICANT_DIGITS)


def get_effective_comparison_digits() -> Optional[int]:
    """
    Get the effective number of significant digits for comparisons.

    Returns:
        - None if the feature is disabled (DISABLED_VALUE was set)
        - The configured value, or DEFAULT_SIGNIFICANT_DIGITS if not set
    """
    value = get_comparison_significant_digits()
    if value == DISABLED_VALUE:
        return None
    return value if value is not None else DEFAULT_SIGNIFICANT_DIGITS


def get_effective_output_digits() -> Optional[int]:
    """
    Get the effective number of significant digits for output.

    Returns:
        - None if the feature is disabled (DISABLED_VALUE was set)
        - The configured value, or DEFAULT_SIGNIFICANT_DIGITS if not set
    """
    value = get_output_significant_digits()
    if value == DISABLED_VALUE:
        return None
    return value if value is not None else DEFAULT_SIGNIFICANT_DIGITS


def get_effective_numeric_digits() -> Optional[int]:
    """
    Get the effective number of significant digits for numeric operations.

    This affects the precision of arithmetic operations (division, multiplication, etc.)
    by setting the Decimal context precision.

    Uses the OUTPUT_NUMBER_SIGNIFICANT_DIGITS environment variable.

    Returns:
        - None if the feature is disabled (DISABLED_VALUE was set)
        - The configured value, or DEFAULT_SIGNIFICANT_DIGITS if not set
    """
    value = get_output_significant_digits()
    if value == DISABLED_VALUE:
        return None
    return value if value is not None else DEFAULT_SIGNIFICANT_DIGITS


def get_float_format() -> Optional[str]:
    """
    Get the float_format string for pandas to_csv.

    Returns:
        - None if the feature is disabled
        - A format string like ".10g" for the configured significant digits
    """
    digits = get_effective_output_digits()
    if digits is None:
        return None
    return f"%.{digits}g"


def _get_rel_tol(significant_digits: Optional[int]) -> Optional[float]:
    """
    Calculate the relative tolerance for number comparisons based on significant digits.

    For n significant digits, the last digit is in position 10^(-(n-1)) relative to the
    leading digit. Rounding at that position gives uncertainty of ±0.5 in the last digit,
    which translates to a relative tolerance of 0.5 * 10^(-(n-1)).

    Args:
        significant_digits: Number of significant digits, or None if disabled.

    Returns:
        Relative tolerance value, or None if feature is disabled.
    """
    if significant_digits is None:
        return None
    return 5 * (10 ** (-(significant_digits)))


def numbers_are_equal(a: float, b: float, significant_digits: Optional[int] = None) -> bool:
    """
    Compare two numbers for equality using significant digits tolerance.

    Args:
        a: First number to compare.
        b: Second number to compare.
        significant_digits: Number of significant digits to use. If None,
            uses get_effective_comparison_digits().

    Returns:
        True if the numbers are considered equal within the tolerance.
    """
    if significant_digits is None:
        significant_digits = get_effective_comparison_digits()

    rel_tol = _get_rel_tol(significant_digits)

    if rel_tol is None:
        return a == b

    if a == b:  # Handles exact matches, infinities
        return True

    max_abs = max(abs(a), abs(b))
    if max_abs == 0:
        return True

    # Calculate absolute tolerance based on the magnitude
    abs_tol = rel_tol * max_abs

    # Implementation of math.isclose function logic with relative tolerance and absolute tolerance
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def numbers_are_less_equal(a: float, b: float, significant_digits: Optional[int] = None) -> bool:
    """
    Compare a <= b using significant digits tolerance for equality.

    Args:
        a: First number.
        b: Second number.
        significant_digits: Number of significant digits to use. If None,
            uses get_effective_comparison_digits().

    Returns:
        True if a <= b (with tolerance for equality).
    """
    if significant_digits is None:
        significant_digits = get_effective_comparison_digits()

    rel_tol = _get_rel_tol(significant_digits)

    if rel_tol is None:
        return a <= b

    if numbers_are_equal(a, b, significant_digits):
        return True

    return a < b


def numbers_are_greater_equal(a: float, b: float, significant_digits: Optional[int] = None) -> bool:
    """
    Compare a >= b using significant digits tolerance for equality.

    Args:
        a: First number.
        b: Second number.
        significant_digits: Number of significant digits to use. If None,
            uses get_effective_comparison_digits().

    Returns:
        True if a >= b (with tolerance for equality).
    """
    if significant_digits is None:
        significant_digits = get_effective_comparison_digits()

    rel_tol = _get_rel_tol(significant_digits)

    if rel_tol is None:
        return a >= b

    if numbers_are_equal(a, b, significant_digits):
        return True

    return a > b
