"""
Configuration utilities for VTL Number type handling.

This module provides functions to read and validate environment variables
that control Number type behavior in comparisons and output formatting.
"""

import os
from typing import Optional

# Environment variable names
ENV_COMPARISON_THRESHOLD = "COMPARISON_ABSOLUTE_THRESHOLD"
ENV_OUTPUT_SIGNIFICANT_DIGITS = "OUTPUT_NUMBER_SIGNIFICANT_DIGITS"

# Default value for significant digits
DEFAULT_SIGNIFICANT_DIGITS = 14

# Valid range for significant digits
MIN_SIGNIFICANT_DIGITS = 6
MAX_SIGNIFICANT_DIGITS = 14

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

    # If feature is disabled, use exact comparison
    if significant_digits is None:
        return a == b

    # Handle special cases
    if a == b:  # Handles infinities and exact matches
        return True

    # Calculate relative tolerance based on significant digits
    # For N significant digits, tolerance is 0.5 * 10^(-N+1) relative to magnitude
    rel_tol = 0.5 * (10 ** (-(significant_digits - 1)))

    # Use the larger absolute value as the reference for relative comparison
    max_abs = max(abs(a), abs(b))

    if max_abs == 0:
        return True  # Both are zero

    # Calculate absolute tolerance based on the magnitude
    abs_tol = rel_tol * max_abs

    return abs(a - b) <= abs_tol
