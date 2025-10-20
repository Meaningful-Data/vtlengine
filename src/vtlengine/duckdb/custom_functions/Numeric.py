import _random
import math
from decimal import Decimal
from typing import Optional, Union

from vtlengine.Exceptions import RunTimeError


def round_duck(
    value: Optional[Union[int, float]], decimals: Optional[Union[int, float]]
) -> Optional[float]:
    """
    Custom round function for DuckDB that handles None values and rounding.
    Applied on Round operator in VTL.
    If decimals is None, it rounds the value to 0 decimals (nearest integer).
    If value is None, it returns None.

    Args:
        value (Optional[Union[int, float]]): The value to round.
        decimals (Optional[int]): The number of decimal places to round to.
          If None, rounds to the nearest integer.

    Returns:
        Optional[float]: The rounded value, or None if the input value is None.
    """
    if value is None or math.isnan(value):
        return None
    multiplier = 1.0
    if decimals is not None:
        multiplier = 10**decimals

    if value >= 0.0:
        rounded_value = math.floor(value * multiplier + 0.5) / multiplier
    else:
        rounded_value = math.ceil(value * multiplier - 0.5) / multiplier

    if decimals is not None:
        return rounded_value

    return int(rounded_value)


def count_decimals(x: Optional[float]) -> int:
    if x is None:
        return 0
    d = Decimal(str(x)).normalize()
    return -int(d.as_tuple().exponent) if int(d.as_tuple().exponent) < 0 else 0


def round_to_ref(x: Optional[float], ref: Optional[float]) -> Optional[float]:
    """
    Rounds a numeric value to match the number of decimal places of a reference value.
    If x is None or ref is None, it returns x.
    If x has more decimal places than ref, it rounds x to the number of decimal places in ref.
    If ref is None, it returns x unchanged.
    Args:
        x (float): The value to round.
        ref (float): The reference value to match decimal places.

    Returns:
        float: The rounded value of x, or x if x is None or ref is None.
    """
    if x is None or ref is None:
        return x
    dec_x = count_decimals(x)
    dec_ref = count_decimals(ref)
    return round(x, dec_ref) if dec_x > dec_ref else x


def trunc_duck(
    value: Optional[Union[int, float]], decimals: Optional[Union[int, float]]
) -> Optional[float]:
    """
    Truncates a numeric value to a specified number of decimals.
    If the input value is None, the function will return None.
    When decimals are specified, the function calculates the truncated value
    based on the number of decimals.
    If decimals are not provided, the function truncates the value like decimals = 0.

    Args:
        value (Optional[Union[int, float]]): The numeric value to truncate.
        decimals (Optional[int]): The number of decimal places to truncate to.
          If None, truncates to the nearest integer.

    Returns:
        Optional[float]: The truncated value, or None if the input value is None.
        If decimals is None, returns an integer.
    """

    if value is None:
        return None

    multiplier = 1.0

    if decimals is not None:
        multiplier = 10**decimals

    truncated_value = int(value * multiplier) / multiplier

    if decimals is not None:
        return truncated_value

    return int(truncated_value)


class PseudoRandom(_random.Random):
    def __init__(self, seed: Union[int, float]) -> None:
        super().__init__()
        self.seed(seed)


def random_duck(seed: Optional[Union[float, int]], index: Optional[int]) -> Optional[float]:
    """Generates a pseudo-random number based on a seed and an index.

    It initializes a PseudoRandom instance with the seed,
    and generates a random number after advancing the random state by the specified index.

    Args:
        seed (float): The seed for the random number generator.
        index (int): The number of times to advance the random state before generating a number.

    Returns:
        A pseudo-random number rounded to 6 decimal places.
    """
    if index is None or seed is None:
        return None

    instance: PseudoRandom = PseudoRandom(seed)
    for _ in range(index):
        instance.random()
    return instance.random().__round__(6)


def division_duck(
    a: Optional[Union[int, float]], b: Optional[Union[int, float]]
) -> Optional[float]:
    """
    Custom DuckDB-safe division function.
    Handles division by zero and overflow.

    Args:
        a (Optional[Union[int, float]]): Dividend
        b (Optional[Union[int, float]]): Divisor

    Returns:
        Optional[float]: Result of a / b, or raises ValueError
    """
    if a is None or b is None:
        return None

    if b == 0:
        raise RunTimeError(code="2-1-15-6", value_1=a, value_2=b, op="/")
    result = a / b

    return result
