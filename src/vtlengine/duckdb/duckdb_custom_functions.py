import _random
from typing import Optional, Union

def round_duck(value: Optional[Union[int, float]], decimals: Optional[int]) -> Optional[float]:
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
    if value is None:
        return None
    if decimals is None:
        return int(round(value, 0))
    return round(value, decimals)


def trunc_duck(value: Optional[Union[int, float]], decimals: Optional[int]) -> Optional[float]:
    """
    Truncates a numeric value to a specified number of decimals.
    If the input value is None, the function will return None.
    When decimals are specified, the function calculates the truncated value
    based on the number of decimals.
    If decimals are not provided, the function truncates the value like decimals = 0.

    Parameters:
        value (Optional[Union[int, float]]): The numeric value to be truncated. Can be None,
        an integer, or a float.

        decimals (Optional[int]): The number of decimal places to truncate the value to.
        If None, the value will be truncated to the nearest integer.

    Returns:
        Optional[float]: The truncated float value if decimals are specified. Returns
        an integer if decimals are not specified. Returns None if the input value is None.
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

def random_duck(seed: Optional[Union[int, float]], index : int) -> Optional[float]:
    instance: PseudoRandom = PseudoRandom(seed)
    for _ in range(index):
        instance.random()
    return instance.random().__round__(6)