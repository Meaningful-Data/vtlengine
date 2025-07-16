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
