from datetime import date
from typing import Union, Optional


def year_duck(value: Optional[Union[date, str]])-> Optional[int]:
    """
    Extracts the year from a date or time value.
    If the input value is None or NaN, it returns None.
    If the input value is not a valid date or time, it raises a SemanticError.

    Args:
        value (str): The date or time value in string format.

    Returns:
        int | None: The extracted year as an integer, or None if the input is None or NaN.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return int(value[:4])
    return value.year

def month_duck(value: str) -> int | None:
    pass