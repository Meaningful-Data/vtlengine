from datetime import date
from typing import Any, Optional, Union

from vtlengine.DataTypes import Number, String


def isnull_duck(x: Any) -> bool:
    return x is None


BASIC_SCALAR_TYPES = Optional[Union[str, float, int, bool, date]]


def between_duck(
    x: BASIC_SCALAR_TYPES, y: BASIC_SCALAR_TYPES, z: BASIC_SCALAR_TYPES
) -> Optional[bool]:
    if x is None or y is None or z is None:
        return None
    x, y = _comparison_cast_values(x, y)
    z, _ = _comparison_cast_values(z, y)
    return y <= x <= z  # type: ignore[operator]


def _comparison_cast_values(
    x: BASIC_SCALAR_TYPES,
    y: BASIC_SCALAR_TYPES,
) -> Any:
    # Cast values to compatible types for comparison
    try:
        if isinstance(x, str) and isinstance(y, bool):
            y = String.cast(y)
        elif isinstance(x, bool) and isinstance(y, str):
            x = String.cast(x)
        elif isinstance(x, str) and isinstance(y, (int, float)):
            x = Number.cast(x)
        elif isinstance(x, (int, float)) and isinstance(y, str):
            y = Number.cast(y)
        elif isinstance(x, date) and isinstance(y, (int, float)):
            x = Number.cast(x.year)
    except ValueError:
        x = str(x)
        y = str(y)

    return x, y
