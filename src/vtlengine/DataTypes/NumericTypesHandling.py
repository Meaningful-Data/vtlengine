import operator
from decimal import Decimal
from typing import Union


def decimal_add(a: Union[float, int], b: Union[float, int]) -> float:
    """
    Adds two numbers, if they are floats, converts them to Decimal and then to float
    :param a: first number
    :param b: second number
    :return: the sum of the two numbers
    """
    if isinstance(a, float) and isinstance(b, float):
        decimal_value = Decimal(a) + Decimal(b)
        return float(decimal_value)

    return operator.add(a, b)


def decimal_sub(a: Union[float, int], b: Union[float, int]) -> float:
    if isinstance(a, float) and isinstance(b, float):
        decimal_value = Decimal(a) - Decimal(b)
        return float(decimal_value)
    return operator.sub(a, b)


def decimal_mul(a: Union[float, int], b: Union[float, int]) -> float:
    if isinstance(a, float) and isinstance(b, float):
        decimal_value = Decimal(a) * Decimal(b)
        return float(decimal_value)
    return operator.mul(a, b)


def decimal_div(a: Union[float, int], b: Union[float, int]) -> float:
    if isinstance(a, float) and isinstance(b, float):
        decimal_value = Decimal(a) / Decimal(b)
        return float(decimal_value)
    return operator.truediv(a, b)
