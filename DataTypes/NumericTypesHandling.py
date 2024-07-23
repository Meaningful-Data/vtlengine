from decimal import Decimal, getcontext
# from functools import reduce
import operator

def decimal_add(a, b):
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

def decimal_sub(a, b):
    if isinstance(a, float) and isinstance(b, float):
        decimal_value = Decimal(a) - Decimal(b)
        return float(decimal_value)
    return operator.sub(a, b)

def decimal_mul(a, b):
    if isinstance(a, float) and isinstance(b, float):
        decimal_value = Decimal(a) * Decimal(b)
        return float(decimal_value)
    return operator.mul(a, b)

def decimal_div(a, b):
    if isinstance(a, float) and isinstance(b, float):
        decimal_value = Decimal(a) / Decimal(b)
        return float(decimal_value)
    return operator.truediv(a, b)