from typing import Type

import pandas as pd

DTYPE_MAPPING = {
    'String': 'string',
    'Number': 'Float64',
    'Integer': 'Int64',
    'TimeInterval': 'string',
    'Date': 'string',
    'TimePeriod': 'string',
    'Duration': 'string',
    'Boolean': 'boolean',
}

CAST_MAPPING = {
    'String': str,
    'Number': float,
    'Integer': int,
    'TimeInterval': str,
    'Date': str,
    'TimePeriod': str,
    'Duration': str,
    'Boolean': bool,
}


class ScalarType:
    """
    """

    default = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    def strictly_same_class(self, obj) -> bool:
        if not isinstance(obj, ScalarType):
            raise Exception("Not use strictly_same_class")
        return self.__class__ == obj.__class__

    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__

    def __ne__(self, other):
        return not self.__eq__(other)

    def instance_is_included(self, set_: set) -> bool:
        return self.__class__ in set_

    @classmethod
    def is_included(cls, set_: set) -> bool:
        return cls in set_

    @classmethod
    def promotion_changed_type(cls, promoted: Type['ScalarType']) -> bool:
        return not issubclass(cls, promoted)

    @classmethod
    def is_subtype(cls, obj: Type["ScalarType"]) -> bool:
        return issubclass(cls, obj)

    def is_null_type(self) -> bool:
        return False

    @classmethod
    def check_type(cls, value):
        if isinstance(value, CAST_MAPPING[cls.__name__]):
            return True

        raise Exception(f"Value {value} is not a {cls.__name__}")

    @classmethod
    def cast(cls, value):
        if pd.isnull(value):
            return None
        return CAST_MAPPING[cls.__name__](value)

    @classmethod
    def dtype(cls):
        return DTYPE_MAPPING[cls.__name__]

    __str__ = __repr__


class String(ScalarType):
    """

    """
    default = ""


class Number(ScalarType):
    """
    """

    def __eq__(self, other):
        return (self.__class__.__name__ == other.__class__.__name__ or
                other.__class__.__name__ == Integer.__name__)

    def __ne__(self, other):
        return not self.__eq__(other)

    @classmethod
    def cast(cls, value):
        if pd.isnull(value):
            return None
        if isinstance(value, str):
            if value.lower() == "true":
                return 1.0
            elif value.lower() == "false":
                return 0.0
        return float(value)


class Integer(Number):
    """
    """

    def __eq__(self, other):
        return (self.__class__.__name__ == other.__class__.__name__ or
                other.__class__.__name__ == Number.__name__)

    def __ne__(self, other):
        return not self.__eq__(other)

    @classmethod
    def cast(cls, value):
        if pd.isnull(value):
            return None
        if isinstance(value, float):
            # Check if the float has decimals
            if value.is_integer():
                return int(value)
            else:
                raise Exception(f"Value {value} has decimals, cannot cast to integer")
        if isinstance(value, str):
            if value.lower() == "true":
                return 1
            elif value.lower() == "false":
                return 0
        return int(value)


class TimeInterval(ScalarType):
    """

    """
    default = None


class Date(TimeInterval):
    """

    """
    default = None


class TimePeriod(TimeInterval):
    """

    """
    default = None


class Duration(ScalarType):
    pass


class Boolean(ScalarType):
    """
    """
    default = None

    def cast(self, value):
        if pd.isnull(value):
            return None
        if isinstance(value, str):
            if value.lower() == "true":
                return True
            elif value.lower() == "false":
                return False
            elif value.lower() == "1":
                return True
            elif value.lower() == "0":
                return False
            else:
                return None
        if isinstance(value, int):
            if value != 0:
                return True
            else:
                return False
        if isinstance(value, float):
            if value != 0.0:
                return True
            else:
                return False
        if isinstance(value, bool):
            return value
        return value


class Null(ScalarType):
    """
    """

    def is_null_type(self) -> bool:
        return True

    def check_type(self, value):
        return True

    def cast(self, value):
        return None

    def dtype(self):
        return 'string'


SCALAR_TYPES = {
    'String': String,
    'Number': Number,
    'Integer': Integer,
    'Time': TimeInterval,
    'Date': Date,
    'Time_Period': TimePeriod,
    'Duration': Duration,
    'Boolean': Boolean,
}

BASIC_TYPES = {
    str: String,
    int: Integer,
    float: Number,
    bool: Boolean,
    type(None): Null
}

COMP_NAME_MAPPING = {
    String: 'str_var',
    Number: 'num_var',
    Integer: 'int_var',
    TimeInterval: 'time_var',
    TimePeriod: 'time_period_var',
    Date: 'date_var',
    Duration: 'duration_var',
    Boolean: 'bool_var'
}

IMPLICIT_TYPE_PROMOTION_MAPPING = {
    String: {String, Boolean},
    Number: {String, Number, Integer},
    Integer: {String, Number, Integer},
    TimeInterval: {String, TimeInterval},
    Date: {String, TimeInterval, Date},
    TimePeriod: {String, TimeInterval, TimePeriod},
    Duration: {String, Duration},
    Boolean: {String, Boolean},
    Null: {String, Number, Integer, TimeInterval, Date, TimePeriod, Duration, Boolean, Null}
}


def binary_implicit_promotion(left_type: ScalarType,
                              right_type: ScalarType,
                              type_to_check: ScalarType = None,
                              return_type: ScalarType = None
                              ) -> ScalarType:
    """
    Validates the compatibility between the types of the operands and the operator
    (implicit type promotion : check_binary_implicit_type_promotion)
    :param left: The left operand
    :param right: The right operand
    :return: The resulting type of the operation, after the implicit type promotion
    """
    left_implicities = IMPLICIT_TYPE_PROMOTION_MAPPING[left_type]
    right_implicities = IMPLICIT_TYPE_PROMOTION_MAPPING[right_type]
    if type_to_check is not None:
        if type_to_check.is_included(left_implicities.intersection(right_implicities)):
            if return_type is not None:
                return return_type
            if left_type.is_included(right_implicities):
                if left_type.is_subtype(right_type): # For Integer and Number
                    return right_type
                elif right_type.is_subtype(left_type):
                    return left_type
                return left_type
            if right_type.is_included(left_implicities):
                return right_type
            return type_to_check
        raise Exception(f"Implicit cast not allowed from {left_type} and {right_type} to {type_to_check}")

    if return_type and (left_type.is_included(
            right_implicities) or right_type.is_included(left_implicities)):
        return return_type
    if left_type.is_included(right_implicities):
        return left_type
    if right_type.is_included(left_implicities):
        return right_type

    raise Exception(f"Implicit cast not allowed from {left_type} to {right_type}")


def check_binary_implicit_promotion(
        left: ScalarType, right: ScalarType,
        type_to_check: ScalarType = None, return_type: ScalarType = None
) -> bool:
    """
    Validates the compatibility between the types of the operands and the operator
    (implicit type promotion : check_binary_implicit_type_promotion)
    :param left: The left operand
    :param right: The right operand
    :param type_to_check: The type of the operator (from the operator if any)
    :param return_type: The type of the result (from the operator if any)
    :return: True if the types are compatible, False otherwise
    """
    left_implicities = IMPLICIT_TYPE_PROMOTION_MAPPING[left]
    right_implicities = IMPLICIT_TYPE_PROMOTION_MAPPING[right]
    if type_to_check:
        return type_to_check.is_included(set_=left_implicities.intersection(right_implicities))

    return left.is_included(right_implicities) or right.is_included(left_implicities)


def unary_implicit_promotion(
        operand_type: ScalarType, type_to_check: ScalarType = None, return_type: ScalarType = None
) -> ScalarType:
    """
    Validates the compatibility between the type of the operand and the operator
    param operand: The operand
    param type_to_check: The type of the operator (from the operator if any)
    param return_type: The type of the result (from the operator if any)
    return: The resulting type of the operation, after the implicit type promotion
    """
    operand_implicities = IMPLICIT_TYPE_PROMOTION_MAPPING[operand_type]
    if type_to_check:
        if not type_to_check.is_included(operand_implicities):
            raise Exception(f"Implicit cast not allowed from {operand_type} to {type_to_check}")

    if return_type:
        return return_type
    if type_to_check and not issubclass(operand_type, type_to_check):
        return type_to_check
    return operand_type


def check_unary_implicit_promotion(
        operand_type: ScalarType, type_to_check: ScalarType = None, return_type: ScalarType = None
) -> bool:
    """
    Validates the compatibility between the type of the operand and the operator
    :param operand: The operand
    :param type_to_check: The type of the operator (from the operator if any)
    :param return_type: The type of the result (from the operator if any)
    :return: True if the types are compatible, False otherwise
    """
    operand_implicities = IMPLICIT_TYPE_PROMOTION_MAPPING[operand_type]
    return not (type_to_check and not type_to_check.is_included(operand_implicities))
