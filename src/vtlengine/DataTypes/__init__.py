from typing import Any, Type

import pandas as pd

from vtlengine.DataTypes.TimeHandling import str_period_to_date, check_max_date, date_to_period_str
from vtlengine.Exceptions import SemanticError

DTYPE_MAPPING = {
    'String': 'string',
    'Number': 'float64',
    'Integer': 'int64',
    'TimeInterval': 'string',
    'Date': 'string',
    'TimePeriod': 'string',
    'Duration': 'string',
    'Boolean': 'object',
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
    def implicit_cast(cls, value, from_type: Type['ScalarType']) -> Any:
        raise Exception("Method should be implemented by inheritors")

    @classmethod
    def explicit_cast(cls, value, from_type: Type['ScalarType']) -> Any:
        raise Exception("Method should be implemented by inheritors")

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
    def dtype(cls) -> str:
        return DTYPE_MAPPING[cls.__name__]

    __str__ = __repr__


class String(ScalarType):
    """

    """
    default = ""

    @classmethod
    def implicit_cast(cls, value, from_type: Type['ScalarType']) -> str:
        # if pd.isna(value):
        #     return cls.default
        if from_type in {Number, Integer, Boolean, String, Date, TimePeriod, TimeInterval,
                         Duration}:
            return str(value)

        raise SemanticError("2-1-5-1", value=value,
                            type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
                            type_2=SCALAR_TYPES_CLASS_REVERSE[cls])

    @classmethod
    def explicit_cast(cls, value, from_type: Type['ScalarType']) -> str:
        if from_type in {TimePeriod, Date, String}:
            return str(value)

        raise SemanticError("2-1-5-1", value=value,
                            type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
                            type_2=SCALAR_TYPES_CLASS_REVERSE[cls])


class Number(ScalarType):
    """
    """

    def __eq__(self, other):
        return (self.__class__.__name__ == other.__class__.__name__ or
                other.__class__.__name__ == Integer.__name__)

    def __ne__(self, other):
        return not self.__eq__(other)

    @classmethod
    def implicit_cast(cls, value, from_type: Type['ScalarType']) -> float:
        # if pd.isna(value):
        #     return cls.default
        if from_type in {Integer, Number}:
            return float(value)

        raise SemanticError("2-1-5-1", value=value,
                            type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
                            type_2=SCALAR_TYPES_CLASS_REVERSE[cls])

    @classmethod
    def explicit_cast(cls, value, from_type: Type['ScalarType']) -> float:
        if from_type in {Boolean}:
            if value:
                return 1.0
            else:
                return 0.0
        elif from_type in {Integer, Number, String}:
            try:
                return float(value)
            except ValueError:
                pass

        raise SemanticError("2-1-5-1", value=value,
                            type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
                            type_2=SCALAR_TYPES_CLASS_REVERSE[cls])

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
    def implicit_cast(cls, value, from_type: Type['ScalarType']) -> int:
        if from_type.__name__ == "Integer":
            return value

        if from_type.__name__ == "Number":
            if value.is_integer():
                return int(value)
            else:
                raise SemanticError("2-1-5-1", value=value,
                                    type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
                                    type_2=SCALAR_TYPES_CLASS_REVERSE[cls])

        raise SemanticError("2-1-5-1", value=value,
                            type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
                            type_2=SCALAR_TYPES_CLASS_REVERSE[cls])

    @classmethod
    def explicit_cast(cls, value, from_type: Type['ScalarType']) -> int:
        if from_type in {Boolean}:
            if value:
                return 1
            else:
                return 0
        if from_type in {Number, String}:
            try:
                if float(value) - int(value) != 0:
                    raise SemanticError("2-1-5-1", value=value,
                                        type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
                                        type_2=SCALAR_TYPES_CLASS_REVERSE[cls])
            except ValueError:
                raise SemanticError("2-1-5-1", value=value,
                                    type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
                                    type_2=SCALAR_TYPES_CLASS_REVERSE[cls])
            return int(value)

        raise SemanticError("2-1-5-1", value=value,
                            type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
                            type_2=SCALAR_TYPES_CLASS_REVERSE[cls])

    @classmethod
    def cast(cls, value):
        if pd.isnull(value):
            return None
        if isinstance(value, float):
            # Check if the float has decimals
            if value.is_integer():
                return int(value)
            else:
                raise ValueError(f"Value {value} has decimals, cannot cast to integer")
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

    @classmethod
    def implicit_cast(cls, value, from_type: Type['ScalarType']) -> Any:
        # TODO: Remove String, only for compatibility with previous engine
        if from_type in {TimeInterval, String}:
            return value
        if from_type in {Date}:
            value = check_max_date(value)

            return f"{value}/{value}"

        if from_type in {TimePeriod}:
            init_value = str_period_to_date(value, start=True).isoformat()
            end_value = str_period_to_date(value, start=False).isoformat()
            return f"{init_value}/{end_value}"

        raise SemanticError("2-1-5-1", value=value,
                            type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
                            type_2=SCALAR_TYPES_CLASS_REVERSE[cls])

    @classmethod
    def explicit_cast(cls, value, from_type: Type['ScalarType']) -> Any:

        raise SemanticError("2-1-5-1", value=value,
                            type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
                            type_2=SCALAR_TYPES_CLASS_REVERSE[cls])


class Date(TimeInterval):
    """

    """
    default = None

    @classmethod
    def implicit_cast(cls, value, from_type: Type['ScalarType']) -> Any:
        # TODO: Remove String, only for compatibility with previous engine
        if from_type in {Date, String}:
            return value

        raise SemanticError("2-1-5-1", value=value,
                            type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
                            type_2=SCALAR_TYPES_CLASS_REVERSE[cls])

    @classmethod
    def explicit_cast(cls, value, from_type: Type['ScalarType']) -> Any:
        # TODO: Remove String, only for compatibility with previous engine
        if from_type == String:
            return value

        raise SemanticError("2-1-5-1", value=value,
                            type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
                            type_2=SCALAR_TYPES_CLASS_REVERSE[cls])


class TimePeriod(TimeInterval):
    """

    """
    default = None

    @classmethod
    def implicit_cast(cls, value, from_type: Type['ScalarType']) -> Any:
        # TODO: Remove String, only for compatibility with previous engine
        if from_type in {TimePeriod, String}:
            return value

        raise SemanticError("2-1-5-1", value=value,
                            type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
                            type_2=SCALAR_TYPES_CLASS_REVERSE[cls])

    @classmethod
    def explicit_cast(cls, value, from_type: Type['ScalarType']) -> Any:
        if from_type in {Date}:
            try:
                period_str = date_to_period_str(value, "D")
            except ValueError:
                raise SemanticError("2-1-5-1", value=value,
                                    type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
                                    type_2=SCALAR_TYPES_CLASS_REVERSE[cls])
            return period_str
        # TODO: Remove String, only for compatibility with previous engine
        elif from_type == String:
            return value

        raise SemanticError("2-1-5-1", value=value,
                            type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
                            type_2=SCALAR_TYPES_CLASS_REVERSE[cls])


class Duration(ScalarType):

    @classmethod
    def implicit_cast(cls, value, from_type: Type['ScalarType']) -> str:
        if from_type in {Duration, String}:
            return value

        raise SemanticError("2-1-5-1", value=value,
                            type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
                            type_2=SCALAR_TYPES_CLASS_REVERSE[cls])

    @classmethod
    def explicit_cast(cls, value, from_type: Type['ScalarType']) -> Any:
        if from_type == String:
            return value

        raise SemanticError("2-1-5-1", value=value,
                            type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
                            type_2=SCALAR_TYPES_CLASS_REVERSE[cls])


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

    @classmethod
    def implicit_cast(cls, value, from_type: Type['ScalarType']) -> bool:
        if from_type in {Boolean}:
            return value

        raise SemanticError("2-1-5-1", value=value,
                            type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
                            type_2=SCALAR_TYPES_CLASS_REVERSE[cls])

    @classmethod
    def explicit_cast(cls, value, from_type: Type['ScalarType']) -> bool:
        if from_type in {Number, Integer}:
            if value in {0, 0.0}:
                return False
            return True

        raise SemanticError("2-1-5-1", value=value,
                            type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
                            type_2=SCALAR_TYPES_CLASS_REVERSE[cls])


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

SCALAR_TYPES_CLASS_REVERSE = {
    String: 'String',
    Number: 'Number',
    Integer: 'Integer',
    TimeInterval: 'Time',
    Date: 'Date',
    TimePeriod: 'Time_Period',
    Duration: 'Duration',
    Boolean: 'Boolean',
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
    Boolean: 'bool_var',
    Null: 'null_var'
}

IMPLICIT_TYPE_PROMOTION_MAPPING = {
    # TODO: Remove Time types, only for compatibility with previous engine
    String: {String, Boolean, TimePeriod},
    Number: {String, Number, Integer},
    Integer: {String, Number, Integer},
    # TODO: Remove String, only for compatibility with previous engine
    TimeInterval: {TimeInterval},
    Date: {TimeInterval, Date},
    TimePeriod: {TimeInterval, TimePeriod},
    Duration: {Duration},
    Boolean: {String, Boolean},
    Null: {String, Number, Integer, TimeInterval, Date, TimePeriod, Duration, Boolean, Null}
}

# TODO: Implicit are valid as cast without mask
EXPLICIT_WITHOUT_MASK_TYPE_PROMOTION_MAPPING = {
    # TODO: Remove time types, only for compatibility with previous engine
    String: {Integer, String, Date, TimePeriod, TimeInterval, Duration, Number},
    Number: {Integer, Boolean, String, Number},
    Integer: {Number, Boolean, String, Integer},
    # TODO: Remove String on time types, only for compatibility with previous engine
    TimeInterval: {TimeInterval, String},
    Date: {TimePeriod, Date, String},
    TimePeriod: {TimePeriod, String},
    Duration: {Duration, String},
    Boolean: {Integer, Number, String, Boolean},
    Null: {String, Number, Integer, TimeInterval, Date, TimePeriod, Duration, Boolean, Null}
}

EXPLICIT_WITH_MASK_TYPE_PROMOTION_MAPPING = {
    String: {Number, TimeInterval, Date, TimePeriod, Duration},
    Number: {},
    Integer: {},
    TimeInterval: {String},
    Date: {String},
    TimePeriod: {Date},
    Duration: {String},
    Boolean: {},
    Null: {String, Number, Integer, TimeInterval, Date, TimePeriod, Duration, Boolean, Null}
}


def binary_implicit_promotion(left_type: ScalarType,
                              right_type: ScalarType,
                              type_to_check: ScalarType = None,
                              return_type: ScalarType = None) -> ScalarType:
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
                if left_type.is_subtype(right_type):  # For Integer and Number
                    return right_type
                elif right_type.is_subtype(left_type):
                    return left_type
                return left_type
            if right_type.is_included(left_implicities):
                return right_type
            return type_to_check
        raise SemanticError(code="1-1-1-2",
                            type_1=SCALAR_TYPES_CLASS_REVERSE[left_type],
                            type_2=SCALAR_TYPES_CLASS_REVERSE[right_type],
                            type_check=SCALAR_TYPES_CLASS_REVERSE[type_to_check])
        # raise Exception(f"Implicit cast not allowed from {left_type} and {right_type} to {type_to_check}")

    if return_type and (left_type.is_included(
            right_implicities) or right_type.is_included(left_implicities)):
        return return_type
    if left_type.is_included(right_implicities):
        return left_type
    if right_type.is_included(left_implicities):
        return right_type

    raise SemanticError(code="1-1-1-1",
                        type_1=SCALAR_TYPES_CLASS_REVERSE[left_type],
                        type_2=SCALAR_TYPES_CLASS_REVERSE[right_type])


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
            raise SemanticError(code="1-1-1-1",
                                type_1=SCALAR_TYPES_CLASS_REVERSE[operand_type],
                                type_2=SCALAR_TYPES_CLASS_REVERSE[type_to_check])

    if return_type:
        return return_type
    if (type_to_check and not issubclass(operand_type, type_to_check)
            and not issubclass(type_to_check, operand_type)):
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
