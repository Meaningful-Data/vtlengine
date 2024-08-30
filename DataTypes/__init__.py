from DataTypes.TimeHandling import str_period_to_date, check_max_date, date_to_period_str
from typing import Any, Type
import numpy as np
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
    def implicit_cast(cls, value, from_type: Type['ScalarType']) -> Any:
        raise Exception("Method should be implemented by inheritors")
    
    @classmethod
    def explicit_cast(cls, value, from_type: Type['ScalarType']) -> Any:
        raise Exception("Method should be implemented by inheritors")

    def is_subtype(self, obj) -> bool:
        if not isinstance(obj, ScalarType):
            raise Exception("Not use is_subtype")
        return issubclass(self.__class__, obj.__class__)

    def is_null_type(self) -> bool:
        return False

    def check_type(self, value):
        if isinstance(value, CAST_MAPPING[self.__class__.__name__]):
            return True

        raise Exception(f"Value {value} is not a {self.__class__.__name__}")

    def cast(self, value):
        return CAST_MAPPING[self.__class__.__name__](value)

    def dtype(self):
        return DTYPE_MAPPING[self.__class__.__name__]

    __str__ = __repr__


class String(ScalarType):
    """

    """
    default = ""

    @classmethod
    def implicit_cast(cls, value, from_type: Type['ScalarType']) -> str:
        # if pd.isna(value):
        #     return cls.default
        if from_type in {Number, Integer, Boolean, String}:
            return str(value)

        raise Exception(f"Cannot implicit cast {from_type} to {cls}")
    
    @classmethod
    def explicit_cast(cls, value, from_type: Type['ScalarType']) -> str:
        if from_type in {TimePeriod, String}:
            return str(value)
        
        raise Exception(f"Cannot explicit without mask cast {from_type} to {cls}")


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

        raise Exception(f"Cannot implicit cast {from_type} to {cls}")
    
    @classmethod
    def explicit_cast(cls, value, from_type: Type['ScalarType']) -> float:
        if from_type in {Boolean}:
            if value:
                return 1.0
            else:
                return 0.0
        
        raise Exception(f"Cannot explicit without mask cast {from_type} to {cls}")


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
        if from_type in {Integer}:
            return value

        raise Exception(f"Cannot implicit cast {from_type} to {cls}")
    
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
                    raise RuntimeError(f"Cannot explicit cast the value {value} from {from_type} to {cls}")
            except ValueError:
                raise RuntimeError(f"Cannot explicit cast the value {value} from {from_type} to {cls}")
            return int(value)
        
        raise Exception(f"Cannot explicit without mask cast {from_type} to {cls}")


class TimeInterval(ScalarType):
    """

    """
    default = None

    @classmethod
    def implicit_cast(cls, value, from_type: Type['ScalarType']) -> Any:
        if from_type in {TimeInterval}:
            return value
        if from_type in {Date}:
            value = check_max_date(value)
            
            return f"{value}/{value}"

        if from_type in {TimePeriod}:
            init_value = str_period_to_date(value, start=True).isoformat()
            end_value = str_period_to_date(value, start=False).isoformat()
            return f"{init_value}/{end_value}"

        raise Exception(f"Cannot implicit cast {from_type} to {cls}")
    
    @classmethod
    def explicit_cast(cls, value, from_type: Type['ScalarType']) -> Any:
        
        raise Exception(f"Cannot explicit without mask cast {from_type} to {cls}")


class Date(TimeInterval):
    """

    """
    default = None

    @classmethod
    def implicit_cast(cls, value, from_type: Type['ScalarType']) -> Any:
        if from_type in {Date}:
            return value

        raise Exception(f"Cannot implicit cast {from_type} to {cls}")
    
    @classmethod
    def explicit_cast(cls, value, from_type: Type['ScalarType']) -> Any:
        
        raise Exception(f"Cannot explicit without mask cast {from_type} to {cls}")


class TimePeriod(TimeInterval):
    """

    """
    default = None

    @classmethod
    def implicit_cast(cls, value, from_type: Type['ScalarType']) -> Any:
        if from_type in {TimePeriod}:
            return value

        raise Exception(f"Cannot implicit cast {from_type} to {cls}")
    
    @classmethod
    def explicit_cast(cls, value, from_type: Type['ScalarType']) -> Any:
        if from_type in {Date}:
            period_str = date_to_period_str(value,"D")
            return period_str
        
        raise Exception(f"Cannot explicit without mask cast {from_type} to {cls}")


class Duration(ScalarType):
    
    @classmethod
    def implicit_cast(cls, value, from_type: Type['ScalarType']) -> str:
        if from_type in {Duration}:
            return value

        raise Exception(f"Cannot implicit cast {from_type} to {cls}")
    
    @classmethod
    def explicit_cast(cls, value, from_type: Type['ScalarType']) -> Any:
        
        raise Exception(f"Cannot explicit without mask cast {from_type} to {cls}")


class Boolean(ScalarType):
    """
    """
    default = None

    def cast(self, value):
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

        raise Exception(f"Cannot implicit cast {from_type} to {cls}")
    
    @classmethod
    def explicit_cast(cls, value, from_type: Type['ScalarType']) -> bool:
        if from_type in {Number, Integer}:
            if value in {0, 0.0}:
                return False
            return True
        
        raise Exception(f"Cannot explicit without mask cast {from_type} to {cls}")


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
    String: {String},
    Number: {String, Number},
    Integer: {String, Number, Integer},
    TimeInterval: {TimeInterval},
    Date: {TimeInterval, Date},
    TimePeriod: {TimeInterval, TimePeriod},
    Duration: {String, Duration},
    Boolean: {String, Boolean},
    Null: {String, Number, Integer, TimeInterval, Date, TimePeriod, Duration, Boolean, Null}
}

# TODO: Implicit are valid as cast without mask
EXPLICIT_WITHOUT_MASK_TYPE_PROMOTION_MAPPING = {
    String: {Integer, String},
    Number: {Integer, Boolean, String, Number},
    Integer: {Number, Boolean, String, Integer},
    TimeInterval: {TimeInterval},
    Date: {TimePeriod, Date},
    TimePeriod: {String, TimePeriod},
    Duration: {Duration},
    Boolean: {Integer, Number, String, Boolean},
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
                return left_type
            if right_type.is_included(left_implicities):
                return right_type
            return type_to_check
        raise Exception("Implicit cast not allowed")

    if return_type and (left_type.is_included(
            right_implicities) or right_type.is_included(left_implicities)):
        return return_type
    if left_type.is_included(right_implicities):
        return left_type
    if right_type.is_included(left_implicities):
        return right_type

    raise Exception("Implicit cast not allowed")


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
            raise Exception("Implicit cast not allowed")

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
