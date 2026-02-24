import re
from typing import Any, Dict, Optional, Set, Type, Union

import pandas as pd

from vtlengine.DataTypes._time_checking import (
    check_date,
    check_time,
    check_time_period,
)
from vtlengine.DataTypes.TimeHandling import (
    PERIOD_IND_MAPPING,
    TimePeriodHandler,
    check_max_date,
    date_to_period_str,
    interval_to_period_str,
    str_period_to_date,
)
from vtlengine.Exceptions import InputValidationException, RunTimeError, SemanticError

DTYPE_MAPPING: Dict[str, str] = {
    "String": "string",
    "Number": "float64",
    "Integer": "int64",
    "TimeInterval": "string",
    "Date": "string",
    "TimePeriod": "string",
    "Duration": "string",
    "Boolean": "object",
}

CAST_MAPPING: Dict[str, type] = {
    "String": str,
    "Number": float,
    "Integer": int,
    "TimeInterval": str,
    "Date": str,
    "TimePeriod": str,
    "Duration": str,
    "Boolean": bool,
}


class DataTypeSimpleRepr(type):
    def __repr__(cls) -> Any:
        return SCALAR_TYPES_CLASS_REVERSE[cls]

    def __hash__(cls) -> int:
        return id(cls)


class ScalarType(metaclass=DataTypeSimpleRepr):
    """ """

    default: Optional[Union[str, "ScalarType"]] = None

    def strictly_same_class(self, obj: "ScalarType") -> bool:
        if not isinstance(obj, ScalarType):
            raise Exception("Not use strictly_same_class")
        return self.__class__ == obj.__class__

    def __eq__(self, other: Any) -> bool:
        return self.__class__.__name__ == other.__class__.__name__

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def instance_is_included(self, set_: Set[Any]) -> bool:
        return self.__class__ in set_

    @classmethod
    def is_included(cls, set_: Set[Any]) -> bool:
        return cls in set_

    @classmethod
    def promotion_changed_type(cls, promoted: Any) -> bool:
        return not issubclass(cls, promoted)

    @classmethod
    def implicit_cast(cls, value: Any, from_type: Any) -> Any:
        raise Exception("Method should be implemented by inheritors")

    @classmethod
    def explicit_cast(cls, value: Any, from_type: Any) -> Any:
        raise Exception("Method should be implemented by inheritors")

    @classmethod
    def is_subtype(cls, obj: Any) -> bool:
        return issubclass(cls, obj)

    @classmethod
    def is_null_type(cls) -> bool:
        return False

    @classmethod
    def check_type(cls, value: Any) -> bool:
        if isinstance(value, CAST_MAPPING[cls.__name__]):
            return True
        raise Exception(f"Value {value} is not a {cls.__name__}")

    @classmethod
    def cast(cls, value: Any) -> Any:
        if pd.isnull(value):
            return None
        class_name: str = cls.__name__.__str__()
        return CAST_MAPPING[class_name](value)

    @classmethod
    def dtype(cls) -> str:
        class_name: str = cls.__name__.__str__()
        return DTYPE_MAPPING[class_name]

    @classmethod
    def check(cls, value: Any) -> bool:
        try:
            cls.cast(value)
            return True
        except Exception:
            return False


class String(ScalarType):
    """ """

    default = ""

    @classmethod
    def implicit_cast(cls, value: Any, from_type: Any) -> str:
        if from_type in {Boolean, String}:
            return str(value)

        raise SemanticError(
            "2-1-5-1",
            value=value,
            type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
            type_2=SCALAR_TYPES_CLASS_REVERSE[cls],
        )

    @classmethod
    def explicit_cast(cls, value: Any, from_type: Any) -> str:
        if from_type in {
            Integer,
            Number,
            Boolean,
            String,
            Date,
            TimePeriod,
            TimeInterval,
        }:
            return str(value)
        if from_type == Duration:
            return _SHORTCODE_TO_ISO.get(str(value), str(value))

        raise RunTimeError(
            "2-1-5-1",
            value=value,
            type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
            type_2=SCALAR_TYPES_CLASS_REVERSE[cls],
        )

    @classmethod
    def check(cls, value: Any) -> bool:
        return True


class Number(ScalarType):
    """ """

    def __eq__(self, other: Any) -> bool:
        return (
            self.__class__.__name__ == other.__class__.__name__
            or other.__class__.__name__ == Integer.__name__
        )

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    @classmethod
    def implicit_cast(cls, value: Any, from_type: Any) -> float:
        # if pd.isna(value):
        #     return cls.default
        if from_type in {Integer, Number}:
            return float(value)

        raise RunTimeError(
            "2-1-5-1",
            value=value,
            type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
            type_2=SCALAR_TYPES_CLASS_REVERSE[cls],
        )

    @classmethod
    def explicit_cast(cls, value: Any, from_type: Any) -> float:
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

        raise RunTimeError(
            "2-1-5-1",
            value=value,
            type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
            type_2=SCALAR_TYPES_CLASS_REVERSE[cls],
        )

    @classmethod
    def cast(cls, value: Any) -> Optional[float]:
        if pd.isnull(value):
            return None
        if isinstance(value, str):
            if value.lower() == "true":
                return 1.0
            elif value.lower() == "false":
                return 0.0
        return float(value)

    @classmethod
    def check(cls, value: Any) -> bool:
        if pd.isnull(value):
            return True
        if isinstance(value, (int, float, bool)):
            return True
        if isinstance(value, str):
            v = value.strip()
            if v.lower() in {"true", "false"}:
                return True
            return bool(re.match(r"^\d+(\.\d*)?$|^\.\d+$", v))
        return False


class Integer(Number):
    """ """

    def __eq__(self, other: Any) -> bool:
        return (
            self.__class__.__name__ == other.__class__.__name__
            or other.__class__.__name__ == Number.__name__
        )

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    @classmethod
    def implicit_cast(cls, value: Any, from_type: Any) -> int:
        if from_type.__name__ == "Integer":
            return value

        if from_type.__name__ == "Number":
            # TODO: VTL 2.2 specifies truncation toward zero (return int(value)),
            #  pending discussion
            if value.is_integer():
                return int(value)
            # else:
            #     raise RunTimeError(
            #         "2-1-5-1",
            #         value=value,
            #         type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
            #         type_2=SCALAR_TYPES_CLASS_REVERSE[cls],
            #     )
            return int(value)

        raise RunTimeError(
            "2-1-5-1",
            value=value,
            type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
            type_2=SCALAR_TYPES_CLASS_REVERSE[cls],
        )

    @classmethod
    def explicit_cast(cls, value: Any, from_type: Any) -> int:
        if from_type in {Boolean}:
            if value:
                return 1
            else:
                return 0
        if from_type in {Number, String}:
            try:
                if float(value) - int(value) != 0:
                    raise RunTimeError(
                        "2-1-5-1",
                        value=value,
                        type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
                        type_2=SCALAR_TYPES_CLASS_REVERSE[cls],
                    )
            except ValueError:
                raise RunTimeError(
                    "2-1-5-1",
                    value=value,
                    type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
                    type_2=SCALAR_TYPES_CLASS_REVERSE[cls],
                )
            return int(value)

        raise RunTimeError(
            "2-1-5-1",
            value=value,
            type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
            type_2=SCALAR_TYPES_CLASS_REVERSE[cls],
        )

    @classmethod
    def cast(cls, value: Any) -> Optional[int]:
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

    @classmethod
    def check(cls, value: Any) -> bool:
        if pd.isnull(value):
            return True
        if isinstance(value, str):
            return value.isdigit() or value.lower() in {"true", "false"}
        if isinstance(value, float):
            return value.is_integer()
        return isinstance(value, (int, bool))


class TimeInterval(ScalarType):
    """ """

    default = None

    @classmethod
    def implicit_cast(cls, value: Any, from_type: Any) -> Any:
        if from_type in {TimeInterval}:
            return value
        if from_type in {Date}:
            value = check_max_date(value)

            return f"{value}/{value}"

        if from_type in {TimePeriod}:
            init_value = str_period_to_date(value, start=True)
            end_value = str_period_to_date(value, start=False)
            return f"{init_value}/{end_value}"

        raise RunTimeError(
            "2-1-5-1",
            value=value,
            type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
            type_2=SCALAR_TYPES_CLASS_REVERSE[cls],
        )

    @classmethod
    def explicit_cast(cls, value: Any, from_type: Any) -> Any:
        if from_type == String:
            return value
        raise RunTimeError(
            "2-1-5-1",
            value=value,
            type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
            type_2=SCALAR_TYPES_CLASS_REVERSE[cls],
        )

    @classmethod
    def check(cls, value: Any) -> bool:
        if pd.isnull(value):
            return True
        try:
            check_time(value)
        except Exception:
            return False
        return True


class Date(TimeInterval):
    """ """

    default = None

    @classmethod
    def implicit_cast(cls, value: Any, from_type: Any) -> Any:
        if from_type in {Date}:
            return check_max_date(value)

        raise RunTimeError(
            "2-1-5-1",
            value=value,
            type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
            type_2=SCALAR_TYPES_CLASS_REVERSE[cls],
        )

    @classmethod
    def explicit_cast(cls, value: Any, from_type: Any) -> Any:
        if from_type == String:
            return check_max_date(value)
        if from_type == TimePeriod:
            handler = TimePeriodHandler(str(value))
            if handler.period_indicator == "D":
                return handler.start_date(as_date=False)
            raise RunTimeError(
                "2-1-5-1",
                value=value,
                type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
                type_2=SCALAR_TYPES_CLASS_REVERSE[cls],
            )
        if from_type == TimeInterval:
            parts = str(value).split("/", maxsplit=1)
            if len(parts) == 2 and parts[0] == parts[1]:
                return check_max_date(parts[0])
            raise RunTimeError(
                "2-1-5-1",
                value=value,
                type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
                type_2=SCALAR_TYPES_CLASS_REVERSE[cls],
            )

        raise RunTimeError(
            "2-1-5-1",
            value=value,
            type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
            type_2=SCALAR_TYPES_CLASS_REVERSE[cls],
        )

    @classmethod
    def check(cls, value: Any) -> bool:
        if pd.isnull(value):
            return True
        try:
            check_date(value)
        except Exception:
            return False
        return True


class TimePeriod(TimeInterval):
    """ """

    default = None

    @classmethod
    def implicit_cast(cls, value: Any, from_type: Any) -> Any:
        if from_type in {TimePeriod}:
            return value

        raise RunTimeError(
            "2-1-5-1",
            value=value,
            type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
            type_2=SCALAR_TYPES_CLASS_REVERSE[cls],
        )

    @classmethod
    def explicit_cast(cls, value: Any, from_type: Any) -> Any:
        if from_type in {Date}:
            try:
                period_str = date_to_period_str(value, "D")
            except ValueError:
                raise RunTimeError(
                    "2-1-5-1",
                    value=value,
                    type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
                    type_2=SCALAR_TYPES_CLASS_REVERSE[cls],
                )
            return period_str
        if from_type == String:
            s = str(value)
            if "/" in s:
                result = interval_to_period_str(s)
                if result is not None:
                    return result
                raise RunTimeError(
                    "2-1-5-1",
                    value=value,
                    type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
                    type_2=SCALAR_TYPES_CLASS_REVERSE[cls],
                )
            return value
        if from_type == TimeInterval:
            result = interval_to_period_str(str(value))
            if result is not None:
                return result
            raise RunTimeError(
                "2-1-5-1",
                value=value,
                type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
                type_2=SCALAR_TYPES_CLASS_REVERSE[cls],
            )

        raise RunTimeError(
            "2-1-5-1",
            value=value,
            type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
            type_2=SCALAR_TYPES_CLASS_REVERSE[cls],
        )

    @classmethod
    def check(cls, value: Any) -> bool:
        if pd.isnull(value):
            return True
        try:
            check_time_period(value)
        except Exception:
            return False
        return True


_SHORTCODE_TO_ISO: dict[str, str] = {
    "A": "P1Y",
    "S": "P6M",
    "Q": "P3M",
    "M": "P1M",
    "W": "P1W",
    "D": "P1D",
}

_ISO_TO_SHORTCODE: dict[str, str] = {
    "P1Y": "A",
    "P6M": "S",
    "P3M": "Q",
    "P1M": "M",
    "P1W": "W",
    "P7D": "W",
    "P1D": "D",
}


class Duration(ScalarType):
    @classmethod
    def validate_duration(cls, value: Any) -> bool:
        if isinstance(value, str):
            if value in PERIOD_IND_MAPPING:
                return True
            else:
                raise InputValidationException(
                    code="2-1-5-1", value=value, type_1=type(value).__name__, type_2="Duration"
                )
        return False

    @classmethod
    def implicit_cast(cls, value: Any, from_type: Any) -> str:
        if from_type == Duration:
            return value

        raise RunTimeError(
            "2-1-5-1",
            value=value,
            type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
            type_2=SCALAR_TYPES_CLASS_REVERSE[cls],
        )

    @classmethod
    def explicit_cast(cls, value: Any, from_type: Any) -> Any:
        if from_type == String:
            s = str(value).strip().upper()
            # ISO-8601 duration format (P1Y, P3M, etc.)
            if s.startswith("P"):
                shortcode = _ISO_TO_SHORTCODE.get(s)
                if shortcode is not None:
                    return shortcode
                raise RunTimeError(
                    "2-1-5-1",
                    value=value,
                    type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
                    type_2=SCALAR_TYPES_CLASS_REVERSE[cls],
                )
            # VTL shortcode (A, S, Q, M, W, D)
            if cls.validate_duration(s):
                return s
        raise RunTimeError(
            "2-1-5-1",
            value=value,
            type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
            type_2=SCALAR_TYPES_CLASS_REVERSE[cls],
        )

    @classmethod
    def check(cls, value: Any) -> bool:
        if pd.isnull(value):
            return True

        if isinstance(value, str):
            match = cls.validate_duration(value)
            return bool(match)
        return False


class Boolean(ScalarType):
    """ """

    default = None

    @classmethod
    def cast(cls, value: Any) -> Optional[bool]:
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
            return value != 0
        if isinstance(value, float):
            return value != 0.0
        if isinstance(value, bool):
            return value
        return value

    @classmethod
    def implicit_cast(cls, value: Any, from_type: Any) -> bool:
        if from_type in {Boolean}:
            return value

        raise RunTimeError(
            "2-1-5-1",
            value=value,
            type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
            type_2=SCALAR_TYPES_CLASS_REVERSE[cls],
        )

    @classmethod
    def explicit_cast(cls, value: Any, from_type: Any) -> bool:
        if from_type in {Number, Integer}:
            return value not in {0}
        if from_type == String and isinstance(value, str):
            return value.strip().lower() == "true"

        raise RunTimeError(
            "2-1-5-1",
            value=value,
            type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
            type_2=SCALAR_TYPES_CLASS_REVERSE[cls],
        )

    @classmethod
    def check(cls, value: Any) -> bool:
        if pd.isnull(value):
            return True
        if isinstance(value, str):
            return value.lower() in {"true", "false", "1", "0"}
        return isinstance(value, (int, float, bool))


class Null(ScalarType):
    """ """

    @classmethod
    def is_null_type(cls) -> bool:
        return True

    @classmethod
    def check_type(cls, value: Any) -> bool:
        return True

    @classmethod
    def cast(cls, value: Any) -> None:
        return None

    @classmethod
    def dtype(cls) -> str:
        return "string"

    @classmethod
    def check(cls, value: Any) -> bool:
        return True


SCALAR_TYPES: Dict[str, Type[ScalarType]] = {
    "String": String,
    "Number": Number,
    "Integer": Integer,
    "Time": TimeInterval,
    "Date": Date,
    "Time_Period": TimePeriod,
    "Duration": Duration,
    "Boolean": Boolean,
    "Null": Null,
}

SCALAR_TYPES_CLASS_REVERSE: Dict[Any, str] = {
    String: "String",
    Number: "Number",
    Integer: "Integer",
    TimeInterval: "Time",
    Date: "Date",
    TimePeriod: "Time_Period",
    Duration: "Duration",
    Boolean: "Boolean",
    Null: "Null",
}

BASIC_TYPES: Dict[type, Type[ScalarType]] = {
    str: String,
    int: Integer,
    float: Number,
    bool: Boolean,
    type(None): Null,
}

COMP_NAME_MAPPING: Dict[Type[ScalarType], str] = {
    String: "str_var",
    Number: "num_var",
    Integer: "int_var",
    TimeInterval: "time_var",
    TimePeriod: "time_period_var",
    Date: "date_var",
    Duration: "duration_var",
    Boolean: "bool_var",
    Null: "null_var",
}

# Key is the data type, value is the set of types to which it can be implicitly promoted
IMPLICIT_TYPE_PROMOTION_MAPPING: Dict[Type[ScalarType], Any] = {
    String: {String},
    Number: {Number, Integer},
    Integer: {Number, Integer},
    TimeInterval: {TimeInterval},
    Date: {Date, TimeInterval},
    TimePeriod: {TimePeriod, TimeInterval},
    Duration: {Duration},
    Boolean: {Boolean, String},
    Null: {
        String,
        Number,
        Integer,
        TimeInterval,
        Date,
        TimePeriod,
        Duration,
        Boolean,
        Null,
    },
}

EXPLICIT_WITHOUT_MASK_TYPE_PROMOTION_MAPPING: Dict[Type[ScalarType], Any] = {
    String: {Integer, Number, Boolean, String, Date, TimePeriod, TimeInterval, Duration},
    Number: {Integer, Boolean, String, Number},
    Integer: {Number, Boolean, String, Integer},
    TimeInterval: {Date, TimePeriod, String, TimeInterval},
    Date: {TimePeriod, Date, String},
    TimePeriod: {Date, TimePeriod, String},
    Duration: {Duration, String},
    Boolean: {Integer, Number, String, Boolean},
    Null: {
        String,
        Number,
        Integer,
        TimeInterval,
        Date,
        TimePeriod,
        Duration,
        Boolean,
        Null,
    },
}

EXPLICIT_WITH_MASK_TYPE_PROMOTION_MAPPING: Dict[Type[ScalarType], Any] = {
    String: {Integer, Number, TimeInterval, Date, TimePeriod, Duration},
    Number: {String},
    Integer: {String},
    TimeInterval: {String},
    Date: {String},
    TimePeriod: {String},
    Duration: {String},
    Boolean: {},
    Null: {
        String,
        Number,
        Integer,
        TimeInterval,
        Date,
        TimePeriod,
        Duration,
        Boolean,
        Null,
    },
}


def binary_implicit_promotion(
    left_type: Type[ScalarType],
    right_type: Type[ScalarType],
    type_to_check: Optional[Type[ScalarType]] = None,
    return_type: Optional[Type[ScalarType]] = None,
) -> Type[ScalarType]:
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
        raise SemanticError(
            code="1-1-1-2",
            type_1=SCALAR_TYPES_CLASS_REVERSE[left_type],
            type_2=SCALAR_TYPES_CLASS_REVERSE[right_type],
            type_check=SCALAR_TYPES_CLASS_REVERSE[type_to_check],
        )
        # raise Exception(f"Implicit cast not allowed from
        # {left_type} and {right_type} to {type_to_check}")

    if return_type and (
        left_type.is_included(right_implicities) or right_type.is_included(left_implicities)
    ):
        return return_type
    if left_type.is_included(right_implicities):
        return left_type
    if right_type.is_included(left_implicities):
        return right_type

    # Fallback: check if both types can be promoted to a common type
    # e.g. Date → TimeInterval and TimePeriod → TimeInterval
    common = left_implicities.intersection(right_implicities)
    if common:
        if return_type:
            return return_type
        # Return the common promoted type (exclude Null if present)
        common.discard(Null)
        if len(common) == 1:
            return common.pop()

    raise SemanticError(
        code="1-1-1-1",
        type_1=SCALAR_TYPES_CLASS_REVERSE[left_type],
        type_2=SCALAR_TYPES_CLASS_REVERSE[right_type],
    )


def check_binary_implicit_promotion(
    left: Type[ScalarType],
    right: Any,
    type_to_check: Any = None,
    return_type: Any = None,
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

    return (
        left.is_included(right_implicities)
        or right.is_included(left_implicities)
        or bool(left_implicities.intersection(right_implicities))
    )


def unary_implicit_promotion(
    operand_type: Type[ScalarType],
    type_to_check: Optional[Type[ScalarType]] = None,
    return_type: Optional[Type[ScalarType]] = None,
) -> Type[ScalarType]:
    """
    Validates the compatibility between the type of the operand and the operator
    param operand: The operand
    param type_to_check: The type of the operator (from the operator if any)
    param return_type: The type of the result (from the operator if any)
    return: The resulting type of the operation, after the implicit type promotion
    """
    operand_implicities = IMPLICIT_TYPE_PROMOTION_MAPPING[operand_type]
    if type_to_check and not type_to_check.is_included(operand_implicities):
        raise SemanticError(
            code="1-1-1-1",
            type_1=SCALAR_TYPES_CLASS_REVERSE[operand_type],
            type_2=SCALAR_TYPES_CLASS_REVERSE[type_to_check],
        )
    if return_type:
        return return_type
    if (
        type_to_check
        and not issubclass(operand_type, type_to_check)
        and not issubclass(type_to_check, operand_type)
    ):
        return type_to_check
    return operand_type


def check_unary_implicit_promotion(
    operand_type: Type[ScalarType], type_to_check: Any = None, return_type: Any = None
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
