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

    def is_included(self, set_: set) -> bool:
        return self.__class__ in set_

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


class Number(ScalarType):
    """
    """

    def __eq__(self, other):
        return (self.__class__.__name__ == other.__class__.__name__ or
                other.__class__.__name__ == Integer.__name__)

    def __ne__(self, other):
        return not self.__eq__(other)


class Integer(Number):
    """
    """

    def __eq__(self, other):
        return (self.__class__.__name__ == other.__class__.__name__ or
                other.__class__.__name__ == Number.__name__)

    def __ne__(self, other):
        return not self.__eq__(other)


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


SCALAR_TYPES = {
    'String': String,
    'Number': Number,
    'Integer': Integer,
    'TimeInterval': TimeInterval,
    'Date': Date,
    'TimePeriod': TimePeriod,
    'Duration': Duration,
    'Boolean': Boolean,
}

BASIC_TYPES = {
    str: String,
    int: Integer,
    float: Number,
    bool: Boolean,
}

COMP_NAME_MAPPING = {
    String: 'string_var',
    Number: 'num_var',
    Integer: 'int_var',
    TimeInterval: 'time_var',
    TimePeriod: 'time_period_var',
    Date: 'date_var',
    Duration: 'duration_var',
    Boolean: 'bool_var'
}

CONVERSION_VALIDATOR = {
    '-': 'same',
    'I': 'implicit',
    'E': 'explicit',
    'N': 'no'
}

TYPE_MAPPING_POSITION = {
    Integer: 0,
    Number: 1,
    Boolean: 2,
    TimeInterval: 3,
    Date: 4,
    TimePeriod: 5,
    String: 6,
    Duration: 7
}

TYPE_PROMOTION_MATRIX = [['-', 'I', 'E', 'N', 'E', 'E', 'I', 'E'],
                         ['E', '-', 'E', 'N', 'N', 'N', 'I', 'N'],
                         ['E', 'E', '-', 'N', 'N', 'N', 'I', 'N'],
                         ['N', 'N', 'N', '-', 'N', 'N', 'E', 'N'],
                         ['N', 'N', 'N', 'I', '-', 'E', 'E', 'N'],
                         ['N', 'N', 'N', 'I', 'N', '-', 'E', 'N'],
                         ['E', 'E', 'N', 'E', 'E', 'E', '-', 'E'],
                         ['N', 'N', 'N', 'N', 'N', 'N', 'E', '-']]
