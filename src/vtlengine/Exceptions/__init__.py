"""
Exceptions.exceptions.py
========================

Description
-----------
All exceptions exposed by the Vtl engine.
"""
import re
from typing import Any, List, Optional

import duckdb

from vtlengine.Exceptions.duckdb_mapping import DUCKDB_TO_VTL_TYPES
from vtlengine.Exceptions.messages import centralised_messages

dataset_output = None

def map_duckdb_type_to_vtl(duckdb_type: str) -> str:
    return DUCKDB_TO_VTL_TYPES.get(duckdb_type.upper(), duckdb_type)

class VTLEngineException(Exception):
    """Base class for exceptions in this module."""

    def __init__(
        self,
        message: str,
        lino: Optional[str] = None,
        colno: Optional[str] = None,
        code: Optional[str] = None,
    ) -> None:
        if code is not None:
            super().__init__(message, code)
        else:
            super().__init__(message)
        self.lino = lino
        self.colno = colno

    @property
    def pos(self) -> List[Optional[str]]:
        """ """

        return [self.lino, self.colno]


class DataTypeException(VTLEngineException):
    """
    Implement here the exception of DataTypeException.py:
        class DataTypeError(Exception):
            def __init__(self, value, dataType):
                super().__init__("Invalid Scalar value '{}' for data type {}.". format(
                    value, dataType
                    ))
    """

    def __init__(
        self,
        message: str = "default_value",
        lino: Optional[str] = None,
        colno: Optional[str] = None,
    ) -> None:
        super().__init__(message, lino, colno)


class SyntaxError(VTLEngineException):
    """ """

    def __init__(
        self,
        message: str = "default_value",
        lino: Optional[str] = None,
        colno: Optional[str] = None,
    ) -> None:
        super().__init__(message, lino, colno)


class SemanticError(VTLEngineException):
    """ """

    output_message = " Please check transformation with output dataset "
    comp_code = None

    def __init__(self, code: str, comp_code: Optional[str] = None, **kwargs: Any) -> None:
        if dataset_output:
            message = (
                centralised_messages[code].format(**kwargs)
                + self.output_message
                + str(dataset_output)
            )
        else:
            message = centralised_messages[code].format(**kwargs)

        super().__init__(message, None, None, code)

        if comp_code:
            self.comp_code = comp_code


class InterpreterError(VTLEngineException):
    output_message = " Please check transformation with output dataset "

    def __init__(self, code: str, **kwargs: Any) -> None:
        if dataset_output:
            message = (
                centralised_messages[code].format(**kwargs)
                + self.output_message
                + str(dataset_output)
            )
        else:
            message = centralised_messages[code].format(**kwargs)
        super().__init__(message, None, None, code)


class RunTimeError(VTLEngineException):
    output_message = " Please check transformation with output dataset "
    comp_code = None

    def __init__(
        self,
        code: str,
        comp_code: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        message = centralised_messages[code].format(**kwargs)
        if dataset_output:
            message += self.output_message + str(dataset_output)

        super().__init__(message, None, None, code)

        if comp_code:
            self.comp_code = comp_code

    @classmethod
    def map_duckdb_error(cls, e: "duckdb.Error", **kwargs) -> "RunTimeError":
        msg = str(e).lower()
        if isinstance(e, duckdb.ConversionException):
            print(e)
        if '"inf"' in msg:
            return cls("2-1-15-6", op="/", duckdb_msg=str(e), **kwargs)

        match = re.search(r"Could not convert(?: (\w+))? '(.+?)' to (\w+)", str(e), re.IGNORECASE)

        if match:
            from_type, value, target_type_raw = match.groups()
            source_type = map_duckdb_type_to_vtl(from_type).capitalize()
            vtl_type = map_duckdb_type_to_vtl(target_type_raw)

            return cls(
                "2-1-5-1",
                value=value,
                type_1=source_type,
                type_2=vtl_type,
                **kwargs
            )
        return cls("2-0-0-0", duckdb_msg=str(e), **kwargs)


class DataLoadError(VTLEngineException):
    output_message = " Please check loaded file"
    comp_code: Optional[str] = None

    def __init__(
        self,
        code: str,
        comp_code: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        message = centralised_messages[code].format(**kwargs)
        if dataset_output:
            message += self.output_message + " " + str(dataset_output)
        else:
            message += self.output_message

        super().__init__(message, None, None, code)

        if comp_code:
            self.comp_code = comp_code


class InputValidationException(VTLEngineException):
    """ """

    def __init__(
        self,
        message: str = "default_value",
        lino: Optional[str] = None,
        colno: Optional[str] = None,
        code: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if code is not None:
            message = centralised_messages[code].format(**kwargs)
            super().__init__(message, lino, colno, code)
        else:
            super().__init__(message, lino, colno)


def check_key(field: str, dict_keys: Any, key: str) -> None:
    if key not in dict_keys:
        closest_key = find_closest_key(dict_keys, key)
        message_append = f". Did you mean {closest_key}?" if closest_key else ""
        raise SemanticError("0-1-1-13", field=field, key=key, closest_key=message_append)


def find_closest_key(dict_keys: Any, key: str) -> Optional[str]:
    closest_key = None
    max_distance = 3
    min_distance = float("inf")

    for dict_key in dict_keys:
        distance = key_distance(key, dict_key)
        if distance < min_distance:
            min_distance = distance
            closest_key = dict_key

    if min_distance <= max_distance:
        return closest_key
    return None


def key_distance(key: str, objetive: str) -> int:
    dp = [[0] * (len(objetive) + 1) for _ in range(len(key) + 1)]

    for i in range(len(key) + 1):
        dp[i][0] = i
    for j in range(len(objetive) + 1):
        dp[0][j] = j

    for i in range(1, len(key) + 1):
        for j in range(1, len(objetive) + 1):
            cost = 0 if key[i - 1] == objetive[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    return dp[-1][-1]
