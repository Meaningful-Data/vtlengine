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


RUNTIME_ERROR_CODES = {
    "2-1-15-6": {"op": "/"},
}


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
    def map_duckdb_error(cls, e: "duckdb.Error", **kwargs) -> "RunTimeError":  # type: ignore[no-untyped-def]
        msg_str = str(e)
        if (
            isinstance(e, duckdb.InvalidInputException)
            and "Python exception occurred while executing the UDF" in msg_str
        ):
            match = re.search(r"RunTimeError: \('(.+?)', '(\d+-\d+-\d+-\d+)'\)", msg_str)
            if match:
                message, code = match.groups()
                return cls(code, **RUNTIME_ERROR_CODES[code])
        msg = str(e).lower()
        if isinstance(e, duckdb.ConversionException):
            print(e)
        for error_code in RUNTIME_ERROR_CODES:
            if error_code in msg:
                return cls(error_code, **RUNTIME_ERROR_CODES[error_code])

        match = re.search(r"Could not convert(?: (\w+))? '(.+?)' to (\w+)", str(e), re.IGNORECASE)

        if match:
            from_type, value, target_type_raw = match.groups()
            source_type = map_duckdb_type_to_vtl(from_type).capitalize()
            vtl_type = map_duckdb_type_to_vtl(target_type_raw)

            return cls("2-1-5-1", value=value, type_1=source_type, type_2=vtl_type, **kwargs)
        return cls("2-0-0-0", duckdb_msg=str(e), **kwargs)


class DataLoadError(VTLEngineException):
    """
    Errors related to loading data into the engine
    (e.g., reading CSV files, data type mismatches during loading).
    """

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

    @classmethod
    def map_duckdb_error(  # type: ignore[no-untyped-def]
        cls, e: "duckdb.Error", comp_code: Optional[str] = None, **kwargs
    ) -> "DataLoadError":
        msg = str(e)
        dataset_name = kwargs.get("name") or kwargs.get("comp_code") or "unknown"
        if isinstance(e, duckdb.ConversionException):
            match = re.search(
                r'Error when converting column\s+"(\w+)".*?'
                r'Could not convert string\s+"(.+?)"\s+to\s+\'?(\w+)\'?',
                msg,
                re.IGNORECASE | re.DOTALL,
            )
            if match:
                column, value, target_type = match.groups()
                vtl_type = map_duckdb_type_to_vtl(target_type)
                return cls(
                    "0-1-1-12",
                    comp_code=comp_code,
                    column=column,
                    value=value,
                    type=vtl_type,
                    duckdb_msg=msg,
                    name=dataset_name,
                    **kwargs,
                )
        elif isinstance(e, duckdb.BinderException):
            null_identifier = kwargs.get("null_identifier") or "unknown_column"
            if re.search(r"read_csv requires at least a single column", str(e)):
                return cls(
                    "0-1-1-4",
                    comp_code=comp_code,
                    duckdb_msg=msg,
                    name=dataset_name,
                    null_identifier=null_identifier,
                    **kwargs,
                )
        elif isinstance(e, duckdb.InvalidInputException):
            print(e)

        return cls(
            "2-0-0-0",
            comp_code=comp_code,
            duckdb_msg=msg,
            **kwargs,
        )


class InputValidationException(VTLEngineException):
    """
    Errors related to DAG, overwriting datasets, invalid inputs, etc.
    Unknown files, invalid paths, etc.
    """

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


class VtlEngineRemoteExtensionException(VTLEngineException):
    """Exception for errors related to DuckDB extensions like httpfs."""

    @classmethod
    def local_access_disabled(cls) -> "VtlEngineRemoteExtensionException":
        message = "Local access to files and extensions is currently disabled. "
        return cls(message)

    @classmethod
    def remote_access_disabled(cls) -> "VtlEngineRemoteExtensionException":
        message = "Remote access to files and extensions is currently disabled. "
        return cls(message)


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
