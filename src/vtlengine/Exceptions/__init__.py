"""
Exceptions.exceptions.py
========================

Description
-----------
All exceptions exposed by the Vtl engine.
"""
from typing import Optional, Any, List

from vtlengine.Exceptions.messages import centralised_messages

dataset_output = None


class VTLEngineException(Exception):
    """Base class for exceptions in this module."""

    def __init__(self, message: str, lino: Optional[str] = None,
                 colno: Optional[str] = None, code: Optional[str] = None) -> None:
        if code is not None:
            super().__init__(message, code)
        else:
            super().__init__(message)
        self.lino = lino
        self.colno = colno

    @property
    def pos(self) -> List[Optional[str]]:
        """

        """
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

    def __init__(self, message: str = 'default_value', lino: Optional[str] = None,
                 colno: Optional[str] = None) -> None:
        super().__init__(message, lino, colno)


class SyntaxError(VTLEngineException):
    """

    """

    def __init__(self, message: str = 'default_value', lino: Optional[str] = None,
                 colno: Optional[str] = None) -> None:
        super().__init__(message, lino, colno)


class SemanticError(VTLEngineException):
    """

    """
    output_message = " Please check transformation with output dataset "
    comp_code = None

    def __init__(self, code: str, comp_code: Optional[str] = None, **kwargs: Any) -> None:
        if dataset_output:
            message = centralised_messages[code].format(**kwargs) + self.output_message + str(
                dataset_output)
        else:
            message = centralised_messages[code].format(**kwargs)

        super().__init__(message, None, None, code)

        if comp_code:
            self.comp_code = comp_code


class InterpreterError(VTLEngineException):
    output_message = " Please check transformation with output dataset "

    def __init__(self, code: str, **kwargs: Any) -> None:
        if dataset_output:
            message = centralised_messages[code].format(**kwargs) + self.output_message + str(
                dataset_output)
        else:
            message = centralised_messages[code].format(**kwargs)
        super().__init__(message, None, None, code)


class RuntimeError(VTLEngineException):
    """

    """

    def __init__(self, message: str, lino: Optional[str] = None, colno: Optional[str] = None) -> None:
        super().__init__(message, lino, colno)


class InputValidationException(VTLEngineException):
    """

    """

    def __init__(self, message: str = 'default_value', lino: Optional[str] = None,
                 colno: Optional[str] = None, code: Optional[str] = None, **kwargs: Any) -> None:
        if code is not None:
            message = centralised_messages[code].format(**kwargs)
            super().__init__(message, lino, colno, code)
        else:
            super().__init__(message, lino, colno)
