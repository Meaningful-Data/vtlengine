import re
from typing import Optional

from vtlengine.Exceptions import SemanticError


def instr_duck(
    str_value: Optional[str],
    str_to_find: Optional[str],
    start: Optional[int],
    occurrence: Optional[int] = 0,
) -> Optional[int]:
    """
    Custom instr function for DuckDB that handles None values and find a string in others.
    Applied on Instr operator in VTL.
    If str_value is None return None
    If str_to_find is None return 0
    If start or occurrence is None both are set to 1

    Args:
        str_value: Optional[str], value to find in
        str_to_find: Optional[str], value to find
        start: Optional[int], position of the string to start
        occurrence: Optional[int] = 0, the occurrence of the pattern to search

    Returns:
        Optional[int]: The position in the input string of a specified string.
    """
    if str_value is None:
        return None
    if str_to_find is None:
        return 0
    else:
        str_value = str(str_value)
        str_to_find = str(str_to_find)

    if start is None:
        start = 1

    if occurrence is None:
        occurrence = 1

    if start < 1:
        raise SemanticError(
            "1-1-18-4",
            op="instr",
            param_type="Start",
            correct_type=">=1",
        )

    if isinstance(start, int):
        start = int(start - 1)
    else:
        raise SemanticError(
            "1-1-18-4",
            op="instr",
            param_type="Occurrence",
            correct_type="Integer",
        )

    if occurrence < 1:
        raise SemanticError(
            "1-1-18-4",
            op="instr",
            param_type="Occurrence",
            correct_type=">= 1",
        )

    if isinstance(occurrence, int):
        occurrence = int(occurrence - 1)
    else:
        raise SemanticError(
            "1-1-18-4",
            op="instr",
            param_type="Occurrence",
            correct_type="Integer",
        )

    occurrences_list = [m.start() for m in re.finditer(str_to_find, str_value[start:])]

    length = len(occurrences_list)

    position = 0 if occurrence > length - 1 else int(start + occurrences_list[occurrence] + 1)

    return position


def replace_duck(value: str, pattern1: Optional[str], pattern2: Optional[str]) -> Optional[str]:
    """
    Custom replace function for DuckDB that handles None values and replace a string in others.
    Applied on Replace operator in VTL.
    If value is None return None
    If pattern1 is None return ""
    If pattern is None set as ""

    Args:
        value: str: the operand
        pattern1: Optional[str]: the pattern to be replaced
        pattern2: Optional[str]: the replacing pattern

    Returns:
        Optional[str]: The modified string
    """

    if value is None:
        return None
    if pattern1 is None:
        return ""
    if pattern2 is None:
        pattern2 = ""
    return value.replace(pattern1, pattern2)


def substr_duck(
    value: str,
    start: Optional[int] = None,
    length: Optional[int] = None,
) -> Optional[str]:
    """
    Custom substr function for DuckDB that handles None values and extracts a substring.
    Applied on Substr operator in VTL.
    If value is None return None
    If start is None return the same string
    If length is None return the same string

    Args:
        value: str: the operand
        start: Optional[int] = None: the starting digit (first character)
                of the string to be extracted
        length: Optional[int] = None: the length (number of characters)
                of the string to be extracted

    Returns:
        Optional[str]: The substring
    """

    if value is None:
        return None
    if start is None and length is None:
        return value

    if length is not None and length < 0:
        raise SemanticError("1-1-18-4", op="substr", param_type="Lengh", correct_type=">= 0")
    if start is not None and start < 1:
        raise SemanticError("1-1-18-4", op="substr", param_type="Start", correct_type=">= 1")

    if start is None:
        start = 0
    elif start != 0:
        start -= 1
    elif start > (len(value)):
        return ""
    param2 = len(value) if length is None or start + length > len(value) else start + length
    return value[start:param2]
