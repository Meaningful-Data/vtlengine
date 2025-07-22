import re
from typing import Optional

from vtlengine.Exceptions import SemanticError


def instr_duck(
    str_value: Optional[str],
    str_to_find: Optional[str],
    start: Optional[int],
    occurrence: Optional[int] = 0,
) -> Optional[int]:
    if str_value is None:
        return None
    if str_to_find is None:
        return 0
    else:
        str_value = str(str_value)
        str_to_find = str(str_to_find)

    if start is None:
        start = 1

    if isinstance(start, (int, float)):
        start = int(start - 1)
    else:
        raise SemanticError(
            "1-1-18-4",
            # op=cls.op,
            param_type="Start",
            correct_type="Integer",
        )

    if occurrence is None:
        occurrence = 1

    if isinstance(occurrence, (int, float)):
        occurrence = int(occurrence - 1)
    else:
        raise SemanticError(
            "1-1-18-4",
            # op=cls.op,
            param_type="Occurrence",
            correct_type="Integer",
        )

    occurrences_list = [m.start() for m in re.finditer(str_to_find, str_value[start:])]

    length = len(occurrences_list)

    position = 0 if occurrence > length - 1 else int(start + occurrences_list[occurrence] + 1)

    return position

def instr_check_param_value(value: Optional[int], position: int) -> Optional[int]:
    if position == 2 and value is not None and value < 1:
        raise SemanticError("1-1-18-4", op="instr", param_type="Start", correct_type=">= 1")
    elif position == 3 and value is not None and value < 1:
        raise SemanticError("1-1-18-4", op="instr", param_type="Occurrence", correct_type=">= 1")
    return value


def replace_duck(x: str, param1: Optional[str], param2: Optional[str]) -> str:
    if param1 is None:
        return ""
    elif param2 is None:
        param2 = ""
    x = str(x)
    if param2 is not None:
        return x.replace(param1, param2)
    return x


def substr_duck(
    x: str,
    start: Optional[int] = None,
    length: Optional[int] = None,
) -> Optional[str]:
    if x is None:
        return None
    if start is None and length is None:
        return x
    if start is None:
        start = 0
    elif start != 0:
        start -= 1
    elif start > (len(x)):
        return ""
    param2 = len(x) if length is None or start + length > len(x) else start + length
    return x[start:param2]
