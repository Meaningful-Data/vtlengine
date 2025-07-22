import re
from typing import Optional

from vtlengine.Exceptions import SemanticError


def duck_instr(
    str_value: str, str_to_find: str, start: Optional[int], occurrence: Optional[int] = 0
) -> int:
    if str_value is None or str_to_find is None:
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


def duck_replace(x: str, param1: Optional[str], param2: Optional[str]) -> str:
    if param1 is None:
        return ""
    elif param2 is None:
        param2 = ""
    x = str(x)
    if param1 is not None and param2 is not None:
        return x.replace(param1, param2)
    return x
