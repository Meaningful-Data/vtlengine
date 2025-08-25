from typing import Any, Optional


def imbalance_func(x: Any, y: Any) -> Optional[float]:
    return None if x is None or y is None else x - y


def handle_mode(x: Any, hr_mode: str) -> str:
    if x is not None and x == "REMOVE_VALUE":
        return "REMOVE_VALUE"
    if hr_mode == "non_null" and x is None or hr_mode == "non_zero" and x == 0:
        return "REMOVE_VALUE"
    return x
