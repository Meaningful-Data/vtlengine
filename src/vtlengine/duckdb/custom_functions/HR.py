from typing import Any, Optional

INF = float(1e100)
NINF = -INF


def imbalance_func(x: Any, y: Any) -> Optional[float]:
    return None if x is None or y is None else x - y


def handle_mode(x: Any, hr_mode: str) -> float:
    if hr_mode == "non_null" and x is None or hr_mode == "non_zero" and x == 0:
        return NINF
    return x
