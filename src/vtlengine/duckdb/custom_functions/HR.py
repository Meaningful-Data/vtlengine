from typing import Any, Optional, Union


def imbalance_func(x: Optional[Union[int, float]], y: Optional[Union[int, float]]) -> Optional[Union[int, float]]:
    return None if x is None or y is None else x - y