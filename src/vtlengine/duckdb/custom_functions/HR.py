from typing import Any, Optional, Union


def imbalance_func(x: Any, y: Any) -> Optional[float]:
    return None if x is None or y is None else x - y