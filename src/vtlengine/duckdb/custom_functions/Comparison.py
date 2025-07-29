from typing import Any, Optional


def isnull_duck(x: Any) -> bool:
    return x is None


def between_duck(x: Any, y: Any, z: Any) -> Optional[bool]:
    return None if (x is None or y is None or z is None) else y <= x <= z
