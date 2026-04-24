from __future__ import annotations

from typing import Any, Dict, Iterator, Optional, Tuple, TypeVar

V = TypeVar("V")


class CaseInsensitiveDict(Dict[str, V]):
    """A dict subclass that treats string keys as case-insensitive.

    Keys are normalized to lowercase for lookup, but the original key
    (from the first insertion) is preserved for iteration and output.

    This is used for VTL regular names which are case-insensitive per the spec.
    """

    def __init__(self, *args: Any, **kwargs: V) -> None:
        self._key_map: Dict[str, str] = {}  # lowercase -> original key
        super().__init__()
        if args:
            arg = args[0]
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v
            elif hasattr(arg, "__iter__"):
                for k, v in arg:
                    self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    def _normalize(self, key: str) -> str:
        return key.casefold()

    def __setitem__(self, key: str, value: V) -> None:
        norm = self._normalize(key)
        if norm not in self._key_map:
            self._key_map[norm] = key
        original = self._key_map[norm]
        super().__setitem__(original, value)

    def __getitem__(self, key: str) -> V:
        norm = self._normalize(key)
        if norm not in self._key_map:
            raise KeyError(key)
        return super().__getitem__(self._key_map[norm])

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return self._normalize(key) in self._key_map

    def __delitem__(self, key: str) -> None:
        norm = self._normalize(key)
        if norm not in self._key_map:
            raise KeyError(key)
        original = self._key_map.pop(norm)
        super().__delitem__(original)

    def get(self, key: str, default: Optional[V] = None) -> Optional[V]:  # type: ignore[override]
        norm = self._normalize(key)
        if norm not in self._key_map:
            return default
        return super().__getitem__(self._key_map[norm])

    def pop(self, key: str, *args: V) -> V:  # type: ignore[override]
        norm = self._normalize(key)
        if norm not in self._key_map:
            if args:
                return args[0]
            raise KeyError(key)
        original = self._key_map.pop(norm)
        return super().pop(original)

    def setdefault(self, key: str, default: Optional[V] = None) -> V:
        norm = self._normalize(key)
        if norm not in self._key_map:
            self[key] = default  # type: ignore[assignment]
        return self[key]

    def update(self, *args: Any, **kwargs: V) -> None:
        if args:
            other = args[0]
            if isinstance(other, dict):
                for k, v in other.items():
                    self[k] = v
            elif hasattr(other, "__iter__"):
                for k, v in other:
                    self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    def canonical_key(self, key: str) -> str:
        """Return the original-case key for a given (possibly different-case) key.

        Raises KeyError if the key doesn't exist.
        """
        norm = self._normalize(key)
        if norm not in self._key_map:
            raise KeyError(key)
        return self._key_map[norm]

    def __iter__(self) -> Iterator[str]:
        return super().__iter__()

    def copy(self) -> CaseInsensitiveDict[V]:
        result: CaseInsensitiveDict[V] = CaseInsensitiveDict()
        result._key_map = self._key_map.copy()
        for key in dict.keys(self):
            dict.__setitem__(result, key, dict.__getitem__(self, key))
        return result

    def __repr__(self) -> str:
        return f"CaseInsensitiveDict({dict(self.items())})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CaseInsensitiveDict):
            return dict.__eq__(self, other)
        if isinstance(other, dict):
            if len(self) != len(other):
                return False
            return all(k in self and self[k] == v for k, v in other.items())
        return NotImplemented

    def __deepcopy__(self, memo: Dict[int, Any]) -> CaseInsensitiveDict[V]:
        from copy import deepcopy

        new: CaseInsensitiveDict[V] = CaseInsensitiveDict.__new__(CaseInsensitiveDict)
        memo[id(self)] = new
        dict.__init__(new)
        new._key_map = deepcopy(self._key_map, memo)
        for key in dict.keys(self):
            dict.__setitem__(new, key, deepcopy(dict.__getitem__(self, key), memo))
        return new

    def __copy__(self) -> CaseInsensitiveDict[V]:
        new: CaseInsensitiveDict[V] = CaseInsensitiveDict.__new__(CaseInsensitiveDict)
        dict.__init__(new)
        new._key_map = self._key_map.copy()
        for key in dict.keys(self):
            dict.__setitem__(new, key, dict.__getitem__(self, key))
        return new

    @classmethod
    def from_dict(cls, d: Dict[str, V]) -> CaseInsensitiveDict[V]:
        """Create a CaseInsensitiveDict from a regular dict."""
        return cls(d)

    def to_dict(self) -> Dict[str, V]:
        """Convert back to a regular dict with original-cased keys."""
        return dict(self.items())

    def __reduce__(self) -> Tuple[type, Tuple[Dict[str, V]]]:
        return (CaseInsensitiveDict, (dict(self.items()),))
