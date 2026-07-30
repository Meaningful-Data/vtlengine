"""Stack headroom for the recursive passes over a VTL AST."""

import sys
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Generator, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])

# An expression chaining N datasets or components (``a + b + c + ...``) parses into
# a left-nested BinOp. Both building that AST and walking it cost several stack
# frames per operand, so a script that combines a couple of hundred operands in one
# expression exhausts CPython's default limit of 1000 (issue #924). This ceiling
# leaves room for chains several times longer while staying far below the point
# where the C stack is at risk.
MIN_RECURSION_LIMIT = 5000


@contextmanager
def recursion_headroom() -> Generator[None, None, None]:
    """Raise the recursion limit for the enclosed pass, then restore it."""
    previous = sys.getrecursionlimit()
    if previous >= MIN_RECURSION_LIMIT:
        yield
        return
    sys.setrecursionlimit(MIN_RECURSION_LIMIT)
    try:
        yield
    finally:
        # Leave it alone if something else raised it further in the meantime.
        if sys.getrecursionlimit() == MIN_RECURSION_LIMIT:
            sys.setrecursionlimit(previous)


def with_recursion_headroom(fn: F) -> F:
    """Wrap an entry point so every recursive pass inside it has the headroom."""

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with recursion_headroom():
            return fn(*args, **kwargs)

    return cast(F, wrapper)
