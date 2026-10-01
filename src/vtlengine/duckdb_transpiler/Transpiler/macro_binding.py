"""Bind the repeated arguments of VTL macros once (issue #1134).

DuckDB expands a macro by copying each argument into every use of its parameter, and
most VTL macros use theirs several times (``vtl_round_sig`` three, ``vtl_sqrt`` five).
A macro call whose argument holds another one multiplies the SQL DuckDB binds at every
level, so ``sqrt(sqrt(...))`` or a right-nested Number sum take seconds to plan with no
data involved. A call whose repeated argument would expand past a size is rewritten to
compute its arguments once, as the one element of a list the macro is applied to:

    vtl_sqrt(<arg>)  ->  list_transform([<arg>], lambda __vtl_arg: vtl_sqrt(__vtl_arg))[1]

The lambda body only reads the lambda parameter and literals: columns, aggregates and
window functions all stay in the list, outside the lambda. Smaller calls are left as
they are, so most statements keep their SQL.
"""

import re
from typing import List, Optional, Tuple

from vtlengine.duckdb_transpiler.sql import macro_parameter_uses

# A repeated argument whose SQL expands to at least this many characters is bound once.
# DuckDB plans about 1000 of them per millisecond, so copying a smaller argument costs
# the binder a few milliseconds, less than the lambda costs to run over many rows
MIN_BOUND_ARGUMENT_SIZE = 10_000

_BOUND = "__vtl_arg"

_TOKEN = re.compile(
    r"'(?:[^']|'')*'"  # string literal
    r'|"(?:[^"]|"")*"'  # quoted identifier
    r"|([A-Za-z_]\w*)(\s*\()?"  # name, maybe of a call
    r"|[(),]"
    r"|[^'\"(),A-Za-z_]+"
)
_LITERAL = re.compile(r"\s*(?:-?\d+(?:\.\d+)?|'(?:[^']|'')*'|NULL|TRUE|FALSE)\s*", re.IGNORECASE)


class _Frame:
    """The text read so far inside one pair of parentheses, and its expanded size."""

    __slots__ = ("opening", "macro", "args", "parts", "size")

    def __init__(self, opening: str, macro: Optional[str]) -> None:
        self.opening = opening
        self.macro = macro
        self.args: List[Tuple[str, int]] = []
        self.parts: List[str] = []
        self.size = 0

    def add(self, text: str, size: int) -> None:
        self.parts.append(text)
        self.size += size

    def end_argument(self) -> None:
        self.args.append(("".join(self.parts), self.size))
        self.parts = []
        self.size = 0


def bind_repeated_macro_arguments(sql: str) -> str:
    """Return ``sql`` with the macro calls whose repeated arguments are too large bound."""
    uses = macro_parameter_uses()
    stack = [_Frame("", None)]
    changed = False
    for match in _TOKEN.finditer(sql):
        piece = match.group(0)
        name, call = match.group(1), match.group(2)
        top = stack[-1]
        if call is not None:
            macro = name.lower()
            stack.append(_Frame(piece, macro if macro in uses else None))
        elif name is not None or piece not in "(),":
            top.add(piece, len(piece))
        elif piece == "(":
            stack.append(_Frame(piece, None))
        elif piece == "," and top.macro is not None:
            top.end_argument()
        elif piece == ",":
            top.add(piece, 1)
        elif len(stack) == 1:
            return sql  # unbalanced: leave the query as it is
        else:
            stack.pop()
            if top.macro is None:
                stack[-1].add(
                    f"{top.opening}{''.join(top.parts)})", len(top.opening) + top.size + 1
                )
            else:
                top.end_argument()
                text, size, bound = _macro_call(top, *uses[top.macro])
                changed = changed or bound
                stack[-1].add(text, size)
    if len(stack) != 1 or not changed:
        return sql
    return "".join(stack[0].parts)


def _macro_call(
    frame: _Frame, body_size: int, param_uses: Tuple[int, ...]
) -> Tuple[str, int, bool]:
    """Write one macro call, bound when a repeated argument is too large.

    Returns its SQL, the size it expands to and whether it was bound.
    """
    args = frame.args
    if len(args) == 1 and not args[0][0].strip():
        args = []  # a call without arguments

    def times_used(k: int) -> int:
        return param_uses[k] if k < len(param_uses) else 1

    expanded = body_size + sum(times_used(k) * size for k, (_, size) in enumerate(args))
    if not any(
        times_used(k) > 1 and size >= MIN_BOUND_ARGUMENT_SIZE for k, (_, size) in enumerate(args)
    ):
        return f"{frame.opening}{','.join(text for text, _ in args)})", expanded, False

    # Literals stay in the body, where DuckDB folds them; every other argument goes
    # into the list, so that the body only reads the lambda parameter
    moved = [k for k, (text, _) in enumerate(args) if not _LITERAL.fullmatch(text)]
    if len(moved) == 1:
        value = f"[{args[moved[0]][0].strip()}]"
        refs = {moved[0]: _BOUND}
    else:
        value = "[{" + ", ".join(f"'a{k}': {args[k][0].strip()}" for k in moved) + "}]"
        refs = {k: f"{_BOUND}.a{k}" for k in moved}
    call_args = ", ".join(refs.get(k, text.strip()) for k, (text, _) in enumerate(args))
    text = f"list_transform({value}, lambda {_BOUND}: {frame.opening}{call_args}))[1]"
    # The moved arguments expand once, in the list; the body copies the references
    size = len(text) + body_size
    size += sum(args[k][1] - len(args[k][0]) + times_used(k) * len(ref) for k, ref in refs.items())
    return text, size, True
