"""The type links of the operators: what feeds an operator, what it feeds, and the types of a block.

An operator takes and gives the types of all its forms (``tool.model.form_types``). ``type_links``
matches those of one operator against every other operator of the catalogue:

* ``comes_from[T]``, for each type ``T`` that the operator takes: the operators that give ``T``
  or a subtype of ``T``. A producer of an ``integer`` feeds an operator that takes a ``number``;
  a producer of a ``number`` does not feed one that takes an ``integer``.
* ``goes_to[T]``, for each type ``T`` that the operator gives: the operators that take ``T`` or a
  supertype of ``T`` other than ``scalar``. A consumer of a ``number`` takes an ``integer``; a
  consumer of an ``integer`` does not take a ``number``.

``comes_from`` has a list for each type of ``takes`` and ``goes_to`` one for each type of
``gives``, empty when nothing feeds or takes that type. The structural types
(``STRUCTURAL_TYPES``) are outside the hierarchy: each one matches only itself, so
``value_domain`` is not matched by ``set``.

The operators that would be in the list of every type are folded into a list of their own:

* ``any_type``: the operators that give ``any``, whose result is the type of their input. They
  can feed whatever is taken, so no list by type has them.
* ``any_scalar``: the operators that take ``scalar``, which is any value. No list by type has
  them either. A form that takes ``any`` takes any value too, and is the same as one that takes
  ``scalar``: it is in ``any_scalar``, and an operator that takes ``any`` is fed by what feeds
  one that takes ``scalar``.

Only ``any`` follows its input. A form that gives ``scalar`` is matched as it is, like a form
that gives any other type: ``scalar`` is above every value type and below none, so the operator
feeds the operators that take ``scalar`` (and ``any``, which is the same) and none that takes a
type under it, such as ``number``. It is not in ``any_type``, and the operators that take
``scalar`` are in its ``any_scalar``, not in a list by type.

On each side, a folded list leaves out the operators that the lists by type of that side have
(``any_type`` those of ``comes_from``, ``any_scalar`` those of ``goes_to``), so that nothing is
listed twice: an operator with a form that gives ``integer`` and one that gives ``any`` is a
producer of ``integer``, and not in ``any_type`` as well. The folded lists are about values:
``any_type`` is for an operator that takes a value type, ``scalar`` or ``any``, and
``any_scalar`` for one that gives one of them. An operator that takes or gives only structural
types has neither, as a ruleset or a value domain is not a value.

An operator is never in its own lists. The refs of every list are in the order of the catalogue
(the order of its chapters, then of the operators of each), and their status is the one that
``StatusIndex.operators`` has for the operator. An operator that is in a list and has no status
there raises ``ValueError``: the index of a run has one for every operator of the catalogue.

``block_types_table`` is the matrix of a block: its operators against the types that they use.
``by_block`` groups a list of refs by block.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Final

from tool.model import (
    ANY_SCALAR,
    ANY_TYPE,
    STRUCTURAL_TYPES,
    TYPE_PARENTS,
    VALUE_TYPES,
    Catalogue,
    Chapter,
    Operator,
    Status,
    form_types,
)
from tool.status import StatusIndex

# The order of the types in a list and in a table: the value types, the structural ones, then the
# two that stand for many of them, ``scalar`` and ``any``. A type that is not here comes last.
_ORDER: Final = (*VALUE_TYPES, *STRUCTURAL_TYPES, ANY_SCALAR, ANY_TYPE)
_RANK: Final = {type_id: rank for rank, type_id in enumerate(_ORDER)}
# The types that a result that follows its input, or an operand that is any scalar, is about:
# values, never structures.
_VALUES: Final = frozenset((*VALUE_TYPES, ANY_SCALAR, ANY_TYPE))
# What an operator takes to be in ``any_scalar``, and not in the list of a type.
_FOLDED_TAKES: Final = frozenset((ANY_SCALAR, ANY_TYPE))
_IN: Final = "in"
_OUT: Final = "out"
_IN_AND_OUT: Final = f"{_IN} · {_OUT}"


@dataclass(frozen=True)
class Ref:
    """An operator that a list of the type links names, with what a page shows of it."""

    chapter_id: str
    chapter_title: str
    operator_id: str
    name: str
    symbol: str
    status: Status


@dataclass(frozen=True)
class TypeLinks:
    """The type links of one operator.

    ``takes`` and ``gives`` are the types of its forms, in the order of the types: the value types
    of ``VALUE_TYPES``, the structural types of ``STRUCTURAL_TYPES``, then ``scalar`` and ``any``.
    ``comes_from`` has a list for each type of ``takes`` and ``goes_to`` one for each type of
    ``gives``, in that order. ``any_type`` and ``any_scalar`` are the folded lists. The module
    docstring says what is in each.
    """

    takes: tuple[str, ...]
    gives: tuple[str, ...]
    comes_from: dict[str, tuple[Ref, ...]]
    any_type: tuple[Ref, ...]
    goes_to: dict[str, tuple[Ref, ...]]
    any_scalar: tuple[Ref, ...]


@dataclass(frozen=True)
class _Candidate:
    """An operator that a list is made of, with the types that it takes and gives."""

    chapter: Chapter
    operator: Operator
    takes: frozenset[str]  # what it takes by type: all but scalar and any
    gives: frozenset[str]  # what it gives by type: all but any
    takes_any_scalar: bool  # it takes scalar, or any
    gives_any_type: bool  # it gives any


# --- The hierarchy of the types ---


def _supertypes(type_id: str) -> frozenset[str]:
    """Every type above ``type_id``: ``integer`` has ``number`` and ``scalar``."""
    found: set[str] = set()
    pending = list(TYPE_PARENTS.get(type_id, ()))
    while pending:
        parent = pending.pop()
        if parent not in found:
            found.add(parent)
            pending.extend(TYPE_PARENTS.get(parent, ()))
    return frozenset(found)


_SUPERTYPES: Final = {type_id: _supertypes(type_id) for type_id in TYPE_PARENTS}
_SUBTYPES: Final = {
    type_id: frozenset(sub for sub in TYPE_PARENTS if type_id in _SUPERTYPES[sub])
    for type_id in TYPE_PARENTS
}


def _fed_by(type_id: str) -> frozenset[str]:
    """The types that an operator may give to feed an operand of ``type_id``: it and its subtypes.

    An operand of ``any`` is one of ``scalar``: any value. A type that is not in the hierarchy,
    such as a structural type, has no subtypes.
    """
    wanted = ANY_SCALAR if type_id == ANY_TYPE else type_id
    return frozenset((wanted,)) | _SUBTYPES.get(wanted, frozenset())


def _taken_as(type_id: str) -> frozenset[str]:
    """The types that an operator may take to consume a ``type_id``: it and its supertypes.

    ``scalar`` is left out of the supertypes: an operator that takes it takes any value, and is
    folded into ``any_scalar``. This function does not keep those operators out of the lists by
    type, as it answers ``{"scalar"}`` for a ``scalar`` and ``{"any"}`` for an ``any``:
    ``_candidate`` does, by leaving ``scalar`` and ``any`` out of the takes of a candidate
    (``takes - _FOLDED_TAKES``), so that no candidate is ever matched on them. A type that is not
    in the hierarchy has no supertypes.
    """
    return frozenset((type_id,)) | (_SUPERTYPES.get(type_id, frozenset()) - {ANY_SCALAR})


# --- Types of an operator ---


def _types(op: Operator) -> tuple[frozenset[str], frozenset[str]]:
    """The types that the forms of ``op`` take, and those that they give."""
    takes: set[str] = set()
    gives: set[str] = set()
    for variant in op.variants:
        form_takes, form_gives = form_types(op, variant)
        takes.update(form_takes)
        gives.update(form_gives)
    return frozenset(takes), frozenset(gives)


def _ordered(types: Iterable[str]) -> tuple[str, ...]:
    """The types, each once, in the order of ``_ORDER``."""
    return tuple(sorted(set(types), key=lambda type_id: (_RANK.get(type_id, len(_RANK)), type_id)))


def _candidate(chapter: Chapter, op: Operator) -> _Candidate:
    takes, gives = _types(op)
    return _Candidate(
        chapter,
        op,
        takes=takes - _FOLDED_TAKES,
        gives=gives - {ANY_TYPE},
        takes_any_scalar=not takes.isdisjoint(_FOLDED_TAKES),
        gives_any_type=ANY_TYPE in gives,
    )


# --- Type links ---


def _key(candidate: _Candidate) -> tuple[str, str]:
    return candidate.chapter.id, candidate.operator.id


def _ref(index: StatusIndex, candidate: _Candidate) -> Ref:
    chapter, op = candidate.chapter, candidate.operator
    key = f"{chapter.id}/{op.id}"
    if key not in index.operators:
        raise ValueError(f"no status for operator {key}")
    return Ref(chapter.id, chapter.title, op.id, op.name, op.symbol, index.operators[key])


def _refs(index: StatusIndex, candidates: Iterable[_Candidate]) -> tuple[Ref, ...]:
    return tuple(_ref(index, candidate) for candidate in candidates)


def _giving(others: Sequence[_Candidate], types: frozenset[str]) -> list[_Candidate]:
    """The candidates that give one of ``types``."""
    return [c for c in others if not c.gives.isdisjoint(types)]


def _taking(others: Sequence[_Candidate], types: frozenset[str]) -> list[_Candidate]:
    """The candidates that take one of ``types``."""
    return [c for c in others if not c.takes.isdisjoint(types)]


def _folded(
    others: Sequence[_Candidate],
    by_type: Iterable[Sequence[_Candidate]],
    wanted: Callable[[_Candidate], bool],
) -> list[_Candidate]:
    """The candidates that ``wanted`` picks and that are in none of the lists ``by_type``."""
    listed = {_key(candidate) for found in by_type for candidate in found}
    return [c for c in others if wanted(c) and _key(c) not in listed]


def type_links(cat: Catalogue, index: StatusIndex, chapter: Chapter, op: Operator) -> TypeLinks:
    """The operators that feed ``op`` and the operators that it feeds, as the module says.

    ``chapter`` is the chapter of ``op`` in ``cat``, and ``index`` has the status of every
    operator that a list names. The lists have the operators of every chapter of ``cat`` but
    ``op`` itself, in the order of the catalogue.
    """
    own_takes, own_gives = _types(op)
    takes, gives = _ordered(own_takes), _ordered(own_gives)
    own = (chapter.id, op.id)
    others = [
        _candidate(other_chapter, other)
        for other_chapter in cat.chapters
        for other in other_chapter.operators
        if (other_chapter.id, other.id) != own
    ]
    fed = {type_id: _giving(others, _fed_by(type_id)) for type_id in takes}
    taking = {type_id: _taking(others, _taken_as(type_id)) for type_id in gives}
    any_type: list[_Candidate] = []
    if not _VALUES.isdisjoint(takes):
        any_type = _folded(others, fed.values(), lambda c: c.gives_any_type)
    any_scalar: list[_Candidate] = []
    if not _VALUES.isdisjoint(gives):
        any_scalar = _folded(others, taking.values(), lambda c: c.takes_any_scalar)
    return TypeLinks(
        takes=takes,
        gives=gives,
        comes_from={type_id: _refs(index, found) for type_id, found in fed.items()},
        any_type=_refs(index, any_type),
        goes_to={type_id: _refs(index, found) for type_id, found in taking.items()},
        any_scalar=_refs(index, any_scalar),
    )


# --- Block ---


def _cell(taken: bool, given: bool) -> str:
    if taken and given:
        return _IN_AND_OUT
    if taken:
        return _IN
    return _OUT if given else ""


def block_types_table(
    chapter: Chapter,
) -> tuple[tuple[str, ...], list[tuple[Operator, tuple[str, ...]]]]:
    """The operators of a block against the types that they use: ``(columns, rows)``.

    The columns are the types that some form of the block takes or gives, in the order of the
    types (``VALUE_TYPES``, ``STRUCTURAL_TYPES``, then ``scalar`` and ``any`` when a form uses
    them): a type that no operator of the block uses has no column. A row is an operator of the
    chapter, in its order, with a cell for each column: ``in`` when a form of the operator takes
    the type, ``out`` when one gives it, ``in · out`` when both do, and an empty cell when
    neither does. The types are those that the forms declare, so an operator that takes
    ``number`` is not ``in`` under ``integer``.
    """
    sides = [(op, *_types(op)) for op in chapter.operators]
    columns = _ordered(t for _, takes, gives in sides for t in (*takes, *gives))
    rows = [
        (op, tuple(_cell(t in takes, t in gives) for t in columns)) for op, takes, gives in sides
    ]
    return columns, rows


def by_block(refs: Sequence[Ref]) -> list[tuple[str, tuple[Ref, ...]]]:
    """The refs grouped by block: ``(title of the block, its refs)``, in the order of the refs.

    The refs of ``type_links`` are in the order of the catalogue, so the blocks come in the order
    of its chapters, and the refs of a block in the order of its operators. The refs are not
    sorted: a block is where its first ref is. The refs of a block carry its id (``chapter_id``).
    """
    blocks: dict[str, list[Ref]] = {}
    for ref in refs:
        blocks.setdefault(ref.chapter_id, []).append(ref)
    return [(group[0].chapter_title, tuple(group)) for group in blocks.values()]
