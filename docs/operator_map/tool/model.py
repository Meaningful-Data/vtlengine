"""Catalogue model of the VTL operator map and the consistency checks on it."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Final, Literal

Status = Literal["ok", "partial", "missing", "diverge"]
Source = Literal["manual", "authored"]
Check = Literal["exact", "random", "current_date"]

TYPE_IDS: Final[tuple[str, ...]] = (
    "scalar",
    "number",
    "integer",
    "string",
    "boolean",
    "time",
    "date",
    "time_period",
    "duration",
    "dataset",
    "set",
    "value_domain",
    "datapoint_ruleset",
    "hierarchical_ruleset",
    "viral_propagation",
    "operator",
    "external_routine",
)
DL_PAGES: Final[tuple[str, ...]] = ("dl/udo", "dl/datapoint", "dl/hierarchical", "dl/viral")


@dataclass(frozen=True)
class Example:
    """One runnable example of a variant."""

    name: str  # folder name: ``manual_ex_1``, ``minimal``, ``minimal_2``...
    source: Source
    manual_ref: str | None = None  # manual example on the operator's page, e.g. ``ex_1``
    # A slip of the manual that is corrected: in its script, its input or its expected output.
    erratum: str | None = None
    # The manual's representation of an input, or of the name of an output, rewritten to the
    # equivalent representation that vtlengine documents.
    adaptation: str | None = None
    decision: int | None = None  # issue of a team decision the expected output follows
    check: Check = "exact"  # ``random`` and ``current_date`` are checked by property


@dataclass(frozen=True)
class Variant:
    """One syntax form of an operator or type."""

    id: str
    label: str
    examples: tuple[Example, ...]
    returns: tuple[str, ...] = ()  # return types when they differ from the operator's


@dataclass(frozen=True)
class Link:
    """A composition link to another operator, drawn as a labelled dotted edge."""

    target: str  # ``chapter.operator``
    label: str


@dataclass(frozen=True)
class Operator:
    """One operator of a manual page, or one definition of the VTL-DL."""

    id: str
    name: str
    symbol: str
    page: str  # ``<chapter title>/<page>`` of the manual, or one of ``DL_PAGES``
    accepts: tuple[str, ...]
    returns: tuple[str, ...]
    variants: tuple[Variant, ...]
    links: tuple[Link, ...] = ()


@dataclass(frozen=True)
class Chapter:
    """A manual chapter, or the definitions."""

    id: str
    title: str
    operators: tuple[Operator, ...]


@dataclass(frozen=True)
class TypeDef:
    """A data type hub; ``id`` is one of ``TYPE_IDS`` and ``parents`` its supertypes."""

    id: str
    parents: tuple[str, ...]
    variants: tuple[Variant, ...] = ()
    see: str | None = None  # ``chapter.operator`` whose first example the type hub links to


@dataclass(frozen=True)
class Catalogue:
    """Everything the map shows."""

    chapters: tuple[Chapter, ...]
    types: tuple[TypeDef, ...]


def manual(ref: str, **kw: Any) -> Example:
    """A manual example, named ``manual_<ref>`` and mapped to ``<operator.page>/<ref>``."""
    return Example(name=f"manual_{ref}", source="manual", manual_ref=ref, **kw)


def authored(name: str = "minimal", **kw: Any) -> Example:
    """An example written for the map."""
    return Example(name=name, source="authored", **kw)


def load_catalogue() -> Catalogue:
    """Read ``catalogue.CHAPTERS`` and ``catalogue.TYPES``."""
    import catalogue  # late: the catalogue modules import this one

    return Catalogue(chapters=tuple(catalogue.CHAPTERS), types=tuple(catalogue.TYPES))


_ID_PATTERN: Final = re.compile(r"[a-z0-9_]+")
_KNOWN_TYPES: Final = frozenset(TYPE_IDS)
_DEFINITIONS: Final = "definitions"  # the chapter whose pages are ``DL_PAGES``


def _bad_id(owner: str, item_id: str) -> Iterator[str]:
    if _ID_PATTERN.fullmatch(item_id) is None:
        yield f"{owner}: bad id '{item_id}'"


def _duplicate(owner: str, kind: str, item_id: str, seen: dict[str, int]) -> Iterator[str]:
    """Report an id once, when it is met for the second time in ``owner``."""
    seen[item_id] = seen.get(item_id, 0) + 1
    if seen[item_id] == 2:
        yield f"{owner}: duplicate {kind} '{item_id}'"


def _unknown_types(owner: str, type_ids: Iterable[str]) -> Iterator[str]:
    for type_id in dict.fromkeys(type_ids):
        if type_id not in _KNOWN_TYPES:
            yield f"{owner}: unknown type '{type_id}'"


def _manual_keys(operator: Operator) -> Iterator[str]:
    """The ``<page>/<ref>`` key of each example of ``operator`` that has a ``manual_ref``."""
    for variant in operator.variants:
        for example in variant.examples:
            if example.manual_ref is not None:
                yield f"{operator.page}/{example.manual_ref}"


def _variant_problems(owner: str, variants: Iterable[Variant]) -> Iterator[str]:
    seen: dict[str, int] = {}
    for variant in variants:
        yield from _bad_id(owner, variant.id)
        yield from _duplicate(owner, "variant", variant.id, seen)
        variant_owner = f"{owner}.{variant.id}"
        yield from _unknown_types(variant_owner, variant.returns)
        seen_examples: dict[str, int] = {}
        for example in variant.examples:
            yield from _duplicate(variant_owner, "example", example.name, seen_examples)


def _operator_problems(
    chapter: Chapter,
    operator: Operator,
    targets: Collection[str],
    pages: frozenset[str],
    manual_examples: frozenset[str],
) -> Iterator[str]:
    owner = f"{chapter.id}.{operator.id}"
    if operator.page not in pages:
        yield f"{owner}: unknown page '{operator.page}'"
    yield from _unknown_types(owner, (*operator.accepts, *operator.returns))
    for link in operator.links:
        if link.target not in targets:
            yield f"{owner}: link target '{link.target}' not found"
    for key in _manual_keys(operator):
        if key not in manual_examples:
            yield f"{owner}: unknown manual example '{key}'"
    yield from _variant_problems(owner, operator.variants)


def _chapter_problems(
    cat: Catalogue,
    targets: Collection[str],
    pages: frozenset[str],
    manual_examples: frozenset[str],
) -> Iterator[str]:
    seen_chapters: dict[str, int] = {}
    for chapter in cat.chapters:
        yield from _bad_id("catalogue", chapter.id)
        yield from _duplicate("catalogue", "chapter", chapter.id, seen_chapters)
        seen_operators: dict[str, int] = {}
        for operator in chapter.operators:
            yield from _bad_id(chapter.id, operator.id)
            yield from _duplicate(chapter.id, "operator", operator.id, seen_operators)
            yield from _operator_problems(chapter, operator, targets, pages, manual_examples)


def _typedef_problems(cat: Catalogue, targets: Collection[str]) -> Iterator[str]:
    seen_types: dict[str, int] = {}
    for typedef in cat.types:
        owner = f"types.{typedef.id}"
        if typedef.id not in _KNOWN_TYPES:
            yield f"types: unknown type '{typedef.id}'"
        yield from _duplicate("types", "type", typedef.id, seen_types)
        yield from _unknown_types(owner, typedef.parents)
        if typedef.see is not None and typedef.see not in targets:
            yield f"{owner}: see target '{typedef.see}' not found"
        yield from _variant_problems(owner, typedef.variants)


def _in_scope(cat: Catalogue, scope: Collection[str] | None) -> Callable[[str], bool]:
    """Tell whether a page, or a ``<page>/<example>`` key, belongs to a chapter of ``scope``.

    A page belongs to a chapter when it starts with ``<chapter title>/``; the pages of the
    ``definitions`` chapter are ``DL_PAGES``. A scope id with no chapter owns no pages.
    """
    if scope is None:
        return lambda _item: True
    ids = frozenset(scope)
    title_prefixes = tuple(
        f"{c.title}/" for c in cat.chapters if c.id in ids and c.id != _DEFINITIONS
    )
    with_dl_pages = _DEFINITIONS in ids

    def in_scope(item: str) -> bool:
        if item.startswith(title_prefixes):
            return True
        return with_dl_pages and any(item == p or item.startswith(f"{p}/") for p in DL_PAGES)

    return in_scope


def _coverage_problems(
    cat: Catalogue,
    pages: frozenset[str],
    manual_examples: frozenset[str],
    scope: Collection[str] | None,
) -> Iterator[str]:
    in_scope = _in_scope(cat, scope)
    operators = [operator for chapter in cat.chapters for operator in chapter.operators]
    covered = {operator.page for operator in operators}
    for page in sorted(pages - covered):
        if in_scope(page):
            yield f"page not covered: {page}"
    mapped = Counter(key for operator in operators for key in _manual_keys(operator))
    for key in sorted(manual_examples):
        if not in_scope(key):
            continue
        if mapped[key] == 0:
            yield f"manual example not mapped: {key}"
        elif mapped[key] > 1:
            yield f"manual example mapped {mapped[key]} times: {key}"


def validate_catalogue(
    cat: Catalogue,
    pages: Collection[str],
    manual_examples: Collection[str],
    *,
    scope: Collection[str] | None = None,
) -> list[str]:
    """Return the problems of ``cat``, such as unknown types, pages or manual examples.

    Problems come in catalogue order (chapters, then types), followed by the page and manual
    example coverage problems, each sorted. ``pages`` are all the manual pages and
    ``manual_examples`` all its ``<page>/<example>`` keys.

    ``scope`` is a collection of chapter ids. When given, the page coverage and the manual
    example mapping are checked only for the pages of those chapters. All the other checks
    (types, links, ``see`` targets, ids, duplicates, and the pages and manual examples the
    catalogue refers to) always cover the whole catalogue, against all of ``pages`` and
    ``manual_examples``.
    """
    targets = {f"{chapter.id}.{op.id}" for chapter in cat.chapters for op in chapter.operators}
    known_pages = frozenset(pages)
    known_examples = frozenset(manual_examples)
    return [
        *_chapter_problems(cat, targets, known_pages, known_examples),
        *_typedef_problems(cat, targets),
        *_coverage_problems(cat, known_pages, known_examples, scope),
    ]
