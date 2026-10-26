"""Catalogue model of the VTL operator map and the consistency checks on it."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

Status = Literal["ok", "partial", "refused", "missing", "diverge"]
Source = Literal["manual", "authored"]
Check = Literal["exact", "random", "current_date"]
# The kind of the widest operand of a form: a dataset, a component or a scalar (``form_kind``).
Kind = Literal["ds", "comp", "sc"]

# The types that a form takes and gives (``Variant.takes`` and ``Variant.gives``). The value types
# of VTL are matched along their hierarchy, ``TYPE_PARENTS``: a form that gives an ``integer``
# gives a ``number``. ``scalar`` is any value type, for a form that takes any of them (``=``,
# ``isnull``, ``nvl``...), and ``any`` is any type at all, for a form whose result follows its
# input (``nvl``, ``if-then-else``, the joins, the clauses...): it is only for what a form
# gives, as a form that takes a value of any type takes ``scalar``. The structural types are not
# values: each one matches only itself.
VALUE_TYPES: Final[tuple[str, ...]] = (
    "number",
    "integer",
    "string",
    "boolean",
    "time",
    "date",
    "time_period",
    "duration",
)
ANY_SCALAR: Final = "scalar"
ANY_TYPE: Final = "any"
# Each value type with its supertypes: scalar, then number and integer under it; string, boolean
# and duration; time, then date and time_period under it.
TYPE_PARENTS: Final[dict[str, tuple[str, ...]]] = {
    ANY_SCALAR: (),
    "number": (ANY_SCALAR,),
    "integer": ("number",),
    "string": (ANY_SCALAR,),
    "boolean": (ANY_SCALAR,),
    "time": (ANY_SCALAR,),
    "date": ("time",),
    "time_period": ("time",),
    "duration": (ANY_SCALAR,),
}
STRUCTURAL_TYPES: Final[tuple[str, ...]] = (
    "set",
    "value_domain",
    "datapoint_ruleset",
    "hierarchical_ruleset",
    "viral_propagation",
    "operator",
    "external_routine",
)
# A link is shown at both of its ends: at its target, with the reverse of its label.
REVERSE_LABELS: Final[dict[str, str]] = {
    "uses": "used by",
    "used in": "uses",
    "used in over()": "uses, in over()",
    "used in group all": "uses, in group all",
}
DL_PAGES: Final[tuple[str, ...]] = ("dl/udo", "dl/datapoint", "dl/hierarchical", "dl/viral")
# The longest title of a light box that the forms of an operator name (``Variant.box``).
BOX_TITLE_MAX: Final = 30


@dataclass(frozen=True)
class Example:
    """One runnable example of a variant."""

    name: str  # folder name: ``manual_ex_1`` for a manual example, else a slug: ``zero_divisor``
    source: Source
    manual_ref: str | None = None  # manual example on the operator's page, e.g. ``ex_1``
    # A slip of the manual that is corrected: in its script, its input or its expected output.
    erratum: str | None = None
    # The manual's representation of an input or of an output name rewritten to the equivalent
    # representation vtlengine documents, or values that no engine can reproduce (the manual's
    # illustrative random draws) left out of the expected output and property-checked.
    adaptation: str | None = None
    decision: int | None = None  # issue of a team decision the expected output follows
    check: Check = "exact"  # ``random`` and ``current_date`` are checked by property


@dataclass(frozen=True)
class Variant:
    """One syntax form of an operator.

    ``takes`` and ``gives`` are the types of the form, of ``VALUE_TYPES``, ``ANY_SCALAR`` and
    ``STRUCTURAL_TYPES``, and ``ANY_TYPE`` too for ``gives``. Every form declares both
    (``validate_catalogue``): ``None`` is a side that is not declared, which has no types
    (``form_types``) and is a problem; an empty ``takes`` is a form that takes nothing, such as
    ``current_date()``. Whether a form is on datasets, components or scalars is its kind
    (``form_kind``), not a type. ``kind`` says the kind of a form whose id has no token for it,
    such as the conversions of ``cast`` (``integer_to_boolean`` is on components).

    ``box`` is the title of the light box that the diagram of the operator draws the form in, for
    forms that stand for a part of a syntax, such as the "Signature", the "Body" and the "Return
    type" of a definition of the VTL-DL. The forms of an operator all have a ``box`` or none has
    (``validate_catalogue``), and a title is not empty and at most ``BOX_TITLE_MAX`` characters. An
    operator whose forms have boxes is drawn in those boxes, whatever the kinds of its forms.
    """

    id: str
    label: str
    examples: tuple[Example, ...]
    returns: tuple[str, ...] = ()  # not the types of the form, which are its ``gives``
    takes: tuple[str, ...] | None = None  # None: not declared; (): takes nothing
    gives: tuple[str, ...] | None = None  # None: not declared
    kind: Kind | None = None  # the kind of its widest operand, when the id does not say it
    box: str | None = None  # the title of its light box, when its forms are in titled boxes


@dataclass(frozen=True)
class Link:
    """A composition link to another operator, shown in the "Used with" box of two diagrams.

    The operator that declares it lists its target in the box of its diagram with the label, and
    the target lists that operator in the box of its own diagram with the reverse of the label
    (``tool.mermaid.used_with``). A definition of the VTL-DL is never an end of a link: it is not
    used inside an operator (``validate_catalogue``).
    """

    target: str  # ``chapter.operator``
    label: str  # a key of ``REVERSE_LABELS``: the target shows the reverse


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
class Catalogue:
    """Everything the map shows."""

    chapters: tuple[Chapter, ...]


@dataclass(frozen=True)
class ViralGroup:
    """The operators that apply the viral propagation rules, and the manual sentence that says so.

    ``page`` is the page of the manual that has the sentence, as ``Operator.page`` names a page, or
    ``typical`` for its Typical behaviours of the ML operators. ``operators`` are addresses,
    ``<chapter>.<operator>``. The groups are kept by hand in ``catalogue.viral``; they are never a
    ``Link``, as a definition of the VTL-DL is not used inside an operator.
    """

    title: str
    page: str
    sentence: str  # quoted from ``page``
    operators: tuple[str, ...]


def manual(ref: str, **kw: Any) -> Example:
    """A manual example, named ``manual_<ref>`` and mapped to ``<operator.page>/<ref>``."""
    return Example(name=f"manual_{ref}", source="manual", manual_ref=ref, **kw)


def authored(name: str = "minimal", **kw: Any) -> Example:
    """An example written for the map."""
    return Example(name=name, source="authored", **kw)


def load_catalogue() -> Catalogue:
    """Read ``catalogue.CHAPTERS``."""
    import catalogue  # late: the catalogue modules import this one

    return Catalogue(chapters=tuple(catalogue.CHAPTERS))


def load_viral_groups() -> tuple[ViralGroup, ...]:
    """Read ``catalogue.viral.GROUPS``: the operators that apply the viral propagation rules."""
    from catalogue import viral  # late: the catalogue modules import this one

    return viral.GROUPS


# --- Forms, links and examples ---


_DATASET: Final = "dataset"
# The kinds of the forms, the widest first: the first one whose token the id has is its kind.
_KINDS: Final[tuple[Kind, ...]] = ("ds", "comp", "sc")
_MANUAL_NAME: Final = re.compile(r"manual_ex_(\d+)")  # the folder of the manual example N


def form_kind(v: Variant) -> Kind | None:
    """The kind of the widest operand of a form: its own ``kind``, else the tokens of its id.

    A form that sets ``kind`` has it, whatever its id says. Otherwise the id is split on ``_``:
    any ``ds`` makes a dataset form, else any ``comp`` a component form, else any ``sc`` a scalar
    form, so ``sc_comp_digits`` is ``comp`` and ``analytic_ds`` is ``ds``. An id that has none of
    the three gives no kind (``identity``, ``param_dataset``, ``ds2``).
    """
    if v.kind is not None:
        return v.kind
    tokens = v.id.split("_")
    return next((kind for kind in _KINDS if kind in tokens), None)


def has_titled_boxes(op: Operator) -> bool:
    """Whether the forms of ``op`` name the light boxes that they are drawn in (``Variant.box``).

    One form that names a box is enough: the diagram of the operator then has those boxes, and the
    kinds of its forms say nothing. ``validate_catalogue`` wants all the forms to name one, or none.
    """
    return any(variant.box is not None for variant in op.variants)


def form_types(v: Variant) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """What the form ``v`` takes and gives: the types that it declares, and no others.

    A side that the form does not declare (``None``) has none, whatever its operator accepts and
    returns: ``validate_catalogue`` refuses a form that does not declare both. A declared empty
    ``takes`` takes nothing, as ``current_date()`` does.
    """
    return v.takes or (), v.gives or ()


def reverse_label(label: str) -> str:
    """The label of a link as its target shows it: ``used in`` is ``uses`` there.

    A label that ``REVERSE_LABELS`` does not have raises ``KeyError``; ``validate_catalogue``
    reports a link that has one.
    """
    return REVERSE_LABELS[label]


def example_title(e: Example) -> str:
    """The title of an example: ``Manual example N`` for ``manual_ex_N``, else its name.

    The name is a slug of what the example tests, and its title that slug in sentence case:
    ``zero_divisor`` is ``Zero divisor``.
    """
    found = _MANUAL_NAME.fullmatch(e.name)
    if found is not None:
        return f"Manual example {found.group(1)}"
    words = e.name.replace("_", " ")
    return words[:1].upper() + words[1:]


# --- Validation ---


_ID_PATTERN: Final = re.compile(r"[a-z0-9_]+")
# The name of an authored example: a slug of what it tests, which is not one of the names that
# the first version of the map gave them (``minimal``, ``minimal_2``).
_SLUG: Final = re.compile(r"[a-z][a-z0-9_]{0,39}")
_SLUG_RULE: Final = "a letter, then up to 39 letters, digits or underscores"
_PLACEHOLDER: Final = re.compile(r"minimal(?:_\d+)?")
_FORM_TYPES: Final = frozenset((*VALUE_TYPES, *STRUCTURAL_TYPES, ANY_SCALAR, ANY_TYPE))
# The types of an operator, and the ``returns`` of a form, may name ``dataset`` too.
_OPERATOR_TYPES: Final = _FORM_TYPES | {_DATASET}
DEFINITIONS: Final = "definitions"  # the chapter whose pages are ``DL_PAGES``
_DEFINITION_LINK: Final = "VTL-DL definitions are never linked from or to operators"


def _bad_id(owner: str, item_id: str, what: str = "id") -> Iterator[str]:
    """``item_id`` against the id rule: the ids, example names and manual refs name folders."""
    if _ID_PATTERN.fullmatch(item_id) is None:
        yield f"{owner}: bad {what} '{item_id}'"


def _duplicate(owner: str, kind: str, item_id: str, seen: dict[str, int]) -> Iterator[str]:
    """Report an id once, when it is met for the second time in ``owner``."""
    seen[item_id] = seen.get(item_id, 0) + 1
    if seen[item_id] == 2:
        yield f"{owner}: duplicate {kind} '{item_id}'"


def _unknown_types(owner: str, type_ids: Iterable[str], known: frozenset[str]) -> Iterator[str]:
    for type_id in dict.fromkeys(type_ids):
        if type_id not in known:
            yield f"{owner}: unknown type '{type_id}'"


def _manual_keys(operator: Operator) -> Iterator[str]:
    """The ``<page>/<ref>`` key of each example of ``operator`` that has a ``manual_ref``."""
    for variant in operator.variants:
        for example in variant.examples:
            if example.manual_ref is not None:
                yield f"{operator.page}/{example.manual_ref}"


def _variant_problems(owner: str, variants: Iterable[Variant]) -> Iterator[str]:
    """The problems of the variants of ``owner`` and of their examples.

    Each form declares the types it takes and gives, and each authored example has a slug of what
    it tests for a name.
    """
    seen: dict[str, int] = {}
    for variant in variants:
        yield from _bad_id(owner, variant.id)
        yield from _duplicate(owner, "variant", variant.id, seen)
        variant_owner = f"{owner}.{variant.id}"
        yield from _unknown_types(variant_owner, variant.returns, _OPERATOR_TYPES)
        declared = (*(variant.takes or ()), *(variant.gives or ()))
        yield from _unknown_types(variant_owner, declared, _FORM_TYPES)
        yield from _type_problems(variant_owner, variant)
        seen_examples: dict[str, int] = {}
        for example in variant.examples:
            # The name of an example names its folder, and so the target of a click.
            yield from _bad_id(variant_owner, example.name, "example name")
            yield from _duplicate(variant_owner, "example", example.name, seen_examples)
            if example.manual_ref is not None:
                yield from _bad_id(variant_owner, example.manual_ref, "manual_ref")
            if example.source == "authored":
                yield from _slug_problems(variant_owner, example.name)


def _mixed_kinds(owner: str, variants: Iterable[Variant]) -> Iterator[str]:
    """The forms of an operator that have no kind, when others have one.

    The forms of an operator all have a kind, or none has: a diagram puts each form in the box
    of its kind, or has no boxes.
    """
    kinds = {variant.id: form_kind(variant) for variant in variants}
    kindless = [variant_id for variant_id, kind in kinds.items() if kind is None]
    if kindless and len(kindless) < len(kinds):
        yield f"{owner}: these forms have no kind, but other forms have one: {', '.join(kindless)}"


def _box_problems(owner: str, variants: Sequence[Variant]) -> Iterator[str]:
    """The problems of the boxes that the forms of an operator name.

    The forms of an operator all name a box, or none does: a diagram draws the boxes that its
    forms name, and a form that names none would hang outside them. A title is not empty (blank
    is empty) and has at most ``BOX_TITLE_MAX`` characters, for it is the title of a box.
    """
    unboxed = [variant.id for variant in variants if variant.box is None]
    if unboxed and len(unboxed) < len(variants):
        yield f"{owner}: these forms have no box, but other forms have one: {', '.join(unboxed)}"
    for variant in variants:
        title = variant.box
        if title is None:
            continue
        if not title.strip():
            yield f"{owner}.{variant.id}: the box title is empty"
        elif len(title) > BOX_TITLE_MAX:
            problem = f"box title '{title}' is longer than {BOX_TITLE_MAX} characters"
            yield f"{owner}.{variant.id}: {problem}"


def _type_problems(owner: str, variant: Variant) -> Iterator[str]:
    """What is wrong with the types of a form: neither its ``returns`` nor its operator's count.

    It declares ``takes``, which may be empty for a form that takes nothing, and ``gives``,
    which holds a type: every form gives something. It does not take ``any``, which is for what a
    form gives (the result follows the input): a value of any type is a ``scalar``.
    """
    if variant.takes is None:
        yield f"{owner}: the form does not declare takes"
    elif ANY_TYPE in variant.takes:
        yield (
            f"{owner}: the form takes '{ANY_TYPE}': {ANY_TYPE} is only for gives, so a value of "
            f"any type takes '{ANY_SCALAR}'"
        )
    if variant.gives is None:
        yield f"{owner}: the form does not declare gives"
    elif not variant.gives:
        yield f"{owner}: the form gives nothing: gives needs a type"


def _slug_problems(owner: str, name: str) -> Iterator[str]:
    """Why ``name`` does not name an authored example after what it tests, if it does not.

    A name that breaks the id rule is reported as such, and only so.
    """
    if _ID_PATTERN.fullmatch(name) is None:
        return
    if _PLACEHOLDER.fullmatch(name) is not None:
        yield f"{owner}: example name '{name}' does not say what the example tests"
    elif _MANUAL_NAME.fullmatch(name) is not None:
        yield f"{owner}: example name '{name}' is the name of a manual example"
    elif _SLUG.fullmatch(name) is None:
        yield f"{owner}: example name '{name}' is not a slug ({_SLUG_RULE})"


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
    yield from _unknown_types(owner, (*operator.accepts, *operator.returns), _OPERATOR_TYPES)
    for link in operator.links:
        if link.target not in targets:
            yield f"{owner}: link target '{link.target}' not found"
        if link.label not in REVERSE_LABELS:
            yield f"{owner}: link label '{link.label}' has no reverse label"
        # A link of a definition, or to one, or both: one line, as it is one link.
        from_definition = chapter.id == DEFINITIONS
        to_definition = link.target.partition(".")[0] == DEFINITIONS
        if from_definition or to_definition:
            yield f"{owner}: link target '{link.target}': {_DEFINITION_LINK}"
    for key in _manual_keys(operator):
        if key not in manual_examples:
            yield f"{owner}: unknown manual example '{key}'"
    yield from _box_problems(owner, operator.variants)
    # The kinds of forms that are in boxes of their own say nothing: no box is drawn by kind.
    if not has_titled_boxes(operator):
        yield from _mixed_kinds(owner, operator.variants)
    yield from _variant_problems(owner, operator.variants)


def _chapter_problems(
    cat: Catalogue,
    targets: Collection[str],
    pages: frozenset[str],
    manual_examples: frozenset[str],
) -> Iterator[str]:
    """The problems of every chapter."""
    seen_chapters: dict[str, int] = {}
    for chapter in cat.chapters:
        yield from _bad_id("catalogue", chapter.id)
        yield from _duplicate("catalogue", "chapter", chapter.id, seen_chapters)
        seen_operators: dict[str, int] = {}
        for operator in chapter.operators:
            yield from _bad_id(chapter.id, operator.id)
            yield from _duplicate(chapter.id, "operator", operator.id, seen_operators)
            yield from _operator_problems(chapter, operator, targets, pages, manual_examples)


def _in_scope(cat: Catalogue, scope: Collection[str] | None) -> Callable[[str], bool]:
    """Tell whether a page, or a ``<page>/<example>`` key, belongs to a chapter of ``scope``.

    A page belongs to a chapter when it starts with ``<chapter title>/``; the pages of the
    ``definitions`` chapter are ``DL_PAGES``. A scope id with no chapter owns no pages.
    """
    if scope is None:
        return lambda _item: True
    ids = frozenset(scope)
    title_prefixes = tuple(
        f"{c.title}/" for c in cat.chapters if c.id in ids and c.id != DEFINITIONS
    )
    with_dl_pages = DEFINITIONS in ids

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

    Problems come in catalogue order, followed by the page and manual example coverage problems,
    each sorted. ``pages`` are all the manual pages and ``manual_examples`` all its
    ``<page>/<example>`` keys.

    ``scope`` is a collection of chapter ids. When given, the page coverage and the manual
    example mapping are checked only for the pages of those chapters. All the other checks (types,
    links, ids, duplicates, the forms, and the pages and manual examples the catalogue refers to)
    always cover the whole catalogue, against all of ``pages`` and ``manual_examples``: a run that
    is limited to some chapters renders the pages of all of them.

    Ids, example names and manual refs follow the id rule, ``[a-z0-9_]+``: they name folders
    and the targets of the clicks of the diagrams. The ``takes`` and ``gives`` of a form are of
    ``VALUE_TYPES``, ``STRUCTURAL_TYPES`` and ``ANY_SCALAR``, and its ``gives`` may be ``ANY_TYPE``
    too; the ``accepts`` and ``returns`` of an operator, and the ``returns`` of a form, may name
    ``dataset`` as well. The label of every link has a reverse in ``REVERSE_LABELS``, and no link
    goes from or to an operator of the definitions chapter: the definitions of the VTL-DL are not
    used inside an operator.

    The forms are checked too: every form declares its ``takes``, empty for a form that takes
    nothing, and its ``gives``, with a type at least; ``ANY_TYPE`` is only for ``gives``, a value
    of any type being taken as ``ANY_SCALAR``; the forms of an operator all have a kind
    (``form_kind``) or none has, unless they name light boxes; the forms of an operator all name a
    box (``Variant.box``) or none does, and the title of a box is not empty and has at most
    ``BOX_TITLE_MAX`` characters; and every authored example is named after what it tests, with a
    slug that matches ``[a-z][a-z0-9_]{0,39}`` and is neither ``minimal`` nor ``minimal_<n>``, nor
    the name of a manual example, ``manual_ex_<n>``. The notes of the examples are checked with
    their folders, by the generator.
    """
    targets = {f"{chapter.id}.{op.id}" for chapter in cat.chapters for op in chapter.operators}
    known_pages = frozenset(pages)
    known_examples = frozenset(manual_examples)
    return [
        *_chapter_problems(cat, targets, known_pages, known_examples),
        *_coverage_problems(cat, known_pages, known_examples, scope),
    ]
