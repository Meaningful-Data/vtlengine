"""Mermaid flowcharts of the operator map: the overview, the diagram of a block and of an operator.

Every diagram starts with its ``flowchart`` line and the same ``classDef`` lines (``CLASSES``), so
a diagram can be pasted into a fence on its own. A node is coloured by the status of what it
stands for, and has no emoji of its own: the class says it, and the emoji of the statuses are in
the tallies. ``refused`` is the colour of ``ok`` with a thick dotted border. Nothing here needs the
run to be complete: a key that the ``StatusIndex`` does not have is a node with no status class,
and no part of a tally, never an error. No diagram has a node for a type: the forms carry their
types (``tool.model.form_types``).

* ``overview_block``: VTL 2.2 is the central node, with the blocks split half to the left of it
  and half to the right ("both sides"), joined to it by plain lines. A block shows how many
  operators it has and how many of them are in each status.
* ``block_diagram``: "both sides" again, with the block in the middle and its operators around
  it. An operator shows how many of its forms are in each status.
* ``operator_diagram``: the operator on top, with the tally of its forms. Under it there is a
  light box for each kind of operand that its forms use (``form_kind``): solid for the datasets,
  dashed for the components and dotted for the scalars. The boxes are there only when the forms
  use at least two kinds. An operator whose forms use none, such as a join, or only one, has "both
  sides" and no boxes: its forms to the left of it and to the right, half and half, and the
  "Used with" box under the forms on the right. The columns are balanced: the box counts as one
  form more than it has nodes (``_left``). A form with no kind among forms that have one is a
  mixture that the strict checks of the catalogue refuse, but it is still drawn: it hangs from the
  operator, outside the boxes.

The "Used with" box of an operator diagram holds the operators that this one is combined with,
from the links of the whole catalogue. A link is shown at both of its ends: the operator that
declares it shows its target with the label that the link has, and the target shows that operator
with the reverse of the label (``tool.model.reverse_label``). An operator that is linked in both
ways is shown once, with each of its labels. An entry is short: the symbol of the operator (its
name when the symbol is not a word, or there is none), an arrow and the label, and its tooltip has
the name and the symbol. With more than ``USED_WITH_MAX`` operators the box has one node for each
block instead, with how many of its operators are there. ``used_with`` has the operators of the
box, for a page that lists them in full under the diagram.

The ``click`` targets are relative to the page that has the diagram. The overview is on the index
page of the map: a block is ``<chapter>/index.html``. A block diagram is on that page, and an
operator is ``<operator>.html``. An operator diagram is on the page of the operator: a form is the
anchor of its section on that page (``form_anchor``), and what the operator is used with is
``../<chapter>/<operator>.html``, or ``../<chapter>/index.html`` for a whole block.

The output follows the order of the catalogue, so a rerun gives the same text.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

from tool.model import (
    Catalogue,
    Chapter,
    Example,
    Kind,
    Operator,
    Status,
    Variant,
    form_kind,
    reverse_label,
)
from tool.status import EMOJI, StatusIndex, counts

# The defaults of Mermaid (``maxEdges`` and ``maxTextSize``). A diagram stays strictly under both,
# as the design asks, so a diagram never sits on a limit.
MAX_EDGES: Final = 500
MAX_CHARS: Final = 50_000

# What follows the name of a page in the target of a click: its page in the built site. The limits
# above count it, as it is part of the text of a diagram.
PAGE_SUFFIX: Final = ".html"

# The most operators that the "Used with" box shows one by one: with more, it shows the blocks.
USED_WITH_MAX: Final = 8

# The text is black on the light fills, so that it also reads on a dark theme. ``refused`` is the
# green of ``ok`` with a thick dotted border: a deliberate refusal, which the border alone says on
# a node. ``block`` is the central node of the overview and of a block, and ``ext`` an entry of
# "Used with", an operator of another page.
CLASSES: Final = (
    "classDef ok fill:#d4edda,stroke:#28a745,color:#000",
    "classDef partial fill:#fff3cd,stroke:#ffc107,color:#000",
    (
        "classDef refused fill:#d4edda,stroke:#28a745,stroke-width:3px,"
        "stroke-dasharray: 2 2,color:#000"
    ),
    "classDef missing fill:#f8d7da,stroke:#dc3545,color:#000",
    "classDef diverge fill:#cce5ff,stroke:#007bff,color:#000",
    "classDef block fill:#2b3a4a,stroke:#2b3a4a,color:#fff,font-weight:bold",
    "classDef ext fill:#fff,stroke:#666,color:#000,stroke-dasharray: 5 5",
)

# The three boxes of an operator diagram have the same light fill, and a border of their own.
_BOX: Final = "fill:#f8f9fa,stroke:#495057,stroke-width:2px"
BOX_STYLES: Final[dict[Kind, str]] = {
    "ds": _BOX,
    "comp": f"{_BOX},stroke-dasharray: 8 5",
    "sc": f"{_BOX},stroke-dasharray: 2 3",
}
BOX_TITLES: Final[dict[Kind, str]] = {"ds": "Dataset", "comp": "Component", "sc": "Scalar"}
# The "Used with" box: a thin, sparsely dotted border.
USED_WITH_STYLE: Final = "fill:#ffffff,stroke:#868e96,stroke-width:1px,stroke-dasharray: 1 4"

# The kinds in the order of their boxes, the widest operand first.
_KINDS: Final[tuple[Kind, ...]] = ("ds", "comp", "sc")
# The ids of the central node of each diagram and of the "Used with" box. The ids of the other
# nodes are made of the ids of the catalogue (``chapter__...``, ``v_...``, ``used__...``), so none
# of them can be these.
_OVERVIEW_ROOT: Final = "R"
_BLOCK_ROOT: Final = "B"
_OPERATOR_ROOT: Final = "O"
_USED_WITH: Final = "G_use"
_OVERVIEW_TITLE: Final = "VTL 2.2<br/>Reference Manual"

_PIN: Final = "📌"  # a node whose expected output follows a recorded team decision
_INDENT: Final = "    "
_EDGE_MARKS: Final = ("-->", "---", ".->", "-.-")  # a line that has one of these is an edge

# The symbols of VTL that would break a label, as Mermaid entity codes. ``:=`` is one symbol.
_ESCAPES: Final = {
    "#": "#35;",
    "|": "#124;",
    '"': "#quot;",
    "<": "#60;",
    ">": "#62;",
    "[": "#91;",
    "]": "#93;",
    "{": "#123;",
    "}": "#125;",
    ";": "#59;",
    ":=": "#58;#61;",
}
_ESCAPE: Final = re.compile("|".join(re.escape(symbol) for symbol in _ESCAPES))
_NOT_ID_CHAR: Final = re.compile(r"[^A-Za-z0-9_]")
# A symbol that is a word, letters, digits and underscores, says what an operator is in few
# characters: ``aggr``, ``inner_join``. ``+``, ``||`` and ``group by`` do not.
_WORD: Final = re.compile(r"[A-Za-z0-9_]+")


class MermaidLimitError(Exception):
    """A diagram is over a limit of Mermaid."""


# --- Escaping, ids and limits ---


def escape_label(text: str) -> str:
    """``text`` with the VTL symbols that break a quoted label as Mermaid entity codes.

    The symbols are ``#``, ``|``, ``"``, ``<``, ``>``, ``[``, ``]``, ``{``, ``}``, ``;`` and
    ``:=``. They are replaced in one pass, so that the ``#`` of an entity code is not escaped
    again (which is the same as replacing ``#`` first).
    """
    return _ESCAPE.sub(lambda match: _ESCAPES[match.group()], text)


def node_id(*parts: str) -> str:
    """The id of a node: ``parts`` joined by ``__``, with anything but ``A-Za-z0-9_`` as ``_``.

    Qualify an operator by its chapter, as the same operator id can be in two chapters.
    """
    return _NOT_ID_CHAR.sub("_", "__".join(parts))


def check_limits(block: str) -> None:
    """Raise ``MermaidLimitError`` unless ``block`` is under both limits of Mermaid.

    The edges are the lines with ``-->``, ``---``, ``.->`` or ``-.-``, so a block holds one edge
    per line. The length is counted in UTF-16 code units, as Mermaid does: a status emoji is two.
    """
    problem = _limit_problem(block)
    if problem is not None:
        raise MermaidLimitError(problem)


def _limit_problem(block: str) -> str | None:
    """What puts ``block`` over a limit, or None."""
    edges = sum(1 for line in block.splitlines() if any(mark in line for mark in _EDGE_MARKS))
    if edges >= MAX_EDGES:
        return f"{edges} edges, and a block needs fewer than {MAX_EDGES}"
    size = len(block.encode("utf-16-le")) // 2
    if size >= MAX_CHARS:
        return f"{size} characters, and a block needs fewer than {MAX_CHARS}"
    return None


def _checked(block: str, what: str) -> str:
    """``block``, or ``MermaidLimitError`` naming ``what`` if it is over a limit."""
    problem = _limit_problem(block)
    if problem is not None:
        raise MermaidLimitError(f"{what}: {problem}")
    return block


# --- Anchors ---


def form_anchor(v: Variant) -> str:
    """The anchor of the section of a form on the page of its operator: ``form-<id>``.

    The underscores of the id are dashes. Sphinx writes the identifier of a target with dashes,
    so the click of a diagram and the target of the page name the same one.
    """
    return "form-" + v.id.replace("_", "-")


def example_anchor(v: Variant, e: Example) -> str:
    """The anchor of an example on the page of its operator: ``ex-<form id>-<example name>``.

    As in ``form_anchor``, the underscores are dashes.
    """
    return f"ex-{v.id}-{e.name}".replace("_", "-")


# --- Lines ---


def _assemble(lines: Sequence[str], direction: str = "LR") -> str:
    """A diagram: the ``flowchart`` line, then the classes and ``lines`` indented under it."""
    body = (*CLASSES, *lines)
    return "\n".join([f"flowchart {direction}", *(f"{_INDENT}{line}" for line in body)])


@dataclass
class _Parts:
    """The lines of a diagram by what they are, in the order that they are written."""

    nodes: list[str] = field(default_factory=list)
    edges: list[str] = field(default_factory=list)
    styles: list[str] = field(default_factory=list)
    clicks: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        return [*self.nodes, *self.edges, *self.styles, *self.clicks]


def _node(node: str, label: str, css: str | None) -> str:
    return f'{node}["{label}"]' + ("" if css is None else f":::{css}")


def _click(node: str, target: str, tooltip: str | None = None) -> str:
    """The line that makes ``node`` a link to ``target``, with a tooltip if it has one.

    The tooltip is a quoted string, as a label is: the caller escapes it with ``escape_label``,
    and Mermaid decodes the entity codes in the text of the tooltip.
    """
    line = f'click {node} href "{target}"'
    return line if tooltip is None else f'{line} "{tooltip}"'


def _subgraph(node: str, title: str, members: Iterable[str]) -> list[str]:
    """The lines of a subgraph with ``members`` in it, laid out from left to right."""
    return [
        f'subgraph {node}["{title}"]',
        f"{_INDENT}direction LR",
        *(f"{_INDENT}{member}" for member in members),
        "end",
    ]


def _both_sides(root: str, nodes: Sequence[str], left: int | None = None) -> list[str]:
    """The lines that join ``nodes`` to ``root``: the first ``left`` of them on its left side.

    The rest are on its right side. Without ``left`` the first half is on the left, with the extra
    node when there is an odd number. The lines have no arrowheads.
    """
    count = (len(nodes) + 1) // 2 if left is None else left
    return [
        *(f"{node} --- {root}" for node in nodes[:count]),
        *(f"{root} --- {node}" for node in nodes[count:]),
    ]


def _left(forms: int, slots: int) -> int:
    """How many of ``forms`` go to the left of the operator, so that its two sides balance.

    The right side has the other forms and then the "Used with" box, which counts as ``slots``
    forms: a slot for each of its nodes and one for its title, and none when there is no box. The
    left side takes ``ceil((forms + slots) / 2)`` of the forms, and all of them when the box is
    longer than they are. Without a box that is half of them, and the extra one if they are odd.
    """
    return min(forms, (forms + slots + 1) // 2)


def _head(text: str, *, pinned: bool = False) -> str:
    """The first line of a node: the escaped ``text`` and the pin, with no emoji of the status."""
    escaped = escape_label(text)
    return f"{escaped} {_PIN}" if pinned else escaped


def _tally(statuses: Iterable[Status | None]) -> str:
    """``🟢6 🟡2``: how many of ``statuses`` are in each status, without the statuses of none.

    A status that is None is not known yet, and is not counted. The statuses come in the order of
    the legend, and the empty string is for a tally with nothing in it.
    """
    known = [status for status in statuses if status is not None]
    return " ".join(
        f"{EMOJI[status]}{number}" for status, number in counts(known).items() if number
    )


def _lines(first: str, second: str) -> str:
    """The text of a node: its first line and, when it has one, its second."""
    return f"{first}<br/>{second}" if second else first


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _title(operator: Operator) -> str:
    """``Name · symbol`` of an operator, and just the name when it has no symbol."""
    return f"{operator.name} · {operator.symbol}" if operator.symbol else operator.name


def _short(operator: Operator) -> str:
    """What an entry of "Used with" calls an operator: its symbol if that is a word, else its name.

    ``aggr`` and ``inner_join`` are words. ``+``, ``||``, ``group by`` and ``name( )`` are not,
    and an operator with no symbol has none to show: they are called by their name.
    """
    return operator.symbol if _WORD.fullmatch(operator.symbol) else operator.name


# --- What the catalogue says ---


def _pinned(variants: Sequence[Variant]) -> bool:
    """Whether an example of ``variants`` follows a recorded team decision."""
    return any(example.decision is not None for variant in variants for example in variant.examples)


def _targets(cat: Catalogue) -> dict[str, tuple[Chapter, Operator]]:
    """Every operator of ``cat`` by its ``<chapter>.<operator>`` address."""
    return {
        f"{chapter.id}.{operator.id}": (chapter, operator)
        for chapter in cat.chapters
        for operator in chapter.operators
    }


def _resolve(
    targets: Mapping[str, tuple[Chapter, Operator]], address: str, owner: str
) -> tuple[Chapter, Operator]:
    """The chapter and the operator at ``address``; ``owner`` says who points there."""
    if address not in targets:
        raise ValueError(f"{owner}: target '{address}' not found")
    return targets[address]


# --- The overview ---


def overview_block(cat: Catalogue, index: StatusIndex) -> str:
    """VTL 2.2 and its blocks on both sides of it, each with its operators and their statuses.

    The first half of the blocks, in the order of the catalogue, is to the left and the rest to
    the right. A block says how many operators it has and how many are in each status, and links
    to its page, ``<chapter>/index.html``. Raises ``MermaidLimitError`` if the overview is over a
    limit.
    """
    nodes = [_node(_OVERVIEW_ROOT, _OVERVIEW_TITLE, "block")]
    blocks: list[str] = []
    clicks: list[str] = []
    for chapter in cat.chapters:
        node = node_id("chapter", chapter.id)
        status = index.chapters.get(chapter.id)
        tally = _tally(index.operators.get(f"{chapter.id}/{op.id}") for op in chapter.operators)
        size = _count(len(chapter.operators), "operator")
        second = f"{size} · {tally}" if tally else size
        nodes.append(_node(node, _lines(_head(chapter.title), second), status))
        blocks.append(node)
        clicks.append(_click(node, f"{chapter.id}/index{PAGE_SUFFIX}"))
    lines = [*nodes, *_both_sides(_OVERVIEW_ROOT, blocks), *clicks]
    return _checked(_assemble(lines), "overview")


# --- The diagram of a block ---


def block_diagram(chapter: Chapter, index: StatusIndex) -> str:
    """The block and its operators on both sides of it, each with its forms and their statuses.

    The first half of the operators, in the order of the chapter, is to the left of the block and
    the rest to its right. An operator says how many of its forms are in each status, is pinned
    when an example of it follows a team decision, and links to its page, ``<operator>.html``.
    Raises ``MermaidLimitError`` if the diagram is over a limit.
    """
    nodes = [_node(_BLOCK_ROOT, escape_label(chapter.title), "block")]
    operators: list[str] = []
    clicks: list[str] = []
    for operator in chapter.operators:
        node = node_id(chapter.id, operator.id)
        key = f"{chapter.id}/{operator.id}"
        status = index.operators.get(key)
        tally = _tally(index.variants.get(f"{key}/{each.id}") for each in operator.variants)
        head = _head(_title(operator), pinned=_pinned(operator.variants))
        nodes.append(_node(node, _lines(head, tally), status))
        operators.append(node)
        clicks.append(_click(node, f"{operator.id}{PAGE_SUFFIX}"))
    lines = [*nodes, *_both_sides(_BLOCK_ROOT, operators), *clicks]
    return _checked(_assemble(lines), chapter.id)


def chapter_blocks(chapter: Chapter, cat: Catalogue, index: StatusIndex) -> list[str]:
    """The diagram of ``chapter`` as the list of blocks that ``tool.render`` still asks for.

    A chapter has one diagram, ``block_diagram``, so this is a list of it, or of nothing for a
    chapter with no operators. ``cat`` is not used. Call ``block_diagram``, not this.
    """
    return [block_diagram(chapter, index)] if chapter.operators else []


# --- The diagram of an operator ---


@dataclass(frozen=True)
class _Form:
    """A form of an operator as a node of its diagram."""

    variant: Variant
    node: str
    line: str


@dataclass(frozen=True)
class UsedWith:
    """An operator that another one is used with, and the labels of the links between them.

    The labels are as the first operator shows them: the label of a link that it declares, as it
    is, and the reverse of the label of a link that points to it.
    """

    chapter: Chapter
    operator: Operator
    labels: tuple[str, ...]


def operator_diagram(chapter: Chapter, op: Operator, cat: Catalogue, index: StatusIndex) -> str:
    """The operator with its forms, and the operators that it is used with.

    The forms are in boxes, one for each kind of operand, when they use at least two kinds, and
    on both sides of the operator otherwise, with the columns balanced when there is a "Used with"
    box (the module docstring has the details). Each form links to its section of the page, and is
    pinned when an example of it follows a team decision. The "Used with" box comes from the links
    of ``cat``, in both directions. Raises ``ValueError`` for a link of the operator to a target
    that is not in ``cat``, or whose label has no reverse, and ``MermaidLimitError`` if the
    diagram is over a limit.
    """
    key = f"{chapter.id}/{op.id}"
    status = index.operators.get(key)
    tally = _tally(index.variants.get(f"{key}/{each.id}") for each in op.variants)
    head = _head(_title(op), pinned=_pinned(op.variants))
    parts = _Parts(nodes=[_node(_OPERATOR_ROOT, _lines(head, tally), status)])
    forms = _forms(chapter, op, index)
    shown = _shown(used_with(chapter, op, cat), cat)
    if _has_boxes(op):
        direction = "TB"
        _boxes(forms, parts)
    else:
        direction = "LR"
        left = _left(len(forms), len(shown) + 1 if shown else 0)
        parts.nodes += [form.line for form in forms]
        parts.edges += _both_sides(_OPERATOR_ROOT, [form.node for form in forms], left)
    parts.clicks += [_click(form.node, f"#{form_anchor(form.variant)}") for form in forms]
    _used_with_box(shown, parts)
    return _checked(_assemble(parts.lines(), direction), f"{chapter.id}.{op.id}")


def _has_boxes(op: Operator) -> bool:
    """Whether the diagram of ``op`` has boxes: only when its forms use two kinds or more.

    An operator whose forms have no kind, such as a join, or all have the same one, such as an
    operator on components alone, has both sides and no boxes. A form with no kind counts for
    none of the kinds.
    """
    return len({kind for kind in map(form_kind, op.variants) if kind is not None}) >= 2


def _forms(chapter: Chapter, op: Operator, index: StatusIndex) -> list[_Form]:
    """The forms of ``op`` as nodes, in order: coloured by their status, pinned or not."""
    forms: list[_Form] = []
    for variant in op.variants:
        status = index.variants.get(f"{chapter.id}/{op.id}/{variant.id}")
        node = node_id(f"v_{variant.id}")
        label = _head(variant.label, pinned=_pinned((variant,)))
        forms.append(_Form(variant, node, _node(node, label, status)))
    return forms


def _boxes(forms: Sequence[_Form], parts: _Parts) -> None:
    """The box of each kind that ``forms`` use, under the operator, with its forms in it.

    A kind that no form has gets no box. A form that has no kind hangs from the operator.
    """
    by_kind: dict[Kind | None, list[_Form]] = {}
    for form in forms:
        by_kind.setdefault(form_kind(form.variant), []).append(form)
    for kind in _KINDS:
        members = by_kind.get(kind, [])
        if not members:
            continue
        box = f"G_{kind}"
        parts.nodes += _subgraph(box, BOX_TITLES[kind], (form.line for form in members))
        parts.edges.append(f"{_OPERATOR_ROOT} --- {box}")
        parts.styles.append(f"style {box} {BOX_STYLES[kind]}")
    for form in by_kind.get(None, []):
        parts.nodes.append(form.line)
        parts.edges.append(f"{_OPERATOR_ROOT} --- {form.node}")


# --- "Used with" ---


def used_with(chapter: Chapter, op: Operator, cat: Catalogue) -> list[UsedWith]:
    """The operators that ``op`` is used with, each once: what its "Used with" box is made of.

    First come the ones that ``op`` links to, in the order of its links, then the ones that link
    to it, in the order of the catalogue. The label of a link is shown as it is at the end that
    declares it, and as its reverse at the other end. An operator that has several links with
    ``op`` has all their labels, without repeats. Raises ``ValueError`` for a link of ``op`` to a
    target that is not in ``cat``, and for a link that points to ``op`` with a label that has no
    reverse.
    """
    owner = f"{chapter.id}.{op.id}"
    targets = _targets(cat)
    found: dict[str, tuple[Chapter, Operator, list[str]]] = {}

    def add(address: str, other_chapter: Chapter, other: Operator, label: str) -> None:
        labels = found.setdefault(address, (other_chapter, other, []))[2]
        if label not in labels:
            labels.append(label)

    for link in op.links:
        target_chapter, target = _resolve(targets, link.target, owner)
        add(link.target, target_chapter, target, link.label)
    for other_chapter in cat.chapters:
        for other in other_chapter.operators:
            address = f"{other_chapter.id}.{other.id}"
            for link in other.links:
                if link.target == owner:
                    add(address, other_chapter, other, _reversed(address, link.label))
    return [
        UsedWith(found_chapter, found_op, tuple(labels))
        for found_chapter, found_op, labels in found.values()
    ]


def _reversed(owner: str, label: str) -> str:
    """``label``, a link of ``owner``, as its target shows it; ``owner`` is named if it has none."""
    try:
        return reverse_label(label)
    except KeyError:
        raise ValueError(f"{owner}: link label '{label}' has no reverse label") from None


@dataclass(frozen=True)
class _Shown:
    """A node of the "Used with" box: its id, its label, the page that it links to and its tooltip.

    The label and the tooltip are escaped. A node for a block has no tooltip.
    """

    node: str
    label: str
    target: str
    tooltip: str | None = None


def _shown(entries: Sequence[UsedWith], cat: Catalogue) -> list[_Shown]:
    """The nodes of the "Used with" box of ``entries``; none for no entry, and then no box.

    Each entry is a node, up to ``USED_WITH_MAX`` of them. With more there is a node for each block.
    """
    return _by_block(entries, cat) if len(entries) > USED_WITH_MAX else _by_operator(entries)


def _used_with_box(shown: Sequence[_Shown], parts: _Parts) -> None:
    """The "Used with" box of the nodes ``shown``, if any, with a dotted line to the operator."""
    if not shown:
        return
    nodes = [_node(each.node, each.label, "ext") for each in shown]
    parts.nodes += _subgraph(_USED_WITH, "Used with", nodes)
    parts.edges.append(f"{_OPERATOR_ROOT} -.- {_USED_WITH}")
    parts.styles.append(f"style {_USED_WITH} {USED_WITH_STYLE}")
    parts.clicks += [_click(each.node, each.target, each.tooltip) for each in shown]


def _by_operator(entries: Sequence[UsedWith]) -> list[_Shown]:
    """A node for each entry: the short name of the operator, an arrow and the labels of its links.

    It links to the page of the operator, and its tooltip is the name and the symbol of it.
    """
    return [
        _Shown(
            node_id("used", entry.chapter.id, entry.operator.id),
            f"{escape_label(_short(entry.operator))} ↗"
            f"<br/><small>{escape_label(' · '.join(entry.labels))}</small>",
            f"../{entry.chapter.id}/{entry.operator.id}{PAGE_SUFFIX}",
            escape_label(_title(entry.operator)),
        )
        for entry in entries
    ]


def _by_block(entries: Sequence[UsedWith], cat: Catalogue) -> list[_Shown]:
    """A node for each block that has entries, in the order of ``cat``: its title and how many."""
    sizes = Counter(entry.chapter.id for entry in entries)
    return [
        _Shown(
            node_id("used", chapter.id),
            f"{escape_label(chapter.title)} ({sizes[chapter.id]})",
            f"../{chapter.id}/index{PAGE_SUFFIX}",
        )
        for chapter in cat.chapters
        if chapter.id in sizes
    ]
