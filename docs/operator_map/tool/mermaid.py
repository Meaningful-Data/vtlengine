"""Mermaid flowcharts of the operator map: the chapters and the overview.

Every block starts with ``flowchart LR`` and the same ``classDef`` lines, so a block can be
pasted into a fence on its own. A node is coloured by the status of what it stands for and
starts with its emoji. Nothing here needs the run to be complete: a key that the ``StatusIndex``
does not have is a node with no emoji and no status class, never an error. No diagram has a node
for a type: the forms carry their types (``tool.model.form_types``).

* ``chapter_blocks``: one unit per operator (the operator, its variants, and its links to other
  operators), packed in order into as few blocks as the limits of Mermaid allow. The stubs of
  the operators it links to are defined once per block.
* ``overview_block``: the chapters, with how many of their variants are in each status.

The ``click`` targets are relative to ``docs/operator_map``, where the pages of the map are. An
operator or a variant links to the folder of its first example, a chapter or a link stub to
the page of the chapter, ``<chapter>`` and ``PAGE_SUFFIX``: the pages are part of the Sphinx
docs, so that is the page of the built site.

The output follows the order of the catalogue, so a rerun gives the same text.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from tool.examples import example_dir
from tool.model import Catalogue, Chapter, Operator, Status, Variant
from tool.status import EMOJI, StatusIndex, counts

# The defaults of Mermaid (``maxEdges`` and ``maxTextSize``). A block stays strictly under both,
# as the design asks, so a block never sits on a limit.
MAX_EDGES: Final = 500
MAX_CHARS: Final = 50_000

# What follows the id of a chapter in the target of a click, and in the label of a link stub:
# its page in the built site. The limits above count it, as it is part of the text of a block.
PAGE_SUFFIX: Final = ".html"

_PIN: Final = "📌"  # a node whose expected output follows a recorded team decision
_INDENT: Final = "    "
_EXAMPLES: Final = Path("examples")
# The statuses of the tally of a node, which are the four that the classes below colour: the
# diagrams do not know ``refused`` yet, and a tally that lists it would change every label.
_STATUSES: Final[tuple[Status, ...]] = ("ok", "partial", "missing", "diverge")
_EDGE_MARKS: Final = ("-->", "---", ".->")  # a line that has one of these is an edge

# The text is black on the light fills, so that it also reads on a dark theme.
_CLASSES: Final = (
    "classDef ok fill:#d4edda,stroke:#28a745,color:#000",
    "classDef partial fill:#fff3cd,stroke:#ffc107,color:#000",
    "classDef missing fill:#f8d7da,stroke:#dc3545,color:#000",
    "classDef diverge fill:#cce5ff,stroke:#007bff,color:#000",
    "classDef ext stroke-dasharray: 5 5",
)

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


class MermaidLimitError(Exception):
    """A block is over a limit of Mermaid, or a single operator is, so it cannot be split."""


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

    The edges are the lines with ``-->``, ``---`` or ``.->``, so a block holds one edge per line.
    The length is counted in UTF-16 code units, as Mermaid does: a status emoji is two.
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


# --- Lines ---


def _assemble(lines: Sequence[str]) -> str:
    """A block: the diagram type, the classes, then ``lines``, all but the first indented."""
    return "\n".join(["flowchart LR", *(f"{_INDENT}{line}" for line in (*_CLASSES, *lines))])


def _css(name: str | None) -> str:
    return "" if name is None else f":::{name}"


def _box(node: str, label: str, css: str | None) -> str:
    return f'{node}["{label}"]{_css(css)}'


def _click(node: str, target: str) -> str:
    return f'click {node} "{target}"'


def _label(status: Status | None, *parts: str, pinned: bool = False) -> str:
    """The text of a node: the emoji of its status, the escaped ``parts`` and the pin."""
    words = [EMOJI[status]] if status is not None else []
    words.extend(escape_label(part) for part in parts if part)
    if pinned:
        words.append(_PIN)
    return " ".join(words)


# --- What the catalogue says ---


def _pinned(variants: Sequence[Variant]) -> bool:
    """Whether an example of ``variants`` follows a recorded team decision."""
    return any(example.decision is not None for variant in variants for example in variant.examples)


def _first_example(variants: Sequence[Variant]) -> tuple[str, str] | None:
    """The variant id and the example name of the first example of ``variants``, if any."""
    return next(
        ((variant.id, example.name) for variant in variants for example in variant.examples),
        None,
    )


def _folder(chapter_id: str, operator_id: str, variant_id: str, example_name: str) -> str:
    """The folder of an example, relative to ``docs/operator_map``."""
    folder = example_dir(_EXAMPLES, chapter_id, operator_id, variant_id, example_name)
    return f"{folder.as_posix()}/"


def _operator_folder(chapter_id: str, operator: Operator) -> str | None:
    """The folder of the first example of ``operator``, if it has one."""
    first = _first_example(operator.variants)
    return None if first is None else _folder(chapter_id, operator.id, *first)


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


# --- Chapters ---


@dataclass(frozen=True)
class _Unit:
    """One operator of a chapter as lines of a block, and the nodes it shares with the others."""

    owner: str  # ``<chapter>.<operator>``
    lines: tuple[str, ...]
    shared: tuple[tuple[str, str], ...]  # (key, line) of the stubs, once per block


def chapter_blocks(chapter: Chapter, cat: Catalogue, index: StatusIndex) -> list[str]:
    """The blocks of ``chapter``: its operators in order, as many to a block as the limits allow.

    An operator is never split across blocks. Its links point to stubs of the operators of
    ``cat`` that they name; a link to an operator that is not in ``cat`` is a ``ValueError``.
    A single operator that is over a limit by itself raises ``MermaidLimitError``.
    """
    targets = _targets(cat)
    return _pack([_unit(chapter, operator, targets, index) for operator in chapter.operators])


def _unit(
    chapter: Chapter,
    operator: Operator,
    targets: Mapping[str, tuple[Chapter, Operator]],
    index: StatusIndex,
) -> _Unit:
    """The nodes, the edges and the clicks of ``operator``, in that order."""
    owner = f"{chapter.id}.{operator.id}"
    key = f"{chapter.id}/{operator.id}"
    op_node = node_id(chapter.id, operator.id)
    status = index.operators.get(key)
    label = _label(status, operator.name, operator.symbol, pinned=_pinned(operator.variants))
    nodes = [_box(op_node, label, status)]
    forms: list[tuple[Variant, str]] = []
    for variant in operator.variants:
        form_node = node_id(chapter.id, operator.id, variant.id)
        form_status = index.variants.get(f"{key}/{variant.id}")
        form_label = _label(form_status, variant.label, pinned=_pinned((variant,)))
        nodes.append(_box(form_node, form_label, form_status))
        forms.append((variant, form_node))

    shared: dict[str, str] = {}
    edges = [f"{op_node} --- {form_node}" for _, form_node in forms]
    for link in operator.links:
        target_chapter, target = _resolve(targets, link.target, owner)
        stub = node_id(f"ext_{target_chapter.id}", target.id)
        page = f"{target_chapter.id}{PAGE_SUFFIX}"
        shared.setdefault(stub, _box(stub, escape_label(f"↗ {target.name} ({page})"), "ext"))
        shared.setdefault(f"click {stub}", _click(stub, page))
        edges.append(f'{op_node} -. "{escape_label(link.label)}" .-> {stub}')

    clicks = _unit_clicks(chapter, operator, op_node, forms)
    return _Unit(owner, (*nodes, *edges, *clicks), tuple(shared.items()))


def _unit_clicks(
    chapter: Chapter, operator: Operator, op_node: str, forms: Sequence[tuple[Variant, str]]
) -> list[str]:
    """The ``click`` lines of an operator and its variants: the first example of each."""
    clicks: list[str] = []
    operator_folder = _operator_folder(chapter.id, operator)
    if operator_folder is not None:
        clicks.append(_click(op_node, operator_folder))
    for variant, form_node in forms:
        if variant.examples:
            folder = _folder(chapter.id, operator.id, variant.id, variant.examples[0].name)
            clicks.append(_click(form_node, folder))
    return clicks


def _chapter_text(units: Sequence[_Unit]) -> str:
    """The block of ``units``: the nodes they share, once each, then their own lines."""
    shared: dict[str, str] = {}
    for unit in units:
        for key, line in unit.shared:
            shared.setdefault(key, line)
    return _assemble([*shared.values(), *(line for unit in units for line in unit.lines)])


def _pack(units: Sequence[_Unit]) -> list[str]:
    """The blocks of ``units`` in order, each as full as the limits allow."""
    blocks: list[str] = []
    group: list[_Unit] = []
    text = ""
    for unit in units:
        candidate = _chapter_text([*group, unit])
        if _limit_problem(candidate) is None:
            group.append(unit)
            text = candidate
            continue
        if group:
            blocks.append(text)
        group = [unit]
        text = _checked(_chapter_text(group), unit.owner)  # alone, it cannot be split further
    if group:
        blocks.append(text)
    return blocks


# --- The overview ---


def overview_block(cat: Catalogue, index: StatusIndex) -> str:
    """One node per chapter, with how many of its variants are in each status.

    A chapter links to its page. Raises ``MermaidLimitError`` if the overview is over a limit.
    """
    nodes: list[str] = []
    clicks: list[str] = []
    for chapter in cat.chapters:
        node = node_id("chapter", chapter.id)
        status = index.chapters.get(chapter.id)
        label = f"{_label(status, chapter.title)}<br/>{_tally(chapter, index)}"
        nodes.append(_box(node, label, status))
        clicks.append(_click(node, f"{chapter.id}{PAGE_SUFFIX}"))
    return _checked(_assemble([*nodes, *clicks]), "overview")


def _tally(chapter: Chapter, index: StatusIndex) -> str:
    """``🟢 n · 🟡 n · 🔴 n · 🔵 n``: how many variants of ``chapter`` are in each status."""
    keys = (
        f"{chapter.id}/{operator.id}/{variant.id}"
        for operator in chapter.operators
        for variant in operator.variants
    )
    tally = counts([index.variants[key] for key in keys if key in index.variants])
    return " · ".join(f"{EMOJI[status]} {tally[status]}" for status in _STATUSES)
