"""Mermaid flowcharts of the operator map: chapters, type flows, the type lattice, the overview.

Every block starts with ``flowchart LR`` and the same ``classDef`` lines, so a block can be
pasted into a fence on its own. A node is coloured by the status of what it stands for and
starts with its emoji. Nothing here needs the run to be complete: a key that the ``StatusIndex``
does not have is a node with no emoji and no status class, never an error.

* ``chapter_blocks``: one unit per operator (the operator, its variants, the types it takes and
  gives, and its links to other operators), packed in order into as few blocks as the limits of
  Mermaid allow. The type hubs ``in_<type>`` and ``out_<type>`` and the stubs of the operators it
  links to are defined once per block.
* ``type_flow_block``: the operators that give a type, the type, the operators that take it, and
  its subtypes.
* ``lattice_block`` and ``overview_block``: the types, and the chapters with their types.

The ``click`` targets are relative to ``docs/operator_map``, where the pages of the map are. An
operator or a variant links to the folder of its first example, a chapter or a link stub to
the page of the chapter, ``<chapter>`` and ``PAGE_SUFFIX``: the pages are part of the Sphinx
docs, so that is the page of the built site. A type, wherever it is drawn (a hub, a subtype, a
node of the lattice or of the overview), links to its first example, or to the first example of
the operator or the variant it ``see``s, and has no link if it has neither.

The output follows the order of the catalogue, so a rerun gives the same text.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, get_args

from tool.examples import example_dir
from tool.model import Catalogue, Chapter, Operator, Status, TypeDef, Variant
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
_STATUSES: Final[tuple[Status, ...]] = get_args(Status)
_EDGE_MARKS: Final = ("-->", "---", ".->")  # a line that has one of these is an edge

# The text is black on the light fills, so that it also reads on a dark theme.
_CLASSES: Final = (
    "classDef ok fill:#d4edda,stroke:#28a745,color:#000",
    "classDef partial fill:#fff3cd,stroke:#ffc107,color:#000",
    "classDef missing fill:#f8d7da,stroke:#dc3545,color:#000",
    "classDef diverge fill:#cce5ff,stroke:#007bff,color:#000",
    "classDef type fill:#eeeeee,color:#000",
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


def _circle(node: str, label: str, css: str | None) -> str:
    return f'{node}(("{label}")){_css(css)}'


def _click(node: str, target: str) -> str:
    return f'click {node} "{target}"'


def _label(status: Status | None, *parts: str, pinned: bool = False) -> str:
    """The text of a node: the emoji of its status, the escaped ``parts`` and the pin."""
    words = [EMOJI[status]] if status is not None else []
    words.extend(escape_label(part) for part in parts if part)
    if pinned:
        words.append(_PIN)
    return " ".join(words)


def _type_node(type_id: str) -> str:
    """The node of a type in the lattice and the overview, where chapters have nodes too."""
    return node_id(f"type_{type_id}")


def _hub(
    shared: dict[str, str],
    direction: str,
    type_id: str,
    type_folder: Callable[[str], str | None],
) -> str:
    """The id of the hub ``in_<type>`` or ``out_<type>``, defined in ``shared`` if it is not.

    The definition is followed by the ``click`` to the folder of the type, if it has one.
    """
    hub = node_id(f"{direction}_{type_id}")
    shared.setdefault(hub, _circle(hub, escape_label(type_id), "type"))
    folder = type_folder(type_id)
    if folder is not None:
        shared.setdefault(f"click {hub}", _click(hub, folder))
    return hub


# --- What the catalogue says ---


def _pinned(variants: Sequence[Variant]) -> bool:
    """Whether an example of ``variants`` follows a recorded team decision."""
    return any(example.decision is not None for variant in variants for example in variant.examples)


def _returned(operator: Operator) -> tuple[str, ...]:
    """The types that ``operator`` gives: its own ``returns`` and those of its variants."""
    from_variants = (type_id for variant in operator.variants for type_id in variant.returns)
    return tuple(dict.fromkeys([*operator.returns, *from_variants]))


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
    shared: tuple[tuple[str, str], ...]  # (key, line) of the hubs and stubs, once per block


def chapter_blocks(chapter: Chapter, cat: Catalogue, index: StatusIndex) -> list[str]:
    """The blocks of ``chapter``: its operators in order, as many to a block as the limits allow.

    An operator is never split across blocks. Its links point to stubs of the operators of
    ``cat`` that they name; a link to an operator that is not in ``cat`` is a ``ValueError``.
    A single operator that is over a limit by itself raises ``MermaidLimitError``. The hubs link
    to the example of their type, which is looked for only for the types that the chapter uses.
    """
    targets = _targets(cat)
    type_folder = _type_folders(cat)
    units = [
        _unit(chapter, operator, targets, type_folder, index) for operator in chapter.operators
    ]
    return _pack(units)


def _unit(
    chapter: Chapter,
    operator: Operator,
    targets: Mapping[str, tuple[Chapter, Operator]],
    type_folder: Callable[[str], str | None],
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
    for type_id in dict.fromkeys(operator.accepts):
        edges.append(f"{_hub(shared, 'in', type_id, type_folder)} --> {op_node}")
    for type_id in dict.fromkeys(operator.returns):
        edges.append(f"{op_node} --> {_hub(shared, 'out', type_id, type_folder)}")
    for variant, form_node in forms:
        for type_id in dict.fromkeys(variant.returns):
            edges.append(f"{form_node} --> {_hub(shared, 'out', type_id, type_folder)}")
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


# --- Types and the overview ---


def type_flow_block(type_id: str, cat: Catalogue, index: StatusIndex) -> str:
    """The flow of a type: its producers, the type ``t``, its consumers and its subtypes.

    A producer is an operator whose ``returns``, or those of any of its variants, hold the type,
    and a consumer one whose ``accepts`` do. An operator is a node once, with an edge for each
    way it uses the type. The nodes are the names of the operators, coloured by their status,
    and each links to the folder of its first example, as in a chapter block; the type and its
    subtypes link to the example of their type. Raises ``MermaidLimitError`` if the flow is over
    a limit: a flow is a single block.
    """
    children = [typedef.id for typedef in cat.types if type_id in typedef.parents]
    nodes = [_circle("t", escape_label(type_id), "type")]
    subs = {child: node_id(f"sub_{child}") for child in children}
    nodes.extend(_circle(sub, escape_label(child), "type") for child, sub in subs.items())
    edges = [f'{sub} -. "subtype of" .-> t' for sub in subs.values()]
    consumers: list[str] = []
    types = [("t", type_id), *((sub, child) for child, sub in subs.items())]
    clicks = _type_clicks(types, _type_folders(cat))
    for chapter in cat.chapters:
        for operator in chapter.operators:
            makes = type_id in _returned(operator)
            takes = type_id in operator.accepts
            if not (makes or takes):
                continue
            node = node_id(chapter.id, operator.id)
            status = index.operators.get(f"{chapter.id}/{operator.id}")
            nodes.append(_box(node, _label(status, operator.name), status))
            folder = _operator_folder(chapter.id, operator)
            if folder is not None:
                clicks.append(_click(node, folder))
            if makes:
                edges.append(f"{node} --> t")
            if takes:
                consumers.append(f"t --> {node}")
    block = _assemble([*nodes, *edges, *consumers, *clicks])
    return _checked(block, f"type flow of {type_id}")


def lattice_block(cat: Catalogue, index: StatusIndex) -> str:
    """The lattice of the types: ``child --> parent``, coloured by the status of each type.

    A parent that has no ``TypeDef`` is drawn too, with no status. Raises ``MermaidLimitError``
    if the lattice is over a limit.
    """
    parents = (parent for typedef in cat.types for parent in typedef.parents)
    types = dict.fromkeys([*(typedef.id for typedef in cat.types), *parents])
    nodes: list[str] = []
    for type_id in types:
        status = index.operators.get(f"types/{type_id}")
        nodes.append(_circle(_type_node(type_id), _label(status, type_id), status or "type"))
    edges = [
        f"{_type_node(typedef.id)} --> {_type_node(parent)}"
        for typedef in cat.types
        for parent in typedef.parents
    ]
    clicks = _type_clicks([(_type_node(type_id), type_id) for type_id in types], _type_folders(cat))
    return _checked(_assemble([*nodes, *edges, *clicks]), "type lattice")


def _type_folder(
    typedef: TypeDef | None, targets: Mapping[str, tuple[Chapter, Operator]]
) -> str | None:
    """The folder a type links to: its first example, else that of what it ``see``s."""
    if typedef is None:
        return None
    own = _first_example(typedef.variants)
    if own is not None:
        return _folder("types", typedef.id, *own)
    if typedef.see is None:
        return None
    return _seen_folder(targets, typedef.see, f"types.{typedef.id}")


def _seen_folder(
    targets: Mapping[str, tuple[Chapter, Operator]], address: str, owner: str
) -> str | None:
    """The folder of the first example of ``address``: ``chapter.operator`` or a variant of it.

    A variant is ``chapter.operator.variant``. One that the operator does not have raises a
    ``ValueError``, as an operator that is not in ``targets`` does.
    """
    parts = address.split(".")
    chapter, operator = _resolve(targets, ".".join(parts[:2]), owner)
    if len(parts) <= 2:
        return _operator_folder(chapter.id, operator)
    variant = next((each for each in operator.variants if [each.id] == parts[2:]), None)
    if variant is None:
        raise ValueError(f"{owner}: target '{address}' not found")
    first = _first_example((variant,))
    return None if first is None else _folder(chapter.id, operator.id, *first)


def _type_folders(cat: Catalogue) -> Callable[[str], str | None]:
    """The function that gives the folder a type of ``cat`` links to, or None.

    The ``see`` of a type is looked for when the type is asked for, so only the types that are
    drawn need a target that exists. A type that is not in ``cat`` has no folder.
    """
    typedefs = {typedef.id: typedef for typedef in cat.types}
    targets = _targets(cat)

    def type_folder(type_id: str) -> str | None:
        return _type_folder(typedefs.get(type_id), targets)

    return type_folder


def _type_clicks(
    types: Iterable[tuple[str, str]], type_folder: Callable[[str], str | None]
) -> list[str]:
    """The ``click`` lines of the nodes of types, given as ``(node, type id)``.

    A type that has no folder to link to gets no line.
    """
    folders = [(node, type_folder(type_id)) for node, type_id in types]
    return [_click(node, folder) for node, folder in folders if folder is not None]


def overview_block(cat: Catalogue, index: StatusIndex) -> str:
    """One node per chapter, with how many of its variants are in each status.

    Each chapter has an edge to every type that its operators, or their variants, return. A
    chapter links to its page and a type to the example of the type. Raises
    ``MermaidLimitError`` if the overview is over a limit.
    """
    nodes: list[str] = []
    edges: list[str] = []
    clicks: list[str] = []
    type_nodes: dict[str, str] = {}  # the node of each type, in the order the types are met
    for chapter in cat.chapters:
        node = node_id("chapter", chapter.id)
        status = index.chapters.get(chapter.id)
        label = f"{_label(status, chapter.title)}<br/>{_tally(chapter, index)}"
        nodes.append(_box(node, label, status))
        clicks.append(_click(node, f"{chapter.id}{PAGE_SUFFIX}"))
        for type_id in dict.fromkeys(
            t for operator in chapter.operators for t in _returned(operator)
        ):
            hub = type_nodes.setdefault(type_id, _type_node(type_id))
            edges.append(f"{node} --> {hub}")
    hubs = [_circle(hub, escape_label(type_id), "type") for type_id, hub in type_nodes.items()]
    types = [(hub, type_id) for type_id, hub in type_nodes.items()]
    clicks.extend(_type_clicks(types, _type_folders(cat)))
    return _checked(_assemble([*nodes, *hubs, *edges, *clicks]), "overview")


def _tally(chapter: Chapter, index: StatusIndex) -> str:
    """``🟢 n · 🟡 n · 🔴 n · 🔵 n``: how many variants of ``chapter`` are in each status."""
    keys = (
        f"{chapter.id}/{operator.id}/{variant.id}"
        for operator in chapter.operators
        for variant in operator.variants
    )
    tally = counts([index.variants[key] for key in keys if key in index.variants])
    return " · ".join(f"{EMOJI[status]} {tally[status]}" for status in _STATUSES)
