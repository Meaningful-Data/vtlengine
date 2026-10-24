"""reStructuredText of the operator map: the index, a page per block and a page per operator.

``index_rst``, ``block_rst`` and ``operator_rst`` take the catalogue and the ``StatusIndex`` of a
run and return the text of a page. The index is ``docs/operator_map/index.rst``, the page of a
block is ``<chapter>/index.rst`` and the page of an operator ``<chapter>/<operator>.rst``: each is
in the hidden ``toctree`` of the page above it, so every operator has a URL of its own. Nothing
here runs an engine. The page of an operator reads the files of its examples, in
``examples_root`` (``docs/operator_map/examples`` unless it is given): which there are, and the
data of their CSV and JSON files.

**Operator page.** The title (``Name · symbol``), a chip with the status of the operator and a
link to its page of the manual; the legend (``LEGEND_HTML``) and, when the diagram has boxes, the
key of their borders (``BORDER_KEY_HTML``); the diagram of the operator, and under it the full
list of what it is used with when the "Used with" box of the diagram is grouped by block. Then:

* **Types**: what its forms take and give, and two columns, "Comes from" and "Goes to", with the
  operators that give what it takes and those that take what it gives (``tool.typelinks``),
  grouped by block in collapsible lists. The operators that return the type of their input, and
  those that take any scalar, are a closed list of their own. An operator that takes any scalar
  is fed by any operator that returns a value, and one that returns the type of its input feeds
  whatever takes that type: each says so in one line instead of a list.
* **Forms**: a section per form, in the order of the diagram (``forms_in_diagram_order``), headed
  by its label, its status and its types. Each example is a collapsible block: its title, its
  status and the mark of each engine (``tool.status.mark``), then its notes
  (``.. include::``), its script (``.. literalinclude::``), its data, its expected scalars or
  error, and links to its files and to its own page in the built site.

The data are HTML tables made here from the CSV files, not ``csv-table``: docutils reads a cell of
a ``csv-table`` as RST, and Sphinx typesets it, so ``x"y`` would show ``x”y``, ``*a*`` an emphasis
and ``-`` a list; the spaces around a value would be lost, and a file with a header only would not
build. A cell of an ``om-data`` table is escaped HTML, and keeps its spaces. The anchor of a form
(``form_anchor``) is an HTML id written before the heading of the form, not an RST target: a
target is a label of Sphinx, which is global to the docs, and the same form ids on many pages
(``ds_ds``...) would each be a "duplicate label" warning.

**Block page.** The title and the status chip, the legend, the diagram of the block, a table of
its operators, a table of the types that they take and give, the forms that are not green with
the example that shows why, and the hidden ``toctree`` of the pages of its operators.

**Index.** The title and the introduction, the hidden ``toctree`` of the block pages, a line on
how the results were made, the legend, how to read the map, the overview diagram, a table of the
blocks, the codes that say that something is not implemented (``NOT_IMPLEMENTED``), the lists of
team decisions, errata and adaptations, which link to the examples on the pages of their
operators, and the attribution.

**Diagrams.** They come from ``tool.mermaid``, in ``.. mermaid::`` directives that ask Mermaid
for their natural size (``useMaxWidth`` off), inside an ``om-diagram`` block. The docs set a
width of at most 700 px for a diagram, which makes the text of a wide one too small to read: the
CSS of the map (``docs/_static/operator_map.css``) lets these diagrams take their natural width,
and a diagram wider than the page scrolls in its block. The links between the pages are ``:doc:``
roles, and the clicks of the diagrams, the links to an anchor and the links to the files of the
examples are relative URLs of the built site.

**Escaping.** The text that comes from the catalogue or from the engines is free text: ``_inline``
and ``_text`` escape what RST would read as markup, and what Sphinx would typeset (see
``_TYPESET``), so that a message of an engine reads as it was written. ``_literal`` puts a piece
of VTL in an inline literal. Inside raw HTML, the text is escaped with ``html.escape``.
"""

from __future__ import annotations

import csv
import html
import json
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final
from urllib.parse import quote

from tool.examples import example_dir
from tool.mermaid import (
    USED_WITH_MAX,
    block_diagram,
    example_anchor,
    form_anchor,
    forms_in_diagram_order,
    has_boxes,
    operator_diagram,
    overview_block,
    used_with,
)
from tool.model import (
    ANY_SCALAR,
    ANY_TYPE,
    STRUCTURAL_TYPES,
    Catalogue,
    Chapter,
    Example,
    Operator,
    Status,
    Variant,
    ViralGroup,
    example_title,
    form_types,
    load_viral_groups,
)
from tool.status import (
    EMOJI,
    NOT_IMPLEMENTED,
    EngineVerdict,
    ExampleResult,
    StatusIndex,
    counts,
    mark,
    rollup,
)
from tool.typelinks import Ref, TypeLinks, block_types_table, by_block, type_links

MANUAL_BASE: Final = "https://sdmx-twg.github.io/vtl/v2.2/reference_manual"
SPEC_REPO: Final = "https://github.com/sdmx-twg/vtl"
ISSUES: Final = "https://github.com/Meaningful-Data/vtlengine/issues"
MAX_DETAIL: Final = 200  # characters of what went wrong with an example, the ellipsis included
DEFAULT_EXAMPLES_ROOT: Final = Path(__file__).resolve().parents[1] / "examples"

# The manual pages that are not under ``operators/``: those of the VTL-DL, and the typical
# behaviours of the operators, which the groups of ``catalogue.viral`` quote.
_DEFINITION_PAGES: Final = {
    "dl/udo": "vtl_dl_udo.html",
    "dl/datapoint": "vtl_dl_rulesets/datapoint_rulesets.html",
    "dl/hierarchical": "vtl_dl_rulesets/hierarchical_rulesets.html",
    "dl/viral": "vtl_dl_rulesets/viral_attributes.html",
    "typical": "typical_behaviour.html",
}
_PAGE_TITLES: Final = {"typical": "Typical behaviours of the ML operators"}
_VIRAL_PAGE: Final = "dl/viral"  # the page of ``define viral propagation``

_STATUSES: Final[tuple[Status, ...]] = ("ok", "partial", "refused", "missing", "diverge")
# What each status means, as the legend says it, and the tooltip of a chip.
_MEANING: Final[dict[str, str]] = {
    "ok": "all ok",
    "partial": "partially implemented",
    "refused": "refused with an error",
    "missing": "not implemented or crash",
    "diverge": "engines differ",
}
_MARK_KEY: Final = "✅ match · ❌ wrong result · 🛡 guarded error (code) · 💥 crash"

# The legend of the statuses and of the marks, above every diagram. A status is a chip of its
# colour; ``refused`` is the green of ``ok`` with a dotted border, as in the diagrams.
LEGEND_HTML: Final = (
    '<div class="om-legend">'
    + "".join(
        f'<span class="om-chip {status}">{EMOJI[status]} {_MEANING[status]}</span>'
        for status in _STATUSES
    )
    + f' <span class="om-key">{_MARK_KEY}</span></div>'
)
# The key of the borders of the boxes of an operator diagram, above the diagrams that have them.
BORDER_KEY_HTML: Final = (
    '<div class="om-legend om-borders">'
    '<span class="om-box solid">Dataset forms</span>'
    '<span class="om-box dashed">Component forms</span>'
    '<span class="om-box dotted">Scalar forms</span>'
    "</div>"
)

# The adornments of the titles: a page, a section, and a section of a section.
_PAGE: Final = "="
_SECTION: Final = "-"
_SUBSECTION: Final = "~"
_INDENT: Final = "   "  # the content of a directive, and the options of a table

# The examples, from the page of a block or of an operator: the docs build copies them next to
# the pages, ``docs/operator_map/examples``.
_EXAMPLES: Final = "../examples"
# A diagram is drawn at its natural size: the CSS of the map scrolls one that is too wide.
_DIAGRAM_CONFIG: Final = '{"flowchart": {"useMaxWidth": false}}'
# The files of an example in the order a reader takes them: the script, its inputs, what it
# expects, the other inputs of ``run``, then anything else. The notes are on the page already.
_FILE_ORDER: Final = ("script.vtl", "input", "expected", "value_domains", "external_routines")
_NOT_LINKED: Final = frozenset({"notes.rst", "notes.md"})
# How a list of types names the types that stand for many: ``scalar`` and ``any``.
_TYPE_NAMES: Final = {ANY_SCALAR: "any scalar", ANY_TYPE: "any type"}
# The takes of an operator that are folded: it takes any value.
_FOLDED_TAKES: Final = frozenset((ANY_SCALAR, ANY_TYPE))

_NEWLINE: Final = re.compile(r"\r\n|\r|\n")
# What RST reads as markup in a line of text: emphasis and strong, literals and roles, a
# substitution, the end of a reference or a target, and the escape itself.
_MARKUP: Final = re.compile(r"([\\*`|_])")
# What the smart quotes of Sphinx would change in a text that is not a literal: the straight
# quotes, a run of hyphens (a dash) and an ellipsis, ``...`` or ``. . .``. Every character of such
# a sequence is escaped, which keeps it as it was written and leaves the rest of the text alone.
# A hyphen or a dot right after a backslash of the text is escaped too: the SmartQuotes of
# docutils, which the checks parse with, drops that backslash otherwise (its ``get_tokens``
# pattern). Sphinx's own transform keeps it, and the escape does no harm there.
_TYPESET: Final = re.compile(r"""['"]|-{2,}|\.{3,}|\.(?: \.){2}|(?<=\\)[-.]""")
_TYPESET_CHARACTER: Final = re.compile(r"""['".-]""")
# A paragraph that starts like a list, a field, a table, a comment... is read as one, unless its
# first character is escaped.
_ENUMERATOR: Final = re.compile(r"\(?(?:#|\d+|[A-Za-z]|[ivxlcdmIVXLCDM]+)[.)](?: |$)")


@dataclass(frozen=True)
class RunMeta:
    """When and with what the results were made."""

    date: str
    vtlengine_commit: str
    vtlengine_version: str
    spec_commit: str  # the commit of the manual, in sdmx-twg/vtl
    pandas_version: str  # the engines, as the children of the run imported them
    duckdb_version: str


def manual_url(page: str) -> str:
    """The URL of a manual page: ``<Chapter>/<Page>``, a definition, ``dl/...``, or ``typical``.

    A page of an operator lives under ``operators/`` with its spaces encoded, the pages of the
    definitions are the ``vtl_dl_*`` pages, and ``typical`` is the page of the typical behaviours
    of the operators.
    """
    if page in _DEFINITION_PAGES:
        return f"{MANUAL_BASE}/{_DEFINITION_PAGES[page]}"
    return f"{MANUAL_BASE}/operators/{quote(page)}/index.html"


# --- The page of an operator ---


def operator_rst(
    chapter: Chapter,
    op: Operator,
    cat: Catalogue,
    index: StatusIndex,
    meta: RunMeta,
    *,
    examples_root: Path = DEFAULT_EXAMPLES_ROOT,
    viral: Sequence[ViralGroup] | None = None,
) -> str:
    """The page of ``op``, an operator of ``chapter``, with the examples in ``examples_root``.

    ``viral`` are the groups of operators that apply the viral propagation rules, the catalogue's
    (``catalogue.viral``) unless it is given. An operator of a group says so under its types, and
    the page of ``define viral propagation`` lists them all. It raises ``MermaidLimitError`` when
    the diagram cannot fit the limits of Mermaid, and ``ValueError`` when an operator that a list
    names has no status in ``index``.
    """
    groups = load_viral_groups() if viral is None else viral
    key = f"{chapter.id}/{op.id}"
    statuses = [index.variants.get(f"{key}/{variant.id}") for variant in op.variants]
    summary = f"{_all_ok(statuses)} of {_count(len(statuses), 'form')} all ok"
    title = _text(op.name) + (f" · {_literal(op.symbol)}" if op.symbol else "")
    parts = [
        _title(title, _PAGE, overline=True),
        _raw(_head(index.operators.get(key), summary, meta, manual_url(op.page))),
        _raw(LEGEND_HTML),
        *([_raw(BORDER_KEY_HTML)] if has_boxes(op) else []),
        _diagram(operator_diagram(chapter, op, cat, index)),
        *_used_with_list(chapter, op, cat, index),
        _section("Types", _raw(*_types(chapter, op, cat, index, groups))),
        *([_applied_by(cat, index, groups)] if op.page == _VIRAL_PAGE else []),
        _section(
            "Forms",
            *(
                _form_section(chapter, op, variant, index, examples_root)
                for variant in forms_in_diagram_order(op)
            ),
        ),
    ]
    return "\n\n".join(parts) + "\n"


def _head(status: Status | None, summary: str, meta: RunMeta, manual: str | None = None) -> str:
    """The line under the title: the status chip, the manual page, and what made the results."""
    chip = _chip(status, f"{EMOJI[status]} {summary}" if status is not None else summary)
    parts = [chip]
    if manual is not None:
        parts.append(f'<a href="{_attribute(manual)}">VTL 2.2 Reference Manual ↗</a>')
    engines = (
        f"vtlengine {meta.vtlengine_version} ({meta.vtlengine_commit}) · "
        f"pandas {meta.pandas_version} · DuckDB {meta.duckdb_version}"
    )
    parts.append(f'<span class="om-gen">{html.escape(engines)}</span>')
    return '<div class="om-head">' + " · ".join(parts) + "</div>"


def _all_ok(statuses: Sequence[Status | None]) -> int:
    return sum(1 for status in statuses if status == "ok")


# --- "Used with" ---


def _used_with_list(
    chapter: Chapter, op: Operator, cat: Catalogue, index: StatusIndex
) -> list[str]:
    """The full list of the operators that ``op`` is used with, when its box shows the blocks.

    The box of the diagram has a node for each block when it has more than ``USED_WITH_MAX``
    operators (``tool.mermaid``). The list is under the diagram, by block, and each operator has
    the labels of its links.
    """
    entries = used_with(chapter, op, cat)
    if len(entries) <= USED_WITH_MAX:
        return []
    labels = {(e.chapter.id, e.operator.id): " · ".join(e.labels) for e in entries}
    refs = _in_catalogue_order((_ref(e.chapter, e.operator, index) for e in entries), cat)
    return [_raw(_listing("Used with", refs, opened=True, labels=labels))]


def _ref(chapter: Chapter, op: Operator, index: StatusIndex) -> Ref:
    """``op`` as a list names it; ``ValueError`` when the index has no status for it."""
    key = f"{chapter.id}/{op.id}"
    if key not in index.operators:
        raise ValueError(f"no status for operator {key}")
    return Ref(chapter.id, chapter.title, op.id, op.name, op.symbol, index.operators[key])


def _in_catalogue_order(refs: Iterable[Ref], cat: Catalogue) -> list[Ref]:
    """``refs`` each once, in the order of the catalogue: its chapters, then their operators."""
    place = {
        (chapter.id, op.id): n
        for n, (chapter, op) in enumerate(
            (chapter, op) for chapter in cat.chapters for op in chapter.operators
        )
    }
    unique = {(ref.chapter_id, ref.operator_id): ref for ref in refs}
    return sorted(unique.values(), key=lambda ref: place.get((ref.chapter_id, ref.operator_id), 0))


# --- Types ---


def _types(
    chapter: Chapter,
    op: Operator,
    cat: Catalogue,
    index: StatusIndex,
    groups: Sequence[ViralGroup],
) -> list[str]:
    """The lines of HTML of the Types section: what it takes and gives, and the two columns.

    An operator that applies the viral propagation rules says so under the first line.
    """
    links = type_links(cat, index, chapter, op)
    takes, gives = _type_list(links.takes), _type_list(links.gives)
    applies = any(f"{chapter.id}.{op.id}" in group.operators for group in groups)
    return [
        f'<p class="om-takes"><b>Takes</b> {html.escape(takes)} · '
        f"<b>Gives</b> {html.escape(gives)}</p>",
        *([_viral_line(cat)] if applies else []),
        '<div class="om-cols"><div class="om-col">'
        + "".join(_comes_from(links, cat))
        + '</div><div class="om-col">'
        + "".join(_goes_to(links, cat))
        + "</div></div>",
    ]


def _comes_from(links: TypeLinks, cat: Catalogue) -> list[str]:
    """The column of the operators that give what this one takes."""
    values = [t for t in links.takes if t not in STRUCTURAL_TYPES and t not in _FOLDED_TAKES]
    structures = [t for t in links.takes if t in STRUCTURAL_TYPES]
    found: list[str] = []
    if not _FOLDED_TAKES.isdisjoint(links.takes):
        found.append(_line("Comes from:", "any operator that returns a value"))
    elif values:
        found += _typed(
            "Comes from",
            values,
            links.comes_from,
            (links.any_type, "operators that return any type (their input's)"),
            cat,
        )
    if structures:
        found += _typed("Comes from", structures, links.comes_from, ((), ""), cat)
    if not found:
        found.append(_line("Comes from:", "nothing: it takes no operand"))
    return found


def _goes_to(links: TypeLinks, cat: Catalogue) -> list[str]:
    """The column of the operators that take what this one gives."""
    values = [t for t in links.gives if t not in STRUCTURAL_TYPES and t != ANY_TYPE]
    structures = [t for t in links.gives if t in STRUCTURAL_TYPES]
    found: list[str] = []
    if ANY_TYPE in links.gives:
        found.append(_line("Goes to:", "whatever takes its input's type"))
    if values:
        found += _typed(
            "Goes to",
            values,
            links.goes_to,
            (links.any_scalar, "operators that take any scalar"),
            cat,
        )
    if structures:
        found += _typed("Goes to", structures, links.goes_to, ((), ""), cat)
    return found


def _typed(
    side: str,
    types: Sequence[str],
    by_type: Mapping[str, Sequence[Ref]],
    folded: tuple[Sequence[Ref], str],
    cat: Catalogue,
) -> list[str]:
    """The list of the operators of ``types``, each once, and the closed list ``folded``.

    The folded list is open when there is nothing by type to show above it.
    """
    title = f"{side} · {_type_list(types)}"
    refs = _in_catalogue_order((ref for t in types for ref in by_type.get(t, ())), cat)
    found = [_listing(title, refs, opened=True)] if refs else [_line(f"{title}:", "none")]
    extra, extra_title = folded
    if extra:
        found.append(_listing(f"{side} · {extra_title}", extra, opened=not refs))
    return found


def _type_list(types: Sequence[str]) -> str:
    """``number, integer``: the types as a page names them; ``nothing`` for none."""
    return ", ".join(_type_name(t) for t in types) or "nothing"


def _type_name(type_id: str) -> str:
    """A type as a page names it: ``any scalar``, ``any type``, ``hierarchical ruleset``."""
    if type_id in _TYPE_NAMES:
        return _TYPE_NAMES[type_id]
    return type_id.replace("_", " ") if type_id in STRUCTURAL_TYPES else type_id


def _listing(
    title: str,
    refs: Sequence[Ref],
    *,
    opened: bool,
    labels: Mapping[tuple[str, str], str] | None = None,
) -> str:
    """A collapsible list of operators, by block: how many, and the tally of their statuses.

    Each block is closed, with its count and tally, and holds a chip for each operator, which
    links to its page. ``labels`` adds a small label to the chip of an operator.
    """
    groups = by_block(refs)
    inner = "".join(
        f'<details class="om-blk"><summary>{html.escape(block)} <b>({len(group)})</b> '
        f"{_tally(ref.status for ref in group)}</summary><div>"
        + "".join(_ref_chip(ref, labels) for ref in group)
        + "</div></details>"
        for block, group in groups
    )
    count = f"{_count(len(refs), 'operator')} in {_count(len(groups), 'block')}"
    attribute = " open" if opened else ""
    return (
        f'<details class="om-dir"{attribute}><summary><b>{html.escape(title)}</b> — {count} '
        f"{_tally(ref.status for ref in refs)}</summary>{inner}</details>"
    )


def _ref_chip(ref: Ref, labels: Mapping[tuple[str, str], str] | None) -> str:
    """A chip that links to the page of an operator: its name, its symbol and its label."""
    symbol = f" <code>{html.escape(ref.symbol)}</code>" if ref.symbol else ""
    label = (labels or {}).get((ref.chapter_id, ref.operator_id))
    small = f" <small>{html.escape(label)}</small>" if label else ""
    target = f"../{ref.chapter_id}/{ref.operator_id}.html"
    return (
        f'<a class="om-chip {ref.status}" href="{_attribute(target)}" '
        f'title="{_attribute(_MEANING[ref.status])}">{html.escape(ref.name)}{symbol}{small}</a>'
    )


def _line(head: str, text: str) -> str:
    return f'<p class="om-line"><b>{html.escape(head)}</b> {html.escape(text)}</p>'


# --- Viral propagation ---


def _viral_line(cat: Catalogue) -> str:
    """The line of an operator that applies the viral propagation rules, under its types.

    It links to the page of ``define viral propagation``, when the catalogue has it.
    """
    found = _viral_definition(cat)
    target = "Viral propagation"
    if found is not None:
        chapter, op = found
        url = _attribute(f"../{chapter.id}/{op.id}.html")
        target = f'<a href="{url}">Viral propagation</a>'
    return (
        '<p class="om-viral"><b>Viral attributes:</b> combined with the propagation rules → '
        f"{target}</p>"
    )


def _viral_definition(cat: Catalogue) -> tuple[Chapter, Operator] | None:
    """The chapter and the operator of the page of ``define viral propagation``, if any."""
    for chapter in cat.chapters:
        for op in chapter.operators:
            if op.page == _VIRAL_PAGE:
                return chapter, op
    return None


def _applied_by(cat: Catalogue, index: StatusIndex, groups: Sequence[ViralGroup]) -> str:
    """The "Applied by" section of the page of ``define viral propagation``.

    The groups of operators that apply its rules, each with the sentence of the manual that says
    so, then all of them by block, as the lists of the Types section. An operator that the
    catalogue does not have is left out.
    """
    operators = {
        f"{chapter.id}.{op.id}": (chapter, op)
        for chapter in cat.chapters
        for op in chapter.operators
    }
    found = [operators[a] for group in groups for a in group.operators if a in operators]
    refs = _in_catalogue_order((_ref(chapter, op, index) for chapter, op in found), cat)
    intro = (
        "Every operator that propagates viral attributes applies the rules that this definition "
        "registers, as the manual says: "
        + _inline(
            '"e.g., the typical behaviour rules for binary, aggregate, analytic, or join operators"'
        )
        + ". These operators combine viral attributes, and the manual says so for each group:"
    )
    items = [
        f"- **{_inline(group.title)}**: {_inline(chr(34) + group.sentence + chr(34))} "
        f"({_link(_page_title(group.page), manual_url(group.page))})"
        for group in groups
    ]
    return _section(
        "Applied by", intro, "\n".join(items), _raw(_listing("Applied by", refs, opened=True))
    )


def _page_title(page: str) -> str:
    """The title of a page of the manual: its last part, or the title of a page of its own."""
    return _PAGE_TITLES.get(page, page.rsplit("/", 1)[-1])


# --- Forms and examples ---


def _form_section(
    chapter: Chapter, op: Operator, variant: Variant, index: StatusIndex, root: Path
) -> str:
    """The anchor, the heading and the examples of a form.

    The heading is the label of the form, the emoji of its status, and its types. The anchor is
    an HTML id of its own, right before the heading: the target of the click of the form in the
    diagram.
    """
    status = index.variants.get(f"{chapter.id}/{op.id}/{variant.id}")
    takes, gives = form_types(op, variant)
    emoji = f" {EMOJI[status]}" if status is not None else ""
    heading = f"{_text(variant.label)}{emoji} · {_type_list(takes)} → {_type_list(gives)}"
    parts = [
        _raw(f'<span class="om-anchor" id="{form_anchor(variant)}"></span>'),
        _title(heading, _SUBSECTION),
    ]
    for n, example in enumerate(variant.examples):
        parts.append(_example(chapter, op, variant, example, index, root, opened=n == 0))
    return "\n\n".join(parts)


def _example(
    chapter: Chapter,
    op: Operator,
    variant: Variant,
    example: Example,
    index: StatusIndex,
    root: Path,
    *,
    opened: bool,
) -> str:
    """The collapsible block of an example: open for the first example of its form."""
    names = (chapter.id, op.id, variant.id, example.name)
    folder = example_dir(root, *names)
    url = f"{_EXAMPLES}/{'/'.join(names)}"
    result = index.examples.get("/".join(names))
    attributes = f'class="om-example" id="{example_anchor(variant, example)}"'
    if opened:
        attributes += " open"
    opening = [f"<details {attributes}><summary>{_summary(example, result)}</summary>"]
    if result is not None and result.status != "ok":
        opening.append(f'<p class="om-detail">{html.escape(_detail_text(result))}</p>')
    parts = [_raw(*opening)]
    if (folder / "notes.rst").is_file():
        parts.append(f".. include:: {url}/notes.rst")
    else:
        parts.append("No notes.")
    parts.append(
        f".. literalinclude:: {url}/script.vtl\n{_INDENT}:caption: script.vtl\n"
        f"{_INDENT}:language: none"
    )
    tables = [
        *_data_tables(folder / "input", "Input"),
        *_data_tables(folder / "expected", "Expected"),
    ]
    if tables:
        parts.append(_raw(*tables))
    scalars = folder / "expected" / "scalars.json"
    if scalars.is_file():
        parts.append(_scalars(scalars))
    error = folder / "expected" / "error.json"
    if error.is_file():
        parts.append(f"Expected error {_literal(_error_code(error))}")
    links = [
        f'<a href="{_attribute(f"{url}/{quote(name)}")}">{html.escape(name)}</a>'
        for name in _files(folder)
    ]
    links.append(f'<a href="{_attribute(f"{url}/index.html")}">example page ↗</a>')
    parts.append(_raw(f'<p class="om-files">Files: {" · ".join(links)}</p>', "</details>"))
    return "\n\n".join(parts)


def _summary(example: Example, result: ExampleResult | None) -> str:
    """The summary of an example: its title, its status, and what each engine gave."""
    title = f'<span class="om-ex-title">{html.escape(example_title(example))}</span>'
    if result is None:
        return f"{title} {_chip(None, 'not run')}"
    chip = _chip(result.status, EMOJI[result.status], _MEANING[result.status])
    marks = f"pandas {_mark(result.pandas)} · DuckDB {_mark(result.duckdb)}"
    return f'{title} {chip} <span class="om-marks">{marks}</span>'


def _mark(verdict: EngineVerdict) -> str:
    """The mark of a verdict in HTML, with the code of an error or the type of a crash as code."""
    symbol, _, rest = mark(verdict).partition(" ")
    return html.escape(symbol) + (f" <code>{html.escape(rest)}</code>" if rest else "")


def _detail(result: ExampleResult) -> str:
    """What went wrong, from the details of the engines: said once if they say the same.

    An engine that has a detail on its own is named, so that it is clear which one it is. Two
    engines that agree on a verdict and still give different outputs have nothing else to say.
    """
    pandas, duckdb = result.pandas, result.duckdb
    parts: list[str] = []
    if pandas.detail and pandas.detail == duckdb.detail:
        parts.append(pandas.detail)
    else:
        if pandas.detail:
            parts.append(f"pandas: {pandas.detail}")
        if duckdb.detail:
            parts.append(f"DuckDB: {duckdb.detail}")
    if not result.engines_agree and pandas.outcome == duckdb.outcome:
        parts.append("the engines give different outputs")
    return "; ".join(parts)


def _detail_text(result: ExampleResult) -> str:
    """``_detail`` on one line, cut at ``MAX_DETAIL`` characters."""
    text = _NEWLINE.sub(" ", _detail(result))
    return text if len(text) <= MAX_DETAIL else text[: MAX_DETAIL - 1] + "…"


def _data_tables(folder: Path, what: str) -> list[str]:
    """An HTML table for each CSV file of ``folder``, by name; none if there is no folder."""
    if not folder.is_dir():
        return []
    return [_data_table(path, what) for path in sorted(folder.glob("*.csv")) if path.is_file()]


def _data_table(path: Path, what: str) -> str:
    """The data of a CSV file as a table on one line: its header, then its rows, as written.

    A file with a header only is an empty table that says it has no data points.
    """
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    header, body = (rows[0], rows[1:]) if rows else ([], [])
    caption = f"{html.escape(what)} <code>{html.escape(path.stem)}</code>"
    if not body:
        caption += " (no data points)"
    head = "".join(f'<th class="head">{_cell(cell)}</th>' for cell in header)
    lines = "".join("<tr>" + "".join(f"<td>{_cell(c)}</td>" for c in row) + "</tr>" for row in body)
    return (
        '<table class="docutils align-default om-data">'
        f'<caption><span class="caption-text">{caption}</span></caption>'
        f"<thead><tr>{head}</tr></thead><tbody>{lines}</tbody></table>"
    )


def _cell(text: str) -> str:
    """A cell as HTML on one line: escaped, with its line breaks as character references."""
    return html.escape(text).replace("\r", "&#13;").replace("\n", "&#10;")


class _Written(str):
    """A number of a JSON file, as the file writes it: ``1.50`` stays ``1.50``."""


def _scalars(path: Path) -> str:
    """The expected scalars of ``scalars.json``: name, type and value, as the file writes them.

    A string is quoted, so that its spaces show, and an empty one is ``""``.
    """
    document = json.loads(
        path.read_text(encoding="utf-8-sig"), parse_int=_Written, parse_float=_Written
    )
    rows = [
        [_literal(str(item.get("name", ""))), _text(str(item.get("type", ""))), _value(item)]
        for item in document.get("scalars", [])
    ]
    return _table(("Name", "Type", "Value"), rows, title="Expected scalars")


def _value(item: Mapping[str, Any]) -> str:
    raw = item.get("value")
    if raw is None:
        return "``null``"
    if isinstance(raw, bool):
        return "``true``" if raw else "``false``"
    if isinstance(raw, _Written):
        return _literal(str(raw))
    return _literal(json.dumps(raw, ensure_ascii=False))


def _error_code(path: Path) -> str:
    """The code of ``error.json``, ``{"code": "2-1-15-6"}``."""
    return str(json.loads(path.read_text(encoding="utf-8-sig")).get("code", ""))


def _files(folder: Path) -> list[str]:
    """The files of an example folder, by their path in it, but the notes, in reading order."""
    if not folder.is_dir():
        return []
    names = [
        path.relative_to(folder).as_posix()
        for path in folder.rglob("*")
        if path.is_file() and not path.name.startswith(".")
    ]

    def order(name: str) -> tuple[int, str]:
        first = name.split("/", 1)[0]
        place = _FILE_ORDER.index(first) if first in _FILE_ORDER else len(_FILE_ORDER)
        return place, name

    return sorted((name for name in names if name not in _NOT_LINKED), key=order)


# --- The page of a block ---


def block_rst(chapter: Chapter, cat: Catalogue, index: StatusIndex, meta: RunMeta) -> str:
    """The page of ``chapter``: its diagram, its operators, their types and what is not green.

    ``cat`` is the catalogue of the map, of which ``chapter`` is one. It raises
    ``MermaidLimitError`` when the diagram cannot fit the limits of Mermaid.
    """
    statuses = [index.operators.get(f"{chapter.id}/{op.id}") for op in chapter.operators]
    summary = f"{_all_ok(statuses)} of {_count(len(statuses), 'operator')} all ok"
    parts = [
        _title(_text(chapter.title), _PAGE, overline=True),
        _raw(_head(index.chapters.get(chapter.id), summary, meta)),
        _raw(LEGEND_HTML),
    ]
    if chapter.operators:
        parts.append(_diagram(block_diagram(chapter, index)))
    parts += [
        _section("Operators", _operator_table(chapter, index)),
        _section("Types", _types_table(chapter)),
        _section("Forms that are not green", *_not_green(chapter, index)),
    ]
    if chapter.operators:
        parts.append(_toctree(op.id for op in chapter.operators))
    return "\n\n".join(parts) + "\n"


def _operator_table(chapter: Chapter, index: StatusIndex) -> str:
    """One row per operator: a link to its page and its symbol, its status, its forms."""
    if not chapter.operators:
        return "This block has no operators."
    rows = []
    for op in chapter.operators:
        status = index.operators.get(f"{chapter.id}/{op.id}")
        link = _doc(op.name, op.id) + (f" {_literal(op.symbol)}" if op.symbol else "")
        rows.append([link, EMOJI[status] if status else "", str(len(op.variants))])
    return _table(("Operator", "Status", "Forms"), rows)


def _types_table(chapter: Chapter) -> str:
    """The operators against the types that their forms take (``in``) and give (``out``)."""
    columns, rows = block_types_table(chapter)
    if not columns:
        return "The forms of this block declare no types."
    head = ("Operator", *(_type_name(column) for column in columns))
    return _table(head, [[_doc(op.name, op.id), *cells] for op, cells in rows])


def _not_green(chapter: Chapter, index: StatusIndex) -> list[str]:
    """The forms that are not 🟢, in the order of the diagrams, with the example that shows why.

    That is the first example that is not green, or for a 🔵 form the first that diverges.
    """
    rows: list[list[str]] = []
    for op in chapter.operators:
        for variant in forms_in_diagram_order(op):
            key = f"{chapter.id}/{op.id}/{variant.id}"
            status = index.variants.get(key)
            if status is None or status == "ok":
                continue
            shown = _shown_example(status, variant, key, index)
            if shown is None:
                continue
            example, result = shown
            rows.append(
                [
                    _doc(op.name, op.id),
                    _link(variant.label, f"{op.id}.html#{form_anchor(variant)}"),
                    EMOJI[status],
                    _link(
                        example_title(example), f"{op.id}.html#{example_anchor(variant, example)}"
                    ),
                    _inline(mark(result.pandas)),
                    _inline(mark(result.duckdb)),
                    _text(_detail_text(result)),
                ]
            )
    if not rows:
        return ["Every form is green."]
    head = ("Operator", "Form", "Status", "Example", "pandas", "DuckDB", "Detail")
    intro = (
        "Every form that is not 🟢, with the example that shows its status: the first that is "
        "not green, or the first that diverges for a 🔵 form."
    )
    return [intro, _table(head, rows)]


def _shown_example(
    status: Status, variant: Variant, key: str, index: StatusIndex
) -> tuple[Example, ExampleResult] | None:
    """The example that shows why the form ``key`` has ``status``, with its result."""
    for example in variant.examples:
        result = index.examples.get(f"{key}/{example.name}")
        if result is not None and _shows(status, result):
            return example, result
    return None


def _shows(status: Status, result: ExampleResult) -> bool:
    """Whether an example is the one to show for a form that has ``status``."""
    if status == "diverge":
        return result.status == "diverge"
    return result.status != "ok"


# --- Index ---


_INTRO: Final = (
    "Which operators, syntax forms and definitions of the VTL 2.2 Reference Manual vtlengine "
    "implements, on both of its engines, pandas and DuckDB. Every form is backed by runnable "
    "examples: a VTL script, its input data and the expected output.\n"
    "\n"
    "These pages are generated by ``docs/operator_map/generate.py``. Do not edit them by hand."
)

# How to run the generator; ``generate.py --help`` and its docstring say the same.
_REGENERATE: Final = (
    "- **Regenerating.** Run ``docs/operator_map/generate.py`` with a Python that can import "
    "vtlengine, with the compiled parser in ``src/`` (see *Building the C++ parser from source* "
    "in the contributing guide of the repository). Where systemd is available, run it under a "
    "memory cap: "
    "``systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 python "
    "docs/operator_map/generate.py``. Elsewhere the cap is optional."
)

_HOW_TO_READ: Final = (
    """\
- **Pages.** This page, then a page for each block (a chapter of the manual, or its \
definitions), then a page for each operator. The sidebar, the tables and the diagrams lead to \
them.
- **Diagrams.** The overview and the page of a block have the map or the block in the middle, \
and its blocks or its operators on both sides. The page of an operator has the operator on top. \
When its forms use two kinds of operands or more, they are in a light box for each kind: a \
solid border for the forms on datasets, a dashed one for components and a dotted one for \
scalars, as the key above the diagram shows. Otherwise its forms are on both sides of it. A box \
with a thin, sparsely dotted border, "Used with", has the operators that it is combined with. \
Where the viewer supports links in diagrams, a node opens its page, and a form its section; the \
tables and the lists are the reliable way.
- **Statuses.** The colour of a node, and the emoji of a chip or a heading, say its status: 🟢 \
all its examples give the expected result on both engines; 🟡 partially implemented, a mix, or \
a result that is wrong on both engines; 🛡️ refused with an error: both engines raise an error \
of vtlengine with a code where the manual defines a result, and the code is not one of those \
that say that something is not implemented (the list below); 🔴 not implemented, or a crash; \
🔵 the engines differ. A form takes the status of its examples, an operator that of its forms, \
and a block that of its operators: 🔵 when any of them is 🔵, 🟢, 🛡️ or 🔴 when all of them \
are, and 🟡 otherwise. The tallies count the forms of an operator, and the operators of a block.
- **Marks.** An example shows what each engine gave: ✅ a match, the expected output or the \
expected error; ❌ a wrong result; 🛡 a guarded error, an error that vtlengine raises on \
purpose, with its code, such as 🛡 2-1-15-6; 💥 a crash, an exception that vtlengine does not \
handle (such as ``KeyError``), or a run that did not end.
- **Examples.** An example is a folder with a ``script.vtl``, its ``input/`` data and its \
``expected/`` output, or the error that it expects, and its notes: what it tests, the rule of \
the manual and how the result follows from it. The expected outputs come from the manual, or \
are derived by hand from its semantics, and never from the output of an engine. The page of an \
operator shows all of it, with links to the files.
- **Types.** A form takes and gives types. The page of an operator lists the operators that \
give what it takes ("Comes from") and those that take what it gives ("Goes to"), by block. An \
integer is a number, so a producer of an integer feeds an operator that takes a number.
- **Comparison.** Only the outputs that the example expects are compared: the components, \
their roles and types, and the data points, in any order. A number written with decimals \
matches a value within one unit of its last digit: an expected ``0.7`` accepts 0.6001–0.7999. \
The tolerance is never finer than what a double holds. A number written without decimals \
matches the same number, within a relative 1e-9, and one written with an exponent within a \
relative 1e-6. ``random`` and ``current_date`` are checked by a property of their values (for \
``random``, those of the columns that are empty in the expected file), and the two engines must \
still return the same values.
"""
    + _REGENERATE
)

_OVERVIEW: Final = (
    "A node for each block, with how many operators it has, and how many of them are in each "
    "status. Where the viewer supports links in diagrams, a block opens its page. The table of "
    "the blocks below is the reliable way to every block page."
)
_BLOCKS: Final = "The counts are over the forms of the operators of each block."
_NOT_IMPLEMENTED: Final = (
    "The codes of the errors of vtlengine that say that it does not implement something. A form "
    "whose examples fail with one of them on both engines is 🔴, not implemented, rather than "
    "🛡️, refused. The list is kept by hand, in ``docs/operator_map/tool/status.py``, with the "
    "messages as the error catalogue of vtlengine writes them."
)


def index_rst(cat: Catalogue, index: StatusIndex, meta: RunMeta) -> str:
    """The index page: the legend, the overview, the blocks and the lists of the whole map.

    It has a hidden ``toctree`` of the block pages. It raises ``MermaidLimitError`` when the
    overview cannot fit the limits of Mermaid.
    """
    entries = list(_entries(cat))
    parts = [
        _title("VTL 2.2 operator map", _PAGE, overline=True),
        _INTRO,
        *([_toctree(f"{chapter.id}/index" for chapter in cat.chapters)] if cat.chapters else []),
        _generation(meta),
        _raw(LEGEND_HTML),
        _section("How to read", _HOW_TO_READ),
        _section("Overview", _OVERVIEW, _diagram(overview_block(cat, index))),
        _section("Blocks", _BLOCKS, _block_table(cat, index)),
        _section("Not implemented", _NOT_IMPLEMENTED, _not_implemented_table()),
        _decisions(entries),
        _errata(entries),
        _adaptations(entries),
        _attribution(meta),
    ]
    return "\n\n".join(parts) + "\n"


def _generation(meta: RunMeta) -> str:
    spec = meta.spec_commit[:12] or "unknown"
    if meta.spec_commit:
        spec = _link(f"sdmx-twg/vtl @ {spec}", f"{SPEC_REPO}/tree/{meta.spec_commit}")
    commit = _literal(meta.vtlengine_commit) or "unknown"
    return (
        f"Generated on {_inline(meta.date)} from vtlengine {_inline(meta.vtlengine_version)} "
        f"(commit {commit}), with pandas {_inline(meta.pandas_version)} and DuckDB "
        f"{_inline(meta.duckdb_version)}, against the VTL 2.2 Reference Manual ({spec})."
    )


def _block_table(cat: Catalogue, index: StatusIndex) -> str:
    """A row per block: its status, operators and forms, and its forms by status; then a total."""
    if not cat.chapters:
        return "The map has no blocks."
    rows = []
    every: list[Status] = []
    for chapter in cat.chapters:
        statuses = _form_statuses(chapter, index)
        every += statuses
        link = _doc(chapter.title, f"{chapter.id}/index")
        rows.append(
            _count_row(link, index.chapters.get(chapter.id), len(chapter.operators), statuses)
        )
    rolled = [index.chapters[c.id] for c in cat.chapters if c.id in index.chapters]
    total = rollup(rolled) if rolled else None
    operators = sum(len(chapter.operators) for chapter in cat.chapters)
    rows.append(_count_row("**Total**", total, operators, every))
    head = ("Block", "Status", "Operators", "Forms", *(EMOJI[status] for status in _STATUSES))
    return _table(head, rows)


def _count_row(
    first: str, status: Status | None, operators: int, statuses: Sequence[Status]
) -> list[str]:
    tally = counts(statuses)
    emoji = EMOJI[status] if status is not None else ""
    numbers = (str(tally[status]) for status in _STATUSES)
    return [first, emoji, str(operators), str(len(statuses)), *numbers]


def _form_statuses(chapter: Chapter, index: StatusIndex) -> list[Status]:
    """The statuses of the forms of ``chapter`` that the index has."""
    keys = (
        f"{chapter.id}/{op.id}/{variant.id}" for op in chapter.operators for variant in op.variants
    )
    return [index.variants[key] for key in keys if key in index.variants]


def _not_implemented_table() -> str:
    """The codes of ``NOT_IMPLEMENTED`` with their messages, written as the error catalogue."""
    rows = [[_literal(code), message] for code, message in NOT_IMPLEMENTED.items()]
    return _table(("Code", "Message"), rows)


# --- The lists of the index ---


@dataclass(frozen=True)
class _Entry:
    """One example of the catalogue, with the chapter, the operator and the form it belongs to."""

    chapter: Chapter
    operator: Operator
    variant: Variant
    example: Example

    def link(self) -> str:
        """The title of the example, linking to it on the page of its operator."""
        target = f"{self.chapter.id}/{self.operator.id}.html"
        anchor = example_anchor(self.variant, self.example)
        return _link(example_title(self.example), f"{target}#{anchor}")

    def where(self) -> str:
        """``(Block › Operator › ``form``)``."""
        names = f"{_inline(self.chapter.title)} › {_inline(self.operator.name)}"
        return f"({names} › {_literal(self.variant.label)})"


def _entries(cat: Catalogue) -> Iterator[_Entry]:
    """Every example of the catalogue, in catalogue order."""
    for chapter in cat.chapters:
        for op in chapter.operators:
            for variant in op.variants:
                for example in variant.examples:
                    yield _Entry(chapter, op, variant, example)


def _item(entry: _Entry, head: str, text: str) -> str:
    """A list item: ``head``, the example and where it is, and ``text``."""
    line = f"- {head}{entry.link()} {entry.where()}"
    return f"{line}: {_inline(text)}" if text else line


def _list_section(title: str, intro: str, none: str, items: Sequence[str]) -> str:
    return _section(title, intro, "\n".join(items) if items else none)


def _decisions(entries: Sequence[_Entry]) -> str:
    items = [
        _item(entry, f"{_link(f'📌 #{issue}', f'{ISSUES}/{issue}')} · ", "")
        for entry in entries
        if (issue := entry.example.decision) is not None
    ]
    return _list_section(
        "📌 Team decisions",
        "Expected outputs that follow a recorded team decision. The notes of an example say what "
        "the manual says and how its output was derived.",
        "None: no expected output follows a recorded team decision.",
        items,
    )


def _errata(entries: Sequence[_Entry]) -> str:
    items = [
        _item(entry, "", erratum)
        for entry in entries
        if (erratum := entry.example.erratum) is not None
    ]
    return _list_section(
        "Manual errata (corrected expected outputs, scripts or inputs)",
        "Examples where the manual gets an expected output, a script or an input wrong. The "
        "example holds the corrected version, and its notes quote the manual's. These are the "
        "ones to report upstream.",
        "None: nothing in the manual's examples had to be corrected.",
        items,
    )


def _adaptations(entries: Sequence[_Entry]) -> str:
    items = [
        _item(entry, "", adaptation)
        for entry in entries
        if (adaptation := entry.example.adaptation) is not None
    ]
    return _list_section(
        "Adaptations",
        "Examples where the manual's representation of an input or of an output name was "
        "rewritten to the equivalent representation vtlengine documents (such as the ``time`` "
        "values ``2010M1/2010M12`` of an input, or the default measure name ``string_var`` of "
        "an output, which vtlengine writes ``str_var``), or values that no engine can reproduce "
        "(the manual's illustrative random draws) were left out of the expected output and are "
        "property-checked. The notes of the example keep the original. These are not errata.",
        "None: no example needed an adaptation.",
        items,
    )


def _attribution(meta: RunMeta) -> str:
    manual = _link("sdmx-twg.github.io/vtl/v2.2", f"{MANUAL_BASE}/index.html")
    source = _link("sdmx-twg/vtl", SPEC_REPO)
    return _section(
        "Attribution",
        "Examples marked *manual* are taken from the VTL 2.2 Reference Manual "
        f"(sdmx-twg/vtl @ {meta.spec_commit}), converted to the vtlengine format.",
        f"The manual is published at {manual}, and its source is the repository {source}.",
    )


# --- Mermaid ---


def mermaid_blocks(text: str) -> list[str]:
    """The Mermaid blocks of a page, in order: what the ``.. mermaid::`` directives hold.

    The options of a directive are not part of its block. The reverse of what the pages write,
    for the checks on the files.
    """
    lines = text.split("\n")
    blocks: list[str] = []
    n = 0
    while n < len(lines):
        if lines[n] != ".. mermaid::":
            n += 1
            continue
        n += 1
        while n < len(lines) and lines[n].startswith(f"{_INDENT}:"):
            n += 1  # an option, such as ``:config:``
        n += 1  # the blank line after the options
        body: list[str] = []
        while n < len(lines) and (not lines[n] or lines[n].startswith(_INDENT)):
            body.append(lines[n][len(_INDENT) :])
            n += 1
        blocks.append("\n".join(body).rstrip("\n"))
    return blocks


def _diagram(block: str) -> str:
    """A ``.. mermaid::`` directive of ``block`` at its natural size, in an ``om-diagram`` block."""
    lines = block.split("\n")
    directive = f".. mermaid::\n{_INDENT}:config: {_DIAGRAM_CONFIG}\n\n" + "\n".join(
        f"{_INDENT}{line}" if line else "" for line in lines
    )
    return "\n\n".join([_raw('<div class="om-diagram">'), directive, _raw("</div>")])


# --- HTML ---


def _raw(*lines: str) -> str:
    """A ``.. raw:: html`` directive with ``lines``, each of them one line of HTML."""
    return ".. raw:: html\n\n" + "\n".join(f"{_INDENT}{line}" for line in lines)


def _chip(status: Status | None, text: str, tip: str | None = None) -> str:
    """A chip of the colour of ``status`` with ``text``, escaped, and ``tip`` as its tooltip."""
    classes = "om-chip" if status is None else f"om-chip {status}"
    title = f' title="{_attribute(tip)}"' if tip else ""
    return f'<span class="{classes}"{title}>{html.escape(text)}</span>'


def _attribute(text: str) -> str:
    """``text`` escaped for the value of an HTML attribute in double quotes."""
    return html.escape(text, quote=True)


def _tally(statuses: Iterable[Status]) -> str:
    """``🟢12 🟡4 🔵1``: how many of ``statuses`` are in each status, in the order of the legend."""
    tally = counts(list(statuses))
    return " ".join(f"{EMOJI[status]}{tally[status]}" for status in _STATUSES if tally[status])


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


# --- Text ---


def _title(text: str, char: str, *, overline: bool = False) -> str:
    """``text`` as a title, ruled with ``char`` (and above too, for the title of a page).

    The rule has to be as wide as the title in columns, and an emoji is two columns wide: twice
    its length in characters is always enough.
    """
    rule = char * (2 * len(text))
    return f"{rule}\n{text}\n{rule}" if overline else f"{text}\n{rule}"


def _section(title: str, *parts: str) -> str:
    """A section of a page: its title, then ``parts``, with a blank line between all."""
    return "\n\n".join([_title(title, _SECTION), *parts])


def _toctree(names: Iterable[str]) -> str:
    """A hidden table of contents of Sphinx: the pages ``names``, relative to this one."""
    pages = "\n".join(f"{_INDENT}{name}" for name in names)
    return f".. toctree::\n{_INDENT}:maxdepth: 1\n{_INDENT}:hidden:\n\n{pages}"


def _inline(text: str) -> str:
    """``text`` as a piece of a line of RST: on one line, with the markup characters escaped.

    The quotes, the dashes and the ellipses that Sphinx would typeset are escaped too: the text
    reads as it was written, in the message of an engine too. A text that ends with two colons
    would make the next lines a literal block: the last colon is escaped.
    """
    escaped = _MARKUP.sub(r"\\\1", _NEWLINE.sub(" ", text).strip())
    escaped = _TYPESET.sub(lambda found: _TYPESET_CHARACTER.sub(r"\\\g<0>", found[0]), escaped)
    return escaped[:-1] + "\\:" if escaped.endswith("::") else escaped


def _text(text: str) -> str:
    """``text`` as RST that starts a paragraph, a cell or a title: ``_inline``, and guarded.

    A text that starts like a list, a field, a doctest, a comment or a table would be read as
    one of them, so it gets an escaped space in front, which RST drops.
    """
    one_line = _NEWLINE.sub(" ", text).strip()
    escaped = _inline(one_line)
    # An escaped first character protects the paragraph, but a line of backslashes only is a
    # transition, however they are escaped.
    if not escaped or (escaped.startswith("\\") and escaped.strip("\\")):
        return escaped
    if one_line[0].isalnum() and not _ENUMERATOR.match(one_line):
        return escaped
    return f"\\ {escaped}"


def _literal(text: str) -> str:
    """``text`` as an inline literal, as it is. A backtick cannot be in one: escaped text then."""
    one_line = _NEWLINE.sub(" ", text).strip()
    if not one_line:
        return ""
    return _inline(one_line) if "`" in one_line else f"``{one_line}``"


def _link(text: str, url: str) -> str:
    """An anonymous hyperlink with its address inside, so that no two of them can clash."""
    return f"`{_inline(text)} <{url}>`__"


def _doc(title: str, target: str) -> str:
    """A link to another page of the docs, by its name relative to this one.

    The title is escaped: ``<`` too, which would start the target otherwise.
    """
    escaped = _inline(title).replace("<", "\\<")
    return f":doc:`{escaped} <{target}>`"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]], *, title: str = "") -> str:
    """A ``list-table`` with a header row, columns that fit their content, and ``rows``.

    A cell is RST that is already escaped. One of several lines goes on under its first line.
    """
    lines = [
        f".. list-table::{f' {title}' if title else ''}",
        f"{_INDENT}:header-rows: 1",
        f"{_INDENT}:widths: auto",
        "",
    ]
    for row in (header, *rows):
        for position, cell in enumerate(row):
            marker = f"{_INDENT}* -" if position == 0 else f"{_INDENT}  -"
            first, *rest = cell.split("\n")
            lines.append(f"{marker} {first}".rstrip())
            lines.extend(f"{_INDENT}    {line}".rstrip() for line in rest)
    return "\n".join(lines)
