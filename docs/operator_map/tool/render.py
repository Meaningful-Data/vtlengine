"""reStructuredText of the operator map: one Sphinx page per chapter, and the index page.

``chapter_rst`` and ``index_rst`` take the catalogue and the ``StatusIndex`` of a run and return
the text of a page. The diagrams come from ``tool.mermaid`` and sit in ``.. mermaid::``
directives; every example has a row in a table, since a viewer may ignore the links of a
diagram. The name of an example in its row links to its folder, whose page in the built site
shows the text of its files, and the Example cell links to the files themselves. Nothing here
runs an engine. The only thing read from disk is whether the folder of an example has an
``input/`` folder and a ``notes.md`` to link, in ``examples_root`` (``docs/operator_map/examples``
unless it is given).

The links to the files of the examples are relative to ``docs/operator_map``, where the pages are
written: the docs build copies the examples next to the pages of the built site. The pages are
documents of Sphinx, so the links between them are ``:doc:`` roles. The diagrams come as
``tool.mermaid`` writes them, and its clicks to a chapter already name the page of the site.

The text that comes from the catalogue or from the engines is free text: ``_inline`` and
``_text`` escape what RST would read as markup, and what Sphinx would typeset (see
``_TYPESET``), so that a message of an engine reads as it was written. ``_literal`` puts a piece
of VTL in an inline literal. A cell of a table is one line, and the Detail of a row is what the
engines said went wrong, cut at 200 characters.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import quote

from tool.examples import example_dir
from tool.mermaid import chapter_blocks, overview_block
from tool.model import Catalogue, Chapter, Example, Operator, Status, Variant
from tool.status import EMOJI, ExampleResult, StatusIndex, counts, rollup

MANUAL_BASE: Final = "https://sdmx-twg.github.io/vtl/v2.2/reference_manual"
SPEC_REPO: Final = "https://github.com/sdmx-twg/vtl"
ISSUES: Final = "https://github.com/Meaningful-Data/vtlengine/issues"
MAX_DETAIL: Final = 200  # characters of the Detail cell, the ellipsis included
DEFAULT_EXAMPLES_ROOT: Final = Path(__file__).resolve().parents[1] / "examples"

# The manual pages of the VTL-DL: the definitions are not under ``operators/``.
_DEFINITION_PAGES: Final = {
    "dl/udo": "vtl_dl_udo.html",
    "dl/datapoint": "vtl_dl_rulesets/datapoint_rulesets.html",
    "dl/hierarchical": "vtl_dl_rulesets/hierarchical_rulesets.html",
    "dl/viral": "vtl_dl_rulesets/viral_attributes.html",
}

_INDEX_DOC: Final = "index"  # the page that the chapters link back to
_EXAMPLES: Final = Path("examples")  # relative to ``docs/operator_map``
_ORDER: Final[tuple[Status, ...]] = ("ok", "partial", "missing", "diverge")
# The statuses that are not green, worst first, with the title of their group in the index.
_GROUPS: Final[tuple[tuple[Status, str], ...]] = (
    ("diverge", "Engines diverge"),
    ("missing", "Not implemented"),
    ("partial", "Partially implemented"),
    ("refused", "Refused with an error"),
)
_MARKS: Final = {"match": "✅", "mismatch": "❌", "error": "💥"}
_NEWLINE: Final = re.compile(r"\r\n|\r|\n")
_TABLE_HEAD: Final = ("Variant", "Status", "pandas", "DuckDB", "Detail", "Example")

# The adornments of the titles: a page, a section, and a section of a section.
_PAGE: Final = "="
_SECTION: Final = "-"
_SUBSECTION: Final = "~"
_INDENT: Final = "   "  # the content of a directive, and the options of a table

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
    """The URL of a manual page: ``<Chapter>/<Page>``, or one of the definitions, ``dl/...``.

    A page of an operator lives under ``operators/`` with its spaces encoded, and the pages of
    the definitions are the ``vtl_dl_*`` pages.
    """
    if page in _DEFINITION_PAGES:
        return f"{MANUAL_BASE}/{_DEFINITION_PAGES[page]}"
    return f"{MANUAL_BASE}/operators/{quote(page)}/index.html"


# --- Chapters ---


def chapter_rst(
    chapter: Chapter,
    cat: Catalogue,
    index: StatusIndex,
    meta: RunMeta,
    *,
    examples_root: Path = DEFAULT_EXAMPLES_ROOT,
) -> str:
    """The page of ``chapter``: its diagrams, then a table of examples per operator.

    It raises ``MermaidLimitError`` when a diagram cannot fit the limits of Mermaid.
    """
    title = _titled(index.chapters.get(chapter.id), _text(chapter.title))
    version, commit = _inline(meta.vtlengine_version), _inline(meta.vtlengine_commit)
    engines = f"pandas {_inline(meta.pandas_version)} · DuckDB {_inline(meta.duckdb_version)}"
    parts = [
        _title(title, _PAGE, overline=True),
        _counts_line(_chapter_statuses(chapter, index)),
        f"← Back to :doc:`{_INDEX_DOC}` · vtlengine {version} ({commit}) · {engines}",
        *(_mermaid(block) for block in chapter_blocks(chapter, cat, index)),
        *(_operator_section(chapter, op, index, examples_root) for op in chapter.operators),
    ]
    return "\n\n".join(parts) + "\n"


def _operator_section(chapter: Chapter, operator: Operator, index: StatusIndex, root: Path) -> str:
    """The heading, the manual link and the table of the examples of an operator."""
    status = index.operators.get(f"{chapter.id}/{operator.id}")
    heading = _titled(status, _text(operator.name))
    if operator.symbol:
        heading += f" {_literal(operator.symbol)}"
    parts = [
        _title(heading, _SECTION),
        _link("VTL 2.2 Reference Manual", manual_url(operator.page)),
    ]
    rows = [
        _example_row(chapter.id, operator.id, variant, example, index, root)
        for variant in operator.variants
        for example in variant.examples
    ]
    if rows:
        parts.append(_table(_TABLE_HEAD, rows))
    return "\n\n".join(parts)


def _example_row(
    chapter_id: str,
    operator_id: str,
    variant: Variant,
    example: Example,
    index: StatusIndex,
    root: Path,
) -> list[str]:
    """The row of an example: its variant, its status, what each engine gave, and its files."""
    result = index.examples.get(f"{chapter_id}/{operator_id}/{variant.id}/{example.name}")
    # A line block: the variant, and under it the name of the example, as two lines of a cell.
    # The name links to the folder, whose page shows the text of every file of the example.
    names = (chapter_id, operator_id, variant.id, example.name)
    folder = quote(example_dir(_EXAMPLES, *names).as_posix())
    lines = [_literal(variant.label), _link(example.name, f"{folder}/")]
    label = "\n".join(f"| {line}" for line in lines if line)
    files = _example_links(chapter_id, operator_id, variant.id, example, root)
    if result is None:  # the index of a run that did not cover it
        return [label, "", "", "", "not run", files]
    verdicts = [_MARKS[result.pandas.outcome], _MARKS[result.duckdb.outcome]]
    return [label, EMOJI[result.status], *verdicts, _detail_cell(result), files]


def _example_links(
    chapter_id: str, operator_id: str, variant_id: str, example: Example, root: Path
) -> str:
    """The Example cell: the links to the files, then the marks of the example.

    ``input/`` is linked when it exists (an example of scalars has none), and so is ``notes.md``.
    """
    names = (chapter_id, operator_id, variant_id, example.name)
    folder = quote(example_dir(_EXAMPLES, *names).as_posix())
    on_disk = example_dir(root, *names)
    links = [_link("script", f"{folder}/script.vtl")]
    if (on_disk / "input").is_dir():
        links.append(_link("input", f"{folder}/input/"))
    links.append(_link("expected", f"{folder}/expected/"))
    if (on_disk / "notes.md").is_file():
        links.append(_link("notes", f"{folder}/notes.md"))
    if example.source == "manual":
        links.append("*manual*")
    if example.decision is not None:
        links.append(_decision_link(example.decision))
    if example.erratum is not None:
        links.append("erratum")
    if example.adaptation is not None:
        links.append("adapted")
    return " · ".join(links)


def _decision_link(issue: int) -> str:
    return _link(f"📌 #{issue}", f"{ISSUES}/{issue}")


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


def _detail_cell(result: ExampleResult) -> str:
    """The Detail of a row: one line, cut at ``MAX_DETAIL`` characters, then escaped."""
    text = _NEWLINE.sub(" ", _detail(result))
    if len(text) > MAX_DETAIL:
        text = text[: MAX_DETAIL - 1] + "…"
    return _text(text)


# --- Index ---


_INTRO: Final = (
    "Which operators, syntax variants and definitions of the VTL 2.2 Reference Manual vtlengine "
    "implements, on both of its engines, pandas and DuckDB. Every node of the diagrams is backed "
    "by a runnable example: a VTL script, its input data and the expected output.\n"
    "\n"
    "This page and the chapter pages are generated by ``docs/operator_map/generate.py``. "
    "Do not edit them by hand."
)

_STATUS_LEGEND: Final = (
    ("🟢 ok", "Both engines give the expected output on every example."),
    (
        "🟡 partial",
        "The engines agree with each other, but not every example gives the expected output: "
        "a wrong or incomplete result, or a mix of matching and failing examples.",
    ),
    ("🔴 missing", "Both engines fail on every example."),
    (
        "🔵 diverge",
        "On some example the two engines differ: one matches and the other does not, one fails "
        "and the other runs, or both run and give different outputs.",
    ),
)
_MARK_LEGEND: Final = (
    ("📌 #n", "The expected output follows a recorded team decision, in the issue n."),
    ("✅", "The engine gave the expected output."),
    ("❌", "The engine ran, but gave another output."),
    ("💥", "The engine failed: an error, a timeout or a crash."),
)

_HOW_TO_READ: Final = """\
- **Chapter pages.** One page per chapter of the manual, and one for its definitions. Each \
starts with diagrams, and has a table per operator with a row for every example.
- **Diagrams.** An operator shows its status, its name and its symbol, and a plain line joins \
it to its syntax variants. A dotted, labelled arrow joins an operator to one that it is used in \
or with, and a dashed box with an arrow is an operator of another chapter.
- **Status.** The colour of a node and its emoji say the same. A variant is 🟢 when both engines \
give the expected output on all of its examples, 🔵 when the engines differ on any of them, 🔴 \
when both engines fail on all of them, and 🟡 otherwise. An operator and a chapter take the \
status of their variants by the same rule, and the counts are over variants.
- **Examples.** An example is a folder with a ``script.vtl``, its ``input/`` data and its \
``expected/`` output. The expected outputs come from the manual, or are derived by hand from its \
semantics, and never from the output of an engine. A viewer may ignore the links of the \
diagrams: the tables are the reliable way to the files.
- **Comparison.** Only the outputs that the example expects are compared: the components, \
their roles and types, and the data points, in any order. A number written with decimals matches \
a value within one unit of its last digit: an expected ``0.7`` accepts 0.6001–0.7999. The \
tolerance is never finer than what a double holds. A number written without decimals matches \
the same number, within a relative 1e-9, and one written with an exponent within a relative \
1e-6. ``random`` and ``current_date`` are checked by a property of their values (for \
``random``, those of the columns that are empty in the expected file), and the two engines must \
still return the same values."""

# How to run the generator; ``generate.py --help`` and its docstring say the same.
_REGENERATE: Final = (
    "- To regenerate, run ``docs/operator_map/generate.py`` with a Python that can import "
    "vtlengine, with the compiled parser in ``src/`` (see *Building the C++ parser from source* "
    "in the contributing guide of the repository). Where systemd is available, run it under a "
    "memory cap: "
    "``systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 python "
    "docs/operator_map/generate.py``. Elsewhere the cap is optional."
)

_OVERVIEW: Final = (
    "One node per chapter, with how many of its variants are in each status. Where the viewer "
    "supports links in diagrams, a chapter node opens the page of the chapter. The table of "
    "chapters below is the reliable way to every chapter page."
)
_CHAPTERS: Final = "The counts are over syntax variants."


def index_rst(
    cat: Catalogue,
    index: StatusIndex,
    meta: RunMeta,
    *,
    examples_root: Path = DEFAULT_EXAMPLES_ROOT,
) -> str:
    """The index page: legend, diagrams, chapter table and the lists of what needs attention.

    It starts with a hidden ``toctree`` of the chapter pages. It raises ``MermaidLimitError``
    when a diagram cannot fit the limits of Mermaid.
    """
    entries = list(_entries(cat))
    parts = [
        _title("VTL 2.2 operator map", _PAGE, overline=True),
        _INTRO,
        *([_toctree(cat)] if cat.chapters else []),
        _generation(meta),
        _legend(),
        _section("How to read", _HOW_TO_READ),
        _section("Overview", _OVERVIEW, _mermaid(overview_block(cat, index))),
        _section("Chapters", _CHAPTERS, _chapter_table(cat, index)),
        _decisions(entries, examples_root),
        _errata(entries, examples_root),
        _adaptations(entries, examples_root),
        _not_green(entries, index),
        _attribution(meta),
    ]
    return "\n\n".join(parts) + "\n"


def _toctree(cat: Catalogue) -> str:
    """The hidden table of contents of Sphinx: the chapter pages, in the order of the catalogue."""
    pages = "\n".join(f"{_INDENT}{chapter.id}" for chapter in cat.chapters)
    return f".. toctree::\n{_INDENT}:maxdepth: 1\n{_INDENT}:hidden:\n\n{pages}"


def _generation(meta: RunMeta) -> str:
    spec = meta.spec_commit[:12] or "unknown"
    if meta.spec_commit:
        spec = _link(f"sdmx-twg/vtl @ {spec}", f"{SPEC_REPO}/tree/{meta.spec_commit}")
    return _section(
        "Generation",
        "\n".join(
            [
                f"- Generated on {_inline(meta.date)}.",
                f"- Engine: vtlengine {_inline(meta.vtlengine_version)}, "
                f"commit {_literal(meta.vtlengine_commit)}, "
                f"with pandas {_inline(meta.pandas_version)} "
                f"and DuckDB {_inline(meta.duckdb_version)}.",
                f"- Manual: VTL 2.2 Reference Manual, {spec}.",
                _REGENERATE,
            ]
        ),
    )


def _legend() -> str:
    return _section(
        "Legend",
        _table(("Status", "Meaning"), _STATUS_LEGEND),
        _table(("Mark", "Meaning"), _MARK_LEGEND),
    )


def _chapter_table(cat: Catalogue, index: StatusIndex) -> str:
    """One row per chapter, and the total."""
    rows = [
        _count_row(
            _doc(chapter.title, chapter.id),
            index.chapters.get(chapter.id),
            _chapter_statuses(chapter, index),
        )
        for chapter in cat.chapters
    ]
    every = [status for chapter in cat.chapters for status in _chapter_statuses(chapter, index)]
    group_statuses = [index.chapters[c.id] for c in cat.chapters if c.id in index.chapters]
    total = rollup(group_statuses) if group_statuses else None
    rows.append(_count_row("**Total**", total, every))
    head = ("Chapter", "Status", "Variants", *(EMOJI[status] for status in _ORDER))
    return _table(head, rows)


def _count_row(first: str, status: Status | None, statuses: Sequence[Status]) -> list[str]:
    tally = counts(statuses)
    emoji = EMOJI[status] if status is not None else ""
    return [first, emoji, str(len(statuses)), *(str(tally[each]) for each in _ORDER)]


# --- The lists of the index ---


@dataclass(frozen=True)
class _Entry:
    """One example of the catalogue, with the operator and the variant it belongs to."""

    group: str  # the chapter id
    title: str  # the title of the chapter
    operator_id: str
    operator_name: str
    page: str  # the manual page
    variant: Variant
    example: Example

    @property
    def variant_key(self) -> str:
        return f"{self.group}/{self.operator_id}/{self.variant.id}"

    @property
    def key(self) -> str:
        return f"{self.variant_key}/{self.example.name}"

    def chapter(self) -> str:
        """The title of the chapter, linking to its page."""
        return _doc(self.title, self.group)

    def where(self) -> str:
        """``Chapter › Operator › ``variant`` (example)``."""
        label, name = _literal(self.variant.label), _inline(self.example.name)
        return f"{self.chapter()} › {_inline(self.operator_name)} › {label} ({name})"

    def folder(self) -> str:
        names = (self.group, self.operator_id, self.variant.id, self.example.name)
        return quote(example_dir(_EXAMPLES, *names).as_posix())

    def on_disk(self, root: Path) -> Path:
        return example_dir(root, self.group, self.operator_id, self.variant.id, self.example.name)


def _entries(cat: Catalogue) -> Iterator[_Entry]:
    """Every example of the catalogue, in catalogue order."""
    for chapter in cat.chapters:
        for operator in chapter.operators:
            for variant in operator.variants:
                for example in variant.examples:
                    yield _Entry(
                        chapter.id,
                        chapter.title,
                        operator.id,
                        operator.name,
                        operator.page,
                        variant,
                        example,
                    )


def _item(entry: _Entry, head: str, text: str, root: Path) -> str:
    """A list item: ``head``, where the example is, ``text``, and the links of the example."""
    links = [_link("manual", manual_url(entry.page))]
    if (entry.on_disk(root) / "notes.md").is_file():
        links.append(_link("notes", f"{entry.folder()}/notes.md"))
    line = f"{head}{entry.where()}"
    if text:
        line += f": {_inline(text)}"
    return "- " + " · ".join([line, *links])


def _list_section(title: str, intro: str, none: str, items: Sequence[str]) -> str:
    return _section(title, intro, "\n".join(items) if items else none)


def _decisions(entries: Sequence[_Entry], root: Path) -> str:
    items = [
        _item(entry, f"{_decision_link(entry.example.decision)} · ", "", root)
        for entry in entries
        if entry.example.decision is not None
    ]
    return _list_section(
        "📌 Team decisions",
        "Expected outputs that follow a recorded team decision. The notes of an example say what "
        "the manual says and how its output was derived.",
        "None: no expected output follows a recorded team decision.",
        items,
    )


def _errata(entries: Sequence[_Entry], root: Path) -> str:
    items = [
        _item(entry, "", entry.example.erratum, root)
        for entry in entries
        if entry.example.erratum is not None
    ]
    return _list_section(
        "Manual errata (corrected expected outputs, scripts or inputs)",
        "Examples where the manual gets an expected output, a script or an input wrong. The "
        "example holds the corrected version, and its notes quote the manual's. These are the "
        "ones to report upstream.",
        "None: nothing in the manual's examples had to be corrected.",
        items,
    )


def _adaptations(entries: Sequence[_Entry], root: Path) -> str:
    items = [
        _item(entry, "", entry.example.adaptation, root)
        for entry in entries
        if entry.example.adaptation is not None
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


def _not_green(entries: Sequence[_Entry], index: StatusIndex) -> str:
    """The variants that are not 🟢, by status, each with the example that shows its status.

    That is the first example that is not green, or for a 🔵 variant the first that diverges.
    """
    groups: dict[Status, list[list[str]]] = {status: [] for status, _ in _GROUPS}
    listed: set[str] = set()
    for entry in entries:
        status = index.variants.get(entry.variant_key)
        result = index.examples.get(entry.key)
        if status is None or status == "ok" or result is None or entry.variant_key in listed:
            continue
        if not _shows(status, result):
            continue
        listed.add(entry.variant_key)
        groups[status].append(
            [
                entry.chapter(),
                _text(entry.operator_name),
                _literal(entry.variant.label),
                _link(entry.example.name, f"{entry.folder()}/"),
                _detail_cell(result),
            ]
        )
    parts = []
    if listed:
        parts.append(
            "Every variant that is not 🟢, by status. A row names the first example that is not "
            "green, or the first that diverges for a 🔵 variant: the chapter pages have them all."
        )
    else:
        parts.append("Every variant is green.")
    head = ("Chapter", "Operator", "Variant", "Example", "Detail")
    for status, title in _GROUPS:
        if groups[status]:
            parts.append(_title(f"{EMOJI[status]} {title} ({len(groups[status])})", _SUBSECTION))
            parts.append(_table(head, groups[status]))
    return _section("Variants that are not green", *parts)


def _shows(status: Status, result: ExampleResult) -> bool:
    """Whether an example is the one to list for a variant that has ``status``."""
    if status == "diverge":
        return result.status == "diverge"
    return result.status != "ok"


def _attribution(meta: RunMeta) -> str:
    manual = _link("sdmx-twg.github.io/vtl/v2.2", f"{MANUAL_BASE}/index.html")
    source = _link("sdmx-twg/vtl", SPEC_REPO)
    return _section(
        "Attribution",
        "Examples marked *manual* are taken from the VTL 2.2 Reference Manual "
        f"(sdmx-twg/vtl @ {meta.spec_commit}), converted to the vtlengine format.",
        f"The manual is published at {manual}, and its source is the repository {source}.",
    )


# --- Counting ---


def _titled(status: Status | None, text: str) -> str:
    """``text`` after the emoji of ``status``, if it has one."""
    return text if status is None else f"{EMOJI[status]} {text}"


def _counts_line(statuses: Sequence[Status]) -> str:
    """``3 variants: 🟢 1 · 🟡 0 · 🔴 1 · 🔵 1``."""
    tally = counts(statuses)
    noun = "variant" if len(statuses) == 1 else "variants"
    shown = " · ".join(f"{EMOJI[status]} {tally[status]}" for status in _ORDER)
    return f"{len(statuses)} {noun}: {shown}"


def _chapter_statuses(chapter: Chapter, index: StatusIndex) -> list[Status]:
    """The statuses of the variants of ``chapter`` that the index has."""
    keys = (
        f"{chapter.id}/{operator.id}/{variant.id}"
        for operator in chapter.operators
        for variant in operator.variants
    )
    return _statuses_of(keys, index)


def _statuses_of(keys: Iterator[str], index: StatusIndex) -> list[Status]:
    return [index.variants[key] for key in keys if key in index.variants]


# --- Mermaid ---


def mermaid_blocks(text: str) -> list[str]:
    """The Mermaid blocks of a page, in order: what the ``.. mermaid::`` directives hold.

    The reverse of what ``chapter_rst`` and ``index_rst`` write, for the checks on the files.
    """
    lines = text.split("\n")
    blocks: list[str] = []
    n = 0
    while n < len(lines):
        if lines[n] != ".. mermaid::":
            n += 1
            continue
        body: list[str] = []
        n += 2  # the directive and the blank line after it
        while n < len(lines) and (not lines[n] or lines[n].startswith(_INDENT)):
            body.append(lines[n][len(_INDENT) :])
            n += 1
        blocks.append("\n".join(body).rstrip("\n"))
    return blocks


def _mermaid(block: str) -> str:
    """A ``.. mermaid::`` directive of ``block``, with its content indented by three spaces."""
    lines = block.split("\n")
    return ".. mermaid::\n\n" + "\n".join(f"{_INDENT}{line}" if line else "" for line in lines)


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
    """A link to another page of the docs, by its name relative to this one."""
    return f":doc:`{title} <{target}>`"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """A ``list-table`` with a header row, columns that fit their content, and ``rows``.

    A cell is RST that is already escaped. One of several lines goes on under its first line.
    """
    lines = [".. list-table::", f"{_INDENT}:header-rows: 1", f"{_INDENT}:widths: auto", ""]
    for row in (header, *rows):
        for position, cell in enumerate(row):
            marker = f"{_INDENT}* -" if position == 0 else f"{_INDENT}  -"
            first, *rest = cell.split("\n")
            lines.append(f"{marker} {first}".rstrip())
            lines.extend(f"{_INDENT}    {line}".rstrip() for line in rest)
    return "\n".join(lines)
