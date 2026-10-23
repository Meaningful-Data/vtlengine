"""Markdown of the operator map: one file per chapter, and the README.

``chapter_markdown`` and ``readme_markdown`` take the catalogue and the ``StatusIndex`` of a run
and return the text of a file. The diagrams come from ``tool.mermaid``; every example has a row
in a table, since GitHub may ignore the links of a diagram. Nothing here runs an engine. The only
thing read from disk is whether the folder of an example has an ``input/`` folder and a
``notes.md`` to link, in ``examples_root`` (``docs/operator_map/examples`` unless it is given).
The links of the text are relative to ``docs/operator_map``, where the files are written.

A cell of a table is one line: a newline is a space and a pipe is escaped. The Detail of a row
is what the engines said went wrong, cut at 200 characters.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import quote

from tool.examples import example_dir
from tool.mermaid import chapter_blocks, lattice_block, overview_block, type_flow_block
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

# The types that get a flow diagram in the README: the basic ones, and the dataset.
FLOW_TYPES: Final = (
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
)

_README: Final = "README.md"
_EXAMPLES: Final = Path("examples")  # relative to ``docs/operator_map``
_TYPES: Final = "types"  # the group, in the index, of the type hubs
_TYPES_TITLE: Final = "Data types"
_ORDER: Final[tuple[Status, ...]] = ("ok", "partial", "missing", "diverge")
# The statuses that are not green, worst first, with the title of their group in the README.
_GROUPS: Final[tuple[tuple[Status, str], ...]] = (
    ("diverge", "Engines diverge"),
    ("missing", "Not implemented"),
    ("partial", "Partially implemented"),
)
_MARKS: Final = {"match": "✅", "mismatch": "❌", "error": "💥"}
_NEWLINE: Final = re.compile(r"\r\n|\r|\n")
_TABLE_HEAD: Final = (
    "| Variant | Status | pandas | DuckDB | Detail | Example |\n|---|---|---|---|---|---|"
)


@dataclass(frozen=True)
class RunMeta:
    """When and with what the results were made."""

    date: str
    vtlengine_commit: str
    vtlengine_version: str
    spec_commit: str  # the commit of the manual, in sdmx-twg/vtl


def manual_url(page: str) -> str:
    """The URL of a manual page: ``<Chapter>/<Page>``, or one of the definitions, ``dl/...``.

    A page of an operator lives under ``operators/`` with its spaces encoded, and the pages of
    the definitions are the ``vtl_dl_*`` pages.
    """
    if page in _DEFINITION_PAGES:
        return f"{MANUAL_BASE}/{_DEFINITION_PAGES[page]}"
    return f"{MANUAL_BASE}/operators/{quote(page)}/index.html"


# --- Chapters ---


def chapter_markdown(
    chapter: Chapter,
    cat: Catalogue,
    index: StatusIndex,
    meta: RunMeta,
    *,
    examples_root: Path = DEFAULT_EXAMPLES_ROOT,
) -> str:
    """The md file of ``chapter``: its diagrams, then a table of examples per operator.

    It raises ``MermaidLimitError`` when a diagram cannot fit the limits of Mermaid.
    """
    parts = [
        f"# {_titled(index.chapters.get(chapter.id), chapter.title)}",
        _counts_line(_chapter_statuses(chapter, index)),
        f"[← Back to the operator map]({_README}) · "
        f"vtlengine {meta.vtlengine_version} ({meta.vtlengine_commit})",
        *(_fence(block) for block in chapter_blocks(chapter, cat, index)),
        *(_operator_section(chapter, op, index, examples_root) for op in chapter.operators),
    ]
    return "\n\n".join(parts) + "\n"


def _operator_section(chapter: Chapter, operator: Operator, index: StatusIndex, root: Path) -> str:
    """The heading, the manual link and the table of the examples of an operator."""
    status = index.operators.get(f"{chapter.id}/{operator.id}")
    heading = f"## {_titled(status, operator.name)}"
    if operator.symbol:
        heading += f" `{operator.symbol}`"
    parts = [heading, f"[VTL 2.2 Reference Manual]({manual_url(operator.page)})"]
    rows = [
        _example_row(chapter.id, operator.id, variant, example, index, root)
        for variant in operator.variants
        for example in variant.examples
    ]
    if rows:
        parts.append("\n".join([_TABLE_HEAD, *rows]))
    return "\n\n".join(parts)


def _example_row(
    chapter_id: str,
    operator_id: str,
    variant: Variant,
    example: Example,
    index: StatusIndex,
    root: Path,
) -> str:
    """The row of an example: its variant, its status, what each engine gave, and its files."""
    result = index.examples.get(f"{chapter_id}/{operator_id}/{variant.id}/{example.name}")
    label = _cell(f"`{variant.label}`") + f"<br><sub>{example.name}</sub>"
    files = _example_links(chapter_id, operator_id, variant.id, example, root)
    if result is None:  # the index of a run that did not cover it
        return _row([label, "", "", "", "not run", files])
    verdicts = [_MARKS[result.pandas.outcome], _MARKS[result.duckdb.outcome]]
    return _row([label, EMOJI[result.status], *verdicts, _detail_cell(result), files])


def _example_links(
    chapter_id: str, operator_id: str, variant_id: str, example: Example, root: Path
) -> str:
    """The Example cell: the links to the files, then the marks of the example.

    ``input/`` is linked when it exists (an example of scalars has none), and so is ``notes.md``.
    """
    names = (chapter_id, operator_id, variant_id, example.name)
    folder = quote(example_dir(_EXAMPLES, *names).as_posix())
    on_disk = example_dir(root, *names)
    links = [f"[script]({folder}/script.vtl)"]
    if (on_disk / "input").is_dir():
        links.append(f"[input]({folder}/input/)")
    links.append(f"[expected]({folder}/expected/)")
    if (on_disk / "notes.md").is_file():
        links.append(f"[notes]({folder}/notes.md)")
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
    return f"[📌 #{issue}]({ISSUES}/{issue})"


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
    text = _NEWLINE.sub(" ", _detail(result))
    if len(text) > MAX_DETAIL:
        text = text[: MAX_DETAIL - 1] + "…"
    return _escape_pipes(text)


# --- README ---


_INTRO: Final = (
    "Which operators, syntax variants and definitions of the VTL 2.2 Reference Manual vtlengine "
    "implements, on both of its engines, pandas and DuckDB. Every node of the diagrams is backed "
    "by a runnable example: a VTL script, its input data and the expected output.\n"
    "\n"
    "This page and the chapter pages are generated by `docs/operator_map/generate.py`. "
    "Do not edit them by hand."
)

_LEGEND: Final = """## Legend

| Status | Meaning |
|---|---|
| 🟢 ok | Both engines give the expected output on every example. |
| 🟡 partial | The engines agree with each other, but not every example gives the expected output: \
a wrong or incomplete result, or a mix of matching and failing examples. |
| 🔴 missing | Both engines fail on every example. |
| 🔵 diverge | On some example the two engines differ: one matches and the other does not, one \
fails and the other runs, or both run and give different outputs. |

| Mark | Meaning |
|---|---|
| 📌 #n | The expected output follows a recorded team decision, in the issue n, and not the \
manual. |
| ✅ | The engine gave the expected output. |
| ❌ | The engine ran, but gave another output. |
| 💥 | The engine failed: an error, a timeout or a crash. |"""

_HOW_TO_READ: Final = """## How to read

- **Chapter pages.** One page per chapter of the manual, and one for its definitions. Each \
starts with diagrams, and has a table per operator with a row for every example.
- **Diagrams.** An operator shows its status, its name and its symbol, and a plain line joins \
it to its syntax variants. A solid arrow from a type to an operator means that the operator \
accepts the type, and from an operator to a type that it returns it. A dotted, labelled arrow \
joins an operator to one that it is used in or with, and a dashed box with an arrow is an \
operator of another chapter. Grey circles are data types.
- **Status.** The colour of a node and its emoji say the same. A variant is 🟢 when both engines \
give the expected output on all of its examples, 🔵 when the engines differ on any of them, 🔴 \
when both engines fail on all of them, and 🟡 otherwise. An operator and a chapter take the \
status of their variants by the same rule, and the counts are over variants.
- **Examples.** An example is a folder with a `script.vtl`, its `input/` data and its \
`expected/` output. The expected outputs come from the manual, or are derived by hand from its \
semantics, and never from the output of an engine. GitHub may ignore the links of the \
diagrams: the tables are the reliable way to the files.
- **Comparison.** Only the outputs that the example expects are compared: the components, \
their roles and types, and the data points, in any order. A number is compared to the digits \
written in the expected file. `random` and `current_date` are checked by a property of their \
values, and the two engines must still return the same values."""

_OVERVIEW: Final = (
    "One node per chapter, with how many of its variants are in each status, and an arrow to "
    "each type that its operators return. Click a chapter to open its page."
)
_CHAPTERS: Final = "The counts are over syntax variants."
_LATTICE: Final = (
    "An arrow reads *is a subtype of*. A type has the status of its own examples, when it has "
    "any, and links to the first of them."
)
_FLOWS: Final = (
    "One diagram for each basic type and one for the dataset: the operators that return the "
    "type on the left, the operators that accept it on the right, and its subtypes as dotted "
    "arrows."
)


def readme_markdown(
    cat: Catalogue,
    index: StatusIndex,
    meta: RunMeta,
    *,
    examples_root: Path = DEFAULT_EXAMPLES_ROOT,
) -> str:
    """The README: legend, diagrams, chapter index and the lists of what needs attention.

    It raises ``MermaidLimitError`` when a diagram cannot fit the limits of Mermaid.
    """
    entries = list(_entries(cat))
    parts = [
        "# VTL 2.2 operator map for vtlengine",
        _INTRO,
        _generation(meta),
        _LEGEND,
        _HOW_TO_READ,
        f"## Overview\n\n{_OVERVIEW}\n\n{_fence(overview_block(cat, index))}",
        f"## Chapters\n\n{_CHAPTERS}\n\n{_chapter_table(cat, index)}",
        f"## Type lattice\n\n{_LATTICE}\n\n{_fence(lattice_block(cat, index))}",
        f"## Type flows\n\n{_FLOWS}\n\n{_flow_blocks(cat, index)}",
        _decisions(entries, examples_root),
        _errata(entries, examples_root),
        _adaptations(entries, examples_root),
        _not_green(entries, index),
        _attribution(meta),
    ]
    return "\n\n".join(parts) + "\n"


def _generation(meta: RunMeta) -> str:
    spec = meta.spec_commit[:12] or "unknown"
    if meta.spec_commit:
        spec = f"[sdmx-twg/vtl @ `{spec}`]({SPEC_REPO}/tree/{meta.spec_commit})"
    return "\n".join(
        [
            "## Generation",
            "",
            f"- Generated on {meta.date}.",
            f"- Engine: vtlengine {meta.vtlengine_version}, commit `{meta.vtlengine_commit}`.",
            f"- Manual: VTL 2.2 Reference Manual, {spec}.",
            "- To regenerate, run `docs/operator_map/generate.py` under "
            "`systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0`.",
        ]
    )


def _chapter_table(cat: Catalogue, index: StatusIndex) -> str:
    """One row per chapter, a row for the type hubs when they have examples, and the total."""
    rows = [
        _count_row(
            f"[{chapter.title}]({chapter.id}.md)",
            index.chapters.get(chapter.id),
            _chapter_statuses(chapter, index),
        )
        for chapter in cat.chapters
    ]
    type_statuses = _statuses_of(_type_variant_keys(cat), index)
    if type_statuses:
        rows.append(_count_row(_TYPES_TITLE, index.chapters.get(_TYPES), type_statuses))
    every = [status for chapter in cat.chapters for status in _chapter_statuses(chapter, index)]
    group_statuses = [index.chapters[key] for key in _index_chapters(cat) if key in index.chapters]
    total = rollup(group_statuses) if group_statuses else None
    rows.append(_count_row("**Total**", total, [*every, *type_statuses]))
    head = f"| Chapter | Status | Variants | {' | '.join(EMOJI[status] for status in _ORDER)} |"
    return "\n".join([head, "|---|---|---|---|---|---|---|", *rows])


def _count_row(first: str, status: Status | None, statuses: Sequence[Status]) -> str:
    tally = counts(statuses)
    emoji = EMOJI[status] if status is not None else ""
    return _row([first, emoji, str(len(statuses)), *(str(tally[each]) for each in _ORDER)])


def _flow_blocks(cat: Catalogue, index: StatusIndex) -> str:
    return "\n\n".join(
        f"### {type_id}\n\n{_fence(type_flow_block(type_id, cat, index))}" for type_id in FLOW_TYPES
    )


# --- The lists of the README ---


@dataclass(frozen=True)
class _Entry:
    """One example of the catalogue, with the operator and the variant it belongs to."""

    group: str  # the chapter id, or ``types``
    title: str  # the title of the chapter
    operator_id: str
    operator_name: str
    page: str | None  # the manual page, if it has one
    variant: Variant
    example: Example

    @property
    def variant_key(self) -> str:
        return f"{self.group}/{self.operator_id}/{self.variant.id}"

    @property
    def key(self) -> str:
        return f"{self.variant_key}/{self.example.name}"

    def chapter(self) -> str:
        """The title of the chapter, linking to its page; the types have no page."""
        return self.title if self.group == _TYPES else f"[{self.title}]({self.group}.md)"

    def where(self) -> str:
        """``Chapter › Operator › `variant` (example)``."""
        label = _NEWLINE.sub(" ", self.variant.label)
        return f"{self.chapter()} › {self.operator_name} › `{label}` ({self.example.name})"

    def folder(self) -> str:
        names = (self.group, self.operator_id, self.variant.id, self.example.name)
        return quote(example_dir(_EXAMPLES, *names).as_posix())

    def on_disk(self, root: Path) -> Path:
        return example_dir(root, self.group, self.operator_id, self.variant.id, self.example.name)


def _entries(cat: Catalogue) -> Iterator[_Entry]:
    """Every example of the catalogue, chapters first and then the types, in catalogue order."""
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
    for typedef in cat.types:
        for variant in typedef.variants:
            for example in variant.examples:
                yield _Entry(_TYPES, _TYPES_TITLE, typedef.id, typedef.id, None, variant, example)


def _item(entry: _Entry, head: str, text: str, root: Path) -> str:
    """A list item: ``head``, where the example is, ``text``, and the links of the example."""
    links = []
    if entry.page is not None:
        links.append(f"[manual]({manual_url(entry.page)})")
    if (entry.on_disk(root) / "notes.md").is_file():
        links.append(f"[notes]({entry.folder()}/notes.md)")
    line = f"{head}{entry.where()}"
    if text:
        line += f": {_NEWLINE.sub(' ', text)}"
    return "- " + " · ".join([line, *links])


def _list_section(title: str, intro: str, none: str, items: Sequence[str]) -> str:
    body = "\n".join(items) if items else none
    return f"## {title}\n\n{intro}\n\n{body}"


def _decisions(entries: Sequence[_Entry], root: Path) -> str:
    items = [
        _item(entry, f"{_decision_link(entry.example.decision)} · ", "", root)
        for entry in entries
        if entry.example.decision is not None
    ]
    return _list_section(
        "📌 Team decisions",
        "Expected outputs that follow a recorded team decision, and not the text of the manual. "
        "The notes of an example say what the manual says and how its output was derived.",
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
        "Manual errata",
        "Expected outputs that the manual gets wrong. The example holds the corrected output, "
        "and its notes quote the manual's version. These are the ones to report upstream.",
        "None: no expected output of the manual had to be corrected.",
        items,
    )


def _adaptations(entries: Sequence[_Entry], root: Path) -> str:
    items = [
        _item(entry, "", entry.example.adaptation, root)
        for entry in entries
        if entry.example.adaptation is not None
    ]
    return _list_section(
        "Input adaptations",
        "Manual inputs that vtlengine cannot load as they are written, such as the `time` "
        "values `2010M1/2010M12`. The example uses the equivalent representation that "
        "vtlengine documents, and its notes keep the original. These are not errata.",
        "None: every manual input loads as written.",
        items,
    )


def _not_green(entries: Sequence[_Entry], index: StatusIndex) -> str:
    """The variants that are not 🟢, by status, each with the example that shows its status.

    That is the first example that is not green, or for a 🔵 variant the first that diverges.
    """
    groups: dict[Status, list[str]] = {status: [] for status, _ in _GROUPS}
    listed: set[str] = set()
    for entry in entries:
        status = index.variants.get(entry.variant_key)
        result = index.examples.get(entry.key)
        if status is None or status == "ok" or result is None or entry.variant_key in listed:
            continue
        if not _shows(status, result):
            continue
        listed.add(entry.variant_key)
        row = [
            entry.chapter(),
            _cell(entry.operator_name),
            _cell(f"`{entry.variant.label}`"),
            f"[{entry.example.name}]({entry.folder()}/)",
            _detail_cell(result),
        ]
        groups[status].append(_row(row))
    parts = ["## Variants that are not green"]
    if listed:
        parts.append(
            "Every variant that is not 🟢, by status. A row names the first example that is not "
            "green, or the first that diverges for a 🔵 variant: the chapter pages have them all."
        )
    else:
        parts.append("Every variant is green.")
    for status, title in _GROUPS:
        if groups[status]:
            head = "| Chapter | Operator | Variant | Example | Detail |\n|---|---|---|---|---|"
            parts.append(f"### {EMOJI[status]} {title} ({len(groups[status])})")
            parts.append("\n".join([head, *groups[status]]))
    return "\n\n".join(parts)


def _shows(status: Status, result: ExampleResult) -> bool:
    """Whether an example is the one to list for a variant that has ``status``."""
    if status == "diverge":
        return result.status == "diverge"
    return result.status != "ok"


def _attribution(meta: RunMeta) -> str:
    return (
        "## Attribution\n\n"
        "Examples marked *manual* are taken from the VTL 2.2 Reference Manual "
        f"(sdmx-twg/vtl @ {meta.spec_commit}), converted to the vtlengine format.\n\n"
        f"The manual is published at [sdmx-twg.github.io/vtl/v2.2]({MANUAL_BASE}/index.html), "
        f"and its source is the repository [sdmx-twg/vtl]({SPEC_REPO})."
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


def _type_variant_keys(cat: Catalogue) -> Iterator[str]:
    """The index keys of the variants of the types."""
    for typedef in cat.types:
        for variant in typedef.variants:
            yield f"{_TYPES}/{typedef.id}/{variant.id}"


def _statuses_of(keys: Iterator[str], index: StatusIndex) -> list[Status]:
    return [index.variants[key] for key in keys if key in index.variants]


def _index_chapters(cat: Catalogue) -> list[str]:
    """The keys of ``index.chapters`` that ``cat`` can have: its chapters and the types."""
    return [*(chapter.id for chapter in cat.chapters), _TYPES]


# --- Text ---


def _fence(block: str) -> str:
    return f"```mermaid\n{block}\n```"


def _escape_pipes(text: str) -> str:
    return text.replace("|", "\\|")


def _cell(text: str) -> str:
    """``text`` as the content of a table cell: on one line, with its pipes escaped."""
    return _escape_pipes(_NEWLINE.sub(" ", text))


def _row(cells: Sequence[str]) -> str:
    """A table row of cells that are already escaped."""
    return "| " + " | ".join(cells) + " |"
