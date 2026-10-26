# ruff: noqa: S101
"""Checks for the reStructuredText of the map: the index, the block pages and the operator pages.

The catalogue, the results and the example folders are small hand-made values: no engine runs
here. The checks look at what matters on each page: the order of its parts, the anchors that the
diagrams click to, the tables that are built from the files of an example, the type lists, the
links of the lists of the index, and that what is written as free text survives the reading of
RST. Two checks need the docs: one parses the pages with docutils, and one builds a small tree of
pages with Sphinx, its theme and the Mermaid extension, as ``docs/conf.py`` does. They are skipped
without them; run them in the venv of the docs::

    docsvenv/bin/python -m pytest -p no:cacheprovider -o python_files='check_*.py' \\
        -o pythonpath="src docs/operator_map" docs/operator_map/tool/tests/check_render.py

Every check that renders an operator page says where its examples are: a folder of its own under
``tmp_path``, never the ``examples`` of this repository.
"""

from __future__ import annotations

import html
import io
import json
import re
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from catalogue import manual_index

from tool.mermaid import (
    USED_WITH_MAX,
    block_diagram,
    example_anchor,
    form_anchor,
    operator_diagram,
    overview_block,
)
from tool.model import (
    Catalogue,
    Chapter,
    Example,
    Link,
    Operator,
    Variant,
    ViralGroup,
    authored,
    load_catalogue,
    load_viral_groups,
    manual,
)
from tool.render import (
    BORDER_KEY_HTML,
    LEGEND_HTML,
    RunMeta,
    block_rst,
    index_rst,
    manual_url,
    mermaid_blocks,
    operator_rst,
    operators_rst,
    overview_rst,
)
from tool.status import NOT_IMPLEMENTED, EngineVerdict, ExampleResult, StatusIndex, build_index

BASE = "https://sdmx-twg.github.io/vtl/v2.2/reference_manual"
ISSUES = "https://github.com/Meaningful-Data/vtlengine/issues"
SPEC = "140e946ed3f888951adb8bb66ea45030be84e2aa"
META = RunMeta(
    "2026-10-23", "abc1234", "9.9.9", SPEC, pandas_version="2.3.3", duckdb_version="1.5.5"
)
ERRATA_TITLE = "Manual errata (corrected expected outputs, scripts or inputs)"
DOCS = Path(__file__).resolve().parents[3]
CSS = DOCS / "_static" / "operator_map.css"
CONF = DOCS / "conf.py"


# --- Builders ---


MATCH = EngineVerdict("match", "")
WRONG = EngineVerdict("mismatch", "row Id_1=1: Me_1 expected 'a' got 'b'")
GUARDED = EngineVerdict(
    "error", "RunTimeError 2-1-15-6: Scalar division by Zero.", "RunTimeError", "2-1-15-6"
)
CRASH = EngineVerdict("error", "KeyError: 'Me_1'", "KeyError")
OK = ExampleResult(MATCH, MATCH, True, "ok")
PARTIAL = ExampleResult(WRONG, WRONG, True, "partial")
REFUSED = ExampleResult(GUARDED, GUARDED, True, "refused")
DIVERGE = ExampleResult(MATCH, CRASH, False, "diverge")
NUMBER = ("number",)


def form(
    variant_id: str,
    label: str,
    takes: tuple[str, ...],
    gives: tuple[str, ...],
    *examples: Example,
    box: str | None = None,
) -> Variant:
    """A form that declares its types, with ``examples``, or one named ``two_rows``."""
    return Variant(
        variant_id,
        label,
        examples or (authored("two_rows"),),
        takes=takes,
        gives=gives,
        box=box,
    )


def operator(
    operator_id: str,
    name: str,
    symbol: str,
    *variants: Variant,
    links: tuple[Link, ...] = (),
    page: str = "",
) -> Operator:
    page = page or f"Some operators/{name}"
    return Operator(operator_id, name, symbol, page, (), (), variants, links)


def keys(cat: Catalogue) -> Iterator[str]:
    """The key of every example of ``cat``, in catalogue order."""
    for chapter in cat.chapters:
        for op in chapter.operators:
            for variant in op.variants:
                for example in variant.examples:
                    yield f"{chapter.id}/{op.id}/{variant.id}/{example.name}"


def index_of(cat: Catalogue, results: dict[str, ExampleResult] | None = None) -> StatusIndex:
    """The index of ``cat`` where every example is ok, but those that ``results`` has."""
    found = results or {}
    return build_index(cat, {key: found.get(key, OK) for key in keys(cat)})


# A small map. Modulo has a form of each kind, so its diagram has boxes, and the order of its
# diagram (dataset, component, scalar) is not the order of the catalogue. Rounding takes a number
# and an integer, which Ceiling gives. Nvl takes any scalar and gives the type of its input.
MODULO = operator(
    "modulo",
    "Modulo",
    "mod",
    form(
        "sc_sc",
        "mod(sc, sc)",
        NUMBER,
        NUMBER,
        authored("zero_divisor"),
        authored("positive_operands"),
    ),
    form(
        "ds_ds",
        "mod(DS, DS)",
        NUMBER,
        NUMBER,
        manual("ex_1", erratum="the manual shows 'ba' where 'ab' is meant"),
    ),
    form("comp_comp", "mod(comp, comp)", NUMBER, NUMBER, authored("two_components", decision=1049)),
)
ROUND = operator(
    "round",
    "Rounding",
    "round",
    form("ds", "round(DS, n)", ("number", "integer"), NUMBER),
    form("sc", "round(sc, n)", ("number", "integer"), NUMBER),
)
CEIL = operator("ceil", "Ceiling", "ceil", form("sc", "ceil(sc)", NUMBER, ("integer",)))
LENGTH = operator(
    "length",
    "String length",
    "length",
    form(
        "sc",
        "length(sc)",
        ("string",),
        ("integer",),
        authored("plain", adaptation="the time values were rewritten"),
    ),
)
NVL = operator("nvl", "Nvl", "nvl", form("sc_sc", "nvl(sc, sc)", ("scalar",), ("any",)))
NUMERIC = Chapter("numeric", "Numeric operators", (MODULO, ROUND, CEIL))
STRING = Chapter("string", "String operators", (LENGTH,))
CONDITIONAL = Chapter("conditional", "Conditional operators", (NVL,))
MAP = Catalogue((NUMERIC, STRING, CONDITIONAL))
INDEX = index_of(
    MAP,
    {
        "numeric/modulo/sc_sc/zero_divisor": REFUSED,
        "numeric/modulo/ds_ds/manual_ex_1": DIVERGE,
        "string/length/sc/plain": PARTIAL,
    },
)

# A definition whose forms name the light boxes that they are in. The catalogue lists the forms in
# another order than the boxes: the Body, the Signature, the Body, the Signature and the Return
# type. The diagram draws the Body first, as its title is the first to appear, then the Signature
# and the Return type, and the page lists the forms in that order.
DEFINE = operator(
    "define_x",
    "Define x",
    "define x",
    form("body_when", "when c then v", ("boolean",), ("viral_propagation",), box="Body"),
    form(
        "sign_variable",
        "define x (variable v)",
        ("scalar",),
        ("viral_propagation",),
        box="Signature",
    ),
    form("body_aggr", "aggr sum", ("number",), ("viral_propagation",), box="Body"),
    form(
        "sign_domain",
        "define x (valuedomain d)",
        ("value_domain",),
        ("viral_propagation",),
        box="Signature",
    ),
    form("result", "returns r", ("scalar",), ("viral_propagation",), box="Return type"),
)
DEFINE_CHAPTER = Chapter("definitions", "Definitions", (DEFINE,))
DEFINE_MAP = Catalogue((DEFINE_CHAPTER,))
DEFINE_INDEX = index_of(DEFINE_MAP)
DEFINE_ORDER = ["body_when", "body_aggr", "sign_variable", "sign_domain", "result"]
DEFINE_BOXES = ["Body", "Signature", "Return type"]

NOTES = (
    ':Tests: "quotes" *stars* `code` | pipe :: colons.\n:Manual: The rule.\n:Expected: ``DS_r``.\n'
)
STRUCTURE = {
    "datasets": [
        {
            "name": "DS_1",
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False}
            ],
        }
    ]
}
# Cells that RST would read as markup, typeset or strip, and that a table must show as written.
EDGE_CSV = 'Id_1,Me_1,Me_2\n1,"a, b",x"y\n2,,"  two  spaces "\n3,*stars* a_ `t` |p|,-\n'
EDGE_ROWS = [
    ["Id_1", "Me_1", "Me_2"],
    ["1", "a, b", 'x"y'],
    ["2", "", "  two  spaces "],
    ["3", "*stars* a_ `t` |p|", "-"],
]


def write_example(root: Path, key: str, files: dict[str, str]) -> None:
    """The folder of the example ``key`` under ``root``, with a script, notes and ``files``."""
    folder = root / key
    for name, text in {"script.vtl": "DS_r := DS_1;\n", "notes.rst": NOTES, **files}.items():
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_text(text, encoding="utf-8")


def write_modulo(root: Path) -> None:
    """The examples of Modulo: datasets (with the cells of ``EDGE_CSV``), scalars and an error."""
    structure = json.dumps(STRUCTURE)
    base = "numeric/modulo"
    write_example(
        root,
        f"{base}/ds_ds/manual_ex_1",
        {
            "input/DS_1.csv": EDGE_CSV,
            "input/DS_1.json": structure,
            "input/DS_2.csv": "Id_1,Me_1\n1,2\n",
            "input/DS_2.json": structure,
            "expected/DS_r.csv": "Id_1,Me_1\n",  # no data point: a header only
            "expected/DS_r.json": structure,
            "notes.md": "The notes of the first version, which the page does not link.\n",
        },
    )
    write_example(
        root,
        f"{base}/comp_comp/two_components",
        {
            "input/DS_1.csv": "Id_1,Me_1\n1,2\n",
            "input/DS_1.json": structure,
            "expected/DS_r.csv": "Id_1,Me_1\n1,0\n",
            "expected/DS_r.json": structure,
        },
    )
    write_example(
        root, f"{base}/sc_sc/zero_divisor", {"expected/error.json": '{"code": "2-1-15-6"}'}
    )
    scalars = {
        "scalars": [
            {"name": "SC_r", "type": "Integer", "value": 1.50},
            {"name": "SC_s", "type": "String", "value": "  Hello"},
            {"name": "SC_t", "type": "String", "value": None},
        ]
    }
    # The value is written as the file writes it: 1.50 keeps its zero.
    text = json.dumps(scalars).replace("1.5", "1.50")
    write_example(root, f"{base}/sc_sc/positive_operands", {"expected/scalars.json": text})


def write_define(root: Path, chapter_id: str = "definitions") -> None:
    """The examples of ``DEFINE``: a script and notes for each of its forms."""
    for variant in DEFINE.variants:
        write_example(root, f"{chapter_id}/define_x/{variant.id}/two_rows", {})


# --- Reading a page back ---


def blocks(text: str) -> list[str]:
    """The Mermaid blocks of a page, in order, as the ``.. mermaid::`` directives hold them."""
    bodies = re.findall(
        r"^\.\. mermaid::\n(?:   :.*\n)*\n((?:   .*\n)+)", text + "\n", flags=re.MULTILINE
    )
    return ["\n".join(line[3:] for line in body.splitlines()).rstrip("\n") for body in bodies]


def headings(text: str) -> list[tuple[str, str]]:
    """The ``(adornment, title)`` of every title of a page, in order."""
    lines = text.split("\n")
    found: list[tuple[str, str]] = []
    for n in range(len(lines) - 1):
        title, rule = lines[n], lines[n + 1]
        is_rule = len(rule) >= 4 and len(set(rule)) == 1 and rule[0] in "=-~"
        if title and is_rule and set(title) != set(rule):
            found.append((rule[0], title))
    return found


def titles(text: str, char: str) -> list[str]:
    """The titles that are underlined with ``char``, in order."""
    return [title for rule, title in headings(text) if rule == char]


def section(text: str, title: str) -> str:
    """The text of the ``-`` section or rubric ``title``, up to the next of either or the end."""
    lines = text.split("\n")
    rubric = f".. rubric:: {title}"
    start = next(
        n
        for n, line in enumerate(lines[:-1])
        if line == rubric or (line == title and set(lines[n + 1]) == {"-"})
    )
    skip = 1 if lines[start] == rubric else 2
    end = len(lines)
    for n in range(start + skip, len(lines) - 1):
        heading = lines[n] and lines[n + 1] and set(lines[n + 1]) == {"-"}
        if lines[n].startswith(".. rubric:: ") or (heading and set(lines[n]) != {"-"}):
            end = n
            break
    return "\n".join(lines[start + skip : end])


def skeleton(text: str) -> list[str]:
    """The parts of a page in order: its titles, its directives, and the first tag of a raw block.

    A title is ``<adornment> <title>``, a directive its line without the leading ``.. `` (an
    option is ``  :<option>: <value>``), and a raw block of HTML ``raw <first tag>``.
    """
    lines = text.split("\n")
    marks = {n: f"{rule} {title}" for n, (rule, title) in _numbered_headings(lines)}
    found: list[str] = []
    for n, line in enumerate(lines):
        if n in marks:
            found.append(marks[n])
        elif line == ".. raw:: html":
            first = lines[n + 2].strip()
            found.append("raw " + re.match(r"<[^>]*>", first).group())  # type: ignore[union-attr]
        elif line.startswith(".. "):
            found.append(line[3:])
        elif line.startswith("   :") and found and not found[-1].startswith("raw "):
            found.append("  " + line.strip())
    return found


def _numbered_headings(lines: list[str]) -> Iterator[tuple[int, tuple[str, str]]]:
    for n in range(len(lines) - 1):
        title, rule = lines[n], lines[n + 1]
        is_rule = len(rule) >= 4 and len(set(rule)) == 1 and rule[0] in "=-~"
        if title and is_rule and set(title) != set(rule):
            yield n, (rule[0], title)


def table_rows(text: str) -> list[list[str]]:
    """The rows of every ``list-table`` of ``text``, header rows too, as lists of cells."""
    rows: list[list[str]] = []
    for line in text.split("\n"):
        if line == "   * -" or line.startswith("   * - "):
            rows.append([line[7:]])
        elif rows and (line == "     -" or line.startswith("     - ")):
            rows[-1].append(line[7:])
        elif rows and line.startswith("       "):
            rows[-1][-1] += "\n" + line[7:]
    return rows


def html_tables(text: str) -> list[tuple[str, list[list[str]]]]:
    """The caption and the cells of every HTML table of ``text``, unescaped, header row first."""
    found: list[tuple[str, list[list[str]]]] = []
    for table in re.findall(r"<table class=\"[^\"]*om-data[^\"]*\">.*?</table>", text):
        caption = re.search(r"<caption>(.*?)</caption>", table)
        assert caption is not None
        rows = [
            [html.unescape(cell) for cell in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row)]
            for row in re.findall(r"<tr>(.*?)</tr>", table)
        ]
        found.append((re.sub(r"<[^>]+>", "", caption.group(1)), rows))
    return found


def one_form_block(result: ExampleResult) -> str:
    """The page of a block with one operator of one form, whose only example has ``result``."""
    chapter = Chapter(
        "c", "C", (operator("o", "O", "o", form("v", "v", NUMBER, NUMBER, authored("minimal"))),)
    )
    cat = Catalogue((chapter,))
    return block_rst(chapter, cat, build_index(cat, {"c/o/v/minimal": result}), META)


def detail_of(result: ExampleResult) -> str:
    """The Detail cell of the row of the only form of ``one_form_block(result)``, not green."""
    rows = table_rows(section(one_form_block(result), "Forms that are not green"))
    assert len(rows) == 2, rows  # the header, and the form
    return rows[1][6]


def detail_cell(detail: str) -> str:
    """The Detail cell of a form whose example both engines got wrong with ``detail``."""
    verdict = EngineVerdict("mismatch", detail)
    return detail_of(ExampleResult(verdict, verdict, True, "partial"))


# --- Manual URLs ---


def test_manual_url() -> None:
    page = "String operators/Concatenation"
    assert manual_url(page) == f"{BASE}/operators/String%20operators/Concatenation/index.html"
    # Spaces are encoded in the page too, and a hyphen is left alone.
    assert manual_url("Join operators/Inner Join") == (
        f"{BASE}/operators/Join%20operators/Inner%20Join/index.html"
    )
    assert manual_url("Conditional operators/if-then-else") == (
        f"{BASE}/operators/Conditional%20operators/if-then-else/index.html"
    )


def test_manual_url_of_the_definitions() -> None:
    assert manual_url("dl/udo") == f"{BASE}/vtl_dl_udo.html"
    assert manual_url("dl/datapoint") == f"{BASE}/vtl_dl_rulesets/datapoint_rulesets.html"
    assert manual_url("dl/hierarchical") == f"{BASE}/vtl_dl_rulesets/hierarchical_rulesets.html"
    assert manual_url("dl/viral") == f"{BASE}/vtl_dl_rulesets/viral_attributes.html"
    # The typical behaviours of the operators, which the viral groups quote.
    assert manual_url("typical") == f"{BASE}/typical_behaviour.html"


# --- The legend ---


def test_legend_has_five_statuses(tmp_path: Path) -> None:
    chips = re.findall(r'<span class="om-chip (\w+)">([^<]*)</span>', LEGEND_HTML)
    assert chips == [
        ("ok", "🟢 all ok"),
        ("partial", "🟡 partially implemented"),
        ("refused", "🛡️ refused with an error"),
        ("missing", "🔴 not implemented or crash"),
        ("diverge", "🔵 engines differ"),
    ]
    # The key of the marks, and of the pin that diagrams draw: a recorded team decision.
    assert (
        "✅ match · ❌ wrong result · 🛡 guarded error (code) · 💥 crash · 📌 team decision"
    ) in LEGEND_HTML
    borders = re.findall(r'<span class="om-box (\w+)">([^<]*)</span>', BORDER_KEY_HTML)
    assert borders == [
        ("solid", "Dataset forms"),
        ("dashed", "Component forms"),
        ("dotted", "Scalar forms"),
    ]
    # Every page but the index has the legend; the border key is on the pages of operators with
    # boxes only.
    write_modulo(tmp_path)
    assert LEGEND_HTML not in index_rst(MAP, INDEX, META)
    for page in (
        overview_rst(MAP, INDEX, META),
        operators_rst(MAP, INDEX),
        block_rst(NUMERIC, MAP, INDEX, META),
        operator_rst(NUMERIC, MODULO, MAP, INDEX, META, examples_root=tmp_path),
        operator_rst(NUMERIC, ROUND, MAP, INDEX, META, examples_root=tmp_path),
    ):
        assert page.count(LEGEND_HTML) == 1
    with_boxes = operator_rst(NUMERIC, MODULO, MAP, INDEX, META, examples_root=tmp_path)
    assert with_boxes.count(BORDER_KEY_HTML) == 1
    one_kind = operator_rst(NUMERIC, CEIL, MAP, INDEX, META, examples_root=tmp_path)
    assert BORDER_KEY_HTML not in one_kind
    # The captions of boxes are for the forms that name their boxes: the boxes of the kinds have
    # the key of their borders, and no caption.
    assert "om-box-title" not in with_boxes
    assert "om-box-title" not in one_kind


# --- The page of an operator ---


def test_operator_page_structure(tmp_path: Path) -> None:
    write_modulo(tmp_path / "examples")
    text = operator_rst(NUMERIC, MODULO, MAP, INDEX, META, examples_root=tmp_path / "examples")
    assert text.endswith("\n")
    rule = "=" * 32  # twice the 16 characters of the title
    assert text.startswith(f"{rule}\nModulo · ``mod``\n{rule}\n\n")

    def example(form_id: str, name: str, *, opened: bool, data: bool, scalars: bool) -> list[str]:
        folder = f"../examples/numeric/modulo/{form_id}/{name}"
        anchor = f"ex-{form_id}-{name}".replace("_", "-")
        return [
            f'raw <details class="om-example" id="{anchor}"{" open" if opened else ""}>',
            f"include:: {folder}/notes.rst",
            f"literalinclude:: {folder}/script.vtl",
            "  :caption: script.vtl",
            "  :language: none",
            *(['raw <table class="docutils align-default om-data">'] if data else []),
            *(
                ["list-table:: Expected scalars", "  :header-rows: 1", "  :widths: auto"]
                if scalars
                else []
            ),
            'raw <p class="om-files">',
        ]

    # The forms come in the order of the diagram: dataset, component, then scalar.
    assert skeleton(text) == [
        "= Modulo · ``mod``",
        'raw <div class="om-head">',
        'raw <div class="om-legend">',
        'raw <div class="om-legend om-borders">',
        'raw <div class="om-diagram">',
        "mermaid::",
        '  :config: {"flowchart": {"useMaxWidth": false}}',
        "raw </div>",
        "- Types",
        'raw <p class="om-takes">',
        "- Forms",
        'raw <span class="om-anchor" id="form-ds-ds">',
        "~ mod(DS, DS) 🔵 · number → number",
        *example("ds_ds", "manual_ex_1", opened=True, data=True, scalars=False),
        'raw <span class="om-anchor" id="form-comp-comp">',
        "~ mod(comp, comp) 🟢 · number → number",
        *example("comp_comp", "two_components", opened=True, data=True, scalars=False),
        'raw <span class="om-anchor" id="form-sc-sc">',
        "~ mod(sc, sc) 🟡 · number → number",
        *example("sc_sc", "zero_divisor", opened=True, data=False, scalars=False),
        *example("sc_sc", "positive_operands", opened=False, data=False, scalars=True),
    ]
    # The anchors are those of the diagram: every click to a form lands on its section.
    assert blocks(text) == [operator_diagram(NUMERIC, MODULO, MAP, INDEX)]
    assert mermaid_blocks(text) == blocks(text)
    clicks = re.findall(r'click \w+ href "#([^"]+)"', blocks(text)[0])
    anchors = re.findall(r'<span class="om-anchor" id="([^"]+)"></span>', text)
    assert sorted(clicks) == sorted(anchors) == sorted(form_anchor(v) for v in MODULO.variants)
    # The paths are relative to the page, docs/operator_map/<chapter>/<operator>.rst, and the
    # files are there.
    page_folder = tmp_path / "numeric"
    paths = re.findall(r"^\.\. (?:include|literalinclude):: (\S+)$", text, flags=re.MULTILINE)
    assert len(paths) == 8
    for path in paths:
        assert (page_folder / path).resolve().is_file(), path
    # The files are linked, the notes are not: they are on the page already.
    folder = "../examples/numeric/modulo/ds_ds/manual_ex_1"
    files = re.search(r'<p class="om-files">Files: (.*)</p>', text)
    assert files is not None
    first_files = files.group(1)
    links = re.findall(r'<a href="([^"]+)">([^<]+)</a>', first_files)
    assert links == [
        (f"{folder}/script.vtl", "script.vtl"),
        (f"{folder}/input/DS_1.csv", "input/DS_1.csv"),
        (f"{folder}/input/DS_1.json", "input/DS_1.json"),
        (f"{folder}/input/DS_2.csv", "input/DS_2.csv"),
        (f"{folder}/input/DS_2.json", "input/DS_2.json"),
        (f"{folder}/expected/DS_r.csv", "expected/DS_r.csv"),
        (f"{folder}/expected/DS_r.json", "expected/DS_r.json"),
        (f"{folder}/index.html", "example page ↗"),
    ]
    assert "notes.rst" not in "".join(re.findall(r'<p class="om-files">.*', text))
    assert "notes.md" not in text


def test_an_example_shows_its_title_status_and_the_mark_of_each_engine(tmp_path: Path) -> None:
    write_modulo(tmp_path)
    text = operator_rst(NUMERIC, MODULO, MAP, INDEX, META, examples_root=tmp_path)
    summaries = dict(
        re.findall(r'<details class="om-example" id="([^"]+)"[^>]*><summary>(.*?)</summary>', text)
    )
    refused = summaries["ex-sc-sc-zero-divisor"]
    assert '<span class="om-ex-title">Zero divisor</span>' in refused
    assert '<span class="om-chip refused" title="refused with an error">🛡️</span>' in refused
    assert "pandas 🛡 <code>2-1-15-6</code> · DuckDB 🛡 <code>2-1-15-6</code>" in refused
    diverging = summaries["ex-ds-ds-manual-ex-1"]
    assert '<span class="om-ex-title">Manual example 1</span>' in diverging
    assert "pandas ✅ · DuckDB 💥 <code>KeyError</code>" in diverging
    assert "pandas ✅ · DuckDB ✅" in summaries["ex-comp-comp-two-components"]
    # What went wrong follows, on its own line, escaped.
    assert '<p class="om-detail">DuckDB: KeyError: &#x27;Me_1&#x27;</p>' in text
    assert text.count('class="om-detail"') == 2  # the refused and the diverging example


def test_the_tables_show_the_cells_of_the_files_as_written(tmp_path: Path) -> None:
    write_modulo(tmp_path)
    text = operator_rst(NUMERIC, MODULO, MAP, INDEX, META, examples_root=tmp_path)
    tables = html_tables(text)
    assert [caption for caption, _ in tables] == [
        "Input DS_1",
        "Input DS_2",
        "Expected DS_r (no data points)",
        "Input DS_1",
        "Expected DS_r",
    ]
    assert tables[0][1] == EDGE_ROWS
    assert tables[2][1] == [["Id_1", "Me_1"]]  # a header only, and a table that builds
    # A raw block of HTML is one line per table: a cell holds its line breaks as entities.
    assert all("\n" not in table for table in re.findall(r"<table.*?</table>", text))


def test_the_input_scalars_are_a_table_captioned_for_them_and_both_files_are_linked(
    tmp_path: Path,
) -> None:
    base = "numeric/modulo/sc_sc/zero_divisor"
    declared = [{"name": "SC_1", "type": "Integer"}, {"name": "SC_2", "type": "String"}]
    write_example(
        tmp_path,
        base,
        {
            "input/scalars.json": json.dumps({"scalars": declared}),
            "input/scalars.csv": 'SC_1,SC_2\n9,"a, b"\n',
            "expected/error.json": '{"code": "2-1-15-6"}',
        },
    )
    text = operator_rst(NUMERIC, MODULO, MAP, INDEX, META, examples_root=tmp_path)

    # The values are a table like the data of a dataset, but it is captioned for the input scalars
    # and not as the data of a dataset named ``scalars``, which no script could use.
    assert html_tables(text) == [("Input scalars", [["SC_1", "SC_2"], ["9", "a, b"]])]
    assert '<caption><span class="caption-text">Input scalars</span></caption>' in text
    assert "<code>scalars</code>" not in text
    # The structure of the scalars is not a table: it is a file of the example, like the others.
    (files,) = [
        line
        for line in re.findall(r'<p class="om-files">Files: (.*)</p>', text)
        if "scalars" in line
    ]
    folder = f"../examples/{base}"
    assert re.findall(r'<a href="([^"]+)">([^<]+)</a>', files) == [
        (f"{folder}/script.vtl", "script.vtl"),
        (f"{folder}/input/scalars.csv", "input/scalars.csv"),
        (f"{folder}/input/scalars.json", "input/scalars.json"),
        (f"{folder}/expected/error.json", "expected/error.json"),
        (f"{folder}/index.html", "example page ↗"),
    ]


def test_expected_scalars_and_an_expected_error(tmp_path: Path) -> None:
    write_modulo(tmp_path)
    text = operator_rst(NUMERIC, MODULO, MAP, INDEX, META, examples_root=tmp_path)
    scalars = text.split(".. list-table:: Expected scalars\n", 1)[1].split("\n\n.. raw::", 1)[0]
    # The values as the file writes them: a number keeps its digits, a string its quotes and
    # spaces, and null is null.
    assert table_rows(scalars) == [
        ["Name", "Type", "Value"],
        ["``SC_r``", "Integer", "``1.50``"],
        ["``SC_s``", "String", '``"  Hello"``'],
        ["``SC_t``", "String", "``null``"],
    ]
    assert "\n\nExpected error ``2-1-15-6``\n\n" in text


def test_the_types_say_what_a_form_takes_and_gives_and_list_the_operators_by_block(
    tmp_path: Path,
) -> None:
    text = operator_rst(NUMERIC, ROUND, MAP, INDEX, META, examples_root=tmp_path)
    types = section(text, "Types")
    assert '<p class="om-takes"><b>Takes</b> number, integer · <b>Gives</b> number</p>' in types
    # Comes from: the producers of a number and of an integer, in one list by block, each once
    # (Ceiling gives an integer, which feeds both). Nvl returns its input's type: it is in the
    # closed list of the operators that return any type.
    comes_from = re.search(r'<div class="om-col">(.*?)</div><div class="om-col">', types, re.S)
    assert comes_from is not None
    left = comes_from.group(1)
    assert "<b>Comes from · number, integer</b> — 3 operators in 2 blocks 🟢1 🟡1 🔵1" in left
    assert left.count('href="../numeric/ceil.html"') == 1
    assert '<details class="om-dir"><summary><b>Comes from · operators that return any type' in left
    assert 'href="../conditional/nvl.html"' in left
    # Goes to: the takers of a number, and the closed list of the takers of any scalar.
    right = types.split('<div class="om-col">', 2)[2]
    assert "<b>Goes to · number</b> — 2 operators in 1 block" in right
    assert '<a class="om-chip diverge" href="../numeric/modulo.html"' in right
    assert '<details class="om-dir"><summary><b>Goes to · operators that take any scalar' in right
    # A block of a list is closed, and says how many it has and their statuses.
    assert '<details class="om-blk"><summary>Numeric operators <b>(2)</b> 🟢1 🔵1</summary>' in (
        left
    )


def test_an_operator_on_any_scalar_says_so_in_one_line(tmp_path: Path) -> None:
    types = section(
        operator_rst(CONDITIONAL, NVL, MAP, INDEX, META, examples_root=tmp_path), "Types"
    )
    assert "<b>Takes</b> any scalar · <b>Gives</b> any type" in types
    assert '<p class="om-line"><b>Comes from:</b> any operator that returns a value</p>' in types
    assert '<p class="om-line"><b>Goes to:</b> whatever takes its input&#x27;s type</p>' in types
    assert "om-dir" not in types  # no list: one line on each side


def test_a_grouped_used_with_box_is_listed_in_full_under_the_diagram(tmp_path: Path) -> None:
    users = tuple(
        operator(
            f"o{n}",
            f"Operator {n}",
            f"op{n}",
            form("sc", "f(sc)", NUMBER, NUMBER),
            links=(Link("clause.calc", "used in"),),
        )
        for n in range(USED_WITH_MAX + 1)
    )
    calc = operator("calc", "Calculation", "calc", form("measure", "calc", ("scalar",), ("any",)))
    clause = Chapter("clause", "Clause operators", (calc,))
    cat = Catalogue((Chapter("numeric", "Numeric operators", users), clause))
    index = index_of(cat)
    text = operator_rst(clause, calc, cat, index, META, examples_root=tmp_path)
    after_diagram = text.split(".. raw:: html\n\n   </div>\n\n", 1)[1].split("\nTypes\n", 1)[0]
    assert "<b>Used with</b> — 9 operators in 1 block 🟢9" in after_diagram
    assert after_diagram.count("<small>uses</small>") == USED_WITH_MAX + 1
    assert 'href="../numeric/o0.html"' in after_diagram
    # With fewer entries, the box shows them all: no list.
    small = Catalogue((Chapter("numeric", "Numeric operators", users[:USED_WITH_MAX]), clause))
    text = operator_rst(clause, calc, small, index_of(small), META, examples_root=tmp_path)
    assert "<b>Used with</b>" not in text


def test_mermaid_blocks_reads_back_the_blocks_of_a_page(tmp_path: Path) -> None:
    text = operator_rst(NUMERIC, ROUND, MAP, INDEX, META, examples_root=tmp_path)
    assert mermaid_blocks(text) == [operator_diagram(NUMERIC, ROUND, MAP, INDEX)]
    assert mermaid_blocks(block_rst(NUMERIC, MAP, INDEX, META)) == [block_diagram(NUMERIC, INDEX)]
    assert mermaid_blocks(overview_rst(MAP, INDEX, META)) == [overview_block(MAP, INDEX)]
    assert mermaid_blocks(index_rst(MAP, INDEX, META)) == []


# --- The boxes that the forms name ---


def define_page(root: Path) -> str:
    """The page of ``DEFINE``, whose forms name the boxes that they are in."""
    write_define(root)
    return operator_rst(DEFINE_CHAPTER, DEFINE, DEFINE_MAP, DEFINE_INDEX, META, examples_root=root)


def test_the_forms_in_titled_boxes_have_a_caption_before_the_first_form_of_each_box(
    tmp_path: Path,
) -> None:
    text = define_page(tmp_path)
    # The legend of the statuses is there once. The key of the borders is for the boxes of the
    # kinds: the titles tell these boxes apart.
    assert text.count(LEGEND_HTML) == 1
    assert BORDER_KEY_HTML not in text
    assert "om-borders" not in text
    assert "om-box " not in text
    # The diagram is the one of the operator, drawn in the boxes that its forms name.
    assert blocks(text) == [operator_diagram(DEFINE_CHAPTER, DEFINE, DEFINE_MAP, DEFINE_INDEX)]
    assert 'subgraph G_b1["Body"]' in blocks(text)[0]
    # One caption for each box, in the order of the boxes of the diagram.
    assert re.findall(r'<p class="om-box-title">(.*?)</p>', text) == DEFINE_BOXES
    # The sections of the forms are in the order of the diagram, with a caption before the first
    # form of each box, and before the anchor of that form.
    forms = {variant.id: variant for variant in DEFINE.variants}
    taken = {
        "body_when": "boolean",
        "body_aggr": "number",
        "sign_variable": "any scalar",
        "sign_domain": "value domain",
        "result": "any scalar",
    }

    def heading(form_id: str) -> str:
        return f"{forms[form_id].label} 🟢 · {taken[form_id]} → viral propagation"

    def form_part(form_id: str) -> list[str]:
        anchor = f'raw <span class="om-anchor" id="{form_anchor(forms[form_id])}">'
        return [anchor, f"~ {heading(form_id)}"]

    caption = 'raw <p class="om-box-title">'
    assert [
        item
        for item in skeleton(text)
        if item.startswith(("- ", "~ ", caption, 'raw <span class="om-anchor"'))
    ] == [
        "- Types",
        "- Forms",
        caption,
        *form_part("body_when"),
        *form_part("body_aggr"),
        caption,
        *form_part("sign_variable"),
        *form_part("sign_domain"),
        caption,
        *form_part("result"),
    ]
    first = (
        '.. raw:: html\n\n   <p class="om-box-title">Body</p>\n\n'
        '.. raw:: html\n\n   <span class="om-anchor" id="form-body-when"></span>\n\n'
    )
    assert first in text
    # The captions are not headings: the sections of the page are what they were.
    assert titles(text, "-") == ["Types", "Forms"]
    assert titles(text, "~") == [heading(form_id) for form_id in DEFINE_ORDER]
    # Every click of the diagram still lands on the anchor of its form.
    clicks = re.findall(r'click \w+ href "#([^"]+)"', blocks(text)[0])
    anchors = re.findall(r'<span class="om-anchor" id="([^"]+)"></span>', text)
    assert sorted(clicks) == sorted(anchors) == sorted(form_anchor(v) for v in DEFINE.variants)


def test_the_caption_of_a_box_is_written_once_for_the_forms_that_are_not_next_to_each_other(
    tmp_path: Path,
) -> None:
    # The catalogue lists the forms of the Body and of the Signature in turn: the page lists them
    # box by box, so that no caption is repeated.
    text = define_page(tmp_path)
    for title in DEFINE_BOXES:
        assert text.count(f'<p class="om-box-title">{title}</p>') == 1
    assert [v.id for v in DEFINE.variants] != DEFINE_ORDER
    anchors = re.findall(r'<span class="om-anchor" id="form-([^"]+)"></span>', text)
    assert anchors == [form_id.replace("_", "-") for form_id in DEFINE_ORDER]


def test_the_title_of_a_box_is_escaped_in_its_caption(tmp_path: Path) -> None:
    hostile = 'A & <B> "C"'
    only = operator("o", "O", "o", form("v", "v", NUMBER, NUMBER, box=hostile))
    chapter = Chapter("c", "C", (only,))
    cat = Catalogue((chapter,))
    write_example(tmp_path, "c/o/v/two_rows", {})
    text = operator_rst(chapter, only, cat, index_of(cat), META, examples_root=tmp_path)
    assert '<p class="om-box-title">A &amp; &lt;B&gt; &quot;C&quot;</p>' in text
    assert f'<p class="om-box-title">{hostile}' not in text
    # The same title is in the diagram, escaped as a label is, not as HTML.
    assert 'subgraph G_b1["A & #60;B#62; #quot;C#quot;"]' in blocks(text)[0]


def test_the_captions_of_the_boxes_are_raw_html_and_make_no_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nodes = pytest.importorskip("docutils.nodes")
    examples = tmp_path / "examples"
    text = define_page(examples)
    tree = read_rst(text, tmp_path / "definitions" / "define_x.rst", monkeypatch)
    raws = [node for node in tree.findall(nodes.raw) if "om-box-title" in node.astext()]
    assert [node.astext() for node in raws] == [
        f'<p class="om-box-title">{title}</p>' for title in DEFINE_BOXES
    ]
    assert {node["format"] for node in raws} == {"html"}
    # The page has the sections that it had: its title, Types, Forms, and a section for each form.
    assert [title.astext() for title in tree.findall(nodes.title)] == [
        "Define x · define x",
        "Types",
        "Forms",
        *(title for title in titles(text, "~")),
    ]


# --- Viral propagation, in text ---


VIRAL_DEFINITION = operator(
    "define_viral_propagation",
    "define viral propagation",
    "",
    form("variable", "variable At_1", ("scalar",), ("viral_propagation",)),
    page="dl/viral",
)
DEFINITIONS = Chapter("definitions", "Definitions (VTL-DL)", (VIRAL_DEFINITION,))
VIRAL_MAP = Catalogue((NUMERIC, STRING, CONDITIONAL, DEFINITIONS))
VIRAL_INDEX = index_of(VIRAL_MAP, {"numeric/modulo/ds_ds/manual_ex_1": DIVERGE})
GROUPS = (
    ViralGroup(
        "Operators on two data sets",
        "typical",
        "For each couple (DP_1, DP_2) ... the proper Attribute propagation algorithm",
        ("numeric.modulo", "string.length"),
    ),
    # An operator that the catalogue does not have is not listed.
    ViralGroup("Rounding", "Numeric operators/Rounding", "It's applied", ("numeric.round", "x.y")),
)
VIRAL_LINE = (
    '<p class="om-viral"><b>Viral attributes:</b> combined with the propagation rules → '
    '<a href="../definitions/define_viral_propagation.html">Viral propagation</a></p>'
)


def test_an_operator_that_applies_the_viral_rules_says_so_under_its_types(tmp_path: Path) -> None:
    text = operator_rst(
        NUMERIC, MODULO, VIRAL_MAP, VIRAL_INDEX, META, examples_root=tmp_path, viral=GROUPS
    )
    takes = '<p class="om-takes"><b>Takes</b> number · <b>Gives</b> number</p>'
    assert f'   {takes}\n   {VIRAL_LINE}\n   <div class="om-cols">' in text
    assert text.count("om-viral") == 1
    # Never in a diagram.
    assert "viral" not in blocks(text)[0].lower()
    # An operator that is in no group has no line.
    ceil = operator_rst(
        NUMERIC, CEIL, VIRAL_MAP, VIRAL_INDEX, META, examples_root=tmp_path, viral=GROUPS
    )
    assert "om-viral" not in ceil
    # Without the page of the definition, the line names it without a link.
    alone = operator_rst(NUMERIC, ROUND, MAP, INDEX, META, examples_root=tmp_path, viral=GROUPS)
    assert "→ Viral propagation</p>" in alone


def test_the_viral_propagation_page_lists_the_operators_that_apply_it(tmp_path: Path) -> None:
    text = operator_rst(
        DEFINITIONS,
        VIRAL_DEFINITION,
        VIRAL_MAP,
        VIRAL_INDEX,
        META,
        examples_root=tmp_path,
        viral=GROUPS,
    )
    assert titles(text, "-") == ["Types", "Applied by", "Forms"]
    applied = section(text, "Applied by")
    # Each group, with the sentence of the manual and its page.
    assert (
        '- **Operators on two data sets**: \\"For each couple (DP\\_1, DP\\_2) \\.\\.\\. '
        'the proper Attribute propagation algorithm\\" '
        f"(`Typical behaviours of the ML operators <{BASE}/typical_behaviour.html>`__)"
    ) in applied.splitlines()
    assert (
        '- **Rounding**: \\"It\\\'s applied\\" '
        f"(`Rounding <{BASE}/operators/Numeric%20operators/Rounding/index.html>`__)"
    ) in applied.splitlines()
    # The operators, by block, each block closed, as the type lists: the unknown one is not there.
    assert "<b>Applied by</b> — 3 operators in 2 blocks 🟢2 🔵1" in applied
    assert '<details class="om-blk"><summary>Numeric operators <b>(2)</b> 🟢1 🔵1</summary>' in (
        applied
    )
    for target in ("../numeric/modulo.html", "../numeric/round.html", "../string/length.html"):
        assert f'href="{target}"' in applied
    assert "x.y" not in applied
    assert ">y<" not in applied
    # Only the page of the definition has the section, and its diagram has none of them.
    modulo = operator_rst(
        NUMERIC, MODULO, VIRAL_MAP, VIRAL_INDEX, META, examples_root=tmp_path, viral=()
    )
    assert "Applied by" not in modulo
    assert "modulo" not in blocks(text)[0]


def test_the_viral_groups_name_operators_of_the_catalogue() -> None:
    cat = load_catalogue()
    addresses = {f"{chapter.id}.{op.id}" for chapter in cat.chapters for op in chapter.operators}
    groups = load_viral_groups()
    listed = [address for group in groups for address in group.operators]
    # Each operator once, every one of the catalogue, and no definition of the VTL-DL.
    assert len(listed) == len(set(listed))
    assert set(listed) <= addresses
    assert not [address for address in listed if address.startswith("definitions.")]
    # What the manual leaves out: the time series and the set operators, membership, and
    # if-then-else and case, whose result takes each data point from one operand.
    left_out = {
        "time.fill_time_series",
        "time.flow_to_stock",
        "time.stock_to_flow",
        "time.timeshift",
        "set.union",
        "set.intersect",
        "set.setdiff",
        "set.symdiff",
        "general.membership",
        "conditional.if_then_else",
        "conditional.case",
    }
    assert not left_out & set(listed)
    # The joins, hierarchy and check_hierarchy, as the design lists them.
    joins = {"join.inner_join", "join.left_join", "join.full_join", "join.cross_join"}
    assert joins | {"hierarchy.hierarchy", "validation.check_hierarchy"} <= set(listed)
    # Each group quotes a page of the manual.
    for group in groups:
        assert group.page == "typical" or group.page in manual_index.MANUAL_PAGES, group.page
        assert group.sentence, group.title
        assert group.operators, group.title


# --- The page of a block ---


def test_block_page_toctree_and_types_table() -> None:
    text = block_rst(NUMERIC, MAP, INDEX, META)
    assert text.startswith(f"{'=' * 34}\nNumeric operators\n{'=' * 34}\n\n")
    # Its parts are rubrics, not sections, so that the sidebar has the operators right under the
    # block.
    assert titles(text, "-") == []
    assert re.findall(r"^\.\. rubric:: (.+)$", text, re.MULTILINE) == [
        "Operators",
        "Types",
        "Forms that are not green",
    ]
    assert '<span class="om-chip diverge">🔵 2 of 3 operators all ok</span>' in text
    assert blocks(text) == [block_diagram(NUMERIC, INDEX)]
    # The operator pages are under the block page, in a hidden toctree, in catalogue order.
    assert text.endswith(
        ".. toctree::\n   :maxdepth: 1\n   :hidden:\n\n   modulo\n   round\n   ceil\n"
    )
    assert table_rows(section(text, "Operators")) == [
        ["Operator", "Status", "Forms"],
        [":doc:`Modulo <modulo>` ``mod``", "🔵", "3"],
        [":doc:`Rounding <round>` ``round``", "🟢", "2"],
        [":doc:`Ceiling <ceil>` ``ceil``", "🟢", "1"],
    ]
    # The types that the forms of the block declare, and only those, in the order of the types.
    assert table_rows(section(text, "Types")) == [
        ["Operator", "number", "integer"],
        [":doc:`Modulo <modulo>`", "in · out", ""],
        [":doc:`Rounding <round>`", "in · out", "in"],
        [":doc:`Ceiling <ceil>`", "in", "out"],
    ]


def test_block_page_lists_the_forms_that_are_not_green_with_their_reason() -> None:
    rows = table_rows(section(block_rst(NUMERIC, MAP, INDEX, META), "Forms that are not green"))
    # In the order of the diagram, each with the first example that shows its status.
    assert rows == [
        ["Operator", "Form", "Status", "Example", "pandas", "DuckDB", "Detail"],
        [
            ":doc:`Modulo <modulo>`",
            "`mod(DS, DS) <modulo.html#form-ds-ds>`__",
            "🔵",
            "`Manual example 1 <modulo.html#ex-ds-ds-manual-ex-1>`__",
            "✅",
            "💥 KeyError",
            "DuckDB: KeyError: \\'Me\\_1\\'",
        ],
        [
            ":doc:`Modulo <modulo>`",
            "`mod(sc, sc) <modulo.html#form-sc-sc>`__",
            "🟡",
            "`Zero divisor <modulo.html#ex-sc-sc-zero-divisor>`__",
            "🛡 2-1-15-6",
            "🛡 2-1-15-6",
            "RunTimeError 2-1-15-6: Scalar division by Zero.",
        ],
    ]
    assert "Every form is green." in block_rst(STRING, MAP, index_of(MAP), META)


# --- The index, the overview and the page of the operators ---


def test_index_structure() -> None:
    text = index_rst(MAP, INDEX, META)
    rule = "=" * 40  # twice the 20 characters of the title
    assert text.startswith(f"{rule}\nVTL 2.2 operator map\n{rule}\n\n")
    # The parts of the map are on one level of the sidebar: the map has no definitions.
    toctree = ".. toctree::\n   :maxdepth: 1\n   :hidden:\n\n   overview\n   operators\n"
    assert toctree in text
    assert titles(text, "-") == []
    assert text.index(toctree) < text.index("Generated on 2026-10-23")
    assert (
        "Generated on 2026-10-23 from vtlengine 9.9.9 (commit ``abc1234``), with pandas 2.3.3 and "
        "DuckDB 1.5.5"
    ) in " ".join(text.split())
    assert f"https://github.com/sdmx-twg/vtl/tree/{SPEC}" in text
    parts = [line for line in text.splitlines() if line.startswith("- :doc:")]
    assert [line.split(": ", 1)[0] for line in parts] == [
        "- :doc:`Overview <overview>`",
        "- :doc:`Operators <operators>`",
    ]
    assert "the 3 blocks of operators" in parts[1]
    assert blocks(text) == []
    assert ".md" not in text


def test_the_definitions_are_a_part_of_their_own() -> None:
    cat = Catalogue((NUMERIC, DEFINE_CHAPTER))
    index = index_of(cat)
    text = index_rst(cat, index, META)
    assert (
        ".. toctree::\n   :maxdepth: 1\n   :hidden:\n\n"
        "   overview\n   operators\n   definitions/index\n"
    ) in text
    assert "- :doc:`Definitions <definitions/index>`: the definitions of the VTL-DL" in text
    # The page of the operators has the blocks of operators only, in its table and its toctree.
    operators = operators_rst(cat, index)
    assert ".. toctree::\n   :maxdepth: 1\n   :hidden:\n\n   numeric/index\n" in operators
    assert "definitions" not in operators
    # The overview has every block, the definitions too.
    assert ":doc:`Definitions <definitions/index>`" in section(
        overview_rst(cat, index, META), "Blocks"
    )


def test_index_without_blocks_has_only_the_overview() -> None:
    empty = Catalogue(())
    text = index_rst(empty, StatusIndex({}, {}, {}, {}), META)
    assert ".. toctree::\n   :maxdepth: 1\n   :hidden:\n\n   overview\n" in text
    assert "   overview\n   operators" not in text
    assert ":doc:`Operators <operators>`" not in text
    assert ".. toctree::" not in operators_rst(empty, StatusIndex({}, {}, {}, {}))


def test_overview_structure() -> None:
    text = overview_rst(MAP, INDEX, META)
    rule = "=" * 16  # twice the 8 characters of the title
    assert text.startswith(f"{rule}\nOverview\n{rule}\n\n")
    assert titles(text, "-") == [
        "How to read",
        "Overview diagram",
        "Blocks",
        "Not implemented",
        "📌 Team decisions",
        ERRATA_TITLE,
        "Adaptations",
        "Attribution",
    ]
    assert text.split("\nHow to read\n", 1)[0].count(LEGEND_HTML) == 1
    assert ".. toctree::" not in text
    assert blocks(section(text, "Overview diagram")) == [overview_block(MAP, INDEX)]
    assert len(blocks(text)) == 1
    how = " ".join(section(text, "How to read").split())
    for words in ("✅", "❌", "🛡", "💥", "solid", "dashed", "dotted", "Used with", "0.6001–0.7999"):
        assert words in how, words
    assert "three parts: this overview, the operators and the definitions" in how
    assert "Elsewhere the cap is optional." in how
    assert ".md" not in text


def test_operators_page_structure() -> None:
    text = operators_rst(MAP, INDEX)
    rule = "=" * 18  # twice the 9 characters of the title
    assert text.startswith(f"{rule}\nOperators\n{rule}\n\n")
    assert text.count(LEGEND_HTML) == 1
    assert (
        ".. toctree::\n   :maxdepth: 1\n   :hidden:\n\n"
        "   numeric/index\n   string/index\n   conditional/index\n"
    ) in text
    assert table_rows(text)[1][0] == ":doc:`Numeric operators <numeric/index>`"


def test_overview_block_table_counts_the_forms() -> None:
    rows = table_rows(section(overview_rst(MAP, INDEX, META), "Blocks"))
    assert rows == [
        ["Block", "Status", "Operators", "Forms", "🟢", "🟡", "🛡️", "🔴", "🔵"],
        [":doc:`Numeric operators <numeric/index>`", "🔵", "3", "6", "4", "1", "0", "0", "1"],
        [":doc:`String operators <string/index>`", "🟡", "1", "1", "0", "1", "0", "0", "0"],
        [
            ":doc:`Conditional operators <conditional/index>`",
            "🟢",
            "1",
            "1",
            "1",
            "0",
            "0",
            "0",
            "0",
        ],
        ["**Total**", "🔵", "5", "8", "5", "2", "0", "0", "1"],
    ]
    # The page of the operators has the same rows for the same blocks.
    assert table_rows(operators_rst(MAP, INDEX)) == rows


def test_overview_lists_the_codes_that_are_not_implemented() -> None:
    rows = table_rows(section(overview_rst(MAP, INDEX, META), "Not implemented"))
    assert rows[0] == ["Code", "Message"]
    assert rows[1:] == [[f"``{code}``", message] for code, message in NOT_IMPLEMENTED.items()]


def test_overview_lists_link_to_anchors_by_operator_and_example() -> None:
    text = overview_rst(MAP, INDEX, META)
    decisions = section(text, "📌 Team decisions")
    assert (
        f"- `📌 #1049 <{ISSUES}/1049>`__ · "
        "`Modulo · Two components <numeric/modulo.html#ex-comp-comp-two-components>`__ "
        "(Numeric operators › ``mod(comp, comp)``)"
    ) in decisions.splitlines()
    errata = section(text, ERRATA_TITLE)
    # The quotes are escaped, so that Sphinx does not turn them into typographic ones.
    assert (
        "- `Modulo · Manual example 1 <numeric/modulo.html#ex-ds-ds-manual-ex-1>`__ "
        "(Numeric operators › ``mod(DS, DS)``): "
        "the manual shows \\'ba\\' where \\'ab\\' is meant"
    ) in errata.splitlines()
    adaptations = section(text, "Adaptations")
    assert (
        "- `String length · Plain <string/length.html#ex-sc-plain>`__ "
        "(String operators › ``length(sc)``): the time values were rewritten"
    ) in adaptations.splitlines()
    # Each anchor is the one of the example on its page.
    assert example_anchor(MODULO.variants[1], MODULO.variants[1].examples[0]) in errata
    for body in (decisions, errata, adaptations):
        assert [line[:2] for line in body.splitlines() if line.strip()][-1] == "- "


def test_overview_lists_say_what_they_list_and_when_they_are_empty() -> None:
    text = overview_rst(MAP, INDEX, META)
    assert "an expected output, a script or an input" in section(text, ERRATA_TITLE)
    assert "These are not errata." in section(text, "Adaptations")
    empty = Catalogue((CONDITIONAL,))
    text = overview_rst(empty, index_of(empty), META)
    assert "None: no expected output follows a recorded team decision." in text
    assert "None: nothing in the manual's examples had to be corrected." in text
    assert "None: no example needed an adaptation." in text


def test_overview_escapes_the_free_text_of_the_lists() -> None:
    erratum = "the *manual* prints a_b | c, `x` and \\ and ends with ::"
    variant = form("v", "f(x)", NUMBER, NUMBER, authored("case", erratum=erratum))
    cat = Catalogue((Chapter("c", "C", (operator("o", "Some_op", "f", variant),)),))
    errata = section(overview_rst(cat, index_of(cat), META), ERRATA_TITLE)
    assert (
        "- `Some\\_op · Case <c/o.html#ex-v-case>`__ (C › ``f(x)``): the \\*manual\\* prints "
        "a\\_b \\| c, \\`x\\` and \\\\ and ends with :\\:"
    ) in errata.splitlines()
    typeset = 'it\'s "a" -- b --- c - d... e. . . f'
    variant = form("v", "f(x)", NUMBER, NUMBER, authored("case", erratum=typeset))
    cat = Catalogue((Chapter("c", "C", (operator("o", "Some_op", "f", variant),)),))
    errata = section(overview_rst(cat, index_of(cat), META), ERRATA_TITLE)
    assert ': it\\\'s \\"a\\" \\-\\- b \\-\\-\\- c - d\\.\\.\\. e\\. \\. \\. f' in errata


def test_overview_attribution_sentence_is_verbatim() -> None:
    attribution = section(overview_rst(MAP, INDEX, META), "Attribution")
    assert (
        "Examples marked *manual* are taken from the VTL 2.2 Reference Manual "
        f"(sdmx-twg/vtl @ {SPEC}), converted to the vtlengine format."
    ) in attribution
    assert f"`sdmx-twg.github.io/vtl/v2.2 <{BASE}/index.html>`__" in attribution


# --- Free text: what went wrong with an example, and the labels of the forms ---
#
# These run without docutils: they pin the text that the pages hold for what RST would read as
# markup. The reading of the same texts by docutils is the check of the next section.


def test_detail_is_one_line_escaped_and_cut_at_200_characters() -> None:
    assert detail_cell("line one\nline two | pipe\r\nend\rlast") == (
        "line one line two \\| pipe end last"
    )
    assert detail_cell("x" * 200) == "x" * 200
    assert detail_cell("x" * 201) == "x" * 199 + "…"
    # The markup is escaped after the cut: the 200 characters are the visible ones.
    assert detail_cell("|" * 250) == "\\|" * 199 + "…"
    assert detail_cell("_" * 250) == "\\_" * 199 + "…"
    # A backslash that ends up last before the ellipsis is not read as escaping it.
    assert detail_cell("a" * 198 + "\\\\b") == "a" * 198 + "\\\\…"


@pytest.mark.parametrize(
    ("detail", "cell"),
    [
        ("row Id_1=2: Me_1 expected 3 got 2.0", "row Id\\_1=2: Me\\_1 expected 3 got 2.0"),
        ("a*b and **c**", "a\\*b and \\*\\*c\\*\\*"),
        ("a|b|", "a\\|b\\|"),
        ("`code` and ``more``", "\\`code\\` and \\`\\`more\\`\\`"),
        ("back\\slash and \\* done", "back\\\\slash and \\\\\\* done"),
        ("ends with _", "ends with \\_"),
        ("a :role:`x` b", "a :role:\\`x\\` b"),
        ("[1]_ and name_ and _`target`", "\\ [1]\\_ and name\\_ and \\_\\`target\\`"),
        ("|sub| ref", "\\|sub\\| ref"),
        # What Sphinx would typeset as it reads the text: quotes, runs of hyphens and ellipses.
        ('Parser Error at or near "SELECT"', 'Parser Error at or near \\"SELECT\\"'),
        ("it's a -- b --- c - d -> e", "it\\'s a \\-\\- b \\-\\-\\- c - d -> e"),
        ("wait... then . . . and .. no", "wait\\.\\.\\. then \\. \\. \\. and .. no"),
        ("2.5 and a.b and 1-1-1-1", "2.5 and a.b and 1-1-1-1"),
        # A backslash right before a hyphen or a dot is lost under the smart quotes of docutils.
        ("regex \\d+\\.\\d+ and \\-x", "regex \\\\d+\\\\\\.\\\\d+ and \\\\\\-x"),
        ("a\\'b a\\\"b", "a\\\\\\'b a\\\\\\\"b"),
    ],
)
def test_detail_escapes_what_rst_reads_as_markup(detail: str, cell: str) -> None:
    assert detail_cell(detail) == cell


@pytest.mark.parametrize(
    ("detail", "cell"),
    [
        ("- item", "\\ - item"),
        ("+ item", "\\ + item"),
        ("• item", "\\ • item"),
        ("1. item", "\\ 1. item"),
        ("a) item", "\\ a) item"),
        ("(i) item", "\\ (i) item"),
        ("#. item", "\\ #. item"),
        (":field: item", "\\ :field: item"),
        (">>> item", "\\ >>> item"),
        (".. comment", "\\ .. comment"),
        ("----", "\\-\\-\\-\\-"),
        ("==== ====", "\\ ==== ===="),
        ("'quoted' text", "\\'quoted\\' text"),
        ('"quoted" text', '\\"quoted\\" text'),
        ("-- option", "\\-\\- option"),
        (". . . dots", "\\. \\. \\. dots"),
        ("* item", "\\* item"),
        ("__ item", "\\_\\_ item"),
        ("| item", "\\| item"),
        ("see below::", "see below:\\:"),
        ("\\\\", "\\ \\\\\\\\"),  # two backslashes would be four: a transition
        ("not a list 1. item", "not a list 1. item"),
        ("e.g. a value", "e.g. a value"),
    ],
)
def test_detail_that_would_start_a_block_or_end_with_two_colons_is_guarded(
    detail: str, cell: str
) -> None:
    assert detail_cell(detail) == cell


def test_detail_names_the_engine_that_has_one() -> None:
    error = EngineVerdict("error", "SemanticError 1-1-1-1: boom")
    other = EngineVerdict("error", "RunTimeError 2-1-0-1: not implemented")
    assert detail_of(ExampleResult(MATCH, error, False, "diverge")) == (
        "DuckDB: SemanticError 1-1-1-1: boom"
    )
    assert detail_of(ExampleResult(error, MATCH, False, "diverge")) == (
        "pandas: SemanticError 1-1-1-1: boom"
    )
    # Two errors that differ are both shown; two that are the same are shown once.
    assert detail_of(ExampleResult(error, other, True, "missing")) == (
        "pandas: SemanticError 1-1-1-1: boom; DuckDB: RunTimeError 2-1-0-1: not implemented"
    )
    assert detail_of(ExampleResult(error, error, True, "missing")) == (
        "SemanticError 1-1-1-1: boom"
    )
    # An example that matches has nothing to say, and its form is not in the table.
    assert "Every form is green." in one_form_block(OK)


def test_engines_that_differ_with_the_same_verdict_say_so() -> None:
    # Both match, or both fail the same way, and still give different outputs: nothing else says.
    assert detail_of(ExampleResult(MATCH, MATCH, False, "diverge")) == (
        "the engines give different outputs"
    )
    assert detail_of(ExampleResult(WRONG, WRONG, False, "diverge")) == (
        "row Id\\_1=1: Me\\_1 expected \\'a\\' got \\'b\\'; the engines give different outputs"
    )


def test_a_label_that_a_literal_cannot_hold_is_escaped_text() -> None:
    labels = {"tick": "a `b` c", "lines": "line one\nline two", "empty": "", "slash": "a\\"}
    variants = tuple(
        form(name, label, NUMBER, NUMBER, authored("minimal", erratum="e"))
        for name, label in labels.items()
    )
    cat = Catalogue((Chapter("c", "C", (operator("o", "O", "o", *variants),)),))
    errata = section(overview_rst(cat, index_of(cat), META), ERRATA_TITLE).splitlines()

    def item(variant: str, label: str) -> str:
        return f"- `O · Minimal <c/o.html#ex-{variant}-minimal>`__ (C › {label}): e"

    # Where a form is named the label is a literal, as it is: on one line, and a backslash last
    # is fine. A backtick cannot be in a literal, and the label is escaped text then.
    assert item("tick", "a \\`b\\` c") in errata
    assert item("lines", "``line one line two``") in errata
    assert item("empty", "") in errata
    assert item("slash", "``a\\``") in errata


def test_example_without_a_result_is_not_run(tmp_path: Path) -> None:
    # The statuses of the operators and of the forms, and no result for any example.
    no_results = StatusIndex({}, INDEX.variants, INDEX.operators, INDEX.chapters)
    text = operator_rst(CONDITIONAL, NVL, MAP, no_results, META, examples_root=tmp_path)
    summaries = re.findall(
        r'<details class="om-example" id="([^"]+)"[^>]*><summary>(.*?)</summary>', text
    )
    # The title and a chip of no status that says it: no marks of the engines, and no line of
    # what went wrong.
    assert summaries == [
        (
            "ex-sc-sc-two-rows",
            '<span class="om-ex-title">Two rows</span> <span class="om-chip">not run</span>',
        )
    ]
    assert "om-marks" not in text
    assert "om-detail" not in text


# --- What a reader of RST makes of the pages ---


def read_rst(text: str, source: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The docutils tree of a page at ``source``. The directives of Sphinx that it uses are stubs.

    The stubs are registered with ``monkeypatch``, so that they are gone after the check: a build
    of Sphinx in the same process registers the real ones. Every message of docutils of level 2
    (a warning) or above fails the check. The smart quotes
    are on, as they are in Sphinx: they turn quotes, runs of hyphens and ellipses into
    typographic characters, in every text that is not a literal. ``.. include::`` is the one of
    docutils, so the notes of the examples are read too, relative to ``source``.
    """
    core = pytest.importorskip("docutils.core")
    nodes = pytest.importorskip("docutils.nodes")
    rst = pytest.importorskip("docutils.parsers.rst")
    directives = pytest.importorskip("docutils.parsers.rst.directives")
    roles = pytest.importorskip("docutils.parsers.rst.roles")

    def stub(options: dict[str, Any], arguments: int = 0) -> Any:
        return type(
            "Stub",
            (rst.Directive,),
            {
                "has_content": True,
                "optional_arguments": arguments,
                "option_spec": options,
                "run": lambda self: [],
            },
        )

    def doc_role(*args: Any) -> tuple[list[Any], list[Any]]:
        text = args[2]  # the role is called with (name, rawtext, text, lineno, inliner, ...)
        return [nodes.Text(text.partition(" <")[0])], []

    registered = directives._directives  # what register_directive fills
    monkeypatch.setitem(registered, "mermaid", stub({"config": directives.unchanged}))
    toctree = stub({"maxdepth": directives.unchanged, "hidden": directives.flag})
    monkeypatch.setitem(registered, "toctree", toctree)
    literal = stub({"caption": directives.unchanged, "language": directives.unchanged}, 1)
    monkeypatch.setitem(registered, "literalinclude", literal)
    monkeypatch.setitem(roles._roles, "doc", doc_role)  # as register_local_role does
    messages: list[str] = []

    class Collector:
        def write(self, message: str) -> None:
            messages.append(message)

    tree = core.publish_doctree(
        text,
        source_path=str(source),
        settings_overrides={
            "report_level": 2,
            "halt_level": 5,
            "warning_stream": Collector(),
            "smart_quotes": True,
            "docinfo_xform": False,
        },
    )
    assert not messages, "".join(messages)
    return tree


# Details that RST would read as markup or as the start of another element, and labels that are
# hostile in the same way: they have to come out of the reading as they went in.
HOSTILE = [
    "row Id_1=2: Me_1 expected *x* `t` a|b c\\d",
    "- looks like a bullet",
    "1. looks like a list",
    ":looks: like a field",
    ">>> looks like a doctest",
    ".. looks like a comment",
    "----",
    "ends with a literal block mark::",
    "name_ ref_ [1]_ |sub| _`target` **strong** ``literal``",
    "'quoted' \"text\" --dash... end\\",
    'Parser Error: syntax error at or near "SELECT" -- it\'s a query... ---. . .',
    "DuckDB: Binder Error: column \"Me_1\" not found, candidates: 'Me_2' -- didn't match...",
    "regex \\d+\\.\\d+ and \\-x, \\'s\\' and \\\"q\\\" and \\\\-- and \\...",
    "\\\\",
]
HOSTILE_LABELS = ["f(x) || *y* _z_", "f(`n`) || *y* _z_", "f(x) \\"]


def test_the_pages_are_read_as_written_by_docutils(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("docutils")
    nodes = pytest.importorskip("docutils.nodes")
    forms = tuple(
        form(
            f"v{n}",
            HOSTILE_LABELS[n % len(HOSTILE_LABELS)],
            NUMBER,
            NUMBER,
            authored("case", adaptation=detail, erratum=detail),
        )
        for n, detail in enumerate(HOSTILE)
    )
    hostile = operator("o_p", "Name_with *marks* | `ticks` \\", "||", *forms)
    chapter = Chapter("c", "C_hapter", (hostile,))
    cat = Catalogue((chapter,))
    results = {}
    for n, detail in enumerate(HOSTILE):
        verdict = EngineVerdict("mismatch", detail)
        results[f"c/o_p/v{n}/case"] = ExampleResult(verdict, verdict, True, "partial")
    index = index_of(cat, results)
    examples = tmp_path / "examples"
    for n in range(len(HOSTILE)):
        write_example(examples, f"c/o_p/v{n}/case", {"input/DS_1.csv": EDGE_CSV})

    page = operator_rst(chapter, hostile, cat, index, META, examples_root=examples)
    tree = read_rst(page, tmp_path / "c" / "o_p.rst", monkeypatch)
    assert [title.astext() for title in tree.findall(nodes.title)] == [
        "Name_with *marks* | `ticks` \\ · ||",
        "Types",
        "Forms",
        *(f"{HOSTILE_LABELS[n % 3]} 🟡 · number → number" for n in range(len(HOSTILE))),
    ]
    # The notes are read as fields, from the folders of the examples.
    fields = [field.astext() for field in tree.findall(nodes.field_name)]
    assert fields[:3] == ["Tests", "Manual", "Expected"]

    block = read_rst(
        block_rst(chapter, cat, index, META), tmp_path / "c" / "index.rst", monkeypatch
    )
    rows = [
        [entry.astext() for entry in row.findall(nodes.entry)] for row in block.findall(nodes.row)
    ]
    not_green = [row for row in rows if len(row) == 7]
    assert [row[6] for row in not_green[1:]] == HOSTILE

    listing = read_rst(overview_rst(cat, index, META), tmp_path / "overview.rst", monkeypatch)
    items = [item.astext() for item in listing.findall(nodes.list_item)]
    for detail in HOSTILE:
        # Once as the erratum and once as the adaptation of its form, as typed.
        assert sum(item.endswith(f": {detail}") for item in items) == 2, detail
    read_rst(index_rst(MAP, INDEX, META), tmp_path / "index.rst", monkeypatch)
    read_rst(overview_rst(MAP, INDEX, META), tmp_path / "overview.rst", monkeypatch)
    read_rst(operators_rst(MAP, INDEX), tmp_path / "operators.rst", monkeypatch)


# --- What Sphinx makes of the pages ---


def test_the_docs_conf_loads_the_stylesheet_of_the_map() -> None:
    # The build below writes a conf of its own: this is the line of the real one that gives the
    # pages of the map their stylesheet.
    assert f'html_css_files += ["{CSS.name}"]' in CONF.read_text(encoding="utf-8")


def test_docs_build_anchor_ids(tmp_path: Path) -> None:
    """A small tree of pages, built as the docs build them, with the settings of docs/conf.py.

    Two operators of one block have forms with the same ids, as the real map has (``ds_ds`` is on
    every page of a binary operator): an anchor has to be unique on its page, never in the docs.
    """
    application = pytest.importorskip("sphinx.application")
    docutils_namespace = pytest.importorskip("sphinx.util.docutils").docutils_namespace
    pytest.importorskip("sphinxcontrib.mermaid")
    pytest.importorskip("sphinx_rtd_theme")
    source = tmp_path / "src"
    plus = operator(
        "addition",
        "Addition",
        "+",
        form("ds_ds", "DS + DS", NUMBER, NUMBER, authored("two_rows")),
        form("sc_sc", "sc + sc", NUMBER, NUMBER, authored("two_numbers")),
    )
    # A definition, whose forms name their boxes, is in the block too: it has captions.
    chapter = Chapter("numeric", "Numeric operators", (MODULO, plus, DEFINE))
    cat = Catalogue((chapter,))
    index = index_of(cat, {"numeric/modulo/sc_sc/zero_divisor": REFUSED})
    examples = source / "examples"
    write_modulo(examples)
    write_define(examples, "numeric")
    write_example(examples, "numeric/addition/ds_ds/two_rows", {"input/DS_1.csv": EDGE_CSV})
    write_example(examples, "numeric/addition/sc_sc/two_numbers", {"expected/DS_r.csv": "A\n"})
    pages = {
        "index.rst": index_rst(cat, index, META),
        "overview.rst": overview_rst(cat, index, META),
        "operators.rst": operators_rst(cat, index),
        "numeric/index.rst": block_rst(chapter, cat, index, META),
        **{
            f"numeric/{op.id}.rst": operator_rst(
                chapter, op, cat, index, META, examples_root=examples
            )
            for op in chapter.operators
        },
    }
    for name, text in pages.items():
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        (source / name).write_text(text, encoding="utf-8")
    (source / "_static").mkdir()
    shutil.copy(CSS, source / "_static" / CSS.name)
    (source / "conf.py").write_text(
        "\n".join(
            [
                'project = "map"',
                'extensions = ["sphinx_rtd_theme", "sphinxcontrib.mermaid"]',
                'exclude_patterns = ["examples/**"]',
                'mermaid_init_config = {"startOnLoad": False, "securityLevel": "antiscript"}',
                'mermaid_width = "min(100%, 700px)"',
                'mermaid_height = "auto"',
                'html_theme = "sphinx_rtd_theme"',
                'html_static_path = ["_static"]',
                f'html_css_files = ["{CSS.name}"]',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    warnings = io.StringIO()
    output = tmp_path / "html"
    # The build registers directives, roles and settings of docutils in the process. They are
    # undone when it ends, so that the checks that read RST (read_rst) see the docutils they
    # had before, whatever the order of the checks.
    with docutils_namespace():
        app = application.Sphinx(
            str(source),
            str(source),
            str(output),
            str(tmp_path / "doctrees"),
            "html",
            status=None,
            warning=warnings,
            freshenv=True,
        )
        app.build()
    # No warning at all: no duplicate label, and the notes, with their quotes, stars, backticks,
    # pipe and colons, read cleanly.
    assert warnings.getvalue() == ""

    built = {name: (output / name).with_suffix(".html").read_text("utf-8") for name in pages}
    for name in ("numeric/modulo.rst", "numeric/addition.rst", "numeric/define_x.rst"):
        page = built[name]
        ids = set(re.findall(r'\bid="([^"]+)"', page))
        clicks = re.findall(r"href (?:&quot;|\")#(form-[a-z0-9-]+)(?:&quot;|\")", page)
        assert clicks, name
        assert set(clicks) <= ids, name
        # The notes are rendered as fields, inside the collapsible block of their example.
        assert "<summary>" in page
        assert "Tests" in page
    # The captions of the boxes are in the built page, each before the first form of its box.
    define = built["numeric/define_x.rst"]
    markers = [
        '<p class="om-box-title">Body</p>',
        'id="form-body-when"',
        'id="form-body-aggr"',
        '<p class="om-box-title">Signature</p>',
        'id="form-sign-variable"',
        'id="form-sign-domain"',
        '<p class="om-box-title">Return type</p>',
        'id="form-result"',
    ]
    places = [define.index(marker) for marker in markers]
    assert places == sorted(places)
    assert define.count('<p class="om-box-title">') == len(DEFINE_BOXES)
    # The links of the index and of the block page land on the anchors of the examples.
    modulo_ids = set(re.findall(r'\bid="([^"]+)"', built["numeric/modulo.rst"]))
    for page in (built["index.rst"], built["numeric/index.rst"]):
        for anchor in re.findall(r'href="(?:numeric/)?modulo\.html#([^"]+)"', page):
            assert anchor in modulo_ids, anchor
    # The cells of the files as they are written, and a file with a header only.
    tables = html_tables_of_built(built["numeric/modulo.rst"])
    assert tables[0] == EDGE_ROWS
    assert tables[2] == [["Id_1", "Me_1"]]
    assert html_tables_of_built(built["numeric/addition.rst"])[1] == [["A"]]


def html_tables_of_built(page: str) -> list[list[list[str]]]:
    """The cells of the data tables of a built page, header row first, as the browser shows them."""
    found: list[list[list[str]]] = []
    for table in re.findall(r"<table class=\"[^\"]*om-data[^\"]*\">.*?</table>", page, re.S):
        found.append(
            [
                [
                    html.unescape(cell)
                    for cell in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, re.S)
                ]
                for row in re.findall(r"<tr[^>]*>(.*?)</tr>", table, re.S)
            ]
        )
    return found
