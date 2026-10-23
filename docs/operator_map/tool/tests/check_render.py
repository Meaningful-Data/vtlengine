# ruff: noqa: S101
"""Checks for the reStructuredText of the map: manual URLs, the chapter pages and the index.

The catalogue, the results and the example folders are small hand-made values: no engine runs
here. One hand-written chapter page pins the layout. The other checks look at the pieces that
matter: the cells of a row, the sections of the index, what each section lists, and that what
is written as free text survives the reading of RST. The last one parses the pages with
docutils, when it is installed (it is a dependency of the docs, not of the tool).

Every check that renders the index or a chapter says where the examples are: a folder of its
own under ``tmp_path``, never the ``examples`` of this repository.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from tool.mermaid import chapter_blocks, lattice_block, overview_block, type_flow_block
from tool.model import (
    Catalogue,
    Chapter,
    Link,
    Operator,
    Status,
    TypeDef,
    Variant,
    authored,
    manual,
)
from tool.render import RunMeta, chapter_rst, index_rst, manual_url, mermaid_blocks
from tool.status import EngineVerdict, ExampleResult, StatusIndex, build_index

BASE = "https://sdmx-twg.github.io/vtl/v2.2/reference_manual"
ISSUES = "https://github.com/Meaningful-Data/vtlengine/issues"
SPEC = "140e946ed3f888951adb8bb66ea45030be84e2aa"
META = RunMeta("2026-10-23", "abc1234", "9.9.9", SPEC)
# The types that have a flow in the index: the basic ones and the dataset.
FLOW_TYPES = (
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


# --- Builders ---


MATCH = EngineVerdict("match", "")
WRONG = EngineVerdict("mismatch", "row Id_1=1: Me_1 expected 'a' got 'b'")
OK = ExampleResult(MATCH, MATCH, True, "ok")
PARTIAL = ExampleResult(WRONG, WRONG, True, "partial")
DIVERGE = ExampleResult(
    MATCH, EngineVerdict("error", "SemanticError 1-1-1-1: boom"), False, "diverge"
)
MISSING = ExampleResult(
    EngineVerdict("error", "SemanticError 1-1-1-1: unknown operator length"),
    EngineVerdict("error", "RunTimeError 2-1-0-1: not implemented"),
    True,
    "missing",
)


def make_operator(
    operator_id: str,
    name: str,
    symbol: str,
    variants: tuple[Variant, ...],
    links: tuple[Link, ...] = (),
) -> Operator:
    return Operator(
        operator_id,
        name,
        symbol,
        f"Some operators/{name}",
        ("string",),
        ("integer",),
        variants,
        links,
    )


# A small map: two chapters and a type, with an erratum, an adaptation, a decision and every status.
CONCAT = make_operator(
    "concat",
    "Concatenation",
    "||",
    (
        Variant(
            "ds_ds",
            "DS_1 || DS_2",
            (manual("ex_1", erratum="the manual shows 'ba' where 'ab' is meant"), authored()),
        ),
        Variant("comp_sc", 'Me_1 || "x"', (authored(decision=1049),)),
    ),
    (Link("clause.calc", "used in"),),
)
LENGTH = make_operator(
    "length",
    "String length",
    "length",
    (Variant("comp", "length(Me_1)", (authored(adaptation="the time values were rewritten"),)),),
)
CALC = make_operator(
    "calc", "Calculation", "calc", (Variant("measure", "calc Me_1 := 1", (authored(),)),)
)
STRING = Chapter("string", "String operators", (CONCAT, LENGTH))
CLAUSE = Chapter("clause", "Clause operators", (CALC,))
TYPES = (
    TypeDef("scalar", ()),
    TypeDef("number", ("scalar",), (Variant("identity", "identity", (authored(),)),)),
    TypeDef("dataset", ()),
)
CAT = Catalogue((STRING, CLAUSE), TYPES)
RESULTS = {
    "string/concat/ds_ds/manual_ex_1": OK,
    "string/concat/ds_ds/minimal": DIVERGE,
    "string/concat/comp_sc/minimal": PARTIAL,
    "string/length/comp/minimal": MISSING,
    "clause/calc/measure/minimal": OK,
    "types/number/identity/minimal": OK,
}
INDEX = build_index(CAT, RESULTS)

# One operator, one variant, one example: small enough to write its chapter page by hand.
MINI_OPERATOR = make_operator(
    "length", "String length", "length", (Variant("comp", "length(Me_1)", (authored(),)),)
)
MINI_CHAPTER = Chapter("string", "String operators", (MINI_OPERATOR,))
MINI_CAT = Catalogue((MINI_CHAPTER,), ())
MINI_INDEX = build_index(MINI_CAT, {"string/length/comp/minimal": OK})


def make_tree(root: Path, folder: str, *, inputs: bool = True, notes: bool = False) -> None:
    """The files of an example folder that the page looks for."""
    base = root / folder
    base.mkdir(parents=True, exist_ok=True)
    (base / "script.vtl").write_text("DS_r := DS_1;\n", encoding="utf-8")
    (base / "expected").mkdir(exist_ok=True)
    if inputs:
        (base / "input").mkdir(exist_ok=True)
    if notes:
        (base / "notes.md").write_text("Notes.\n", encoding="utf-8")


# --- Reading a page back ---


def blocks(text: str) -> list[str]:
    """The Mermaid blocks of a page, in order, as the ``.. mermaid::`` directives hold them."""
    bodies = re.findall(r"^\.\. mermaid::\n\n((?:   .*\n)+)", text, flags=re.MULTILINE)
    return ["\n".join(line[3:] for line in body.splitlines()) for body in bodies]


def headings(text: str) -> list[tuple[str | None, str, str]]:
    """The ``(overline, title, underline)`` of every title of a page, in order."""
    lines = text.split("\n")
    found: list[tuple[str | None, str, str]] = []
    for n in range(len(lines) - 1):
        title, rule = lines[n], lines[n + 1]
        # A title that is made of the same character is the overline of another title.
        is_rule = len(rule) >= 4 and len(set(rule)) == 1 and rule[0] in "=-~"
        if title and is_rule and set(title) != set(rule):
            found.append((lines[n - 1] if n and lines[n - 1] == rule else None, title, rule))
    return found


def titles(text: str, char: str) -> list[str]:
    """The titles that are underlined with ``char``, in order."""
    return [title for _, title, rule in headings(text) if rule[0] == char]


def section(text: str, title: str) -> str:
    """The text of the ``-`` section ``title``, up to the next ``-`` title or the end."""
    lines = text.split("\n")
    start = next(
        n for n, line in enumerate(lines[:-1]) if line == title and set(lines[n + 1]) == {"-"}
    )
    end = len(lines)
    for n in range(start + 2, len(lines) - 1):
        if lines[n] and lines[n + 1] and set(lines[n + 1]) == {"-"} and set(lines[n]) != {"-"}:
            end = n
            break
    return "\n".join(lines[start + 2 : end])


def table_rows(text: str) -> list[list[str]]:
    """The rows of every ``list-table`` of ``text``, header rows too, as lists of cells.

    A cell of several lines has them joined with a newline, without their indentation.
    """
    rows: list[list[str]] = []
    for line in text.split("\n"):
        if line == "   * -" or line.startswith("   * - "):
            rows.append([line[7:]])
        elif rows and (line == "     -" or line.startswith("     - ")):
            rows[-1].append(line[7:])
        elif rows and line.startswith("       "):
            rows[-1][-1] += "\n" + line[7:]
    return rows


def row_with(text: str, needle: str) -> list[str]:
    """The only table row of ``text`` that has ``needle`` in a cell."""
    rows = [row for row in table_rows(text) if any(needle in cell for cell in row)]
    assert len(rows) == 1, f"{len(rows)} rows have {needle}"
    return rows[0]


def detail_of(result: ExampleResult, tmp_path: Path) -> str:
    """The Detail cell of the only example of a one-example chapter that has ``result``."""
    cat = Catalogue(
        (Chapter("c", "C", (make_operator("o", "O", "o", (Variant("v", "v", (authored(),)),)),)),),
        (),
    )
    index = build_index(cat, {"c/o/v/minimal": result})
    text = chapter_rst(cat.chapters[0], cat, index, META, examples_root=tmp_path)
    return row_with(text, "/script.vtl>")[4]


def detail_cell(detail: str, tmp_path: Path) -> str:
    """The Detail cell of an example that both engines got wrong with ``detail``."""
    verdict = EngineVerdict("mismatch", detail)
    return detail_of(ExampleResult(verdict, verdict, True, "partial"), tmp_path)


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


# --- Chapter pages ---


def test_chapter_rst_layout(tmp_path: Path) -> None:
    folder = "examples/string/length/comp/minimal"
    make_tree(tmp_path, "string/length/comp/minimal", notes=True)
    block = chapter_blocks(MINI_CHAPTER, MINI_CAT, MINI_INDEX)[0]
    body = "\n".join(f"   {line}" for line in block.split("\n"))
    page_rule = "=" * 36  # twice the 18 characters of the title
    operator_rule = "-" * 52  # twice the 26 characters of the heading
    assert chapter_rst(MINI_CHAPTER, MINI_CAT, MINI_INDEX, META, examples_root=tmp_path) == (
        f"{page_rule}\n"
        "🟢 String operators\n"
        f"{page_rule}\n"
        "\n"
        "1 variant: 🟢 1 · 🟡 0 · 🔴 0 · 🔵 0\n"
        "\n"
        "← Back to :doc:`index` · vtlengine 9.9.9 (abc1234)\n"
        "\n"
        ".. mermaid::\n"
        "\n"
        f"{body}\n"
        "\n"
        "🟢 String length ``length``\n"
        f"{operator_rule}\n"
        "\n"
        f"`VTL 2.2 Reference Manual <{BASE}/operators/Some%20operators/"
        "String%20length/index.html>`__\n"
        "\n"
        ".. list-table::\n"
        "   :header-rows: 1\n"
        "   :widths: auto\n"
        "\n"
        "   * - Variant\n"
        "     - Status\n"
        "     - pandas\n"
        "     - DuckDB\n"
        "     - Detail\n"
        "     - Example\n"
        "   * - | ``length(Me_1)``\n"
        "       | minimal\n"
        "     - 🟢\n"
        "     - ✅\n"
        "     - ✅\n"
        "     -\n"
        f"     - `script <{folder}/script.vtl>`__ · `input <{folder}/input/>`__ · "
        f"`expected <{folder}/expected/>`__ · `notes <{folder}/notes.md>`__\n"
    )


def test_chapter_rst_title_counts_and_operator_headings(tmp_path: Path) -> None:
    text = chapter_rst(STRING, CAT, INDEX, META, examples_root=tmp_path)
    lines = text.splitlines()
    # The chapter is 🔵 (a variant diverges); the counts are over variants: 3, none of them 🟢.
    assert titles(text, "=") == ["🔵 String operators"]
    assert "3 variants: 🟢 0 · 🟡 1 · 🔴 1 · 🔵 1" in lines
    assert "← Back to :doc:`index` · vtlengine 9.9.9 (abc1234)" in lines
    assert titles(text, "-") == ["🔵 Concatenation ``||``", "🔴 String length ``length``"]
    link = f"`VTL 2.2 Reference Manual <{manual_url(CONCAT.page)}>`__"
    assert link in lines


def test_chapter_rst_has_directive_and_table_row_per_example(tmp_path: Path) -> None:
    text = chapter_rst(STRING, CAT, INDEX, META, examples_root=tmp_path)
    assert blocks(text) == chapter_blocks(STRING, CAT, INDEX)
    rows = [row for row in table_rows(text) if "/script.vtl>" in row[5]]
    assert len(rows) == 4  # manual_ex_1 and minimal, comp_sc, comp
    for variant, example in (
        ("concat/ds_ds", "manual_ex_1"),
        ("concat/ds_ds", "minimal"),
        ("concat/comp_sc", "minimal"),
        ("length/comp", "minimal"),
    ):
        row_with(text, f"examples/string/{variant}/{example}/script.vtl>")
    # A table per operator, each with its header.
    assert text.count("   * - Variant\n     - Status\n") == 2
    assert text.count(".. list-table::\n   :header-rows: 1\n   :widths: auto\n\n") == 2


def test_the_row_of_an_example_shows_its_verdicts(tmp_path: Path) -> None:
    text = chapter_rst(STRING, CAT, INDEX, META, examples_root=tmp_path)
    first = row_with(text, "concat/ds_ds/manual_ex_1/script.vtl>")
    # The label is a literal, as it is, and the example name sits under it, escaped.
    assert first[:5] == ["| ``DS_1 || DS_2``\n| manual\\_ex\\_1", "🟢", "✅", "✅", ""]
    diverging = row_with(text, "concat/ds_ds/minimal/script.vtl>")
    assert diverging[1:5] == ["🔵", "✅", "💥", "DuckDB: SemanticError 1-1-1-1: boom"]
    wrong = row_with(text, "concat/comp_sc/minimal/script.vtl>")
    # The quotes of the message are escaped: Sphinx would typeset them otherwise.
    assert wrong[1:5] == ["🟡", "❌", "❌", "row Id\\_1=1: Me\\_1 expected \\'a\\' got \\'b\\'"]


def test_a_label_that_a_literal_cannot_hold_is_escaped_text(tmp_path: Path) -> None:
    labels = {"tick": "a `b` c", "lines": "line one\nline two", "empty": "", "slash": "a\\"}
    variants = tuple(Variant(name, label, (authored(),)) for name, label in labels.items())
    cat = Catalogue((Chapter("c", "C", (make_operator("o", "O", "o", variants),)),), ())
    index = build_index(cat, {f"c/o/{name}/minimal": OK for name in labels})
    text = chapter_rst(cat.chapters[0], cat, index, META, examples_root=tmp_path)
    assert row_with(text, "c/o/tick/minimal/script.vtl>")[0] == "| a \\`b\\` c\n| minimal"
    assert (
        row_with(text, "c/o/lines/minimal/script.vtl>")[0] == "| ``line one line two``\n| minimal"
    )
    assert row_with(text, "c/o/empty/minimal/script.vtl>")[0] == "| minimal"
    assert row_with(text, "c/o/slash/minimal/script.vtl>")[0] == "| ``a\\``\n| minimal"


def test_detail_is_one_line_escaped_and_cut_at_200_characters(tmp_path: Path) -> None:
    assert detail_cell("line one\nline two | pipe\r\nend\rlast", tmp_path) == (
        "line one line two \\| pipe end last"
    )
    assert detail_cell("x" * 200, tmp_path) == "x" * 200
    assert detail_cell("x" * 201, tmp_path) == "x" * 199 + "…"
    # The markup is escaped after the cut: the 200 characters are the visible ones.
    assert detail_cell("|" * 250, tmp_path) == "\\|" * 199 + "…"
    assert detail_cell("_" * 250, tmp_path) == "\\_" * 199 + "…"
    # A backslash that ends up last before the ellipsis is not read as escaping it.
    assert detail_cell("a" * 198 + "\\\\b", tmp_path) == "a" * 198 + "\\\\…"


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
        # A backslash that comes right before a hyphen or a dot would be lost under Sphinx.
        ("regex \\d+\\.\\d+ and \\-x", "regex \\\\d+\\\\\\.\\\\d+ and \\\\\\-x"),
        ("a\\'b a\\\"b", "a\\\\\\'b a\\\\\\\"b"),
    ],
)
def test_detail_escapes_what_rst_reads_as_markup(detail: str, cell: str, tmp_path: Path) -> None:
    assert detail_cell(detail, tmp_path) == cell


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
    detail: str, cell: str, tmp_path: Path
) -> None:
    assert detail_cell(detail, tmp_path) == cell


def test_detail_names_the_engine_that_has_one(tmp_path: Path) -> None:
    error = EngineVerdict("error", "SemanticError 1-1-1-1: boom")
    other = EngineVerdict("error", "RunTimeError 2-1-0-1: not implemented")
    assert detail_of(ExampleResult(MATCH, error, False, "diverge"), tmp_path) == (
        "DuckDB: SemanticError 1-1-1-1: boom"
    )
    assert detail_of(ExampleResult(error, MATCH, False, "diverge"), tmp_path) == (
        "pandas: SemanticError 1-1-1-1: boom"
    )
    # Two errors that differ are both shown; two that are the same are shown once.
    assert detail_of(ExampleResult(error, other, True, "missing"), tmp_path) == (
        "pandas: SemanticError 1-1-1-1: boom; DuckDB: RunTimeError 2-1-0-1: not implemented"
    )
    assert detail_of(ExampleResult(error, error, True, "missing"), tmp_path) == (
        "SemanticError 1-1-1-1: boom"
    )
    assert detail_of(OK, tmp_path) == ""


def test_engines_that_differ_with_the_same_verdict_say_so(tmp_path: Path) -> None:
    # Both match, or both fail the same way, and still give different outputs: nothing else says.
    assert detail_of(ExampleResult(MATCH, MATCH, False, "diverge"), tmp_path) == (
        "the engines give different outputs"
    )
    assert detail_of(ExampleResult(WRONG, WRONG, False, "diverge"), tmp_path) == (
        "row Id\\_1=1: Me\\_1 expected \\'a\\' got \\'b\\'; the engines give different outputs"
    )


def test_example_cell_links_files_notes_and_marks(tmp_path: Path) -> None:
    folder = "examples/string/concat/ds_ds/manual_ex_1"
    make_tree(tmp_path, "string/concat/ds_ds/manual_ex_1", notes=True)
    text = chapter_rst(STRING, CAT, INDEX, META, examples_root=tmp_path)
    cell = row_with(text, f"{folder}/script.vtl>")[5]
    # The links are anonymous hyperlinks with their address inside, so that none can clash.
    assert cell == (
        f"`script <{folder}/script.vtl>`__ · `input <{folder}/input/>`__ · "
        f"`expected <{folder}/expected/>`__ · `notes <{folder}/notes.md>`__ · *manual* · erratum"
    )
    # A recorded decision links its issue, an adaptation says so, and an authored example has no
    # *manual* mark. Nothing is on disk for these: no input and no notes to link.
    decision = row_with(text, "concat/comp_sc/minimal/script.vtl>")[5]
    assert decision.endswith(
        "`expected <examples/string/concat/comp_sc/minimal/expected/>`__ · "
        f"`📌 #1049 <{ISSUES}/1049>`__"
    )
    adapted = row_with(text, "length/comp/minimal/script.vtl>")[5]
    assert adapted.endswith("/expected/>`__ · adapted")
    assert "`input <" not in decision
    assert "`notes <" not in adapted
    assert "*manual*" not in adapted


def test_example_without_a_result_is_not_run(tmp_path: Path) -> None:
    nothing = StatusIndex({}, {}, {}, {})
    text = chapter_rst(STRING, CAT, nothing, META, examples_root=tmp_path)
    lines = text.splitlines()
    assert titles(text, "=") == ["String operators"]
    assert "0 variants: 🟢 0 · 🟡 0 · 🔴 0 · 🔵 0" in lines
    assert "Concatenation ``||``" in titles(text, "-")
    row = row_with(text, "concat/ds_ds/minimal/script.vtl>")
    assert row[1:5] == ["", "", "", "not run"]


def test_every_block_of_a_split_chapter_gets_its_own_directive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split = ["flowchart LR\n    a --> b", "flowchart LR\n    c --> d", "flowchart LR\n    e --> f"]
    monkeypatch.setattr("tool.render.chapter_blocks", lambda *_args: split)
    text = chapter_rst(STRING, CAT, INDEX, META, examples_root=tmp_path)
    assert blocks(text) == split
    assert text.count(".. mermaid::\n\n") == 3


@pytest.mark.parametrize("status", ["ok", "partial", "missing", "diverge"])
def test_chapter_title_carries_the_emoji_of_its_status(status: Status, tmp_path: Path) -> None:
    emoji = {"ok": "🟢", "partial": "🟡", "missing": "🔴", "diverge": "🔵"}[status]
    index = StatusIndex({}, {}, {}, {"string": status})
    text = chapter_rst(MINI_CHAPTER, MINI_CAT, index, META, examples_root=tmp_path)
    assert titles(text, "=") == [f"{emoji} String operators"]


# --- Titles, diagrams and links ---


def test_titles_are_underlined_at_least_twice_as_wide_and_page_titles_have_an_overline(
    tmp_path: Path,
) -> None:
    for text in (
        chapter_rst(STRING, CAT, INDEX, META, examples_root=tmp_path),
        index_rst(CAT, INDEX, META, examples_root=tmp_path),
    ):
        found = headings(text)
        assert len(found) >= 3
        for overline, title, underline in found:
            # An emoji is two columns wide, and the rule has to be as wide as the title.
            assert len(underline) >= 2 * len(title), title
            # A page title has the same rule above and below; the other titles have none above.
            assert (overline == underline) == (underline[0] == "="), title
        assert [rule[0] for _, _, rule in found if rule[0] == "="] == ["="]  # one page title


def test_diagram_clicks_point_to_the_pages_of_the_site_not_to_md_files(tmp_path: Path) -> None:
    chapter = chapter_rst(STRING, CAT, INDEX, META, examples_root=tmp_path)
    index = index_rst(CAT, INDEX, META, examples_root=tmp_path)
    (block,) = blocks(chapter)
    # The link to an operator of another chapter: its click and the page named in its label.
    assert '    click ext_clause__calc "clause.html"' in block.splitlines()
    assert "↗ Calculation (clause.html)" in block
    (overview, *_) = blocks(index)
    assert '    click chapter__string "string.html"' in overview.splitlines()
    assert '    click chapter__clause "clause.html"' in overview.splitlines()
    for text in (chapter, index):
        assert ".md" not in "".join(blocks(text))
    # The clicks to a folder are what the diagram builder wrote.
    assert '    click string__length "examples/string/length/comp/minimal/"' in block.splitlines()


def test_mermaid_blocks_reads_back_the_blocks_of_a_page(tmp_path: Path) -> None:
    for text in (
        chapter_rst(STRING, CAT, INDEX, META, examples_root=tmp_path),
        index_rst(CAT, INDEX, META, examples_root=tmp_path),
    ):
        assert mermaid_blocks(text) == blocks(text)
        assert mermaid_blocks(text)
    assert mermaid_blocks("no diagram here\n") == []
    two = ".. mermaid::\n\n   flowchart LR\n       a --> b\n\ntext\n\n.. mermaid::\n\n   x\n"
    assert mermaid_blocks(two) == ["flowchart LR\n    a --> b", "x"]


# --- Index ---


def test_index_starts_with_the_title_and_a_hidden_toctree_of_the_chapters(tmp_path: Path) -> None:
    text = index_rst(CAT, INDEX, META, examples_root=tmp_path)
    rule = "=" * 40  # twice the 20 characters of the title
    assert text.startswith(f"{rule}\nVTL 2.2 operator map\n{rule}\n\n")
    assert text.endswith("\n")
    toctree = ".. toctree::\n   :maxdepth: 1\n   :hidden:\n\n   string\n   clause\n"
    assert toctree in text
    # The toctree comes before the first section, and there is only one.
    assert text.index(toctree) < text.index("\nGeneration\n")
    assert text.count(".. toctree::") == 1
    assert "README" not in text
    assert ".md" not in text


def test_index_without_chapters_has_no_toctree(tmp_path: Path) -> None:
    empty = Catalogue((), ())
    text = index_rst(empty, StatusIndex({}, {}, {}, {}), META, examples_root=tmp_path)
    assert ".. toctree::" not in text
    assert titles(text, "=") == ["VTL 2.2 operator map"]


def test_index_sections_come_in_the_order_of_the_design(tmp_path: Path) -> None:
    text = index_rst(CAT, INDEX, META, examples_root=tmp_path)
    assert titles(text, "-") == [
        "Generation",
        "Legend",
        "How to read",
        "Overview",
        "Chapters",
        "Type lattice",
        "Type flows",
        "📌 Team decisions",
        "Manual errata (corrected expected outputs, scripts or inputs)",
        "Adaptations",
        "Variants that are not green",
        "Attribution",
    ]


def test_index_metadata_and_legend(tmp_path: Path) -> None:
    text = index_rst(CAT, INDEX, META, examples_root=tmp_path)
    generation = section(text, "Generation")
    for value in ("2026-10-23", "9.9.9", "abc1234", SPEC[:12]):
        assert value in generation
    assert f"https://github.com/sdmx-twg/vtl/tree/{SPEC}" in generation
    legend = section(text, "Legend")
    for mark in ("🟢", "🟡", "🔴", "🔵", "📌", "✅", "❌", "💥"):
        assert mark in legend
    assert len(table_rows(legend)) == 5 + 5  # two tables, each with a header row


def test_the_pin_legend_says_a_decision_is_recorded_not_that_the_manual_disagrees(
    tmp_path: Path,
) -> None:
    text = index_rst(CAT, INDEX, META, examples_root=tmp_path)
    legend = section(text, "Legend")
    # A decision and the manual can agree: the mark says only that a decision is behind the output.
    assert row_with(legend, "📌 #n") == [
        "📌 #n",
        "The expected output follows a recorded team decision, in the issue n.",
    ]
    assert "not the manual" not in legend
    decisions = section(text, "📌 Team decisions")
    assert "Expected outputs that follow a recorded team decision." in decisions
    assert "not the text of the manual" not in decisions


def test_the_lists_of_the_index_say_what_they_list(tmp_path: Path) -> None:
    text = index_rst(CAT, INDEX, META, examples_root=tmp_path)
    errata = section(text, "Manual errata (corrected expected outputs, scripts or inputs)")
    assert "an expected output, a script or an input" in errata
    adaptations = section(text, "Adaptations")
    assert (
        "Examples where the manual's representation of an input, or of the name of an output, "
        "was rewritten"
    ) in adaptations
    assert "equivalent representation that vtlengine documents" in adaptations
    assert "These are not errata." in adaptations
    assert "Manual inputs that vtlengine cannot load" not in adaptations


def test_index_blocks_are_the_overview_the_lattice_and_a_flow_per_basic_type(
    tmp_path: Path,
) -> None:
    text = index_rst(CAT, INDEX, META, examples_root=tmp_path)
    assert blocks(section(text, "Overview")) == [overview_block(CAT, INDEX)]
    assert blocks(section(text, "Type lattice")) == [lattice_block(CAT, INDEX)]
    flows = section(text, "Type flows")
    assert blocks(flows) == [type_flow_block(type_id, CAT, INDEX) for type_id in FLOW_TYPES]
    assert titles(flows, "~") == list(FLOW_TYPES)
    assert len(blocks(text)) == 12


def test_index_overview_does_not_promise_that_a_click_works(tmp_path: Path) -> None:
    overview = section(index_rst(CAT, INDEX, META, examples_root=tmp_path), "Overview")
    assert "Click a chapter" not in overview
    assert "click a chapter" not in overview.lower()
    assert "Where the viewer supports links in diagrams" in overview
    assert "table of chapters" in overview
    how = section(index_rst(CAT, INDEX, META, examples_root=tmp_path), "How to read")
    assert "GitHub" not in how
    assert "A viewer may ignore the links of the diagrams" in how


def test_how_to_read_describes_the_comparison_as_it_is_made(tmp_path: Path) -> None:
    how = section(index_rst(CAT, INDEX, META, examples_root=tmp_path), "How to read")
    # A number is compared to the digits written, never finer than a double; a random example
    # compares what its expected file has, and checks the columns that are empty there.
    assert "written in the expected file, as far as a double holds them" in how
    assert "those of the columns that are empty in the expected file" in how


def test_index_chapter_table_counts_variants(tmp_path: Path) -> None:
    chapters = section(index_rst(CAT, INDEX, META, examples_root=tmp_path), "Chapters")
    assert row_with(chapters, "Chapter") == [
        "Chapter",
        "Status",
        "Variants",
        "🟢",
        "🟡",
        "🔴",
        "🔵",
    ]
    assert row_with(chapters, "<string>") == [
        ":doc:`String operators <string>`",
        "🔵",
        "3",
        "0",
        "1",
        "1",
        "1",
    ]
    assert row_with(chapters, "<clause>") == [
        ":doc:`Clause operators <clause>`",
        "🟢",
        "1",
        "1",
        "0",
        "0",
        "0",
    ]
    # The type hubs have no page of their own; the total counts everything above it.
    assert row_with(chapters, "Data types") == ["Data types", "🟢", "1", "1", "0", "0", "0"]
    assert row_with(chapters, "Total") == ["**Total**", "🔵", "5", "2", "1", "1", "1"]


def test_index_lists_errata_adaptations_and_decisions(tmp_path: Path) -> None:
    text = index_rst(CAT, INDEX, META, examples_root=tmp_path)
    where = ":doc:`String operators <string>` › Concatenation"
    decisions = section(text, "📌 Team decisions")
    assert f"`📌 #1049 <{ISSUES}/1049>`__" in decisions
    assert f'{where} › ``Me_1 || "x"`` (minimal)' in decisions
    errata = section(text, "Manual errata (corrected expected outputs, scripts or inputs)")
    # The quotes are escaped, so that Sphinx does not turn them into typographic ones.
    assert "the manual shows \\'ba\\' where \\'ab\\' is meant" in errata
    assert f"{where} › ``DS_1 || DS_2`` (manual\\_ex\\_1)" in errata
    assert f"`manual <{manual_url(CONCAT.page)}>`__" in errata
    adaptations = section(text, "Adaptations")
    assert "the time values were rewritten" in adaptations
    assert (
        ":doc:`String operators <string>` › String length › ``length(Me_1)`` (minimal)"
        in adaptations
    )
    # Each list is a bullet list that holds what it should and nothing else.
    for body in (decisions, errata, adaptations):
        assert [line[:2] for line in body.splitlines() if line.strip()][-1] == "- "
    assert "1049" not in errata + adaptations
    assert "rewritten" not in errata + decisions
    assert "'ba'" not in adaptations + decisions


def test_index_escapes_the_free_text_of_the_lists(tmp_path: Path) -> None:
    erratum = "the *manual* prints a_b | c, `x` and \\ and ends with ::"
    variant = Variant("v", "f(x)", (authored(erratum=erratum),))
    cat = Catalogue((Chapter("c", "C", (make_operator("o", "Some_op", "f", (variant,)),)),), ())
    index = build_index(cat, {"c/o/v/minimal": OK})
    errata = section(
        index_rst(cat, index, META, examples_root=tmp_path),
        "Manual errata (corrected expected outputs, scripts or inputs)",
    )
    assert (
        "- :doc:`C <c>` › Some\\_op › ``f(x)`` (minimal): the \\*manual\\* prints a\\_b \\| c, "
        "\\`x\\` and \\\\ and ends with :\\: · `manual <" in errata
    )
    # Quotes, runs of hyphens and ellipses are what Sphinx would typeset: each one is escaped.
    typeset = 'it\'s "a" -- b --- c - d... e. . . f'
    variant = Variant("v", "f(x)", (authored(erratum=typeset),))
    cat = Catalogue((Chapter("c", "C", (make_operator("o", "Some_op", "f", (variant,)),)),), ())
    index = build_index(cat, {"c/o/v/minimal": OK})
    errata = section(
        index_rst(cat, index, META, examples_root=tmp_path),
        "Manual errata (corrected expected outputs, scripts or inputs)",
    )
    assert (
        '(minimal): it\\\'s \\"a\\" \\-\\- b \\-\\-\\- c - d\\.\\.\\. e\\. \\. \\. f · `manual <'
        in errata
    )


def test_index_lists_the_notes_that_exist(tmp_path: Path) -> None:
    make_tree(tmp_path, "string/concat/comp_sc/minimal", notes=True)
    text = index_rst(CAT, INDEX, META, examples_root=tmp_path)
    decisions = section(text, "📌 Team decisions")
    assert "`notes <examples/string/concat/comp_sc/minimal/notes.md>`__" in decisions
    assert "`notes <" not in section(
        text, "Manual errata (corrected expected outputs, scripts or inputs)"
    )


def test_index_lists_variants_that_are_not_green_by_status(tmp_path: Path) -> None:
    text = index_rst(CAT, INDEX, META, examples_root=tmp_path)
    listing = section(text, "Variants that are not green")
    assert titles(listing, "~") == [
        "🔵 Engines diverge (1)",
        "🔴 Not implemented (1)",
        "🟡 Partially implemented (1)",
    ]
    diverge, missing, partial = re.split(r"^[^\n]*\n~+\n", listing, flags=re.MULTILINE)[1:]
    assert row_with(diverge, "<string>") == [
        ":doc:`String operators <string>`",
        "Concatenation",
        "``DS_1 || DS_2``",
        "`minimal <examples/string/concat/ds_ds/minimal/>`__",
        "DuckDB: SemanticError 1-1-1-1: boom",
    ]
    assert "``length(Me_1)``" in row_with(missing, "<string>")[2]
    assert '``Me_1 || "x"``' in row_with(partial, "<string>")[2]
    # The green variants are not listed, and neither are the type hubs that are green.
    assert "Calculation" not in listing
    assert "Data types" not in listing


def test_a_diverging_variant_is_listed_with_the_example_that_diverges(tmp_path: Path) -> None:
    # The first example is only 🟡: the row of the 🔵 variant has to point at the second.
    variant = Variant("ds", "DS_1 + DS_2", (authored("minimal"), authored("minimal_2")))
    operator = make_operator("plus", "Addition", "+", (variant,))
    cat = Catalogue((Chapter("numeric", "Numeric operators", (operator,)),), ())
    results = {"numeric/plus/ds/minimal": PARTIAL, "numeric/plus/ds/minimal_2": DIVERGE}
    listing = section(
        index_rst(cat, build_index(cat, results), META, examples_root=tmp_path),
        "Variants that are not green",
    )
    row = row_with(listing, "<numeric>")
    assert row[3] == "`minimal\\_2 <examples/numeric/plus/ds/minimal_2/>`__"
    assert titles(listing, "~") == ["🔵 Engines diverge (1)"]


def test_index_lists_the_variants_of_a_type_that_are_not_green(tmp_path: Path) -> None:
    results = {**RESULTS, "types/number/identity/minimal": PARTIAL}
    listing = section(
        index_rst(CAT, build_index(CAT, results), META, examples_root=tmp_path),
        "Variants that are not green",
    )
    row = row_with(listing, "Data types")
    assert row[:3] == ["Data types", "number", "``identity``"]


def test_index_attribution_sentence_is_verbatim(tmp_path: Path) -> None:
    attribution = section(index_rst(CAT, INDEX, META, examples_root=tmp_path), "Attribution")
    assert (
        "Examples marked *manual* are taken from the VTL 2.2 Reference Manual "
        f"(sdmx-twg/vtl @ {SPEC}), converted to the vtlengine format."
    ) in attribution
    assert f"`sdmx-twg.github.io/vtl/v2.2 <{BASE}/index.html>`__" in attribution
    assert "`sdmx-twg/vtl <https://github.com/sdmx-twg/vtl>`__" in attribution


def test_index_empty_lists_say_so(tmp_path: Path) -> None:
    cat = Catalogue((MINI_CHAPTER,), ())
    text = index_rst(cat, MINI_INDEX, META, examples_root=tmp_path)
    assert "None: no expected output follows a recorded team decision." in text
    assert "None: nothing in the manual's examples had to be corrected." in text
    assert "None: no input or output name of the manual was rewritten." in text
    assert "Every variant is green." in text
    # No type has a variant, so there is no row for the types, and the blocks still render.
    assert "Data types" not in text
    assert len(blocks(text)) == 12


def test_a_page_asks_the_examples_folder_it_is_given(tmp_path: Path) -> None:
    # No input folder and no notes there: none is linked, whatever the repository has.
    make_tree(tmp_path, "string/length/comp/minimal", inputs=False)
    text = chapter_rst(MINI_CHAPTER, MINI_CAT, MINI_INDEX, META, examples_root=tmp_path)
    assert "`input <" not in text
    assert "`notes <" not in text


# --- What a reader of RST makes of the pages ---


def read_rst(text: str) -> Any:
    """The docutils tree of a page. The directives and the role of Sphinx that it uses are stubs.

    Every message of docutils of level 2 (a warning) or above fails the check. The smart quotes
    are on, as they are in Sphinx: they turn quotes, runs of hyphens and ellipses into
    typographic characters, in every text that is not a literal.
    """
    core = pytest.importorskip("docutils.core")
    nodes = pytest.importorskip("docutils.nodes")
    rst = pytest.importorskip("docutils.parsers.rst")
    directives = pytest.importorskip("docutils.parsers.rst.directives")
    roles = pytest.importorskip("docutils.parsers.rst.roles")

    def stub(options: dict[str, Any]) -> Any:
        return type(
            "Stub",
            (rst.Directive,),
            {"has_content": True, "option_spec": options, "run": lambda self: []},
        )

    def doc_role(*args: Any) -> tuple[list[Any], list[Any]]:
        text = args[2]  # the role is called with (name, rawtext, text, lineno, inliner, ...)
        return [nodes.Text(text.partition(" <")[0])], []

    directives.register_directive("mermaid", stub({}))
    toctree = stub({"maxdepth": directives.unchanged, "hidden": directives.flag})
    directives.register_directive("toctree", toctree)
    roles.register_local_role("doc", doc_role)
    messages: list[str] = []

    class Collector:
        def write(self, message: str) -> None:
            messages.append(message)

    tree = core.publish_doctree(
        text,
        settings_overrides={
            "report_level": 2,
            "halt_level": 5,
            "warning_stream": Collector(),
            "smart_quotes": True,
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


def test_the_pages_are_read_as_written_by_docutils(tmp_path: Path) -> None:
    pytest.importorskip("docutils")
    nodes = pytest.importorskip("docutils.nodes")

    forms = tuple(
        Variant(
            f"v{n}",
            HOSTILE_LABELS[n % len(HOSTILE_LABELS)],
            (authored(adaptation=detail, erratum=detail),),
        )
        for n, detail in enumerate(HOSTILE)
    )
    operator = make_operator("o_p", "Name_with *marks* | `ticks` \\", "||", forms)
    cat = Catalogue((Chapter("c", "C_hapter", (operator,)),), ())
    results = {}
    for n, detail in enumerate(HOSTILE):
        verdict = EngineVerdict("mismatch", detail)
        results[f"c/o_p/v{n}/minimal"] = ExampleResult(verdict, verdict, True, "partial")
    index = build_index(cat, results)

    tree = read_rst(chapter_rst(cat.chapters[0], cat, index, META, examples_root=tmp_path))
    cells = [
        [entry.astext() for entry in row.findall(nodes.entry)] for row in tree.findall(nodes.row)
    ]
    assert [row[4] for row in cells[1:]] == HOSTILE
    assert [row[0].splitlines()[0] for row in cells[1:]] == [
        HOSTILE_LABELS[n % len(HOSTILE_LABELS)] for n in range(len(HOSTILE))
    ]
    assert [title.astext() for title in tree.findall(nodes.title)] == [
        "🟡 C_hapter",
        "🟡 Name_with *marks* | `ticks` \\ ||",
    ]

    # The index lists the same details as errata and adaptations, and every page reads cleanly.
    listing = read_rst(index_rst(cat, index, META, examples_root=tmp_path))
    items = [item.astext() for item in listing.findall(nodes.list_item)]
    for detail in HOSTILE:
        # Once as the erratum and once as the adaptation of its variant, as typed.
        assert sum(f": {detail} ·" in item for item in items) == 2, detail
    read_rst(index_rst(CAT, INDEX, META, examples_root=tmp_path))
    read_rst(chapter_rst(STRING, CAT, INDEX, META, examples_root=tmp_path))
