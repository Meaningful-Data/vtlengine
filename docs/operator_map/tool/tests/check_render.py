# ruff: noqa: S101
"""Checks for the markdown of the map: manual URLs, the chapter files and the README.

The catalogue, the results and the example folders are small hand-made values: no engine runs
here. One hand-written chapter file pins the layout. The other checks look at the pieces that
matter: the cells of a row, the sections of the README, and what each section lists.
"""

from __future__ import annotations

import re
from pathlib import Path

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
from tool.render import RunMeta, chapter_markdown, manual_url, readme_markdown
from tool.status import EngineVerdict, ExampleResult, StatusIndex, build_index

BASE = "https://sdmx-twg.github.io/vtl/v2.2/reference_manual"
ISSUES = "https://github.com/Meaningful-Data/vtlengine/issues"
SPEC = "140e946ed3f888951adb8bb66ea45030be84e2aa"
META = RunMeta("2026-10-23", "abc1234", "9.9.9", SPEC)
# The types that have a flow in the README: the basic ones and the dataset.
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

# One operator, one variant, one example: small enough to write its chapter file by hand.
MINI_OPERATOR = make_operator(
    "length", "String length", "length", (Variant("comp", "length(Me_1)", (authored(),)),)
)
MINI_CHAPTER = Chapter("string", "String operators", (MINI_OPERATOR,))
MINI_CAT = Catalogue((MINI_CHAPTER,), ())
MINI_INDEX = build_index(MINI_CAT, {"string/length/comp/minimal": OK})


def make_tree(root: Path, folder: str, *, inputs: bool = True, notes: bool = False) -> None:
    """The files of an example folder that the markdown looks for."""
    base = root / folder
    base.mkdir(parents=True, exist_ok=True)
    (base / "script.vtl").write_text("DS_r := DS_1;\n", encoding="utf-8")
    (base / "expected").mkdir(exist_ok=True)
    if inputs:
        (base / "input").mkdir(exist_ok=True)
    if notes:
        (base / "notes.md").write_text("Notes.\n", encoding="utf-8")


def fences(text: str) -> list[str]:
    """The Mermaid blocks of a md file, in order."""
    return re.findall(r"^```mermaid\n(.*?)\n```$", text, flags=re.MULTILINE | re.DOTALL)


def section(text: str, heading: str) -> str:
    """The text of the ``## heading`` section, up to the next ``## `` heading or the end."""
    pattern = rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)"
    match = re.search(pattern, text, flags=re.MULTILINE | re.DOTALL)
    assert match is not None, f"no section {heading}"
    return match.group(1)


def row_with(text: str, needle: str) -> str:
    """The only table row of ``text`` that has ``needle``."""
    rows = [line for line in text.splitlines() if line.startswith("| ") and needle in line]
    assert len(rows) == 1, f"{len(rows)} rows have {needle}"
    return rows[0]


def cells(row: str) -> list[str]:
    """The cells of a table row, with the escaped pipes still escaped."""
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", row)[1:-1]]


def detail_cell(result: ExampleResult, tmp_path: Path) -> str:
    """The Detail cell of the only example of a one-example chapter that has ``result``."""
    cat = Catalogue(
        (Chapter("c", "C", (make_operator("o", "O", "o", (Variant("v", "v", (authored(),)),)),)),),
        (),
    )
    index = build_index(cat, {"c/o/v/minimal": result})
    text = chapter_markdown(cat.chapters[0], cat, index, META, examples_root=tmp_path)
    return cells(row_with(text, "/script.vtl)"))[4]


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


# --- Chapter files ---


def test_chapter_markdown_layout(tmp_path: Path) -> None:
    folder = "examples/string/length/comp/minimal"
    make_tree(tmp_path, "string/length/comp/minimal", notes=True)
    block = chapter_blocks(MINI_CHAPTER, MINI_CAT, MINI_INDEX)[0]
    assert chapter_markdown(MINI_CHAPTER, MINI_CAT, MINI_INDEX, META, examples_root=tmp_path) == (
        "# 🟢 String operators\n"
        "\n"
        "1 variant: 🟢 1 · 🟡 0 · 🔴 0 · 🔵 0\n"
        "\n"
        "[← Back to the operator map](README.md) · vtlengine 9.9.9 (abc1234)\n"
        "\n"
        "```mermaid\n"
        f"{block}\n"
        "```\n"
        "\n"
        "## 🟢 String length `length`\n"
        "\n"
        f"[VTL 2.2 Reference Manual]({BASE}/operators/Some%20operators/"
        "String%20length/index.html)\n"
        "\n"
        "| Variant | Status | pandas | DuckDB | Detail | Example |\n"
        "|---|---|---|---|---|---|\n"
        "| `length(Me_1)`<br><sub>minimal</sub> | 🟢 | ✅ | ✅ |  | "
        f"[script]({folder}/script.vtl) · [input]({folder}/input/) · "
        f"[expected]({folder}/expected/) · [notes]({folder}/notes.md) |\n"
    )


def test_chapter_markdown_title_counts_and_operator_headings(tmp_path: Path) -> None:
    text = chapter_markdown(STRING, CAT, INDEX, META, examples_root=tmp_path)
    lines = text.splitlines()
    # The chapter is 🔵 (a variant diverges); the counts are over variants: 3, none of them 🟢.
    assert lines[0] == "# 🔵 String operators"
    assert "3 variants: 🟢 0 · 🟡 1 · 🔴 1 · 🔵 1" in lines
    assert "[← Back to the operator map](README.md) · vtlengine 9.9.9 (abc1234)" in lines
    assert [line for line in lines if line.startswith("## ")] == [
        "## 🔵 Concatenation `||`",
        "## 🔴 String length `length`",
    ]
    assert f"[VTL 2.2 Reference Manual]({manual_url(CONCAT.page)})" in lines


def test_chapter_markdown_has_fence_and_table_row_per_example(tmp_path: Path) -> None:
    text = chapter_markdown(STRING, CAT, INDEX, META, examples_root=tmp_path)
    assert fences(text) == chapter_blocks(STRING, CAT, INDEX)
    rows = [line for line in text.splitlines() if line.startswith("| ") and "/script.vtl)" in line]
    assert len(rows) == 4  # manual_ex_1 and minimal, comp_sc, comp
    for variant, example in (
        ("concat/ds_ds", "manual_ex_1"),
        ("concat/ds_ds", "minimal"),
        ("concat/comp_sc", "minimal"),
        ("length/comp", "minimal"),
    ):
        row_with(text, f"examples/string/{variant}/{example}/script.vtl)")
    # A table per operator, each with its header.
    assert text.count("| Variant | Status | pandas | DuckDB | Detail | Example |") == 2


def test_the_row_of_an_example_shows_its_verdicts(tmp_path: Path) -> None:
    text = chapter_markdown(STRING, CAT, INDEX, META, examples_root=tmp_path)
    first = cells(row_with(text, "concat/ds_ds/manual_ex_1/script.vtl)"))
    # The label is code, with its pipes escaped, and the example name sits under it.
    assert first[:5] == ["`DS_1 \\|\\| DS_2`<br><sub>manual_ex_1</sub>", "🟢", "✅", "✅", ""]
    diverging = cells(row_with(text, "concat/ds_ds/minimal/script.vtl)"))
    assert diverging[1:5] == ["🔵", "✅", "💥", "DuckDB: SemanticError 1-1-1-1: boom"]
    wrong = cells(row_with(text, "concat/comp_sc/minimal/script.vtl)"))
    assert wrong[1:5] == ["🟡", "❌", "❌", "row Id_1=1: Me_1 expected 'a' got 'b'"]


def test_detail_is_one_line_escaped_and_cut_at_200_characters(tmp_path: Path) -> None:
    verdict = EngineVerdict("mismatch", "line one\nline two | pipe\r\nend\rlast")
    result = ExampleResult(verdict, verdict, True, "partial")
    assert detail_cell(result, tmp_path) == "line one line two \\| pipe end last"

    def cut(text: str) -> str:
        same = EngineVerdict("mismatch", text)
        return detail_cell(ExampleResult(same, same, True, "partial"), tmp_path)

    assert cut("x" * 200) == "x" * 200
    assert cut("x" * 201) == "x" * 199 + "…"
    # The pipes are escaped after the cut: the 200 characters are the visible ones.
    assert cut("|" * 250) == "\\|" * 199 + "…"


def test_detail_names_the_engine_that_has_one(tmp_path: Path) -> None:
    error = EngineVerdict("error", "SemanticError 1-1-1-1: boom")
    other = EngineVerdict("error", "RunTimeError 2-1-0-1: not implemented")
    assert detail_cell(ExampleResult(MATCH, error, False, "diverge"), tmp_path) == (
        "DuckDB: SemanticError 1-1-1-1: boom"
    )
    assert detail_cell(ExampleResult(error, MATCH, False, "diverge"), tmp_path) == (
        "pandas: SemanticError 1-1-1-1: boom"
    )
    # Two errors that differ are both shown; two that are the same are shown once.
    assert detail_cell(ExampleResult(error, other, True, "missing"), tmp_path) == (
        "pandas: SemanticError 1-1-1-1: boom; DuckDB: RunTimeError 2-1-0-1: not implemented"
    )
    assert detail_cell(ExampleResult(error, error, True, "missing"), tmp_path) == (
        "SemanticError 1-1-1-1: boom"
    )
    assert detail_cell(OK, tmp_path) == ""


def test_engines_that_differ_with_the_same_verdict_say_so(tmp_path: Path) -> None:
    # Both match, or both fail the same way, and still give different outputs: nothing else says.
    assert detail_cell(ExampleResult(MATCH, MATCH, False, "diverge"), tmp_path) == (
        "the engines give different outputs"
    )
    assert detail_cell(ExampleResult(WRONG, WRONG, False, "diverge"), tmp_path) == (
        "row Id_1=1: Me_1 expected 'a' got 'b'; the engines give different outputs"
    )


def test_example_cell_links_files_notes_and_marks(tmp_path: Path) -> None:
    folder = "examples/string/concat/ds_ds/manual_ex_1"
    make_tree(tmp_path, "string/concat/ds_ds/manual_ex_1", notes=True)
    text = chapter_markdown(STRING, CAT, INDEX, META, examples_root=tmp_path)
    cell = cells(row_with(text, f"{folder}/script.vtl)"))[5]
    assert cell == (
        f"[script]({folder}/script.vtl) · [input]({folder}/input/) · "
        f"[expected]({folder}/expected/) · [notes]({folder}/notes.md) · *manual* · erratum"
    )
    # A recorded decision links its issue, an adaptation says so, and an authored example has no
    # *manual* mark. Nothing is on disk for these: no input and no notes to link.
    decision = cells(row_with(text, "concat/comp_sc/minimal/script.vtl)"))[5]
    assert decision.endswith(
        f"[expected](examples/string/concat/comp_sc/minimal/expected/) · [📌 #1049]({ISSUES}/1049)"
    )
    adapted = cells(row_with(text, "length/comp/minimal/script.vtl)"))[5]
    assert adapted.endswith("/expected/) · adapted")
    assert "[input]" not in decision
    assert "[notes]" not in adapted
    assert "*manual*" not in adapted


def test_example_without_a_result_is_not_run(tmp_path: Path) -> None:
    nothing = StatusIndex({}, {}, {}, {})
    text = chapter_markdown(STRING, CAT, nothing, META, examples_root=tmp_path)
    lines = text.splitlines()
    assert lines[0] == "# String operators"
    assert "0 variants: 🟢 0 · 🟡 0 · 🔴 0 · 🔵 0" in lines
    assert "## Concatenation `||`" in lines
    row = cells(row_with(text, "concat/ds_ds/minimal/script.vtl)"))
    assert row[1:5] == ["", "", "", "not run"]


# --- README ---


def test_readme_sections_come_in_the_order_of_the_design() -> None:
    text = readme_markdown(CAT, INDEX, META)
    assert text.startswith("# VTL 2.2 operator map")
    assert re.findall(r"^## (.+)$", text, flags=re.MULTILINE) == [
        "Generation",
        "Legend",
        "How to read",
        "Overview",
        "Chapters",
        "Type lattice",
        "Type flows",
        "📌 Team decisions",
        "Manual errata",
        "Input adaptations",
        "Variants that are not green",
        "Attribution",
    ]
    assert text.endswith("\n")


def test_readme_metadata_and_legend() -> None:
    text = readme_markdown(CAT, INDEX, META)
    generation = section(text, "Generation")
    for value in ("2026-10-23", "9.9.9", "abc1234", SPEC[:12]):
        assert value in generation
    legend = section(text, "Legend")
    for mark in ("🟢", "🟡", "🔴", "🔵", "📌", "✅", "❌", "💥"):
        assert mark in legend


def test_readme_blocks_are_the_overview_the_lattice_and_a_flow_per_basic_type() -> None:
    text = readme_markdown(CAT, INDEX, META)
    assert fences(section(text, "Overview")) == [overview_block(CAT, INDEX)]
    assert fences(section(text, "Type lattice")) == [lattice_block(CAT, INDEX)]
    flows = section(text, "Type flows")
    assert fences(flows) == [type_flow_block(type_id, CAT, INDEX) for type_id in FLOW_TYPES]
    assert re.findall(r"^### (.+)$", flows, flags=re.MULTILINE) == list(FLOW_TYPES)
    assert len(fences(text)) == 12


def test_readme_chapter_index_counts_variants() -> None:
    index = section(readme_markdown(CAT, INDEX, META), "Chapters")
    assert cells(row_with(index, "| Chapter |")) == [
        "Chapter",
        "Status",
        "Variants",
        "🟢",
        "🟡",
        "🔴",
        "🔵",
    ]
    assert cells(row_with(index, "(string.md)")) == [
        "[String operators](string.md)",
        "🔵",
        "3",
        "0",
        "1",
        "1",
        "1",
    ]
    assert cells(row_with(index, "(clause.md)")) == [
        "[Clause operators](clause.md)",
        "🟢",
        "1",
        "1",
        "0",
        "0",
        "0",
    ]
    # The type hubs have no file of their own; the total counts everything above it.
    assert cells(row_with(index, "Data types")) == ["Data types", "🟢", "1", "1", "0", "0", "0"]
    assert cells(row_with(index, "Total")) == ["**Total**", "🔵", "5", "2", "1", "1", "1"]


def test_readme_lists_errata_adaptations_and_decisions() -> None:
    text = readme_markdown(CAT, INDEX, META)
    where = "[String operators](string.md) › Concatenation"
    decisions = section(text, "📌 Team decisions")
    assert f"[📌 #1049]({ISSUES}/1049)" in decisions
    assert f'{where} › `Me_1 || "x"` (minimal)' in decisions
    errata = section(text, "Manual errata")
    assert "the manual shows 'ba' where 'ab' is meant" in errata
    assert f"{where} › `DS_1 || DS_2` (manual_ex_1)" in errata
    assert f"[manual]({manual_url(CONCAT.page)})" in errata
    adaptations = section(text, "Input adaptations")
    assert "the time values were rewritten" in adaptations
    assert "[String operators](string.md) › String length › `length(Me_1)` (minimal)" in adaptations
    # Each list holds what it should and nothing else.
    assert "1049" not in errata + adaptations
    assert "rewritten" not in errata + decisions
    assert "'ba'" not in adaptations + decisions


def test_readme_lists_the_notes_that_exist(tmp_path: Path) -> None:
    make_tree(tmp_path, "string/concat/comp_sc/minimal", notes=True)
    text = readme_markdown(CAT, INDEX, META, examples_root=tmp_path)
    decisions = section(text, "📌 Team decisions")
    assert "[notes](examples/string/concat/comp_sc/minimal/notes.md)" in decisions
    assert "[notes]" not in section(text, "Manual errata")


def test_readme_lists_variants_that_are_not_green_by_status() -> None:
    text = readme_markdown(CAT, INDEX, META)
    listing = section(text, "Variants that are not green")
    assert re.findall(r"^### (.+)$", listing, flags=re.MULTILINE) == [
        "🔵 Engines diverge (1)",
        "🔴 Not implemented (1)",
        "🟡 Partially implemented (1)",
    ]
    diverge, missing, partial = re.split(r"^### .+$", listing, flags=re.MULTILINE)[1:]
    assert cells(row_with(diverge, "(string.md)")) == [
        "[String operators](string.md)",
        "Concatenation",
        "`DS_1 \\|\\| DS_2`",
        "[minimal](examples/string/concat/ds_ds/minimal/)",
        "DuckDB: SemanticError 1-1-1-1: boom",
    ]
    assert "`length(Me_1)`" in row_with(missing, "(string.md)")
    assert '`Me_1 \\|\\| "x"`' in row_with(partial, "(string.md)")
    # The green variants are not listed, and neither are the type hubs that are green.
    assert "Calculation" not in listing
    assert "Data types" not in listing


def test_a_diverging_variant_is_listed_with_the_example_that_diverges() -> None:
    # The first example is only 🟡: the row of the 🔵 variant has to point at the second.
    variant = Variant("ds", "DS_1 + DS_2", (authored("minimal"), authored("minimal_2")))
    operator = make_operator("plus", "Addition", "+", (variant,))
    cat = Catalogue((Chapter("numeric", "Numeric operators", (operator,)),), ())
    results = {"numeric/plus/ds/minimal": PARTIAL, "numeric/plus/ds/minimal_2": DIVERGE}
    listing = section(
        readme_markdown(cat, build_index(cat, results), META), "Variants that are not green"
    )
    [row] = [line for line in listing.splitlines() if "(numeric.md)" in line]
    assert "[minimal_2](examples/numeric/plus/ds/minimal_2/)" in row
    assert re.findall(r"^### (.+)$", listing, flags=re.MULTILINE) == ["🔵 Engines diverge (1)"]


def test_every_block_of_a_split_chapter_gets_its_own_fence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocks = ["flowchart LR\n    a --> b", "flowchart LR\n    c --> d", "flowchart LR\n    e --> f"]
    monkeypatch.setattr("tool.render.chapter_blocks", lambda *_args: blocks)
    text = chapter_markdown(STRING, CAT, INDEX, META, examples_root=tmp_path)
    assert fences(text) == blocks
    assert text.count("```mermaid") == 3


def test_readme_lists_the_variants_of_a_type_that_are_not_green() -> None:
    results = {**RESULTS, "types/number/identity/minimal": PARTIAL}
    listing = section(
        readme_markdown(CAT, build_index(CAT, results), META), "Variants that are not green"
    )
    row = cells(row_with(listing, "Data types"))
    assert row[:3] == ["Data types", "number", "`identity`"]


def test_readme_attribution_sentence_is_verbatim() -> None:
    attribution = section(readme_markdown(CAT, INDEX, META), "Attribution")
    assert (
        "Examples marked *manual* are taken from the VTL 2.2 Reference Manual "
        f"(sdmx-twg/vtl @ {SPEC}), converted to the vtlengine format."
    ) in attribution


def test_readme_empty_lists_say_so() -> None:
    cat = Catalogue((MINI_CHAPTER,), ())
    text = readme_markdown(cat, MINI_INDEX, META)
    assert "None: no expected output follows a recorded team decision." in text
    assert "None: no expected output of the manual had to be corrected." in text
    assert "None: every manual input loads as written." in text
    assert "Every variant is green." in text
    # No type has a variant, so there is no row for the types, and the blocks still render.
    assert "| Data types |" not in text
    assert len(fences(text)) == 12


@pytest.mark.parametrize("status", ["ok", "partial", "missing", "diverge"])
def test_chapter_title_carries_the_emoji_of_its_status(status: Status, tmp_path: Path) -> None:
    emoji = {"ok": "🟢", "partial": "🟡", "missing": "🔴", "diverge": "🔵"}[status]
    index = StatusIndex({}, {}, {}, {"string": status})
    text = chapter_markdown(MINI_CHAPTER, MINI_CAT, index, META, examples_root=tmp_path)
    assert text.startswith(f"# {emoji} String operators\n")
