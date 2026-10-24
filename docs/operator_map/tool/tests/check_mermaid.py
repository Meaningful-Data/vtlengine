# ruff: noqa: S101
"""Checks for the Mermaid rendering: escaping, ids, limits, chapters and the overview.

The catalogues and the statuses are small hand-made values: no engine runs here.
"""

from __future__ import annotations

import re

import pytest

import tool.mermaid
from tool.mermaid import (
    MAX_CHARS,
    MAX_EDGES,
    PAGE_SUFFIX,
    MermaidLimitError,
    chapter_blocks,
    check_limits,
    escape_label,
    node_id,
    overview_block,
)
from tool.model import Catalogue, Chapter, Link, Operator, Status, Variant, authored
from tool.status import EMOJI, StatusIndex

# What every block starts with: the diagram type and the classes of the statuses and the stubs.
CLASSES = [
    "flowchart LR",
    "    classDef ok fill:#d4edda,stroke:#28a745,color:#000",
    "    classDef partial fill:#fff3cd,stroke:#ffc107,color:#000",
    "    classDef missing fill:#f8d7da,stroke:#dc3545,color:#000",
    "    classDef diverge fill:#cce5ff,stroke:#007bff,color:#000",
    "    classDef ext stroke-dasharray: 5 5",
]


# --- Builders ---


def block_of(*lines: str) -> str:
    """The text of a block: the classes, then ``lines`` indented."""
    return "\n".join([*CLASSES, *(f"    {line}" for line in lines)])


def make_variant(
    variant_id: str,
    label: str,
    *,
    returns: tuple[str, ...] = (),
    decision: int | None = None,
    examples: tuple[str, ...] = ("minimal",),
) -> Variant:
    return Variant(
        variant_id,
        label,
        tuple(authored(name, decision=decision) for name in examples),
        returns,
    )


def make_operator(
    operator_id: str,
    name: str,
    symbol: str,
    accepts: tuple[str, ...],
    returns: tuple[str, ...],
    variants: tuple[Variant, ...],
    links: tuple[Link, ...] = (),
) -> Operator:
    page = f"Some operators/{name}"
    return Operator(operator_id, name, symbol, page, accepts, returns, variants, links)


def make_index(
    variants: dict[str, Status] | None = None,
    operators: dict[str, Status] | None = None,
    chapters: dict[str, Status] | None = None,
) -> StatusIndex:
    return StatusIndex({}, variants or {}, operators or {}, chapters or {})


# A small map: string operators that link to a clause.
CONCAT = make_operator(
    "concat",
    "Concatenation",
    "||",
    ("string",),
    ("string",),
    (
        make_variant("ds_ds", "DS_1 || DS_2", examples=("manual_ex_1", "minimal")),
        make_variant("comp_sc", 'Me_1 || "x"', decision=1049),
    ),
    (Link("clause.calc", "used in"),),
)
LENGTH = make_operator(
    "length",
    "String length",
    "length",
    ("string",),
    ("integer",),
    (
        make_variant("comp", "length(Me_1)"),
        make_variant("sc_null", "length(null)", returns=("number",)),
    ),
)
CALC = make_operator(
    "calc",
    "Calculation",
    "calc",
    ("dataset",),
    ("dataset",),
    (make_variant("measure", "calc Me_1 := 1"),),
)
STRING = Chapter("string", "String operators", (CONCAT, LENGTH))
CLAUSE = Chapter("clause", "Clause operators", (CALC,))
CAT = Catalogue((STRING, CLAUSE))
# ``string/length`` has not been run: it has no status at any level.
INDEX = make_index(
    variants={
        "string/concat/ds_ds": "ok",
        "string/concat/comp_sc": "diverge",
        "clause/calc/measure": "missing",
    },
    operators={"string/concat": "diverge", "clause/calc": "missing"},
    chapters={"string": "diverge", "clause": "missing"},
)
NOTHING_RUN = make_index()


# --- Escaping and ids ---


def test_escape_label_vtl_symbols() -> None:
    assert escape_label('DS_1#Me_1 || "x" <> [a]{b};') == (
        "DS_1#35;Me_1 #124;#124; #quot;x#quot; #60;#62; #91;a#93;#123;b#125;#59;"
    )


def test_escape_label_assignment_and_plain_text() -> None:
    assert escape_label("DS_r := DS_1") == "DS_r #58;#61; DS_1"
    # Only the symbol ``:=`` is escaped: a lone ``:`` or ``=``, parentheses and arrows are not.
    assert escape_label("↗ Calc (clause.html): a = b") == "↗ Calc (clause.html): a = b"


def test_node_ids_are_chapter_qualified() -> None:
    assert node_id("aggregate", "sum") != node_id("definitions", "sum")
    assert node_id("numeric", "unary_plus", "ds") == "numeric__unary_plus__ds"


def test_node_id_replaces_characters_outside_the_id_alphabet() -> None:
    assert node_id("a-b", "c d.e", "f#") == "a_b__c_d_e__f_"


# --- Limits ---


def edge_lines(count: int, arrow: str) -> list[str]:
    return [f"a{n} {arrow} b{n}" for n in range(count)]


def test_check_limits_raises() -> None:
    block = "\n".join(["flowchart LR", *edge_lines(501, "-->")])
    with pytest.raises(MermaidLimitError, match="501 edges"):
        check_limits(block)


def test_check_limits_counts_every_edge_style_and_stays_under_the_limit() -> None:
    under = [
        "flowchart LR",
        'n0["a"]',  # a node line is not an edge
        *edge_lines(200, "-->"),
        *edge_lines(150, "---"),
        *edge_lines(149, ".->"),
    ]
    check_limits("\n".join(under))  # 499 edges
    assert MAX_EDGES == 500
    with pytest.raises(MermaidLimitError, match="500 edges"):
        check_limits("\n".join([*under, "a --- b"]))


def test_check_limits_counts_characters_as_mermaid_does() -> None:
    assert MAX_CHARS == 50_000
    check_limits("x" * (MAX_CHARS - 1))
    with pytest.raises(MermaidLimitError, match="characters"):
        check_limits("x" * MAX_CHARS)
    # Mermaid counts UTF-16 code units, so each status emoji, outside the BMP, is two.
    emojis = "🟢" * (MAX_CHARS // 2)
    assert len(emojis) < MAX_CHARS
    with pytest.raises(MermaidLimitError, match="characters"):
        check_limits(emojis)


# --- Chapter blocks ---


def test_every_block_starts_with_the_class_definitions() -> None:
    blocks = [*chapter_blocks(STRING, CAT, INDEX), overview_block(CAT, INDEX)]
    for block in blocks:
        assert block.splitlines()[: len(CLASSES)] == CLASSES


def test_chapter_block_layout() -> None:
    assert chapter_blocks(STRING, CAT, INDEX) == [
        block_of(
            # The stubs of the operators of other chapters, once per block, as they are met.
            'ext_clause__calc["↗ Calculation (clause.html)"]:::ext',
            'click ext_clause__calc "clause.html"',
            # Concatenation: a decision of its second variant pins it and that variant.
            'string__concat["🔵 Concatenation #124;#124; 📌"]:::diverge',
            'string__concat__ds_ds["🟢 DS_1 #124;#124; DS_2"]:::ok',
            'string__concat__comp_sc["🔵 Me_1 #124;#124; #quot;x#quot; 📌"]:::diverge',
            "string__concat --- string__concat__ds_ds",
            "string__concat --- string__concat__comp_sc",
            'string__concat -. "used in" .-> ext_clause__calc',
            'click string__concat "examples/string/concat/ds_ds/manual_ex_1/"',
            'click string__concat__ds_ds "examples/string/concat/ds_ds/manual_ex_1/"',
            'click string__concat__comp_sc "examples/string/concat/comp_sc/minimal/"',
            # String length has not been run: no emoji and no class. What it takes and gives,
            # and what its second variant returns, are no nodes: a diagram has no types.
            'string__length["String length length"]',
            'string__length__comp["length(Me_1)"]',
            'string__length__sc_null["length(null)"]',
            "string__length --- string__length__comp",
            "string__length --- string__length__sc_null",
            'click string__length "examples/string/length/comp/minimal/"',
            'click string__length__comp "examples/string/length/comp/minimal/"',
            'click string__length__sc_null "examples/string/length/sc_null/minimal/"',
        )
    ]


def test_chapter_without_operators_has_no_blocks() -> None:
    empty = Chapter("empty", "Empty", ())
    assert chapter_blocks(empty, Catalogue((empty,)), NOTHING_RUN) == []


def test_operator_without_examples_has_no_click() -> None:
    bare = make_operator("bare", "Bare", "bare", (), (), ())
    chapter = Chapter("numeric", "Numeric operators", (bare,))
    (block,) = chapter_blocks(chapter, Catalogue((chapter,)), NOTHING_RUN)
    assert block == block_of('numeric__bare["Bare bare"]')


def test_link_stub_is_defined_once_for_several_operators() -> None:
    link = (Link("clause.calc", "used in"),)
    first = make_operator("a", "A", "a", (), (), (make_variant("v", "a"),), link)
    second = make_operator("b", "B", "b", (), (), (make_variant("v", "b"),), link)
    chapter = Chapter("numeric", "Numeric operators", (first, second))
    (block,) = chapter_blocks(chapter, Catalogue((chapter, CLAUSE)), NOTHING_RUN)
    assert block.count('ext_clause__calc["') == 1
    assert block.count('click ext_clause__calc "clause.html"') == 1
    assert block.count(".-> ext_clause__calc") == 2


def test_the_pages_of_the_chapters_are_html_pages_of_the_site() -> None:
    # The map is a part of the Sphinx docs: a chapter is the page ``<chapter>.html``, never a
    # file of Markdown. The diagrams name it so, in the clicks and in the labels of the stubs.
    assert PAGE_SUFFIX == ".html"
    (block,) = chapter_blocks(STRING, CAT, INDEX)
    overview = overview_block(CAT, INDEX)
    for text in (block, overview):
        assert ".md" not in text
    assert f'click ext_clause__calc "clause{PAGE_SUFFIX}"' in block
    assert f"(clause{PAGE_SUFFIX})" in block
    assert f'click chapter__string "string{PAGE_SUFFIX}"' in overview


def test_the_size_limits_are_checked_on_the_text_with_the_page_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = [
        make_operator(f"t{n}", f"T{n}", "t", (), (), (make_variant("v", "t"),)) for n in "123"
    ]
    other = Chapter("other", "Other operators", tuple(targets))
    links = tuple(Link(f"other.t{n}", "uses") for n in "123")

    def chapter(padding: int) -> Chapter:
        name = "x" * padding
        pad = make_operator("pad", name, "p", (), (), (make_variant("v", "p"),), links)
        return Chapter("numeric", "Numeric operators", (pad,))

    def size(padding: int) -> int:
        (block,) = chapter_blocks(
            chapter(padding), Catalogue((chapter(padding), other)), NOTHING_RUN
        )
        return len(block.encode("utf-16-le")) // 2

    # With the suffix ``.md`` the block would be one character under the limit (the name of the
    # operator is one character long to begin with, and adds a character for each other one)...
    monkeypatch.setattr(tool.mermaid, "PAGE_SUFFIX", ".md")
    padding = MAX_CHARS - size(1)
    assert size(padding) == MAX_CHARS - 1
    # ...and the name of each page is written twice for each of the three stubs: with the real
    # suffix it is over, and the check says so.
    monkeypatch.setattr(tool.mermaid, "PAGE_SUFFIX", PAGE_SUFFIX)
    with pytest.raises(MermaidLimitError, match="characters"):
        size(padding)


def test_link_labels_stub_names_and_chapter_titles_are_escaped() -> None:
    target = make_operator("calc", 'Calc "x" [y]', "calc", (), (), (make_variant("v", "calc"),))
    clause = Chapter("clause", "Clause # operators", (target,))
    link = Link("clause.calc", "a|b")
    source = make_operator("a", "A", "a", (), (), (make_variant("v", "a"),), (link,))
    chapter = Chapter("numeric", "Numeric operators", (source,))
    cat = Catalogue((chapter, clause))
    (block,) = chapter_blocks(chapter, cat, NOTHING_RUN)
    assert 'numeric__a -. "a#124;b" .-> ext_clause__calc' in block
    assert 'ext_clause__calc["↗ Calc #quot;x#quot; #91;y#93; (clause.html)"]:::ext' in block
    assert 'chapter__clause["Clause #35; operators<br/>' in overview_block(cat, NOTHING_RUN)


def test_link_to_an_unknown_target_raises() -> None:
    lost = make_operator("lost", "Lost", "lost", (), (), (), (Link("clause.nope", "used in"),))
    chapter = Chapter("numeric", "Numeric operators", (lost,))
    with pytest.raises(ValueError, match="numeric.lost: target 'clause.nope' not found"):
        chapter_blocks(chapter, Catalogue((chapter, CLAUSE)), NOTHING_RUN)


def synthetic_chapter(count: int) -> Chapter:
    """``count`` operators with two variants each."""
    operators = tuple(
        make_operator(
            f"op_{n}",
            f"Operator {n}",
            f"op{n}",
            ("number",) if n % 2 else ("string",),
            ("number",),
            (make_variant("ds", "DS"), make_variant("comp", "Me_1")),
        )
        for n in range(count)
    )
    return Chapter("numeric", "Numeric operators", operators)


def test_chapter_blocks_split_under_limits() -> None:
    chapter = synthetic_chapter(300)
    blocks = chapter_blocks(chapter, Catalogue((chapter,)), NOTHING_RUN)
    assert len(blocks) >= 2
    for block in blocks:
        check_limits(block)
    defined = [
        [m.group(1) for m in re.finditer(r"^ *(numeric__op_\d+)\[", block, re.MULTILINE)]
        for block in blocks
    ]
    # Every operator is defined in exactly one block, and the blocks keep the catalogue order.
    assert [node for nodes in defined for node in nodes] == [
        node_id("numeric", operator.id) for operator in chapter.operators
    ]


def test_a_single_operator_over_the_limit_raises() -> None:
    forms = tuple(make_variant(f"v{n}", f"form {n}") for n in range(MAX_EDGES + 1))
    big = make_operator("big", "Big", "big", (), (), forms)
    chapter = Chapter("numeric", "Numeric operators", (big,))
    with pytest.raises(MermaidLimitError, match=r"numeric\.big"):
        chapter_blocks(chapter, Catalogue((chapter,)), NOTHING_RUN)


# --- Overview ---


def test_overview_layout() -> None:
    # One node per chapter, with how many of its variants are in each status, and a click to
    # its page: no node for a type.
    assert overview_block(CAT, INDEX) == block_of(
        'chapter__string["🔵 String operators<br/>🟢 1 · 🟡 0 · 🔴 0 · 🔵 1"]:::diverge',
        'chapter__clause["🔴 Clause operators<br/>🟢 0 · 🟡 0 · 🔴 1 · 🔵 0"]:::missing',
        'click chapter__string "string.html"',
        'click chapter__clause "clause.html"',
    )


# --- Statuses that are not there yet ---


def test_missing_statuses_render_neutral() -> None:
    blocks = chapter_blocks(STRING, CAT, NOTHING_RUN)
    text = "\n".join(line for block in blocks for line in block.splitlines()[len(CLASSES) :])
    for status, emoji in EMOJI.items():
        assert f":::{status}" not in text
        assert emoji not in text
    assert overview_block(CAT, NOTHING_RUN).splitlines()[len(CLASSES)] == (
        '    chapter__string["String operators<br/>🟢 0 · 🟡 0 · 🔴 0 · 🔵 0"]'
    )
