# ruff: noqa: S101
"""Checks for the Mermaid diagrams: escaping, ids, limits, and the three diagrams of the map.

The three diagrams are the overview of the blocks, the diagram of a block and the diagram of an
operator. The catalogues and the statuses are small hand-made values: no engine runs here. The
expected text of a diagram is written out in full where its layout is what is checked.
"""

from __future__ import annotations

import dataclasses
import re

import pytest

from tool.mermaid import (
    BOX_STYLES,
    CLASSES,
    LIGHT_BOX_STYLE,
    MAX_CHARS,
    MAX_EDGES,
    PAGE_SUFFIX,
    USED_WITH_MAX,
    USED_WITH_STYLE,
    MermaidLimitError,
    block_diagram,
    check_limits,
    escape_label,
    example_anchor,
    form_anchor,
    forms_in_diagram_order,
    has_boxes,
    node_id,
    operator_diagram,
    overview_block,
    used_with,
)
from tool.model import Catalogue, Chapter, Kind, Link, Operator, Status, Variant, authored
from tool.status import EMOJI, StatusIndex

# What every diagram has after its ``flowchart`` line: the classes of the statuses, the one of
# the central node and the one of the entries of "Used with". They are written out here, so that
# a change of a colour or of a border is seen.
CLASS_LINES = [
    "    classDef ok fill:#d4edda,stroke:#28a745,color:#000",
    "    classDef partial fill:#fff3cd,stroke:#ffc107,color:#000",
    (
        "    classDef refused fill:#d4edda,stroke:#28a745,stroke-width:3px,"
        "stroke-dasharray: 2 2,color:#000"
    ),
    "    classDef missing fill:#f8d7da,stroke:#dc3545,color:#000",
    "    classDef diverge fill:#cce5ff,stroke:#007bff,color:#000",
    "    classDef block fill:#2b3a4a,stroke:#2b3a4a,color:#fff,font-weight:bold",
    "    classDef ext fill:#fff,stroke:#666,color:#000,stroke-dasharray: 5 5",
]
# The light box of each kind of operand: the same fill, and a border of its own.
SOLID = "fill:#f8f9fa,stroke:#495057,stroke-width:2px"
DASHED = f"{SOLID},stroke-dasharray: 8 5"
DOTTED = f"{SOLID},stroke-dasharray: 2 3"
# The light box that the forms name: the same fill, and a thin solid border.
LIGHT = "fill:#f8f9fa,stroke:#868e96,stroke-width:1px"
USED_WITH = "fill:#ffffff,stroke:#868e96,stroke-width:1px,stroke-dasharray: 1 4"


# --- Builders ---


def block_of(direction: str, *lines: str) -> str:
    """The text of a diagram: its ``flowchart`` line, the classes, then ``lines`` indented."""
    return "\n".join([f"flowchart {direction}", *CLASS_LINES, *(f"    {line}" for line in lines)])


def body(diagram: str) -> list[str]:
    """The lines of a diagram after its ``flowchart`` line and its classes, indentation off."""
    return [line.strip() for line in diagram.splitlines()[1 + len(CLASS_LINES) :]]


def make_variant(
    variant_id: str,
    label: str,
    *,
    returns: tuple[str, ...] = (),
    decision: int | None = None,
    examples: tuple[str, ...] = ("minimal",),
    kind: Kind | None = None,
    box: str | None = None,
) -> Variant:
    return Variant(
        variant_id,
        label,
        tuple(authored(name, decision=decision) for name in examples),
        returns,
        kind=kind,
        box=box,
    )


def make_operator(
    operator_id: str,
    name: str,
    symbol: str,
    variants: tuple[Variant, ...],
    links: tuple[Link, ...] = (),
) -> Operator:
    page = f"Some operators/{name}"
    return Operator(operator_id, name, symbol, page, (), (), variants, links)


def make_index(
    variants: dict[str, Status] | None = None,
    operators: dict[str, Status] | None = None,
    chapters: dict[str, Status] | None = None,
) -> StatusIndex:
    return StatusIndex({}, variants or {}, operators or {}, chapters or {})


def edges_of(diagram: str) -> int:
    """How many edges a diagram has: one per line that has a ``---`` or an arrow."""
    marks = ("-->", "---", ".->", "-.-")
    return sum(1 for line in diagram.splitlines() if any(mark in line for mark in marks))


def size_of(diagram: str) -> int:
    """The length of a diagram as Mermaid counts it, in UTF-16 code units."""
    return len(diagram.encode("utf-16-le")) // 2


# A small map: string operators that link to a clause. Concatenation has a dataset form and a
# component form, so its diagram has two boxes; Calculation has one form with no kind, so its
# diagram has both sides and no box; String length has not been run.
CONCAT = make_operator(
    "concat",
    "Concatenation",
    "||",
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
    (
        make_variant("comp", "length(Me_1)"),
        make_variant("sc_null", "length(null)", returns=("number",)),
    ),
)
CALC = make_operator("calc", "Calculation", "calc", (make_variant("measure", "calc Me_1 := 1"),))
STRING = Chapter("string", "String operators", (CONCAT, LENGTH))
CLAUSE = Chapter("clause", "Clause operators", (CALC,))
CAT = Catalogue((STRING, CLAUSE))
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

# An operator with a form of each kind: the three boxes, in the order of the widest operand.
ROUND = make_operator(
    "round",
    "Rounding",
    "round",
    (
        make_variant("ds", "round(DS_1, 1)"),
        make_variant("comp", "DS_1[calc Me_1 := round(Me_1)]"),
        make_variant("sc", "round(1.5)"),
    ),
)
NUMERIC = Chapter("numeric", "Numeric operators", (ROUND,))
ROUND_CAT = Catalogue((NUMERIC,))
ROUND_INDEX = make_index(
    variants={
        "numeric/round/ds": "ok",
        "numeric/round/comp": "partial",
        "numeric/round/sc": "refused",
    },
    operators={"numeric/round": "partial"},
    chapters={"numeric": "partial"},
)


# A definition of the VTL-DL: its forms are parts of a syntax, and each names the light box that it
# is drawn in. The catalogue lists them form by form, so the boxes are not in the order of the
# forms: the Signature has the first and the third, the Body the second and the fourth. It is used
# in a calc.
DEFINE = make_operator(
    "define_viral",
    "Viral propagation",
    "define viral propagation",
    (
        make_variant("variable", "define x (variable v)", box="Signature"),
        make_variant("enumerated", "when v = x then y", box="Body"),
        make_variant("valuedomain", "define x (valuedomain d)", box="Signature"),
        make_variant("aggregation", "aggr sum", box="Body"),
    ),
    (Link("clause.calc", "used in"),),
)
DEFINE_CHAPTER = Chapter("definitions", "Definitions", (DEFINE,))
DEFINE_CAT = Catalogue((DEFINE_CHAPTER, CLAUSE))
DEFINE_INDEX = make_index(
    variants={
        "definitions/define_viral/variable": "ok",
        "definitions/define_viral/enumerated": "partial",
        "definitions/define_viral/valuedomain": "ok",
        "definitions/define_viral/aggregation": "diverge",
    },
    operators={"definitions/define_viral": "diverge", "clause/calc": "missing"},
    chapters={"definitions": "diverge", "clause": "missing"},
)


def chapters_of(count: int) -> tuple[Chapter, ...]:
    """``count`` chapters, ``c0``... in that order, each with one operator."""
    return tuple(
        Chapter(
            f"c{n}",
            f"Chapter {n}",
            (make_operator(f"o{n}", f"Operator {n}", f"op{n}", (make_variant("v", "v"),)),),
        )
        for n in range(count)
    )


def operators_of(count: int) -> Chapter:
    """A chapter of ``count`` operators, ``o0``..., each with two forms, ``a`` and ``b``."""
    operators = tuple(
        make_operator(
            f"o{n}", f"Operator {n}", f"op{n}", (make_variant("a", "a"), make_variant("b", "b"))
        )
        for n in range(count)
    )
    return Chapter("numeric", "Numeric operators", operators)


def lone(variants: tuple[Variant, ...]) -> tuple[Chapter, Operator, Catalogue]:
    """An operator with ``variants``, alone in its chapter."""
    operator = make_operator("op", "Op", "op", variants)
    chapter = Chapter("numeric", "Numeric operators", (operator,))
    return chapter, operator, Catalogue((chapter,))


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


# --- Anchors ---


def test_form_anchor_has_no_underscore() -> None:
    assert form_anchor(make_variant("ds_group_by", "sum(DS group by Id)")) == "form-ds-group-by"
    assert form_anchor(make_variant("ds", "x")) == "form-ds"


def test_form_anchors_are_what_docutils_makes_of_them() -> None:
    # Sphinx writes the id of a target with ``make_id``: an anchor that it would change is a click
    # that lands nowhere. It needs docutils, which the docs have: the check is skipped without it.
    nodes = pytest.importorskip("docutils.nodes")
    for form_id in ("ds", "ds_group_by", "comp_2", "sc_null", "2d", "a_b_c_d"):
        anchor = form_anchor(make_variant(form_id, "x"))
        assert nodes.make_id(anchor) == anchor


def test_example_anchor_has_the_form_and_the_name_and_no_underscore() -> None:
    form = make_variant("ds_group_by", "x", examples=("zero_divisor", "manual_ex_1", "minimal"))
    assert [example_anchor(form, example) for example in form.examples] == [
        "ex-ds-group-by-zero-divisor",
        "ex-ds-group-by-manual-ex-1",
        "ex-ds-group-by-minimal",
    ]


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
        *edge_lines(150, "-->"),
        *edge_lines(100, "---"),
        *edge_lines(100, "-.-"),  # the dotted line of "Used with"
        *edge_lines(149, ".->"),
    ]
    check_limits("\n".join(under))  # 499 edges
    assert MAX_EDGES == 500
    with pytest.raises(MermaidLimitError, match="500 edges"):
        check_limits("\n".join([*under, "a -.- b"]))


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


# --- The classes and the styles ---


def test_the_classes_and_the_styles_are_the_designed_ones() -> None:
    assert [f"    {line}" for line in CLASSES] == CLASS_LINES
    assert BOX_STYLES == {"ds": SOLID, "comp": DASHED, "sc": DOTTED}
    assert LIGHT_BOX_STYLE == LIGHT
    # The light box has the fill of the boxes of the kinds, and a thin solid border: no dashes.
    assert LIGHT.split(",stroke")[0] == SOLID.split(",stroke")[0]
    assert "stroke-dasharray" not in LIGHT
    assert USED_WITH_STYLE == USED_WITH
    assert USED_WITH_MAX == 8


def test_every_diagram_starts_with_its_type_and_the_classes() -> None:
    sides = [
        overview_block(CAT, INDEX),
        block_diagram(STRING, INDEX),
        operator_diagram(CLAUSE, CALC, CAT, INDEX),
    ]
    boxes = [
        operator_diagram(NUMERIC, ROUND, ROUND_CAT, ROUND_INDEX),
        operator_diagram(DEFINE_CHAPTER, DEFINE, DEFINE_CAT, DEFINE_INDEX),
    ]
    for direction, diagrams in (("LR", sides), ("TB", boxes)):
        for diagram in diagrams:
            assert diagram.splitlines()[: 1 + len(CLASS_LINES)] == [
                f"flowchart {direction}",
                *CLASS_LINES,
            ]


# --- The overview ---


def test_overview_layout() -> None:
    # VTL 2.2 in the middle, the first half of the blocks to the left of it and the rest to the
    # right, with plain lines. A block says how many operators it has and how many are in each
    # status, and links to its page: there is no node for a type.
    assert overview_block(CAT, INDEX) == block_of(
        "LR",
        'R["VTL 2.2<br/>Reference Manual"]:::block',
        'chapter__string["String operators<br/>2 operators · 🔵1"]:::diverge',
        'chapter__clause["Clause operators<br/>1 operator · 🔴1"]:::missing',
        "chapter__string --- R",
        "R --- chapter__clause",
        'click chapter__string href "string/index.html"',
        'click chapter__clause href "clause/index.html"',
    )


@pytest.mark.parametrize(("count", "left"), [(14, 7), (5, 3), (4, 2), (1, 1), (0, 0)])
def test_overview_both_sides(count: int, left: int) -> None:
    cat = Catalogue(chapters_of(count))
    text = overview_block(cat, NOTHING_RUN)
    assert text.splitlines()[0] == "flowchart LR"
    assert "-->" not in text
    lines = body(text)
    ids = [node_id("chapter", chapter.id) for chapter in cat.chapters]
    # The first half of the chapters is written ``X --- R`` and the rest ``R --- X``, in order.
    assert [line for line in lines if line.endswith(" --- R")] == [f"{i} --- R" for i in ids[:left]]
    assert [line for line in lines if line.startswith("R --- ")] == [
        f"R --- {i}" for i in ids[left:]
    ]
    for chapter, node in zip(cat.chapters, ids):
        assert f'click {node} href "{chapter.id}/index.html"' in lines


def test_overview_counts_the_operators_and_tallies_their_statuses() -> None:
    operators = tuple(
        make_operator(f"o{n}", f"O{n}", "o", (make_variant("v", "v"),)) for n in range(5)
    )
    cat = Catalogue((Chapter("c", "C", operators),))
    index = make_index(
        operators={
            "c/o0": "ok",
            "c/o1": "ok",
            "c/o2": "refused",
            "c/o3": "missing",
            "c/o4": "diverge",
        },
        chapters={"c": "diverge"},
    )
    assert body(overview_block(cat, index))[1] == (
        'chapter__c["C<br/>5 operators · 🟢2 🛡️1 🔴1 🔵1"]:::diverge'
    )


# --- The diagram of a block ---


def test_block_diagram_layout() -> None:
    # The block in the middle, half of its operators to the left and half to the right. An
    # operator shows how many of its forms are in each status, and links to its page; a team
    # decision of one of its examples pins it. String length has not been run: no class and
    # no tally. No node has an emoji of its status: the class says it.
    assert block_diagram(STRING, INDEX) == block_of(
        "LR",
        'B["String operators"]:::block',
        'string__concat["Concatenation · #124;#124; 📌<br/>🟢1 🔵1"]:::diverge',
        'string__length["String length · length"]',
        "string__concat --- B",
        "B --- string__length",
        'click string__concat href "concat.html"',
        'click string__length href "length.html"',
    )


def test_block_diagram_both_sides_clicks() -> None:
    chapter = operators_of(9)
    variants: dict[str, Status] = {}
    for n in range(9):
        variants[f"numeric/o{n}/a"] = "ok"
        variants[f"numeric/o{n}/b"] = "partial"
    index = make_index(variants=variants, operators={f"numeric/o{n}": "partial" for n in range(9)})
    text = block_diagram(chapter, index)
    assert text.splitlines()[0] == "flowchart LR"
    assert "-->" not in text
    lines = body(text)
    ids = [node_id("numeric", operator.id) for operator in chapter.operators]
    # Half of the operators to the left of the block, the larger half when they are odd.
    assert [line for line in lines if line.endswith(" --- B")] == [f"{i} --- B" for i in ids[:5]]
    assert [line for line in lines if line.startswith("B --- ")] == [f"B --- {i}" for i in ids[5:]]
    for operator, node in zip(chapter.operators, ids):
        assert f'click {node} href "{operator.id}.html"' in lines
        # The label carries the tally of the forms of the operator.
        label = next(line for line in lines if line.startswith(f"{node}["))
        assert label.endswith('<br/>🟢1 🟡1"]:::partial')


def test_block_diagram_of_a_chapter_without_operators_is_the_block_alone() -> None:
    empty = Chapter("empty", "Empty", ())
    assert block_diagram(empty, NOTHING_RUN) == block_of("LR", 'B["Empty"]:::block')


def test_block_diagram_of_an_operator_without_forms_has_no_tally() -> None:
    bare = make_operator("bare", "Bare", "bare", ())
    chapter = Chapter("numeric", "Numeric operators", (bare,))
    index = make_index(operators={"numeric/bare": "ok"})
    assert body(block_diagram(chapter, index))[1] == 'numeric__bare["Bare · bare"]:::ok'


# --- The diagram of an operator: boxes ---


def test_operator_diagram_boxes() -> None:
    assert operator_diagram(NUMERIC, ROUND, ROUND_CAT, ROUND_INDEX) == block_of(
        "TB",
        'O["Rounding · round<br/>🟢1 🟡1 🛡️1"]:::partial',
        'subgraph G_ds["Dataset"]',
        "    direction LR",
        '    v_ds["round(DS_1, 1)"]:::ok',
        "end",
        'subgraph G_comp["Component"]',
        "    direction LR",
        '    v_comp["DS_1#91;calc Me_1 #58;#61; round(Me_1)#93;"]:::partial',
        "end",
        'subgraph G_sc["Scalar"]',
        "    direction LR",
        '    v_sc["round(1.5)"]:::refused',
        "end",
        "O --- G_ds",
        "O --- G_comp",
        "O --- G_sc",
        f"style G_ds {SOLID}",
        f"style G_comp {DASHED}",
        f"style G_sc {DOTTED}",
        'click v_ds href "#form-ds"',
        'click v_comp href "#form-comp"',
        'click v_sc href "#form-sc"',
    )


def test_operator_diagram_boxes_have_a_border_each_and_every_form_a_click() -> None:
    text = operator_diagram(NUMERIC, ROUND, ROUND_CAT, ROUND_INDEX)
    lines = body(text)
    assert text.splitlines()[0] == "flowchart TB"
    assert "-->" not in text
    for kind, title, style in (
        ("ds", "Dataset", SOLID),
        ("comp", "Component", DASHED),
        ("sc", "Scalar", DOTTED),
    ):
        assert f'subgraph G_{kind}["{title}"]' in lines
        assert f"O --- G_{kind}" in lines
        assert f"style G_{kind} {style}" in lines
    # Each form is a click to its own section of the page, by the anchor of its form.
    for form in ROUND.variants:
        assert f'click v_{form.id} href "#{form_anchor(form)}"' in lines
    # The fill is the same in the three boxes: the border tells them apart.
    assert len({style.split(",stroke-dasharray")[0] for style in BOX_STYLES.values()}) == 1


def test_a_kind_with_no_forms_has_no_box() -> None:
    chapter, operator, cat = lone(
        (make_variant("ds", "substr(DS_1, 1)"), make_variant("sc", 'substr("abc", 1)'))
    )
    text = operator_diagram(chapter, operator, cat, NOTHING_RUN)
    lines = body(text)
    assert text.splitlines()[0] == "flowchart TB"
    assert text.count("subgraph ") == 2
    assert 'subgraph G_ds["Dataset"]' in lines
    assert 'subgraph G_sc["Scalar"]' in lines
    assert "O --- G_ds" in lines
    assert "O --- G_sc" in lines
    assert "G_comp" not in text
    assert "Component" not in text


def test_an_operator_of_two_kinds_gets_boxes_whatever_the_kind_of_most_forms() -> None:
    # ``cast``: many forms on components, one on a dataset, one on a scalar. The ids of the
    # conversions have no token for their kind: the kind of the form says it.
    conversions = tuple(
        make_variant(f"{source}_to_{target}", f"cast(Me_1, {target})", kind="comp")
        for source, target in (("integer", "boolean"), ("string", "date"), ("number", "string"))
    )
    forms = (*conversions, make_variant("ds", "cast(DS_1, string)"), make_variant("sc", "cast(1)"))
    chapter, operator, cat = lone(forms)
    text = operator_diagram(chapter, operator, cat, NOTHING_RUN)
    lines = body(text)
    assert text.splitlines()[0] == "flowchart TB"
    assert text.count("subgraph ") == 3
    start = lines.index('subgraph G_comp["Component"]')
    assert lines[start + 2 : start + 6] == [
        'v_integer_to_boolean["cast(Me_1, boolean)"]',
        'v_string_to_date["cast(Me_1, date)"]',
        'v_number_to_string["cast(Me_1, string)"]',
        "end",
    ]


def test_a_form_that_says_its_kind_goes_to_that_box_whatever_its_id_says() -> None:
    chapter, operator, cat = lone(
        (make_variant("sc_comp_digits", "x", kind="ds"), make_variant("sc", "y"))
    )
    lines = body(operator_diagram(chapter, operator, cat, NOTHING_RUN))
    assert lines[lines.index('subgraph G_ds["Dataset"]') + 2] == 'v_sc_comp_digits["x"]'
    assert lines[lines.index('subgraph G_sc["Scalar"]') + 2] == 'v_sc["y"]'


# --- The diagram of an operator: boxes that the forms name ---


def boxes_of(diagram: str) -> dict[str, list[str]]:
    """The nodes in each subgraph of a diagram, by the id of the subgraph, in order of writing."""
    found: dict[str, list[str]] = {}
    current: str | None = None
    for line in body(diagram):
        opening = re.match(r'subgraph (\w+)\["', line)
        if opening is not None:
            current = opening.group(1)
            found[current] = []
        elif line == "end":
            current = None
        elif current is not None and line != "direction LR":
            node = re.match(r"(\w+)\[", line)
            assert node is not None, line
            found[current].append(node.group(1))
    return found


def test_operator_diagram_titled_boxes() -> None:
    # Two boxes, Signature and Body, two forms each, and what the operator is used with. The
    # forms are in the order of the catalogue inside each box, and the boxes in the order in which
    # their titles first appear: the form that opens the Body is the second of the catalogue.
    assert operator_diagram(DEFINE_CHAPTER, DEFINE, DEFINE_CAT, DEFINE_INDEX) == block_of(
        "TB",
        'O["Viral propagation · define viral propagation<br/>🟢2 🟡1 🔵1"]:::diverge',
        'subgraph G_b1["Signature"]',
        "    direction LR",
        '    v_variable["define x (variable v)"]:::ok',
        '    v_valuedomain["define x (valuedomain d)"]:::ok',
        "end",
        'subgraph G_b2["Body"]',
        "    direction LR",
        '    v_enumerated["when v = x then y"]:::partial',
        '    v_aggregation["aggr sum"]:::diverge',
        "end",
        'subgraph G_use["Used with"]',
        "    direction LR",
        '    used__clause__calc["calc ↗<br/><small>used in</small>"]:::ext',
        "end",
        "O --- G_b1",
        "O --- G_b2",
        "O -.- G_use",
        f"style G_b1 {LIGHT}",
        f"style G_b2 {LIGHT}",
        f"style G_use {USED_WITH}",
        'click v_variable href "#form-variable"',
        'click v_enumerated href "#form-enumerated"',
        'click v_valuedomain href "#form-valuedomain"',
        'click v_aggregation href "#form-aggregation"',
        'click used__clause__calc href "../clause/calc.html" "Calculation · calc"',
    )


def test_titled_boxes_are_in_the_order_of_first_appearance() -> None:
    forms = (
        make_variant("body_a", "a", box="Body"),
        make_variant("signature_a", "b", box="Signature"),
        make_variant("body_b", "c", box="Body"),
        make_variant("return_type", "d", box="Return type"),
        make_variant("signature_b", "e", box="Signature"),
    )
    chapter, operator, cat = lone(forms)
    text = operator_diagram(chapter, operator, cat, NOTHING_RUN)
    lines = body(text)
    assert text.splitlines()[0] == "flowchart TB"
    # One box for each title, numbered as the titles first appear: not by name, not by form.
    assert [line for line in lines if line.startswith("subgraph ")] == [
        'subgraph G_b1["Body"]',
        'subgraph G_b2["Signature"]',
        'subgraph G_b3["Return type"]',
    ]
    # Each box has its forms, in the order of the catalogue.
    assert list(boxes_of(text).items()) == [
        ("G_b1", ["v_body_a", "v_body_b"]),
        ("G_b2", ["v_signature_a", "v_signature_b"]),
        ("G_b3", ["v_return_type"]),
    ]
    assert [line for line in lines if line.startswith("O --- ")] == [
        "O --- G_b1",
        "O --- G_b2",
        "O --- G_b3",
    ]
    assert [line for line in lines if line.startswith("style ")] == [
        f"style G_b{n} {LIGHT}" for n in (1, 2, 3)
    ]
    # The page lists the forms in the order that the diagram draws them.
    assert [form.id for form in forms_in_diagram_order(operator)] == [
        "body_a",
        "body_b",
        "signature_a",
        "signature_b",
        "return_type",
    ]


def test_titles_that_differ_are_boxes_that_differ() -> None:
    forms = (
        make_variant("a", "a", box="Body"),
        make_variant("b", "b", box="body"),
        make_variant("c", "c", box="Body "),
        make_variant("d", "d", box="Body"),
    )
    chapter, operator, cat = lone(forms)
    text = operator_diagram(chapter, operator, cat, NOTHING_RUN)
    assert list(boxes_of(text).items()) == [
        ("G_b1", ["v_a", "v_d"]),
        ("G_b2", ["v_b"]),
        ("G_b3", ["v_c"]),
    ]


def test_kinds_are_ignored_when_the_forms_name_boxes() -> None:
    # The ids have the tokens of the three kinds, and one form says a kind of its own: none of it
    # counts. The box of the first form is the first box, whatever its kind would put first.
    forms = (
        make_variant("sc_x", "a", box="Signature"),
        make_variant("ds_x", "b", box="Body"),
        make_variant("comp_x", "c", box="Signature", kind="ds"),
        make_variant("free", "d", box="Body"),
    )
    chapter, operator, cat = lone(forms)
    text = operator_diagram(chapter, operator, cat, NOTHING_RUN)
    assert text.splitlines()[0] == "flowchart TB"
    for word in ("G_ds", "G_comp", "G_sc", "Dataset", "Component", "Scalar"):
        assert word not in text
    for style in BOX_STYLES.values():
        assert style not in text
    assert list(boxes_of(text).items()) == [
        ("G_b1", ["v_sc_x", "v_comp_x"]),
        ("G_b2", ["v_ds_x", "v_free"]),
    ]
    assert [form.id for form in forms_in_diagram_order(operator)] == [
        "sc_x",
        "comp_x",
        "ds_x",
        "free",
    ]


def test_boxes_that_the_forms_name_are_drawn_whatever_the_kinds() -> None:
    # Boxes by kind need two kinds (R6). Boxes that the forms name need none: forms of one kind,
    # of no kind, or a single box are drawn in boxes all the same.
    for forms in (
        (make_variant("comp_a", "a", box="Body"), make_variant("comp_b", "b", box="Body")),
        (make_variant("free", "a", box="Body"),),
        (make_variant("ds_a", "a", box="Signature"), make_variant("ds_b", "b", box="Body")),
    ):
        chapter, operator, cat = lone(forms)
        text = operator_diagram(chapter, operator, cat, NOTHING_RUN)
        assert has_boxes(operator)
        assert text.splitlines()[0] == "flowchart TB"
        assert "-->" not in text
        assert text.count("subgraph ") == len({form.box for form in forms})
        # The same forms that name no box have none: one kind is not two.
        plain = tuple(dataclasses.replace(form, box=None) for form in forms)
        assert not has_boxes(dataclasses.replace(operator, variants=plain))


def test_a_form_that_names_no_box_among_forms_that_do_hangs_from_the_operator() -> None:
    # The checks of the catalogue refuse this mixture, but a diagram never drops a form.
    forms = (
        make_variant("a", "a", box="Signature"),
        make_variant("other", "o"),
        make_variant("b", "b", box="Body"),
    )
    chapter, operator, cat = lone(forms)
    text = operator_diagram(chapter, operator, cat, NOTHING_RUN)
    lines = body(text)
    assert text.splitlines()[0] == "flowchart TB"
    assert list(boxes_of(text).items()) == [("G_b1", ["v_a"]), ("G_b2", ["v_b"])]
    assert lines.index('v_other["o"]') > lines.index("end")
    assert lines.index("O --- v_other") > lines.index("O --- G_b2")
    assert 'click v_other href "#form-other"' in lines
    # It is the last form of the page, as it is the last that the diagram draws.
    assert [form.id for form in forms_in_diagram_order(operator)] == ["a", "b", "other"]


def test_a_box_title_is_escaped_as_a_label_is() -> None:
    chapter, operator, cat = lone((make_variant("a", "a", box='Sig #1 "x" <b>; [c]'),))
    lines = body(operator_diagram(chapter, operator, cat, NOTHING_RUN))
    assert 'subgraph G_b1["Sig #35;1 #quot;x#quot; #60;b#62;#59; #91;c#93;"]' in lines


def test_titled_boxes_have_a_line_and_a_style_each_and_the_forms_have_the_clicks() -> None:
    forms = tuple(make_variant(f"f{n}", f"form {n}", box=f"Box {n % 3}") for n in range(7))
    chapter, operator, cat = lone(forms)
    text = operator_diagram(chapter, operator, cat, NOTHING_RUN)
    lines = body(text)
    boxes = boxes_of(text)
    assert list(boxes) == ["G_b1", "G_b2", "G_b3"]
    # The ids of the boxes are not those of the kinds, of "Used with", of the operator or of a form.
    assert not set(boxes) & {"G_ds", "G_comp", "G_sc", "G_use", "O"}
    assert not set(boxes) & {f"v_{form.id}" for form in forms}
    for box in boxes:
        assert f"O --- {box}" in lines
        assert f"style {box} {LIGHT}" in lines
        assert not any(line.startswith(f"click {box} ") for line in lines)
    # Every form is in exactly one box, and clicks to its section.
    members = [node for nodes in boxes.values() for node in nodes]
    assert sorted(members) == sorted(f"v_{form.id}" for form in forms)
    for form in forms:
        assert f'click v_{form.id} href "#{form_anchor(form)}"' in lines
    assert edges_of(text) == 3  # a line for each box, and no other


# --- The diagram of an operator: both sides ---


def test_operator_diagram_single_kind() -> None:
    # Join forms have no token for a kind: the diagram has both sides, half of the forms on each,
    # and no box. The same goes for an operator whose forms are all of one kind.
    joins = tuple(make_variant(f"form_{n}", f"inner_join(DS_{n})") for n in range(5))
    only_components = tuple(make_variant(f"comp_{n}", f"Me_{n}") for n in range(4))
    for forms, left in ((joins, 3), (only_components, 2)):
        chapter, operator, cat = lone(forms)
        text = operator_diagram(chapter, operator, cat, NOTHING_RUN)
        assert text.splitlines()[0] == "flowchart LR"
        for word in ("subgraph", "style ", "G_", "-->"):
            assert word not in text
        lines = body(text)
        ids = [f"v_{form.id}" for form in forms]
        assert [line for line in lines if line.endswith(" --- O")] == [
            f"{i} --- O" for i in ids[:left]
        ]
        assert [line for line in lines if line.startswith("O --- ")] == [
            f"O --- {i}" for i in ids[left:]
        ]
        for form in forms:
            assert f'click v_{form.id} href "#{form_anchor(form)}"' in lines


def test_operator_diagram_both_sides_layout() -> None:
    assert operator_diagram(CLAUSE, CALC, Catalogue((CLAUSE,)), INDEX) == block_of(
        "LR",
        'O["Calculation · calc<br/>🔴1"]:::missing',
        'v_measure["calc Me_1 #58;#61; 1"]:::missing',
        "v_measure --- O",
        'click v_measure href "#form-measure"',
    )


def test_a_form_with_no_kind_among_forms_of_two_kinds_hangs_from_the_operator() -> None:
    # The checks of the catalogue refuse this mixture, but a diagram never drops a form.
    chapter, operator, cat = lone(
        (make_variant("ds", "a"), make_variant("comp", "b"), make_variant("other", "c"))
    )
    text = operator_diagram(chapter, operator, cat, NOTHING_RUN)
    lines = body(text)
    assert text.splitlines()[0] == "flowchart TB"
    assert text.count("subgraph ") == 2
    assert 'v_other["c"]' in lines
    assert "O --- v_other" in lines
    assert 'click v_other href "#form-other"' in lines


def test_a_form_with_no_kind_among_forms_of_one_kind_is_drawn_on_the_sides() -> None:
    chapter, operator, cat = lone(
        (make_variant("comp", "a"), make_variant("comp_2", "b"), make_variant("other", "c"))
    )
    text = operator_diagram(chapter, operator, cat, NOTHING_RUN)
    lines = body(text)
    assert text.splitlines()[0] == "flowchart LR"
    assert "subgraph" not in text
    assert [line for line in lines if " --- " in line] == [
        "v_comp --- O",
        "v_comp_2 --- O",
        "O --- v_other",
    ]


def test_an_operator_without_forms_is_the_operator_alone() -> None:
    bare = make_operator("bare", "Bare", "", ())
    chapter = Chapter("definitions", "Definitions", (bare,))
    text = operator_diagram(chapter, bare, Catalogue((chapter,)), NOTHING_RUN)
    assert text == block_of("LR", 'O["Bare"]')


# --- "Used with" ---


def test_used_with_both_ends() -> None:
    # Concatenation says that it is used in a calc. The diagram of the calc has the concatenation,
    # with the reverse of "used in", and the diagram of the concatenation has the calc, with the
    # label that it declared. Each is a link to the page of the other operator, with its name and
    # its symbol as the tooltip: the symbol of the concatenation is not a word, so its entry says
    # its name.
    calc = body(operator_diagram(CLAUSE, CALC, CAT, INDEX))
    assert 'subgraph G_use["Used with"]' in calc
    assert 'used__string__concat["Concatenation ↗<br/><small>uses</small>"]:::ext' in calc
    assert (
        'click used__string__concat href "../string/concat.html" "Concatenation · #124;#124;"'
        in calc
    )
    concat = body(operator_diagram(STRING, CONCAT, CAT, INDEX))
    assert 'subgraph G_use["Used with"]' in concat
    assert 'used__clause__calc["calc ↗<br/><small>used in</small>"]:::ext' in concat
    assert 'click used__clause__calc href "../clause/calc.html" "Calculation · calc"' in concat


def test_used_with_layout_with_boxes() -> None:
    assert operator_diagram(STRING, CONCAT, CAT, INDEX) == block_of(
        "TB",
        'O["Concatenation · #124;#124; 📌<br/>🟢1 🔵1"]:::diverge',
        'subgraph G_ds["Dataset"]',
        "    direction LR",
        '    v_ds_ds["DS_1 #124;#124; DS_2"]:::ok',
        "end",
        'subgraph G_comp["Component"]',
        "    direction LR",
        '    v_comp_sc["Me_1 #124;#124; #quot;x#quot; 📌"]:::diverge',
        "end",
        'subgraph G_use["Used with"]',
        "    direction LR",
        '    used__clause__calc["calc ↗<br/><small>used in</small>"]:::ext',
        "end",
        "O --- G_ds",
        "O --- G_comp",
        "O -.- G_use",
        f"style G_ds {SOLID}",
        f"style G_comp {DASHED}",
        f"style G_use {USED_WITH}",
        'click v_ds_ds href "#form-ds-ds"',
        'click v_comp_sc href "#form-comp-sc"',
        'click used__clause__calc href "../clause/calc.html" "Calculation · calc"',
    )


def test_used_with_layout_on_both_sides() -> None:
    assert operator_diagram(CLAUSE, CALC, CAT, INDEX) == block_of(
        "LR",
        'O["Calculation · calc<br/>🔴1"]:::missing',
        'v_measure["calc Me_1 #58;#61; 1"]:::missing',
        'subgraph G_use["Used with"]',
        "    direction LR",
        '    used__string__concat["Concatenation ↗<br/><small>uses</small>"]:::ext',
        "end",
        "v_measure --- O",
        "O -.- G_use",
        f"style G_use {USED_WITH}",
        'click v_measure href "#form-measure"',
        'click used__string__concat href "../string/concat.html" "Concatenation · #124;#124;"',
    )


def linked(label: str) -> Catalogue:
    """The operator ``alpha.a`` links to ``beta.b`` with ``label``."""
    source = make_operator(
        "a", "Alpha", "a", (make_variant("x", "a(1)"),), (Link("beta.b", label),)
    )
    target = make_operator("b", "Beta", "b", (make_variant("y", "b(1)"),))
    return Catalogue(
        (
            Chapter("alpha", "Alpha operators", (source,)),
            Chapter("beta", "Beta operators", (target,)),
        )
    )


def diagram_of(cat: Catalogue, chapter_id: str, operator_id: str) -> list[str]:
    """The lines of the diagram of an operator of ``cat``, with nothing run."""
    chapter = next(chapter for chapter in cat.chapters if chapter.id == chapter_id)
    operator = next(operator for operator in chapter.operators if operator.id == operator_id)
    return body(operator_diagram(chapter, operator, cat, NOTHING_RUN))


@pytest.mark.parametrize(
    ("declared", "reverse"),
    [
        ("uses", "used by"),
        ("used by", "uses"),
        ("used in", "uses"),
        ("used in over()", "uses, in over()"),
        ("used in group all", "uses, in group all"),
        ("propagates at", "propagation defined by"),
    ],
)
def test_used_with_reverse_labels(declared: str, reverse: str) -> None:
    cat = linked(declared)
    source = diagram_of(cat, "alpha", "a")
    target = diagram_of(cat, "beta", "b")
    assert f'used__beta__b["b ↗<br/><small>{declared}</small>"]:::ext' in source
    assert f'used__alpha__a["a ↗<br/><small>{reverse}</small>"]:::ext' in target
    assert 'click used__beta__b href "../beta/b.html" "Beta · b"' in source
    assert 'click used__alpha__a href "../alpha/a.html" "Alpha · a"' in target


def test_used_with_a_link_declared_at_both_ends_is_shown_once() -> None:
    # The definitions say "used by" where the operators say "uses": at each end, the box has the
    # other operator once, with one label.
    back = make_operator(
        "b", "Beta", "b", (make_variant("y", "b(1)"),), (Link("alpha.a", "used by"),)
    )
    cat = Catalogue((linked("uses").chapters[0], Chapter("beta", "Beta operators", (back,))))
    for chapter_id, operator_id, other, label in (
        ("alpha", "a", "beta__b", "uses"),
        ("beta", "b", "alpha__a", "used by"),
    ):
        entries = [
            line for line in diagram_of(cat, chapter_id, operator_id) if line.startswith("used__")
        ]
        assert len(entries) == 1
        assert entries[0].startswith(f"used__{other}[")
        assert f"<small>{label}</small>" in entries[0]


def test_used_with_two_labels_for_one_operator_are_one_entry() -> None:
    source = make_operator(
        "a",
        "Alpha",
        "a",
        (make_variant("x", "a(1)"),),
        (Link("beta.b", "uses"), Link("beta.b", "used in over()")),
    )
    target = make_operator("b", "Beta", "b", (make_variant("y", "b(1)"),))
    cat = Catalogue(
        (
            Chapter("alpha", "Alpha operators", (source,)),
            Chapter("beta", "Beta operators", (target,)),
        )
    )
    entries = [line for line in diagram_of(cat, "alpha", "a") if line.startswith("used__")]
    assert entries == ['used__beta__b["b ↗<br/><small>uses · used in over()</small>"]:::ext']


def entry_of(symbol: str, name: str = "Target name") -> tuple[str, str]:
    """The node and the click of an operator in the "Used with" box of another one.

    The operator ``beta.b`` has the given ``name`` and ``symbol``, and ``alpha.a`` links to it.
    """
    source = make_operator(
        "a", "Alpha", "a", (make_variant("x", "a(1)"),), (Link("beta.b", "uses"),)
    )
    target = make_operator("b", name, symbol, (make_variant("y", "b(1)"),))
    cat = Catalogue(
        (
            Chapter("alpha", "Alpha operators", (source,)),
            Chapter("beta", "Beta operators", (target,)),
        )
    )
    lines = diagram_of(cat, "alpha", "a")
    node = next(line for line in lines if line.startswith("used__beta__b["))
    click = next(line for line in lines if line.startswith("click used__beta__b "))
    return node, click


@pytest.mark.parametrize("symbol", ["calc", "inner_join", "check_datapoint", "A1", "x_2"])
def test_used_with_entry_is_the_symbol_when_it_is_a_word(symbol: str) -> None:
    # Short, as in the approved mock-up: ``aggr ↗`` and the label in small letters. The tooltip
    # of the click has the name and the symbol.
    node, click = entry_of(symbol)
    assert node == f'used__beta__b["{symbol} ↗<br/><small>uses</small>"]:::ext'
    assert click == f'click used__beta__b href "../beta/b.html" "Target name · {symbol}"'


@pytest.mark.parametrize(
    ("symbol", "tooltip"),
    [
        ("+", "Target name · +"),
        ("||", "Target name · #124;#124;"),
        ("<>", "Target name · #60;#62;"),
        ("group by", "Target name · group by"),
        ("name( )", "Target name · name( )"),
        ("a-b", "Target name · a-b"),
        ("", "Target name"),  # no symbol: the tooltip is the name alone
    ],
)
def test_used_with_entry_is_the_name_when_the_symbol_is_not_a_word(
    symbol: str, tooltip: str
) -> None:
    # A word is letters, digits and underscores. ``+``, ``group by`` and the missing symbol of a
    # definition do not say what the operator is: its name does.
    node, click = entry_of(symbol)
    assert node == 'used__beta__b["Target name ↗<br/><small>uses</small>"]:::ext'
    assert click == f'click used__beta__b href "../beta/b.html" "{tooltip}"'


def test_used_with_tooltip_is_escaped_as_a_label_is() -> None:
    # A quote would end the string of the tooltip: it is an entity code, like the rest.
    node, click = entry_of("a|b", name='Say "hi" [x] {y}; #1')
    said = "Say #quot;hi#quot; #91;x#93; #123;y#125;#59; #35;1"
    assert node == f'used__beta__b["{said} ↗<br/><small>uses</small>"]:::ext'
    assert click == f'click used__beta__b href "../beta/b.html" "{said} · a#124;b"'


def test_used_with_declared_entries_come_before_the_reverse_ones() -> None:
    def operator(operator_id: str, links: tuple[Link, ...] = ()) -> Operator:
        return make_operator(operator_id, operator_id, "", (make_variant("y", "y"),), links)

    hub = operator("hub", (Link("beta.b2", "uses"), Link("beta.b1", "uses")))
    users = (
        operator("u2", (Link("alpha.hub", "uses"),)),
        operator("u1", (Link("alpha.hub", "uses"),)),
    )
    beta = Chapter("beta", "Beta operators", (operator("b1"), operator("b2")))
    cat = Catalogue((Chapter("alpha", "Alpha operators", (hub, *users)), beta))
    entries = [line.split("[")[0] for line in diagram_of(cat, "alpha", "hub")]
    assert [entry for entry in entries if entry.startswith("used__")] == [
        "used__beta__b2",  # declared, in the order of its links
        "used__beta__b1",
        "used__alpha__u2",  # then those that link here, in the order of the catalogue
        "used__alpha__u1",
    ]


def hub_of(declared: int, users: int) -> Catalogue:
    """A hub that links to ``declared`` operators of ``alpha``, and that ``users`` of ``beta``."""
    alpha = tuple(
        make_operator(f"a{n}", f"A{n}", f"a{n}", (make_variant("y", "y"),)) for n in range(declared)
    )
    beta = tuple(
        make_operator(
            f"b{n}", f"B{n}", f"b{n}", (make_variant("y", "y"),), (Link("hub.hub", "used in"),)
        )
        for n in range(users)
    )
    links = tuple(Link(f"alpha.a{n}", "uses") for n in range(declared))
    hub = make_operator("hub", "Hub", "hub", (make_variant("x", "hub(1)"),), links)
    return Catalogue(
        (
            Chapter("alpha", "Alpha # operators", alpha),
            Chapter("hub", "Hub operators", (hub,)),
            Chapter("beta", "Beta operators", beta),
        )
    )


def test_used_with_eight_entries_are_one_node_each() -> None:
    # Five links that it declares and three that point to it: eight entries, a node for each.
    lines = diagram_of(hub_of(5, 3), "hub", "hub")
    nodes = [line for line in lines if line.startswith("used__")]
    assert len(nodes) == USED_WITH_MAX == 8
    assert all("↗<br/><small>" in node for node in nodes)
    assert [line for line in lines if line.startswith("click used__")] == [
        *(f'click used__alpha__a{n} href "../alpha/a{n}.html" "A{n} · a{n}"' for n in range(5)),
        *(f'click used__beta__b{n} href "../beta/b{n}.html" "B{n} · b{n}"' for n in range(3)),
    ]


def test_used_with_grouped_over_8() -> None:
    # Six links that it declares and three that point to it: nine entries become one node for
    # each block, with the title of the block and how many entries it has, linking to the page of
    # the block, in the order of the catalogue.
    lines = diagram_of(hub_of(6, 3), "hub", "hub")
    assert [line for line in lines if line.startswith("used__")] == [
        'used__alpha["Alpha #35; operators (6)"]:::ext',
        'used__beta["Beta operators (3)"]:::ext',
    ]
    assert [line for line in lines if line.startswith("click used__")] == [
        'click used__alpha href "../alpha/index.html"',
        'click used__beta href "../beta/index.html"',
    ]
    assert "O -.- G_use" in lines
    assert f"style G_use {USED_WITH}" in lines
    assert not any("↗" in line for line in lines)


def test_used_with_grouped_counts_the_entries_that_point_to_the_operator_too() -> None:
    assert [
        line for line in diagram_of(hub_of(0, 9), "hub", "hub") if line.startswith("used__")
    ] == ['used__beta["Beta operators (9)"]:::ext']


def test_used_with_lists_the_operators_of_the_box_with_their_labels() -> None:
    # The same operators as the box, in the same order, whether the box shows them one by one or
    # by block: a page that lists them in full agrees with the diagram.
    def listed(cat: Catalogue, chapter_id: str, operator_id: str) -> list[tuple[str, ...]]:
        chapter = next(chapter for chapter in cat.chapters if chapter.id == chapter_id)
        operator = next(operator for operator in chapter.operators if operator.id == operator_id)
        return [
            (entry.chapter.id, entry.operator.id, *entry.labels)
            for entry in used_with(chapter, operator, cat)
        ]

    assert listed(CAT, "string", "concat") == [("clause", "calc", "used in")]
    assert listed(CAT, "clause", "calc") == [("string", "concat", "uses")]
    assert listed(CAT, "string", "length") == []
    assert listed(hub_of(1, 1), "hub", "hub") == [
        ("alpha", "a0", "uses"),
        ("beta", "b0", "uses"),
    ]
    nine = listed(hub_of(6, 3), "hub", "hub")
    assert len(nine) == 9
    assert [entry[0] for entry in nine] == ["alpha"] * 6 + ["beta"] * 3


def crowd(forms: int, entries: int, blocks: int = 1) -> str:
    """The diagram of an operator with ``forms`` forms that have no kind, used with ``entries``.

    It links to ``entries`` operators, one link each, spread over ``blocks`` other blocks.
    """
    variants = tuple(make_variant(f"form_{n}", f"form {n}") for n in range(forms))
    links = tuple(Link(f"other{n % blocks}.t{n}", "uses") for n in range(entries))
    hub = make_operator("hub", "Hub", "hub", variants, links)
    others = tuple(
        Chapter(
            f"other{block}",
            f"Other {block}",
            tuple(
                make_operator(f"t{n}", f"T{n}", f"t{n}", (make_variant("v", "v"),))
                for n in range(entries)
                if n % blocks == block
            ),
        )
        for block in range(blocks)
    )
    chapter = Chapter("numeric", "Numeric operators", (hub,))
    return operator_diagram(chapter, hub, Catalogue((chapter, *others)), NOTHING_RUN)


@pytest.mark.parametrize(
    ("forms", "entries", "blocks", "left"),
    [
        # ``join.inner_join``: 22 forms and 7 entries. The box counts as 8 forms, 7 and its
        # title, so 30 are to be shared: 15 on the left, and 7 and the box on the right.
        (22, 7, 1, 15),
        # A box shorter than the forms: 3 slots, so 9 are shared and the left side takes 5.
        (6, 2, 1, 5),
        (7, 2, 1, 5),
        # A box longer than the forms: they all go to the left, and the box is alone on the right.
        (2, 5, 1, 2),
        (1, 7, 1, 1),
        (0, 3, 1, 0),
        # More than 8 entries: the box has a node for each block, 2 and its title are 3 slots.
        (6, 9, 2, 5),
        (4, 9, 2, 4),
        # No box: half and half, with the extra form on the left.
        (5, 0, 1, 3),
        (4, 0, 1, 2),
        (1, 0, 1, 1),
    ],
)
def test_both_sides_with_a_used_with_box_balance_the_columns(
    forms: int, entries: int, blocks: int, left: int
) -> None:
    text = crowd(forms, entries, blocks)
    lines = body(text)
    assert text.splitlines()[0] == "flowchart LR"
    ids = [f"v_form_{n}" for n in range(forms)]
    # The first forms are to the left, in order, and the others to the right.
    assert [line.split()[0] for line in lines if line.endswith(" --- O")] == ids[:left]
    assert [line.split()[-1] for line in lines if line.startswith("O --- ")] == ids[left:]
    if entries == 0:
        assert "G_use" not in text
        return
    # The box is under the forms on the right: it comes after all of them, and is joined last.
    box = lines.index('subgraph G_use["Used with"]')
    assert all(lines.index(f'{each}["form {n}"]') < box for n, each in enumerate(ids))
    edges = [line for line in lines if " --- " in line or " -.- " in line]
    assert edges[-1] == "O -.- G_use"


def test_both_sides_with_a_used_with_box_layout() -> None:
    # Five forms and one entry: 2 slots, so 7 are shared and the left side takes 4 forms. The
    # fifth is on the right, over the box.
    assert crowd(5, 1) == block_of(
        "LR",
        'O["Hub · hub"]',
        'v_form_0["form 0"]',
        'v_form_1["form 1"]',
        'v_form_2["form 2"]',
        'v_form_3["form 3"]',
        'v_form_4["form 4"]',
        'subgraph G_use["Used with"]',
        "    direction LR",
        '    used__other0__t0["t0 ↗<br/><small>uses</small>"]:::ext',
        "end",
        "v_form_0 --- O",
        "v_form_1 --- O",
        "v_form_2 --- O",
        "v_form_3 --- O",
        "O --- v_form_4",
        "O -.- G_use",
        f"style G_use {USED_WITH}",
        'click v_form_0 href "#form-form-0"',
        'click v_form_1 href "#form-form-1"',
        'click v_form_2 href "#form-form-2"',
        'click v_form_3 href "#form-form-3"',
        'click v_form_4 href "#form-form-4"',
        'click used__other0__t0 href "../other0/t0.html" "T0 · t0"',
    )


def test_the_boxes_layout_is_not_split_by_the_used_with_box() -> None:
    # The balance is for both sides. Under the boxes, the box of "Used with" is one more box.
    text = operator_diagram(STRING, CONCAT, CAT, INDEX)
    assert text.splitlines()[0] == "flowchart TB"
    assert "O -.- G_use" in body(text)
    assert not [line for line in body(text) if line.endswith(" --- O")]


def test_used_with_none_has_no_box() -> None:
    text = operator_diagram(NUMERIC, ROUND, ROUND_CAT, ROUND_INDEX)
    for word in ("G_use", "Used with", "-.-"):
        assert word not in text


def test_a_link_to_an_unknown_target_raises() -> None:
    lost = make_operator("lost", "Lost", "lost", (), (Link("clause.nope", "used in"),))
    chapter = Chapter("numeric", "Numeric operators", (lost,))
    with pytest.raises(ValueError, match="numeric.lost: target 'clause.nope' not found"):
        operator_diagram(chapter, lost, Catalogue((chapter, CLAUSE)), NOTHING_RUN)


def test_a_label_with_no_reverse_raises_at_the_target_end() -> None:
    cat = linked("sees")
    # The end that declares the label shows it as it is; the other end needs its reverse.
    assert 'used__beta__b["b ↗<br/><small>sees</small>"]:::ext' in diagram_of(cat, "alpha", "a")
    with pytest.raises(ValueError, match="alpha.a: link label 'sees' has no reverse label"):
        diagram_of(cat, "beta", "b")


# --- All the diagrams ---


def defined_ids(diagram: str) -> list[str]:
    """The ids that a diagram defines, as nodes and as subgraphs, in order."""
    found = (re.match(r'(?:subgraph )?(\w+)\["', line) for line in body(diagram))
    return [match.group(1) for match in found if match is not None]


def test_every_diagram_is_well_formed() -> None:
    mixed = lone((make_variant("ds", "a"), make_variant("comp", "b"), make_variant("other", "c")))
    # Titled boxes, one of them with a title to escape, and a form that names no box.
    hanging = lone(
        (
            make_variant("a", "a", box='Sig "x"'),
            make_variant("b", "b", box="Body"),
            make_variant("other", "c"),
        )
    )
    diagrams = [
        overview_block(CAT, INDEX),
        block_diagram(STRING, INDEX),
        operator_diagram(NUMERIC, ROUND, ROUND_CAT, ROUND_INDEX),
        operator_diagram(STRING, CONCAT, CAT, INDEX),
        operator_diagram(CLAUSE, CALC, CAT, INDEX),
        operator_diagram(*mixed[:2], mixed[2], NOTHING_RUN),
        operator_diagram(DEFINE_CHAPTER, DEFINE, DEFINE_CAT, DEFINE_INDEX),
        operator_diagram(*hanging[:2], hanging[2], NOTHING_RUN),
    ]
    for declared in (5, 6):  # eight entries of "Used with" are nodes, nine are blocks
        hub = hub_of(declared, 3)
        diagrams.append(operator_diagram(hub.chapters[1], hub.chapters[1].operators[0], hub, INDEX))
    for diagram in diagrams:
        ids = defined_ids(diagram)
        assert len(ids) == len(set(ids)), "an id is defined twice"
        for line in body(diagram):
            if line.startswith(("click ", "style ")):
                assert line.split()[1] in ids, line
            edge = re.fullmatch(r"(\w+) (?:---|-\.-) (\w+)", line)
            if edge is not None:
                assert edge.group(1) in ids, line
                assert edge.group(2) in ids, line


# --- The statuses ---


def test_refused_class() -> None:
    # Refused is the green of "all ok" with a 3 px dotted border, and the shield.
    assert (
        "classDef refused fill:#d4edda,stroke:#28a745,stroke-width:3px,"
        "stroke-dasharray: 2 2,color:#000"
    ) in CLASSES
    assert EMOJI["refused"] == "🛡️"
    chapter = Chapter("c", "C", (make_operator("o", "O", "o", (make_variant("v", "v()"),)),))
    cat = Catalogue((chapter,))
    index = make_index(
        variants={"c/o/v": "refused"}, operators={"c/o": "refused"}, chapters={"c": "refused"}
    )
    operator = body(operator_diagram(chapter, chapter.operators[0], cat, index))
    # The class says it, and the shield is in the tallies: no node has it.
    assert 'O["O · o<br/>🛡️1"]:::refused' in operator
    assert 'v_v["v()"]:::refused' in operator
    assert 'c__o["O · o<br/>🛡️1"]:::refused' in body(block_diagram(chapter, index))
    assert 'chapter__c["C<br/>1 operator · 🛡️1"]:::refused' in body(overview_block(cat, index))


def test_all_the_statuses_are_in_the_tally_in_the_order_of_the_legend() -> None:
    statuses: tuple[Status, ...] = ("ok", "partial", "refused", "missing", "diverge")
    forms = tuple(make_variant(f"f{n}", f"f{n}") for n in range(5))
    chapter, operator, cat = lone(forms)
    index = make_index(
        variants={f"numeric/op/f{n}": status for n, status in enumerate(statuses)},
        operators={"numeric/op": "diverge"},
    )
    node = body(operator_diagram(chapter, operator, cat, index))[0]
    assert node == 'O["Op · op<br/>🟢1 🟡1 🛡️1 🔴1 🔵1"]:::diverge'


def test_a_node_has_no_emoji_of_its_status_but_the_tallies_do() -> None:
    # The class says the status of a node: a refused node is the green of ok with a dotted
    # border, with no shield. The emoji are in the tallies, which count them. A team decision is
    # still pinned.
    diagrams = [
        overview_block(CAT, INDEX),
        block_diagram(STRING, INDEX),
        operator_diagram(NUMERIC, ROUND, ROUND_CAT, ROUND_INDEX),
        operator_diagram(STRING, CONCAT, CAT, INDEX),
        operator_diagram(CLAUSE, CALC, CAT, INDEX),
    ]
    emojis = tuple(EMOJI.values())
    tallies = 0
    pins = 0
    for diagram in diagrams:
        for line in body(diagram):
            found = re.fullmatch(r'\w+\["([^"]*)"\](?::{3}\w+)?', line)
            if found is None:
                continue
            first, _, second = found.group(1).partition("<br/>")
            assert not any(emoji in first for emoji in emojis), line
            tallies += any(emoji in second for emoji in emojis)
            pins += "📌" in first
    # Two blocks, an operator of the block diagram, and the three operators with a diagram.
    assert tallies == 6
    # The concatenation and its second form, in the block diagram and its own diagram.
    assert pins == 3


def test_missing_statuses_render_neutral() -> None:
    diagrams = [
        overview_block(CAT, NOTHING_RUN),
        block_diagram(STRING, NOTHING_RUN),
        operator_diagram(STRING, CONCAT, CAT, NOTHING_RUN),
        operator_diagram(CLAUSE, CALC, CAT, NOTHING_RUN),
    ]
    for diagram in diagrams:
        text = "\n".join(diagram.splitlines()[1 + len(CLASS_LINES) :])
        for status, emoji in EMOJI.items():
            assert f":::{status}" not in text
            assert emoji not in text
    assert body(overview_block(CAT, NOTHING_RUN))[1] == (
        'chapter__string["String operators<br/>2 operators"]'
    )
    assert body(block_diagram(STRING, NOTHING_RUN))[1] == (
        'string__concat["Concatenation · #124;#124; 📌"]'
    )


# --- Pages and escaping ---


def test_the_pages_of_the_diagrams_are_html_pages_of_the_site() -> None:
    # The map is a part of the Sphinx docs: a page is ``<name>.html``, never a file of Markdown.
    # The overview goes to the page of a block, a block to the page of an operator, and an
    # operator to the section of a form or to the page of what it is used with: all relative.
    assert PAGE_SUFFIX == ".html"
    diagrams = [
        overview_block(CAT, INDEX),
        block_diagram(STRING, INDEX),
        operator_diagram(STRING, CONCAT, CAT, INDEX),
        operator_diagram(CLAUSE, CALC, CAT, INDEX),
    ]
    for diagram in diagrams:
        assert ".md" not in diagram
    clicks = [line for diagram in diagrams for line in body(diagram) if line.startswith("click ")]
    found = [re.fullmatch(r'click \S+ href "([^"]+)"(?: "[^"]*")?', line) for line in clicks]
    assert sorted(match.group(1) for match in found if match is not None) == [
        "#form-comp-sc",
        "#form-ds-ds",
        "#form-measure",
        "../clause/calc.html",
        "../string/concat.html",
        "clause/index.html",
        "concat.html",
        "length.html",
        "string/index.html",
    ]
    assert len(found) == len(clicks) == 9


def test_the_titles_names_labels_and_forms_are_escaped() -> None:
    odd = make_operator(
        "odd",
        'Odd "x" [y]',
        "a|b",
        (make_variant("v", "f(#1) | {2}; <3>"),),
        (Link("clause.c", "a|b"),),
    )
    target = make_operator("c", "Calc {z}", "c<>", (make_variant("w", "w"),))
    chapter = Chapter("numeric", "Numeric # operators", (odd,))
    cat = Catalogue((chapter, Chapter("clause", "Clause # operators", (target,))))
    operator = diagram_of(cat, "numeric", "odd")
    assert 'O["Odd #quot;x#quot; #91;y#93; · a#124;b"]' in operator
    assert 'v_v["f(#35;1) #124; #123;2#125;#59; #60;3#62;"]' in operator
    assert 'used__clause__c["Calc #123;z#125; ↗<br/><small>a#124;b</small>"]:::ext' in operator
    assert (
        'click used__clause__c href "../clause/c.html" "Calc #123;z#125; · c#60;#62;"' in operator
    )
    assert 'B["Numeric #35; operators"]:::block' in body(block_diagram(chapter, NOTHING_RUN))
    assert 'chapter__clause["Clause #35; operators<br/>1 operator"]' in body(
        overview_block(cat, NOTHING_RUN)
    )


# --- Limits ---


def big_operator(count: int, *, kinds: bool) -> Operator:
    """An operator of ``count`` forms with a label as long as the longest one of the map.

    ``kinds`` gives them a kind each in turn, so that they are in the boxes; without it they have
    none, and are on both sides. It links to as many operators as "Used with" shows.
    """
    names = ("ds", "comp", "sc")
    forms = tuple(
        make_variant(f"{names[n % 3]}_{n}" if kinds else f"form_{n}", f"{'x' * 83}{n}")
        for n in range(count)
    )
    links = tuple(Link(f"other.t{n}", "used in over()") for n in range(USED_WITH_MAX))
    return make_operator("big", "Big", "big", forms, links)


def targets_of(count: int) -> Chapter:
    operators = tuple(
        make_operator(f"t{n}", f"Target {n}", f"t{n}", (make_variant("v", "t"),))
        for n in range(count)
    )
    return Chapter("other", "Other operators", operators)


@pytest.mark.parametrize("kinds", [True, False])
def test_limits_on_largest(kinds: bool) -> None:
    # Sixty forms, more than the largest operator has, and the most that "Used with" shows.
    big = big_operator(60, kinds=kinds)
    chapter = Chapter("numeric", "Numeric operators", (big,))
    cat = Catalogue((chapter, targets_of(USED_WITH_MAX)))
    text = operator_diagram(chapter, big, cat, NOTHING_RUN)
    assert text.count("subgraph G_ds") == (1 if kinds else 0)
    assert len([line for line in body(text) if line.startswith("used__other__")]) == USED_WITH_MAX
    assert edges_of(text) < MAX_EDGES
    assert size_of(text) < MAX_CHARS
    check_limits(text)


def test_an_operator_over_a_limit_raises() -> None:
    forms = tuple(make_variant(f"v{n}", f"form {n}") for n in range(MAX_EDGES + 1))
    chapter, operator, cat = lone(forms)
    with pytest.raises(MermaidLimitError, match=r"numeric\.op: 501 edges"):
        operator_diagram(chapter, operator, cat, NOTHING_RUN)


def test_a_block_over_a_limit_raises() -> None:
    with pytest.raises(MermaidLimitError, match=r"numeric: 501 edges"):
        block_diagram(operators_of(MAX_EDGES + 1), NOTHING_RUN)


def test_an_overview_over_a_limit_raises() -> None:
    with pytest.raises(MermaidLimitError, match=r"overview: 500 edges"):
        overview_block(Catalogue(chapters_of(MAX_EDGES)), NOTHING_RUN)


def test_the_diagrams_of_the_size_of_the_real_map_are_under_the_limits() -> None:
    # 14 blocks, and 18 operators in the largest block.
    check_limits(overview_block(Catalogue(chapters_of(14)), NOTHING_RUN))
    check_limits(block_diagram(operators_of(18), NOTHING_RUN))


# --- The order of the forms, for the pages ---


def test_has_boxes_needs_forms_of_two_kinds() -> None:
    assert has_boxes(ROUND)  # a dataset, a component and a scalar form
    assert has_boxes(CONCAT)  # a dataset and a component form
    assert not has_boxes(CALC)  # a form with no kind
    # Forms of one kind, and one with no kind, are not two kinds.
    one_kind = make_operator(
        "o", "O", "o", (make_variant("comp", "a"), make_variant("comp_sc", "b"))
    )
    assert not has_boxes(one_kind)
    mixed = make_operator("o", "O", "o", (make_variant("comp", "a"), make_variant("free", "b")))
    assert not has_boxes(mixed)


def test_forms_in_diagram_order_are_the_boxes_in_order_then_the_forms_with_no_kind() -> None:
    modulo = make_operator(
        "modulo",
        "Modulo",
        "mod",
        (
            make_variant("sc_sc", "mod(1, 2)"),
            make_variant("ds_ds", "mod(DS_1, DS_2)"),
            make_variant("comp_sc", "mod(Me_1, 2)"),
            make_variant("free", "mod()"),  # no kind: it hangs from the operator, after the boxes
            make_variant("ds_sc", "mod(DS_1, 2)"),
        ),
    )
    order = ["ds_ds", "ds_sc", "comp_sc", "sc_sc", "free"]
    assert [variant.id for variant in forms_in_diagram_order(modulo)] == order
    # The diagram draws its forms in that order.
    numeric = Chapter("numeric", "Numeric operators", (modulo,))
    diagram = operator_diagram(numeric, modulo, Catalogue((numeric,)), NOTHING_RUN)
    places = [diagram.index(f"v_{variant_id}[") for variant_id in order]
    assert places == sorted(places)
    # Without boxes, the forms are drawn, and come, in the order of the catalogue.
    both_sides = make_operator(
        "o", "O", "o", (make_variant("b", "b"), make_variant("a", "a"), make_variant("c", "c"))
    )
    assert forms_in_diagram_order(both_sides) == list(both_sides.variants)


def test_has_boxes_when_the_forms_name_them_and_forms_in_diagram_order_follow_the_boxes() -> None:
    assert has_boxes(DEFINE)
    assert [form.id for form in forms_in_diagram_order(DEFINE)] == [
        "variable",
        "valuedomain",
        "enumerated",
        "aggregation",
    ]
    # One form that names a box is enough, and the kinds say nothing then: this is a form of
    # each kind, in the boxes it names, the other way round to the order of the kinds.
    forms = (
        make_variant("sc", "a", box="Body"),
        make_variant("comp", "b"),
        make_variant("ds", "c", box="Signature"),
    )
    operator = make_operator("o", "O", "o", forms)
    assert has_boxes(operator)
    assert [form.id for form in forms_in_diagram_order(operator)] == ["sc", "ds", "comp"]
    # The forms of an operator with boxes that the kinds draw are as before.
    assert has_boxes(ROUND)
    assert [form.id for form in forms_in_diagram_order(ROUND)] == ["ds", "comp", "sc"]
