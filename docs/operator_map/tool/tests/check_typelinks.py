# ruff: noqa: S101
"""Checks for the type links: what feeds an operator, what it feeds, and the types of a block.

The catalogues and the statuses are small hand-made values: no engine runs here, and the real
catalogue is not read, as its forms are typed chapter by chapter. The expected lists are worked
out by hand from the matching rules of ``tool.typelinks``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable

import pytest

from tool.model import Catalogue, Chapter, Operator, Status, Variant, authored
from tool.status import StatusIndex
from tool.typelinks import Ref, TypeLinks, block_types_table, by_block, type_links

PAGE = "Some operators/Some page"


# --- Builders ---


def form(form_id: str, takes: tuple[str, ...], gives: tuple[str, ...]) -> Variant:
    """A form that takes and gives these types."""
    return Variant(form_id, form_id, (authored("what_it_tests"),), takes=takes, gives=gives)


def operator(op_id: str, name: str, symbol: str, *forms: Variant) -> Operator:
    return Operator(op_id, name, symbol, PAGE, (), (), forms)


def typed(
    op_id: str, takes: tuple[str, ...], gives: tuple[str, ...], symbol: str | None = None
) -> Operator:
    """An operator with one form, which takes and gives these types."""
    name = op_id.replace("_", " ").capitalize()
    return operator(op_id, name, symbol or op_id, form("only", takes, gives))


def chapter_of(chapter_id: str, title: str, *operators: Operator) -> Chapter:
    return Chapter(chapter_id, title, operators)


def find(chapter: Chapter, operator_id: str) -> Operator:
    """The operator of ``chapter`` with this id: not the one at some position."""
    return next(op for op in chapter.operators if op.id == operator_id)


def index_of(cat: Catalogue, statuses: dict[str, Status] | None = None) -> StatusIndex:
    """An index with the status of every operator of ``cat``: ``ok`` unless ``statuses`` say."""
    operators: dict[str, Status] = {
        f"{chapter.id}/{op.id}": "ok" for chapter in cat.chapters for op in chapter.operators
    }
    operators.update(statuses or {})
    return StatusIndex({}, {}, operators, {})


def ids(refs: Iterable[Ref]) -> list[str]:
    return [ref.operator_id for ref in refs]


def places(refs: Iterable[Ref]) -> list[tuple[str, str]]:
    return [(ref.chapter_id, ref.operator_id) for ref in refs]


def links_in(cat: Catalogue, operator_id: str, index: StatusIndex | None = None) -> TypeLinks:
    """The type links of the operator ``operator_id`` of ``cat``, wherever it is."""
    for chapter in cat.chapters:
        for op in chapter.operators:
            if op.id == operator_id:
                return type_links(cat, index or index_of(cat), chapter, op)
    raise AssertionError(f"no operator {operator_id}")


# The chapters are not in the alphabetical order, nor are their operators, so that the order of a
# list is the one of the catalogue and not a sort of the names.
NUMERIC = chapter_of(
    "numeric",
    "Numeric operators",
    typed("round", ("number", "integer"), ("number",)),
    typed("mod", ("number",), ("number",)),
    typed("ceil", ("number",), ("integer",)),
)
STRING = chapter_of(
    "string",
    "String operators",
    typed("length", ("string",), ("integer",)),
    typed("substr", ("string", "integer"), ("string",)),
    typed("concat", ("string",), ("string",), symbol="||"),
)
COMPARISON = chapter_of(
    "comparison",
    "Comparison operators",
    typed("equal", ("scalar",), ("boolean",), symbol="="),
    operator(
        "in",
        "Membership test",
        "in",
        form("sc_set", ("string", "set"), ("boolean",)),
        form("sc_domain", ("string", "value_domain"), ("boolean",)),
    ),
)
CONDITIONAL = chapter_of(
    "conditional",
    "Conditional operators",
    typed("nvl", ("scalar",), ("any",)),
    typed("if_then_else", ("boolean", "scalar"), ("any",)),
)
DEFINITIONS = chapter_of(
    "definitions",
    "Definitions",
    typed("define_datapoint_ruleset", (), ("datapoint_ruleset",)),
    typed("define_value_domain", (), ("value_domain",)),
    typed("define_set", (), ("set",)),
)
VALIDATION = chapter_of(
    "validation",
    "Validation operators",
    typed("check_datapoint", ("datapoint_ruleset",), ("boolean",)),
)
CAT = Catalogue((NUMERIC, STRING, COMPARISON, CONDITIONAL, DEFINITIONS, VALIDATION))
# All five statuses are in use, and the operators of a list have different ones, so that a ref
# shows whose status it has.
STATUSES: dict[str, Status] = {
    "numeric/round": "partial",
    "numeric/mod": "ok",
    "numeric/ceil": "refused",
    "string/length": "diverge",
    "string/substr": "ok",
    "string/concat": "missing",
    "comparison/equal": "ok",
    "comparison/in": "partial",
    "conditional/nvl": "missing",
    "conditional/if_then_else": "ok",
    "definitions/define_datapoint_ruleset": "ok",
    "definitions/define_value_domain": "refused",
    "definitions/define_set": "diverge",
    "validation/check_datapoint": "partial",
}
INDEX = index_of(CAT, STATUSES)


def links_of(operator_id: str) -> TypeLinks:
    return links_in(CAT, operator_id, INDEX)


# --- Comes from and goes to, by the hierarchy of the types ---


def test_producer_of_subtype_feeds_parent() -> None:
    # round gives a number, and ceil and length give an integer, which is a number too. substr
    # gives a string, equal a boolean, and nvl any type: none of them is a number.
    links = links_of("mod")
    assert ids(links.comes_from["number"]) == ["round", "ceil", "length"]
    # The other way round does not hold: a number is not an integer, so mod does not feed round
    # when round takes an integer.
    assert ids(links_of("round").comes_from["integer"]) == ["ceil", "length"]
    assert ids(links_of("round").comes_from["number"]) == ["mod", "ceil", "length"]


def test_consumer_of_supertype_takes_subtype() -> None:
    # An integer is a number: round and mod take it, and so does substr, which takes an integer.
    assert ids(links_of("ceil").goes_to["integer"]) == ["round", "mod", "substr"]
    assert ids(links_of("length").goes_to["integer"]) == ["round", "mod", "ceil", "substr"]
    # A number is not an integer: substr does not take what mod gives.
    assert ids(links_of("mod").goes_to["number"]) == ["round", "ceil"]


def test_the_time_chain_follows_the_hierarchy() -> None:
    # time is above date and above time_period, which are not above each other.
    cat = Catalogue(
        (
            chapter_of(
                "some",
                "Some operators",
                typed("today", (), ("date",)),
                typed("period_of", (), ("time_period",)),
                typed("from_time", (), ("time",)),
                typed("as_time", ("time",), ("string",)),
                typed("on_date", ("date",), ("string",)),
            ),
        )
    )
    # A date and a time period are times: both feed a taker of time. A taker of date is fed by a
    # date only: not by a time period, nor by a time, which may be either.
    assert ids(links_in(cat, "as_time").comes_from["time"]) == ["today", "period_of", "from_time"]
    assert ids(links_in(cat, "on_date").comes_from["date"]) == ["today"]
    # The other way round: a date goes to the takers of date and of time, a time period to the
    # takers of time, and a time to the takers of time, not to those of date.
    assert ids(links_in(cat, "today").goes_to["date"]) == ["as_time", "on_date"]
    assert ids(links_in(cat, "period_of").goes_to["time_period"]) == ["as_time"]
    assert ids(links_in(cat, "from_time").goes_to["time"]) == ["as_time"]


def test_a_taker_of_scalar_is_fed_by_every_value() -> None:
    # Every value type is under scalar. The results that follow their input (nvl and
    # if_then_else) are in any_type, and the definitions give structures, not values.
    links = links_of("equal")
    assert ids(links.comes_from["scalar"]) == [
        "round",
        "mod",
        "ceil",
        "length",
        "substr",
        "concat",
        "in",
        "check_datapoint",
    ]
    assert ids(links.any_type) == ["nvl", "if_then_else"]


def test_every_taken_and_given_type_has_a_list() -> None:
    # There is a list for each type, in the order of takes and gives, and a type that nothing
    # feeds or takes has an empty one.
    cat = Catalogue(
        (
            chapter_of(
                "some",
                "Some operators",
                typed("needs_duration", ("duration", "string"), ("boolean",)),
                typed("gives_string", (), ("string",)),
            ),
        )
    )
    links = links_in(cat, "needs_duration")
    assert tuple(links.comes_from) == links.takes == ("string", "duration")
    assert tuple(links.goes_to) == links.gives == ("boolean",)
    assert ids(links.comes_from["string"]) == ["gives_string"]
    assert links.comes_from["duration"] == ()
    assert links.goes_to["boolean"] == ()


def test_refs_follow_the_catalogue_order() -> None:
    # numeric, string, comparison, validation: the order of the chapters, then of the operators
    # of each. Neither is the alphabetical one.
    assert places(links_of("equal").comes_from["scalar"]) == [
        ("numeric", "round"),
        ("numeric", "mod"),
        ("numeric", "ceil"),
        ("string", "length"),
        ("string", "substr"),
        ("string", "concat"),
        ("comparison", "in"),
        ("validation", "check_datapoint"),
    ]


def test_ref_carries_the_names_and_the_status_of_the_operator() -> None:
    round_ref, _, length_ref = links_of("mod").comes_from["number"]
    assert round_ref == Ref(
        chapter_id="numeric",
        chapter_title="Numeric operators",
        operator_id="round",
        name="Round",
        symbol="round",
        status="partial",
    )
    assert length_ref.status == "diverge"
    # The symbol is the one of the operator, which is not its id.
    concat = links_of("length").comes_from["string"][1]
    assert concat == Ref("string", "String operators", "concat", "Concat", "||", "missing")


def test_ref_and_type_links_are_frozen() -> None:
    ref = links_of("ceil").goes_to["integer"][0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        ref.status = "ok"  # type: ignore[misc]
    links = links_of("ceil")
    with pytest.raises(dataclasses.FrozenInstanceError):
        links.takes = ()  # type: ignore[misc]


def test_a_listed_operator_without_a_status_is_reported() -> None:
    statuses = {key: status for key, status in STATUSES.items() if key != "numeric/ceil"}
    index = StatusIndex({}, {}, statuses, {})
    with pytest.raises(ValueError, match="numeric/ceil"):
        type_links(CAT, index, NUMERIC, find(NUMERIC, "mod"))
    # ceil is in no list of check_datapoint, so it needs no status there.
    check = find(VALIDATION, "check_datapoint")
    assert type_links(CAT, index, VALIDATION, check).takes == ("datapoint_ruleset",)


# --- Any scalar and any type ---


def test_any_scalar_consumers_folded() -> None:
    # equal, nvl and if_then_else take any scalar: they are in a list of their own, and not in
    # the list of the type that mod, ceil or length give.
    for operator_id in ("mod", "ceil", "length"):
        links = links_of(operator_id)
        assert ids(links.any_scalar) == ["equal", "nvl", "if_then_else"]
        for consumers in links.goes_to.values():
            assert {"equal", "nvl", "if_then_else"}.isdisjoint(ids(consumers))


def test_any_type_producers_folded() -> None:
    # nvl and if_then_else give the type of their input: they are in any_type only, whatever
    # type is taken.
    for operator_id in ("mod", "ceil", "round", "length"):
        links = links_of(operator_id)
        assert ids(links.any_type) == ["nvl", "if_then_else"]
        for producers in links.comes_from.values():
            assert {"nvl", "if_then_else"}.isdisjoint(ids(producers))


def test_an_operator_that_gives_any_type_goes_to_the_takers_of_any_scalar_only() -> None:
    # A consumer takes a type that is given, and any is none of them: there is no list by type
    # for what nvl gives. The takers of any scalar take it, whatever it is.
    links = links_of("nvl")
    assert links.gives == ("any",)
    assert links.goes_to == {"any": ()}
    assert ids(links.any_scalar) == ["equal", "if_then_else"]
    # What nvl takes, any scalar, is fed by every producer of a value. The producer that follows
    # its input, if_then_else, is in any_type.
    assert ids(links.comes_from["scalar"]) == [
        "round",
        "mod",
        "ceil",
        "length",
        "substr",
        "concat",
        "equal",
        "in",
        "check_datapoint",
    ]
    assert ids(links.any_type) == ["if_then_else"]


def test_a_taker_of_any_scalar_listed_by_type_is_not_listed_again() -> None:
    # if_then_else takes a boolean and any scalar, so it is in the list of the booleans that
    # equal gives, and not in the list of the takers of any scalar: nvl is, and equal itself is
    # left out.
    links = links_of("equal")
    assert ids(links.goes_to["boolean"]) == ["if_then_else"]
    assert ids(links.any_scalar) == ["nvl"]


def test_folded_lists_do_not_repeat_what_is_listed_by_type() -> None:
    # "mixed" has a form that gives an integer and a form that follows its input: it is a
    # producer of numbers, and it is not in any_type as well.
    producers = Catalogue(
        (
            chapter_of(
                "some",
                "Some operators",
                typed("target", ("number",), ("number",)),
                operator(
                    "mixed",
                    "Mixed",
                    "mixed",
                    form("typed", ("scalar",), ("integer",)),
                    form("follows", ("scalar",), ("any",)),
                ),
                typed("follows_only", ("scalar",), ("any",)),
            ),
        )
    )
    links = links_in(producers, "target")
    assert ids(links.comes_from["number"]) == ["mixed"]
    assert ids(links.any_type) == ["follows_only"]
    # The same on the other side: "mixed" takes a number in one form and any scalar in another.
    consumers = Catalogue(
        (
            chapter_of(
                "some",
                "Some operators",
                typed("source", ("string",), ("integer",)),
                operator(
                    "mixed",
                    "Mixed",
                    "mixed",
                    form("number", ("number",), ("string",)),
                    form("scalar", ("scalar",), ("string",)),
                ),
                typed("scalar_only", ("scalar",), ("string",)),
            ),
        )
    )
    links = links_in(consumers, "source")
    assert ids(links.goes_to["integer"]) == ["mixed"]
    assert ids(links.any_scalar) == ["scalar_only"]


def test_a_form_that_takes_any_type_takes_any_scalar() -> None:
    cat = Catalogue(
        (
            chapter_of(
                "some",
                "Some operators",
                typed("source", (), ("integer",)),
                typed("takes_anything", ("any",), ("string",)),
                typed("follows", ("scalar",), ("any",)),
            ),
        )
    )
    # It is folded with the takers of any scalar, whatever type is given to it...
    links = links_in(cat, "source")
    assert links.goes_to == {"integer": ()}
    assert ids(links.any_scalar) == ["takes_anything", "follows"]
    # ... and it is fed as a taker of any scalar is: by every producer of a value, and by the
    # operators that follow their input (any_type).
    links = links_in(cat, "takes_anything")
    assert links.takes == ("any",)
    assert ids(links.comes_from["any"]) == ["source"]
    assert ids(links.any_type) == ["follows"]


def test_a_form_that_gives_scalar_is_matched_as_it_is() -> None:
    # Only any follows its input. scalar is a type, above every value type and below none, so an
    # operator that gives it feeds the takers of scalar, and no taker of a type under it. The
    # takers give any, so that they do not feed one another by type.
    cat = Catalogue(
        (
            chapter_of(
                "some",
                "Some operators",
                typed("source", (), ("scalar",)),
                typed("takes_scalar", ("scalar",), ("any",)),
                typed("takes_number", ("number",), ("any",)),
            ),
        )
    )
    assert ids(links_in(cat, "takes_scalar").comes_from["scalar"]) == ["source"]
    number = links_in(cat, "takes_number")
    assert number.comes_from == {"number": ()}
    # It does not follow its input: takes_scalar is in any_type, and source is not.
    assert ids(number.any_type) == ["takes_scalar"]
    # And it goes to the takers of scalar only, which are the ones that take any scalar.
    source = links_in(cat, "source")
    assert source.goes_to == {"scalar": ()}
    assert ids(source.any_scalar) == ["takes_scalar"]


# --- The structural types ---


def test_structural_exact() -> None:
    # A ruleset is fed by the definition that gives it, and a value domain is not a set.
    links = links_of("check_datapoint")
    assert links.takes == ("datapoint_ruleset",)
    assert ids(links.comes_from["datapoint_ruleset"]) == ["define_datapoint_ruleset"]
    in_links = links_of("in")
    assert in_links.takes == ("string", "set", "value_domain")
    assert ids(in_links.comes_from["set"]) == ["define_set"]
    assert ids(in_links.comes_from["value_domain"]) == ["define_value_domain"]
    # And the other way round: a definition is taken by the operators that take its type.
    assert ids(links_of("define_value_domain").goes_to["value_domain"]) == ["in"]
    assert ids(links_of("define_set").goes_to["set"]) == ["in"]
    assert ids(links_of("define_datapoint_ruleset").goes_to["datapoint_ruleset"]) == [
        "check_datapoint"
    ]


def test_structural_types_are_not_values() -> None:
    # What follows its input, and what takes any scalar, is about values: a ruleset is neither
    # fed by the operators that give any type nor taken by those that take any scalar.
    check = links_of("check_datapoint")
    assert check.any_type == ()
    define = links_of("define_datapoint_ruleset")
    assert define.any_scalar == ()
    # An operator that takes a value type and a structural type has both.
    assert ids(links_of("in").any_type) == ["nvl", "if_then_else"]


def test_an_operator_that_takes_nothing_has_nothing_before_it() -> None:
    links = links_of("define_datapoint_ruleset")
    assert links.takes == ()
    assert links.comes_from == {}
    assert links.any_type == ()
    assert links.gives == ("datapoint_ruleset",)


def test_self_excluded() -> None:
    # mod gives what it takes, so it would feed itself: it is in the lists of the operators that
    # take a number, and not in its own.
    assert "mod" in ids(links_of("round").comes_from["number"])
    assert "mod" in ids(links_of("ceil").goes_to["integer"])
    for chapter in CAT.chapters:
        for op in chapter.operators:
            links = type_links(CAT, INDEX, chapter, op)
            listed = [
                *links.any_type,
                *links.any_scalar,
                *(ref for refs in links.comes_from.values() for ref in refs),
                *(ref for refs in links.goes_to.values() for ref in refs),
            ]
            assert (chapter.id, op.id) not in places(listed), op.id


# --- The types of an operator ---


def test_takes_and_gives_are_the_types_of_all_the_forms_in_type_order() -> None:
    links = links_of("in")
    assert (links.takes, links.gives) == (("string", "set", "value_domain"), ("boolean",))
    # Declared in any order, they come in the order of the value types, then the structural
    # ones, then the umbrella types: scalar and any.
    cat = Catalogue(
        (
            chapter_of(
                "some",
                "Some operators",
                typed(
                    "odd",
                    ("scalar", "set", "string", "integer", "number"),
                    ("any", "boolean", "duration"),
                ),
            ),
        )
    )
    odd = links_in(cat, "odd")
    assert odd.takes == ("number", "integer", "string", "set", "scalar")
    assert odd.gives == ("boolean", "duration", "any")


def test_a_type_that_is_not_known_comes_last() -> None:
    # The catalogue checks reject it, but a list is still written.
    cat = Catalogue(
        (chapter_of("some", "Some operators", typed("odd", ("mystery", "string"), ("boolean",))),)
    )
    assert links_in(cat, "odd").takes == ("string", "mystery")


def test_forms_without_types_have_those_of_the_operator() -> None:
    # A form that declares no types takes the accepts of its operator and gives its returns,
    # without the dataset: this is what a chapter that is not typed yet looks like.
    old = Operator(
        "old",
        "Old",
        "old",
        PAGE,
        ("number", "dataset"),
        ("integer", "dataset"),
        (Variant("ds", "old(DS)", (authored("what_it_tests"),)),),
    )
    cat = Catalogue(
        (chapter_of("some", "Some operators", old, typed("target", ("number",), ("string",))),)
    )
    old_links = links_in(cat, "old")
    assert (old_links.takes, old_links.gives) == (("number",), ("integer",))
    # It feeds what takes a number, as an operator of its types does.
    assert ids(links_in(cat, "target").comes_from["number"]) == ["old"]


def test_a_form_that_takes_nothing_is_not_given_the_accepts_of_its_operator() -> None:
    # An empty takes is declared, it is not a missing one: Variant.takes is None when a form does
    # not declare it. Both operators accept a number. The form of "nothing" declares that it
    # takes nothing, as current_date() does, so nothing feeds it. The form of "fallback" does not
    # declare takes, so it takes what its operator accepts.
    accepts, returns = ("number", "dataset"), ("date", "dataset")
    nothing = Operator(
        "nothing",
        "Nothing",
        "nothing",
        PAGE,
        accepts,
        returns,
        (Variant("sc", "nothing()", (authored("what_it_tests"),), takes=()),),
    )
    fallback = Operator(
        "fallback",
        "Fallback",
        "fallback",
        PAGE,
        accepts,
        returns,
        (Variant("sc", "fallback(sc)", (authored("what_it_tests"),)),),
    )
    chapter = chapter_of(
        "some", "Some operators", typed("producer", (), ("number",)), nothing, fallback
    )
    cat = Catalogue((chapter,))
    links = links_in(cat, "nothing")
    assert (links.takes, links.gives) == ((), ("date",))
    assert links.comes_from == {}
    assert links.any_type == ()
    fed = links_in(cat, "fallback")
    assert fed.takes == ("number",)
    assert ids(fed.comes_from["number"]) == ["producer"]
    # The table reads the types of the forms the same way: nothing is "in" for "nothing".
    columns, rows = block_types_table(chapter)
    assert columns == ("number", "date")
    assert [cells for _, cells in rows] == [("out", ""), ("", "out"), ("in", "out")]


# --- The types table of a block ---


def table(chapter: Chapter) -> tuple[tuple[str, ...], list[tuple[str, tuple[str, ...]]]]:
    columns, rows = block_types_table(chapter)
    assert [op for op, _ in rows] == list(chapter.operators)
    return columns, [(op.id, cells) for op, cells in rows]


def test_block_table_cells() -> None:
    columns, rows = table(NUMERIC)
    assert columns == ("number", "integer")
    assert rows == [
        ("round", ("in · out", "in")),
        ("mod", ("in · out", "")),
        ("ceil", ("in", "out")),
    ]


def test_block_table_has_a_column_only_for_the_types_used() -> None:
    # No number, and no boolean. The first operator uses string and integer, in that order: the
    # columns are in the order of the types, not of their use.
    columns, rows = table(STRING)
    assert columns == ("integer", "string")
    assert rows == [
        ("length", ("out", "in")),
        ("substr", ("in", "in · out")),
        ("concat", ("", "in · out")),
    ]


def test_block_table_orders_value_types_then_structural_types_then_umbrella_types() -> None:
    # "in" has two forms, one that takes a set and one that takes a value domain: the types of
    # both are in its row.
    columns, rows = table(COMPARISON)
    assert columns == ("string", "boolean", "set", "value_domain", "scalar")
    assert rows == [
        ("equal", ("", "out", "", "", "in")),
        ("in", ("in", "out", "in", "in", "")),
    ]
    columns, rows = table(CONDITIONAL)
    assert columns == ("boolean", "scalar", "any")
    assert rows == [
        ("nvl", ("", "in", "out")),
        ("if_then_else", ("in", "in", "out")),
    ]
    columns, rows = table(DEFINITIONS)
    assert columns == ("set", "value_domain", "datapoint_ruleset")
    assert rows == [
        ("define_datapoint_ruleset", ("", "", "out")),
        ("define_value_domain", ("", "out", "")),
        ("define_set", ("out", "", "")),
    ]


def test_block_table_of_a_block_with_nothing_to_show() -> None:
    assert block_types_table(chapter_of("empty", "Nothing")) == ((), [])
    # An operator that takes nothing and gives nothing has a row of empty cells.
    bare = chapter_of(
        "bare",
        "Bare",
        typed("nothing", (), ()),
        typed("something", ("string",), ("boolean",)),
    )
    columns, rows = table(bare)
    assert columns == ("string", "boolean")
    assert rows == [("nothing", ("", "")), ("something", ("in", "out"))]


# --- By block ---


def test_by_block_groups_the_refs_by_chapter_in_the_order_of_the_catalogue() -> None:
    refs = links_of("equal").comes_from["scalar"]
    blocks = by_block(refs)
    assert [(title, ids(group)) for title, group in blocks] == [
        ("Numeric operators", ["round", "mod", "ceil"]),
        ("String operators", ["length", "substr", "concat"]),
        ("Comparison operators", ["in"]),
        ("Validation operators", ["check_datapoint"]),
    ]
    # Every ref is in one block, and a block holds the refs of its chapter, as they were given.
    assert [ref for _, group in blocks for ref in group] == list(refs)
    assert all(isinstance(group, tuple) for _, group in blocks)


def test_by_block_follows_the_order_of_the_refs() -> None:
    # The refs of type_links are in the order of the catalogue, and so are the blocks. By block
    # does not sort: a chapter comes where its first ref is.
    refs = links_of("ceil").goes_to["integer"]
    round_ref, mod_ref, substr_ref = refs
    blocks = by_block([substr_ref, round_ref, mod_ref])
    assert [(title, ids(group)) for title, group in blocks] == [
        ("String operators", ["substr"]),
        ("Numeric operators", ["round", "mod"]),
    ]
    blocks = by_block([round_ref, substr_ref, mod_ref])
    assert [(title, ids(group)) for title, group in blocks] == [
        ("Numeric operators", ["round", "mod"]),
        ("String operators", ["substr"]),
    ]


def test_by_block_of_nothing() -> None:
    assert by_block([]) == []
