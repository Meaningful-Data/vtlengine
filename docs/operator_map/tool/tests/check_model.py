"""Checks for the catalogue model, its helpers and ``validate_catalogue``."""

# The repo's ruff config allows ``assert`` only under ``tests/``, not in this tool.
# ruff: noqa: S101
from __future__ import annotations

import dataclasses
from typing import Any

import catalogue.manual_index
import pytest

from tool.loader import load_catalogue
from tool.model import (
    ANY_SCALAR,
    ANY_TYPE,
    BOX_TITLE_MAX,
    DL_PAGES,
    REVERSE_LABELS,
    STRUCTURAL_TYPES,
    TYPE_PARENTS,
    VALUE_TYPES,
    Catalogue,
    Chapter,
    Example,
    Kind,
    Link,
    Operator,
    Variant,
    authored,
    example_title,
    form_kind,
    form_types,
    has_titled_boxes,
    manual,
    reverse_label,
    validate_catalogue,
)

ADD_PAGE = "Numeric operators/Addition"
ADD_EX_1 = f"{ADD_PAGE}/ex_1"
LENGTH_PAGE = "String operators/Length"


def typed_form(
    *examples: Example,
    variant_id: str = "ds_ds",
    label: str = "DS + DS",
    kind: Kind | None = None,
    box: str | None = None,
) -> Variant:
    """A form of the addition that declares its types, with ``examples`` (its manual example)."""
    return Variant(
        variant_id,
        label,
        examples or (manual("ex_1"),),
        takes=("number",),
        gives=("number",),
        kind=kind,
        box=box,
    )


def untyped_form(*examples: Example) -> Variant:
    """A form of the addition that declares no types, with ``examples`` (its manual example)."""
    return Variant("ds_ds", "DS + DS", examples or (manual("ex_1"),))


def make_operator(**changes: Any) -> Operator:
    """Addition with one typed form mapped to its first manual example; ``changes`` override."""
    operator = Operator(
        id="addition",
        name="Addition",
        symbol="+",
        page=ADD_PAGE,
        accepts=("number", "dataset"),
        returns=("number", "dataset"),
        variants=(typed_form(),),
    )
    return dataclasses.replace(operator, **changes)


def make_chapter(*operators: Operator) -> Chapter:
    return Chapter(id="numeric", title="Numeric operators", operators=operators)


def make_catalogue(*operators: Operator) -> Catalogue:
    return Catalogue(chapters=(make_chapter(*operators),))


def make_string_chapter(*variants: Variant) -> Chapter:
    operator = Operator(
        id="length",
        name="String length",
        symbol="length",
        page=LENGTH_PAGE,
        accepts=("string", "dataset"),
        returns=("integer", "dataset"),
        variants=variants,
    )
    return Chapter(id="string", title="String operators", operators=(operator,))


def problems_for(cat: Catalogue, scope: tuple[str, ...] | None = None) -> list[str]:
    """Validate against the Addition page and its first manual example only."""
    return validate_catalogue(cat, {ADD_PAGE}, {ADD_EX_1}, scope=scope)


# --- Validation: types, links, ids and coverage ---


def test_valid_catalogue_has_no_problems() -> None:
    assert validate_catalogue(make_catalogue(make_operator()), {ADD_PAGE}, {ADD_EX_1}) == []


def test_unknown_type() -> None:
    cat = make_catalogue(make_operator(accepts=("numbr", "dataset")))
    assert problems_for(cat) == ["numeric.addition: unknown type 'numbr'"]


def test_unknown_type_is_reported_once_per_place() -> None:
    cat = make_catalogue(
        make_operator(accepts=("numbr", "numbr", "dataset"), returns=("numbr", "dataset"))
    )
    assert problems_for(cat) == ["numeric.addition: unknown type 'numbr'"]


def test_unknown_type_in_returns() -> None:
    variant = dataclasses.replace(typed_form(), returns=("intger",))
    cat = make_catalogue(make_operator(returns=("numbr",), variants=(variant,)))
    assert problems_for(cat) == [
        "numeric.addition: unknown type 'numbr'",
        "numeric.addition.ds_ds: unknown type 'intger'",
    ]


def test_link_target_must_exist() -> None:
    cat = make_catalogue(make_operator(links=(Link("clause.aggr", "used in"),)))
    assert problems_for(cat) == ["numeric.addition: link target 'clause.aggr' not found"]


def test_link_target_may_point_to_another_chapter() -> None:
    clause_operator = make_operator(id="aggr", page="Clause operators/Aggr", variants=())
    clause = Chapter(id="clause", title="Clause operators", operators=(clause_operator,))
    linked = make_operator(links=(Link("clause.aggr", "used in"),))
    cat = Catalogue(chapters=(make_chapter(linked), clause))
    pages = {ADD_PAGE, "Clause operators/Aggr"}
    assert validate_catalogue(cat, pages, {ADD_EX_1}) == []


# --- Validation: the definitions of the VTL-DL are never linked ---


def definition(op_id: str, page: str, *links: Link) -> Operator:
    """A definition of the VTL-DL that has ``links``: one form with one authored example."""
    form = typed_form(authored("one_row"), variant_id="scalar_param", label="x")
    return make_operator(id=op_id, page=page, variants=(form,), links=links)


def definition_problems(addition: Operator, *definitions: Operator) -> list[str]:
    """The problems of the addition and the ``definitions`` chapter, whose pages are all known."""
    chapter = Chapter(id="definitions", title="Definitions", operators=definitions)
    cat = Catalogue(chapters=(make_chapter(addition), chapter))
    pages = {ADD_PAGE, *(each.page for each in definitions)}
    return validate_catalogue(cat, pages, {ADD_EX_1})


NEVER_LINKED = "VTL-DL definitions are never linked from or to operators"


def test_a_definition_with_no_link_is_not_a_problem() -> None:
    udo = definition("define_operator", "dl/udo")
    assert definition_problems(make_operator(), udo) == []
    # An operator may link to the other operators, as the others do.
    assert definition_problems(make_operator(links=(Link("numeric.addition", "uses"),)), udo) == []


def test_a_link_from_a_definition_is_a_problem() -> None:
    owner = "definitions.define_operator"
    linking = definition("define_operator", "dl/udo", Link("numeric.addition", "uses"))
    assert definition_problems(make_operator(), linking) == [
        f"{owner}: link target 'numeric.addition': {NEVER_LINKED}"
    ]
    # Any label does it, and each link has a problem of its own.
    for label in REVERSE_LABELS:
        labelled = definition("define_operator", "dl/udo", Link("numeric.addition", label))
        assert len(definition_problems(make_operator(), labelled)) == 1, label
    twice = definition(
        "define_operator",
        "dl/udo",
        Link("numeric.addition", "uses"),
        Link("numeric.addition", "used in"),
    )
    assert len(definition_problems(make_operator(), twice)) == 2


def test_a_link_to_a_definition_is_a_problem() -> None:
    udo = definition("define_operator", "dl/udo")
    linking = make_operator(links=(Link("definitions.define_operator", "uses"),))
    assert definition_problems(linking, udo) == [
        f"numeric.addition: link target 'definitions.define_operator': {NEVER_LINKED}"
    ]
    for label in REVERSE_LABELS:
        labelled = make_operator(links=(Link("definitions.define_operator", label),))
        assert len(definition_problems(labelled, udo)) == 1, label


def test_a_link_between_two_definitions_is_one_problem() -> None:
    # It goes from a definition and to one, and is one link: one line.
    link = Link("definitions.define_datapoint_ruleset", "uses")
    first = definition("define_operator", "dl/udo", link)
    second = definition("define_datapoint_ruleset", "dl/datapoint")
    assert definition_problems(make_operator(), first, second) == [
        "definitions.define_operator: link target 'definitions.define_datapoint_ruleset': "
        + NEVER_LINKED
    ]


def test_the_links_of_a_definition_are_checked_whatever_the_scope() -> None:
    linking = definition("define_operator", "dl/udo", Link("numeric.addition", "uses"))
    chapter = Chapter(id="definitions", title="Definitions", operators=(linking,))
    cat = Catalogue(chapters=(make_chapter(make_operator()), chapter))
    pages = {ADD_PAGE, "dl/udo"}
    expected = [f"definitions.define_operator: link target 'numeric.addition': {NEVER_LINKED}"]
    for scope in (None, ("numeric",), ("string",), ("definitions",), ()):
        assert validate_catalogue(cat, pages, {ADD_EX_1}, scope=scope) == expected, scope


def test_the_real_catalogue_links_no_definition() -> None:
    cat = load_catalogue()
    # The guard goes by the id of the chapter of the definitions: it is there to guard.
    assert "definitions" in {chapter.id for chapter in cat.chapters}
    problems = validate_catalogue(
        cat, catalogue.manual_index.MANUAL_PAGES, catalogue.manual_index.MANUAL_EXAMPLES
    )
    assert problems == []


def test_page_not_covered() -> None:
    pages = {ADD_PAGE, "Numeric operators/Floor"}
    problems = validate_catalogue(make_catalogue(make_operator()), pages, {ADD_EX_1})
    assert problems == ["page not covered: Numeric operators/Floor"]


def test_unknown_page() -> None:
    variants = (typed_form(authored("one_row")),)
    cat = make_catalogue(make_operator(page="Numeric operators/Addtion", variants=variants))
    assert validate_catalogue(cat, {ADD_PAGE}, set()) == [
        "numeric.addition: unknown page 'Numeric operators/Addtion'",
        f"page not covered: {ADD_PAGE}",
    ]
    # The whole catalogue is checked against all the pages, whatever the scope.
    assert validate_catalogue(cat, {ADD_PAGE}, set(), scope=("string",)) == [
        "numeric.addition: unknown page 'Numeric operators/Addtion'"
    ]


def test_manual_example_mapped_twice_or_never() -> None:
    twice = make_operator(
        variants=(
            typed_form(),
            typed_form(variant_id="sc_sc", label="sc + sc"),
        )
    )
    problems = validate_catalogue(make_catalogue(twice), {ADD_PAGE}, {ADD_EX_1})
    assert problems == [f"manual example mapped 2 times: {ADD_EX_1}"]

    both = {ADD_EX_1, f"{ADD_PAGE}/ex_2"}
    problems = validate_catalogue(make_catalogue(make_operator()), {ADD_PAGE}, both)
    assert problems == [f"manual example not mapped: {ADD_PAGE}/ex_2"]


def test_unknown_manual_example() -> None:
    variant = typed_form(manual("ex_1"), manual("ex_9"))
    cat = make_catalogue(make_operator(variants=(variant,)))
    unknown = [f"numeric.addition: unknown manual example '{ADD_PAGE}/ex_9'"]
    assert problems_for(cat) == unknown
    # The whole catalogue is checked against all the manual examples, whatever the scope.
    assert validate_catalogue(cat, {ADD_PAGE}, {ADD_EX_1}, scope=("string",)) == unknown
    assert validate_catalogue(cat, {ADD_PAGE}, {ADD_EX_1}, scope=()) == unknown


def test_duplicate_and_malformed_ids() -> None:
    duplicated = make_operator(
        variants=(
            typed_form(),
            typed_form(authored("one_row"), label="DS + DS again"),
        )
    )
    assert problems_for(make_catalogue(duplicated)) == [
        "numeric.addition: duplicate variant 'ds_ds'"
    ]

    malformed = make_operator(id="Add")
    assert problems_for(make_catalogue(malformed)) == ["numeric: bad id 'Add'"]


def test_duplicate_example_in_one_variant() -> None:
    twice = typed_form(manual("ex_1"), authored("one_row"), authored("one_row"))
    cat = make_catalogue(make_operator(variants=(twice,)))
    assert problems_for(cat) == ["numeric.addition.ds_ds: duplicate example 'one_row'"]

    # Every variant has its own ``one_row``: the same name in two variants is not a duplicate.
    first = typed_form(manual("ex_1"), authored("one_row"))
    second = typed_form(authored("one_row"), variant_id="sc_sc", label="sc + sc")
    assert problems_for(make_catalogue(make_operator(variants=(first, second)))) == []


def test_bad_and_duplicate_ids_at_every_level() -> None:
    def variant(variant_id: str) -> Variant:
        return typed_form(authored("one_row"), variant_id=variant_id, label="x")

    first = make_operator(variants=(variant("DS_DS"), variant("a"), variant("a"), variant("a")))
    second = make_operator(variants=(variant("trailing\n"),))
    numeric = Chapter("numeric", "Numeric operators", (first, second, make_operator(id="")))
    duplicate = Chapter("numeric", "Numeric operators", ())
    cat = Catalogue(chapters=(numeric, duplicate, Chapter("Num-eric", "Bad", ())))
    assert problems_for(cat) == [
        "numeric.addition: bad id 'DS_DS'",
        "numeric.addition: duplicate variant 'a'",
        "numeric: duplicate operator 'addition'",
        "numeric.addition: bad id 'trailing\n'",
        "numeric: bad id ''",
        "catalogue: duplicate chapter 'numeric'",
        "catalogue: bad id 'Num-eric'",
    ]


def test_problems_follow_catalogue_order_then_sorted() -> None:
    variants = (typed_form(authored("one_row"), label="DS - DS"),)
    second = make_operator(
        id="subtraction",
        page="Numeric operators/Subtraction",
        accepts=("x", "dataset"),
        variants=variants,
    )
    first = make_operator(accepts=("y", "dataset"))
    cat = make_catalogue(first, second)
    pages = {ADD_PAGE, "Numeric operators/Subtraction", "Numeric operators/Floor", "A/B"}
    manual_examples = {ADD_EX_1, f"{ADD_PAGE}/ex_3", f"{ADD_PAGE}/ex_2"}
    assert validate_catalogue(cat, pages, manual_examples) == [
        "numeric.addition: unknown type 'y'",
        "numeric.subtraction: unknown type 'x'",
        "page not covered: A/B",
        "page not covered: Numeric operators/Floor",
        f"manual example not mapped: {ADD_PAGE}/ex_2",
        f"manual example not mapped: {ADD_PAGE}/ex_3",
    ]


# --- Forms: their kind and their types ---


@pytest.mark.parametrize(
    ("variant_id", "kind"),
    [
        ("sc_comp_digits", "comp"),
        ("analytic_ds", "ds"),
        ("bare", None),
        # Any token ``ds`` makes a dataset form, wherever it is and whatever else the id has.
        ("ds", "ds"),
        ("ds_sc", "ds"),
        ("sc_ds", "ds"),
        ("comp_ds", "ds"),
        ("ds_group_by", "ds"),
        ("levenshtein_ds_ds", "ds"),
        # Else any token ``comp`` a component form, else any ``sc`` a scalar form.
        ("comp", "comp"),
        ("sc_comp", "comp"),
        ("comp_sc_digits", "comp"),
        ("sc", "sc"),
        ("sc_sc", "sc"),
        ("sc_null", "sc"),
        # A token is a part of the id between underscores: a part that only starts like one is
        # not one.
        ("param_dataset", None),
        ("non_null_dataset_all", None),
        ("dsl", None),
        ("ds2", None),
        ("component", None),
        ("scalar_param", None),
        ("identity", None),
    ],
)
def test_form_kind(variant_id: str, kind: Kind | None) -> None:
    assert form_kind(Variant(variant_id, "x", (authored(),))) == kind


def test_form_kind_explicit_overrides_tokens() -> None:
    def form(variant_id: str, kind: Kind | None = None) -> Variant:
        return Variant(variant_id, "x", (authored(),), kind=kind)

    # A form whose id has no token says its kind: the conversions of ``cast`` are on components.
    assert form_kind(form("integer_to_boolean")) is None
    assert form_kind(form("integer_to_boolean", "comp")) == "comp"
    # What a form says wins over the tokens of its id, whichever they are.
    assert form_kind(form("ds_ds", "sc")) == "sc"
    assert form_kind(form("sc_comp_digits", "ds")) == "ds"
    assert form_kind(form("comp", "comp")) == "comp"
    # Without a kind of its own, the tokens decide.
    assert form_kind(form("analytic_ds", None)) == "ds"
    assert form_kind(form("identity")) is None


def test_form_types_are_the_ones_the_form_declares() -> None:
    # A form has the types that it declares, whatever its returns say.
    typed = Variant(
        "sc_sc", "x", (authored(),), returns=("number",), takes=("string",), gives=("integer",)
    )
    assert form_types(typed) == (("string",), ("integer",))
    # And no other: a form that declares none has none, whatever its operator accepts and
    # returns (``validate_catalogue`` refuses such a form). Whether a form is on datasets is its
    # kind, not one of its types.
    bare = Variant("ds_ds", "DS + DS", (manual("ex_1"),))
    assert bare.takes is None
    assert bare.gives is None
    assert form_types(bare) == ((), ())
    # Each side on its own: ``takes=()`` is a form that takes nothing, such as ``current_date()``.
    takes_only = Variant("ds_ds", "x", (authored(),), takes=("integer",))
    assert form_types(takes_only) == (("integer",), ())
    gives_only = Variant("ds_ds", "x", (authored(),), gives=("boolean",))
    assert form_types(gives_only) == ((), ("boolean",))
    nothing = Variant("sc", "current_date()", (authored(),), takes=(), gives=("date",))
    assert form_types(nothing) == ((), ("date",))
    # Its own ``returns`` do not give it types either.
    rounded = Variant("ds_no_digits", "round(DS)", (authored(),), returns=("integer", "dataset"))
    assert form_types(rounded) == ((), ())


def test_the_value_and_structural_types_keep_the_order_of_the_design() -> None:
    # The columns of the types table of a block follow this order.
    assert VALUE_TYPES == (
        "number",
        "integer",
        "string",
        "boolean",
        "time",
        "date",
        "time_period",
        "duration",
    )
    assert STRUCTURAL_TYPES == (
        "set",
        "value_domain",
        "datapoint_ruleset",
        "hierarchical_ruleset",
        "viral_propagation",
        "operator",
        "external_routine",
    )
    assert (ANY_SCALAR, ANY_TYPE) == ("scalar", "any")


def test_the_types_follow_the_hierarchy_of_the_design() -> None:
    # Scalar, then number and integer under it; string, boolean and duration; time, then date
    # and time_period under it. The structural types are outside the hierarchy.
    assert TYPE_PARENTS == {
        "scalar": (),
        "number": ("scalar",),
        "integer": ("number",),
        "string": ("scalar",),
        "boolean": ("scalar",),
        "time": ("scalar",),
        "date": ("time",),
        "time_period": ("time",),
        "duration": ("scalar",),
    }
    assert set(TYPE_PARENTS) == {ANY_SCALAR, *VALUE_TYPES}
    assert not set(TYPE_PARENTS) & {ANY_TYPE, *STRUCTURAL_TYPES}


def test_reverse_label_known_and_unknown() -> None:
    # A link is shown at both of its ends: at its target, with the reverse of its label.
    reverses = {
        "uses": "used by",
        "used in": "uses",
        "used in over()": "uses, in over()",
        "used in group all": "uses, in group all",
    }
    for label, reverse in reverses.items():
        assert reverse_label(label) == reverse, label
    assert set(REVERSE_LABELS) == set(reverses)
    with pytest.raises(KeyError):
        reverse_label("feeds")
    with pytest.raises(KeyError):
        reverse_label("Uses")
    # Two labels that no link of the catalogue declares are gone: they were for the links of the
    # definitions of the VTL-DL, which the catalogue no longer has.
    for gone in ("used by", "propagates at"):
        with pytest.raises(KeyError):
            reverse_label(gone)
    # A link whose label has no reverse could not be shown at its target: the catalogue is wrong.
    unknown = make_operator(links=(Link("numeric.addition", "feeds"),))
    assert problems_for(make_catalogue(unknown)) == [
        "numeric.addition: link label 'feeds' has no reverse label"
    ]
    known = make_operator(links=tuple(Link("numeric.addition", label) for label in reverses))
    assert problems_for(make_catalogue(known)) == []


@pytest.mark.parametrize(
    ("example", "title"),
    [
        (manual("ex_1"), "Manual example 1"),
        (manual("ex_12"), "Manual example 12"),
        (manual("ex_3", erratum="E"), "Manual example 3"),
        (authored("zero_divisor"), "Zero divisor"),
        (authored("integer_measure"), "Integer measure"),
        (authored("null"), "Null"),
        (authored("two_ds_2_rows"), "Two ds 2 rows"),
    ],
)
def test_example_title(example: Example, title: str) -> None:
    assert example_title(example) == title


# --- Validation: the names of the examples ---


def test_example_names_and_manual_refs_are_ids() -> None:
    # They name folders and the targets of the clicks of the diagrams.
    examples = (
        Example(name="Minimal", source="authored"),
        Example(name="../x", source="authored"),
        manual("ex-1"),
    )
    cat = make_catalogue(make_operator(variants=(typed_form(*examples),)))
    assert problems_for(cat) == [
        f"numeric.addition: unknown manual example '{ADD_PAGE}/ex-1'",
        "numeric.addition.ds_ds: bad example name 'Minimal'",
        "numeric.addition.ds_ds: bad example name '../x'",
        "numeric.addition.ds_ds: bad example name 'manual_ex-1'",
        "numeric.addition.ds_ds: bad manual_ref 'ex-1'",
        f"manual example not mapped: {ADD_EX_1}",
    ]


# --- Validation: limiting the coverage checks to chapters ---


def test_scope_limits_coverage_checks() -> None:
    string_chapter = make_string_chapter(
        typed_form(manual("ex_1"), variant_id="ds", label="length(DS)"),
        typed_form(manual("ex_1"), variant_id="sc", label="length(sc)"),
    )
    cat = Catalogue(chapters=(make_chapter(make_operator()), string_chapter))
    pages = {ADD_PAGE, "Numeric operators/Floor", LENGTH_PAGE, "String operators/Substr"}
    manual_examples = {
        ADD_EX_1,
        f"{ADD_PAGE}/ex_2",
        f"{LENGTH_PAGE}/ex_1",
        "String operators/Substr/ex_1",
    }
    numeric = [
        "page not covered: Numeric operators/Floor",
        f"manual example not mapped: {ADD_PAGE}/ex_2",
    ]
    string = [
        "page not covered: String operators/Substr",
        f"manual example mapped 2 times: {LENGTH_PAGE}/ex_1",
        "manual example not mapped: String operators/Substr/ex_1",
    ]
    everything = [
        "page not covered: Numeric operators/Floor",
        "page not covered: String operators/Substr",
        f"manual example not mapped: {ADD_PAGE}/ex_2",
        f"manual example mapped 2 times: {LENGTH_PAGE}/ex_1",
        "manual example not mapped: String operators/Substr/ex_1",
    ]

    assert validate_catalogue(cat, pages, manual_examples) == everything
    assert validate_catalogue(cat, pages, manual_examples, scope=None) == everything
    assert (
        validate_catalogue(cat, pages, manual_examples, scope=("numeric", "string")) == everything
    )
    assert validate_catalogue(cat, pages, manual_examples, scope=("numeric",)) == numeric
    assert validate_catalogue(cat, pages, manual_examples, scope={"string"}) == string
    assert validate_catalogue(cat, pages, manual_examples, scope=[]) == []
    # An id with no chapter in the catalogue owns no pages.
    assert validate_catalogue(cat, pages, manual_examples, scope=("nope",)) == []


def test_scope_definitions_covers_dl_pages() -> None:
    cat = make_catalogue(make_operator())
    pages = {ADD_PAGE, *DL_PAGES}
    dl_problems = [f"page not covered: {page}" for page in sorted(DL_PAGES)]
    assert validate_catalogue(cat, pages, {ADD_EX_1}) == dl_problems
    assert validate_catalogue(cat, pages, {ADD_EX_1}, scope=("definitions",)) == dl_problems
    assert validate_catalogue(cat, pages, {ADD_EX_1}, scope=("numeric",)) == []

    keys = {ADD_EX_1, "dl/udo/ex_1"}
    assert validate_catalogue(cat, {ADD_PAGE}, keys, scope=("definitions",)) == [
        "manual example not mapped: dl/udo/ex_1"
    ]
    assert validate_catalogue(cat, {ADD_PAGE}, keys, scope=("numeric",)) == []

    udo = make_operator(
        id="define_operator",
        page="dl/udo",
        variants=(typed_form(authored("one_row"), variant_id="scalar_param", label="x"),),
    )
    definitions = Chapter(id="definitions", title="Definitions", operators=(udo,))
    cat = Catalogue(chapters=(make_chapter(make_operator()), definitions))
    assert validate_catalogue(cat, pages, {ADD_EX_1}, scope=("definitions",)) == [
        "page not covered: dl/datapoint",
        "page not covered: dl/hierarchical",
        "page not covered: dl/viral",
    ]


def test_scope_does_not_limit_type_link_and_id_checks() -> None:
    links = (Link("clause.aggr", "used in"), Link("numeric.addition", "feeds"))
    broken = make_operator(accepts=("numbr", "dataset"), links=links)
    string_chapter = make_string_chapter(
        typed_form(authored("one_row"), variant_id="DS", label="length(DS)")
    )
    cat = Catalogue(chapters=(make_chapter(broken), string_chapter))
    pages = {ADD_PAGE, "Numeric operators/Floor", LENGTH_PAGE}
    problems = validate_catalogue(cat, pages, {ADD_EX_1}, scope=("string",))
    assert problems == [
        "numeric.addition: unknown type 'numbr'",
        "numeric.addition: link target 'clause.aggr' not found",
        "numeric.addition: link label 'feeds' has no reverse label",
        "string.length: bad id 'DS'",
    ]


# --- Validation: the types of the forms ---


def test_the_types_of_a_form_are_known_types() -> None:
    # The value types, any scalar, any type and the structural types, and nothing else; the
    # kind of a form says whether it is on datasets, so ``dataset`` is not one of its types.
    known = Variant(
        "ds_ds",
        "DS + DS",
        (manual("ex_1"),),
        takes=(
            "number",
            "integer",
            "string",
            "boolean",
            "time",
            "date",
            "time_period",
            "duration",
            "scalar",
        ),
        gives=(
            "any",
            "set",
            "value_domain",
            "datapoint_ruleset",
            "hierarchical_ruleset",
            "viral_propagation",
            "operator",
            "external_routine",
        ),
    )
    assert problems_for(make_catalogue(make_operator(variants=(known,)))) == []
    unknown = Variant(
        "ds_ds",
        "DS + DS",
        (manual("ex_1"),),
        takes=("dataset", "numbr", "numbr"),
        gives=("Number", "integer"),
    )
    expected = [
        "numeric.addition.ds_ds: unknown type 'dataset'",
        "numeric.addition.ds_ds: unknown type 'numbr'",
        "numeric.addition.ds_ds: unknown type 'Number'",
    ]
    # Whatever the scope says: the whole catalogue is checked.
    cat = make_catalogue(make_operator(variants=(unknown,)))
    assert problems_for(cat) == expected
    assert validate_catalogue(cat, {ADD_PAGE}, {ADD_EX_1}, scope=("string",)) == expected
    # The operators and the returns of a form still name ``dataset``, as they did.
    assert problems_for(make_catalogue(make_operator())) == []


def test_a_form_declares_the_types_it_takes_and_gives() -> None:
    untyped = make_catalogue(make_operator(variants=(untyped_form(),)))  # it declares no types
    assert problems_for(untyped) == [
        "numeric.addition.ds_ds: the form does not declare takes",
        "numeric.addition.ds_ds: the form does not declare gives",
    ]
    half = Variant("ds_ds", "DS + DS", (manual("ex_1"),), takes=("number",))
    assert problems_for(make_catalogue(make_operator(variants=(half,)))) == [
        "numeric.addition.ds_ds: the form does not declare gives"
    ]
    # Returns of its own, or types of the operator, do not make a form typed.
    legacy = Variant("ds_ds", "DS + DS", (manual("ex_1"),), returns=("integer",))
    assert problems_for(make_catalogue(make_operator(variants=(legacy,)))) == [
        "numeric.addition.ds_ds: the form does not declare takes",
        "numeric.addition.ds_ds: the form does not declare gives",
    ]
    # A form gives something: an empty ``gives`` is declared, and still a problem.
    gives_nothing = Variant("ds_ds", "DS + DS", (manual("ex_1"),), takes=("number",), gives=())
    assert problems_for(make_catalogue(make_operator(variants=(gives_nothing,)))) == [
        "numeric.addition.ds_ds: the form gives nothing: gives needs a type"
    ]
    assert problems_for(make_catalogue(make_operator(variants=(typed_form(),)))) == []
    # An unknown type is a problem too.
    odd = Variant("ds_ds", "DS + DS", (manual("ex_1"),), takes=("numbr",), gives=("dataset",))
    assert problems_for(make_catalogue(make_operator(variants=(odd,)))) == [
        "numeric.addition.ds_ds: unknown type 'numbr'",
        "numeric.addition.ds_ds: unknown type 'dataset'",
    ]


def test_a_form_does_not_take_any() -> None:
    # ``any`` is for what a form gives: the result follows the input. A form that takes a value
    # of any type takes ``scalar``.
    owner = "numeric.addition.ds_ds"
    problem = (
        f"{owner}: the form takes 'any': any is only for gives, so a value of any type takes "
        "'scalar'"
    )

    def with_types(takes: tuple[str, ...], gives: tuple[str, ...]) -> list[str]:
        form = dataclasses.replace(typed_form(), takes=takes, gives=gives)
        return problems_for(make_catalogue(make_operator(variants=(form,))))

    assert with_types(("any",), ("number",)) == [problem]
    # One problem for the form, wherever it takes ``any`` and however many times.
    assert with_types(("number", "any", "any", "string"), ("number",)) == [problem]
    # ``scalar`` is what a value of any type is, and ``any`` is what a result that follows its
    # input gives, alone or with other types.
    assert with_types(("scalar",), ("any",)) == []
    assert with_types(("scalar",), ("number", "any")) == []
    # The other problems of the form are reported too, the unknown types first.
    assert with_types(("any", "numbr"), ("number",)) == [f"{owner}: unknown type 'numbr'", problem]
    # Each form that takes it has a problem of its own, which names it.
    second = dataclasses.replace(
        typed_form(authored("one_row"), variant_id="sc_sc", label="sc + sc"), takes=("any",)
    )
    first = dataclasses.replace(typed_form(), takes=("any",))
    assert problems_for(make_catalogue(make_operator(variants=(first, second)))) == [
        problem,
        problem.replace(".ds_ds:", ".sc_sc:"),
    ]


def test_a_form_may_take_nothing_but_not_give_nothing() -> None:
    # ``current_date()`` takes no operand: an empty ``takes`` says so, and is not a problem.
    def current_date(takes: tuple[str, ...] | None, gives: tuple[str, ...] | None) -> Catalogue:
        form = Variant("sc", "current_date()", (manual("ex_1"),), takes=takes, gives=gives)
        return make_catalogue(make_operator(variants=(form,)))

    assert problems_for(current_date((), ("date",))) == []
    # Undeclared is not the same as nothing.
    assert problems_for(current_date(None, ("date",))) == [
        "numeric.addition.sc: the form does not declare takes"
    ]
    # Taking nothing does not excuse giving nothing.
    assert problems_for(current_date((), ())) == [
        "numeric.addition.sc: the form gives nothing: gives needs a type"
    ]


def test_the_checks_of_the_forms_cover_every_chapter_whatever_the_scope() -> None:
    string_chapter = make_string_chapter(Variant("ds", "length(DS)", (authored("minimal"),)))
    cat = Catalogue(
        chapters=(make_chapter(make_operator(variants=(untyped_form(),))), string_chapter)
    )
    pages = {ADD_PAGE, LENGTH_PAGE}
    expected = [
        "numeric.addition.ds_ds: the form does not declare takes",
        "numeric.addition.ds_ds: the form does not declare gives",
        "string.length.ds: the form does not declare takes",
        "string.length.ds: the form does not declare gives",
        "string.length.ds: example name 'minimal' does not say what the example tests",
    ]
    # A run that is limited to some chapters renders the pages of all of them, so nothing turns
    # these checks off, nor limits them: the scope is for the coverage of the pages.
    for scope in (None, ("numeric", "string"), ("string",), ("numeric",), ()):
        assert validate_catalogue(cat, pages, {ADD_EX_1}, scope=scope) == expected, scope


def test_authored_examples_are_named_after_what_they_test() -> None:
    def named(*names: str) -> Catalogue:
        examples = tuple(authored(name) for name in names)
        return make_catalogue(make_operator(variants=(typed_form(manual("ex_1"), *examples),)))

    owner = "numeric.addition.ds_ds"
    # Named after what it tests: a letter, then up to 39 letters, digits and underscores.
    good = named("zero_divisor", "a", "null_operand", "two_rows_2", "x" * 40)
    assert problems_for(good) == []
    # The names of the first version of the map say nothing about what they test.
    assert problems_for(named("minimal", "minimal_2", "minimal_13")) == [
        f"{owner}: example name 'minimal' does not say what the example tests",
        f"{owner}: example name 'minimal_2' does not say what the example tests",
        f"{owner}: example name 'minimal_13' does not say what the example tests",
    ]
    slug = "a letter, then up to 39 letters, digits or underscores"
    assert problems_for(named("x" * 41, "2_rows", "_rows")) == [
        f"{owner}: example name '{'x' * 41}' is not a slug ({slug})",
        f"{owner}: example name '2_rows' is not a slug ({slug})",
        f"{owner}: example name '_rows' is not a slug ({slug})",
    ]
    # The name of a manual example is for a manual example: an authored one would pass for it.
    assert problems_for(named("manual_ex_2")) == [
        f"{owner}: example name 'manual_ex_2' is the name of a manual example"
    ]
    # A name that is not an id is reported once, by the rule of the ids.
    assert problems_for(named("Zero_divisor")) == [f"{owner}: bad example name 'Zero_divisor'"]
    # The names are unique within a form, and each form has its own.
    assert problems_for(named("zero_divisor", "zero_divisor")) == [
        f"{owner}: duplicate example 'zero_divisor'"
    ]
    other = typed_form(authored("zero_divisor"), variant_id="sc_sc")
    first = typed_form(manual("ex_1"), authored("zero_divisor"))
    assert problems_for(make_catalogue(make_operator(variants=(first, other)))) == []
    # The manual examples keep their names.
    assert problems_for(named()) == []


def test_the_forms_of_an_operator_all_have_a_kind_or_none() -> None:
    def form(variant_id: str, kind: Kind | None = None) -> Variant:
        return typed_form(authored("one_row"), variant_id=variant_id, kind=kind)

    def operator(*forms: Variant) -> Catalogue:
        return make_catalogue(make_operator(variants=(typed_form(manual("ex_1")), *forms)))

    # The form ``ds_ds`` has a kind; these two have none: the operator mixes them.
    mixed = operator(form("integer_to_boolean"), form("sc_sc"), form("expression"))
    assert problems_for(mixed) == [
        "numeric.addition: these forms have no kind, but other forms have one: "
        "integer_to_boolean, expression"
    ]
    # A kind of their own settles them, and so does a kind from a token.
    settled = operator(form("integer_to_boolean", "comp"), form("sc_sc"), form("expression", "sc"))
    assert problems_for(settled) == []
    # Forms with no kind at all are fine: the operator has no boxes in its diagram.
    kindless = make_catalogue(
        make_operator(
            variants=(
                typed_form(manual("ex_1"), variant_id="identity"),
                form("expression"),
            )
        )
    )
    assert problems_for(kindless) == []
    # The scope does not limit it: it is for the coverage of the pages.
    assert problems_for(mixed, scope=("string",)) == problems_for(mixed)


def boxed_operator(*boxes: str | None, ids: tuple[str, ...] = ()) -> Catalogue:
    """The addition with a form for each of ``boxes``, in the box it names (None: no box).

    The forms are called ``one``, ``two``..., or ``ids``. The first one has the manual example.
    """
    names = ids or ("one", "two", "three", "four", "five")
    forms = tuple(
        typed_form(
            *((manual("ex_1"),) if n == 0 else (authored("one_row"),)),
            variant_id=names[n],
            box=box,
        )
        for n, box in enumerate(boxes)
    )
    return make_catalogue(make_operator(variants=forms))


def test_a_form_names_its_box_and_the_default_is_none() -> None:
    assert Variant("x", "x", (authored(),)).box is None
    form = Variant("x", "x", (authored(),), box="Signature")
    assert form.box == "Signature"
    assert not has_titled_boxes(make_operator())
    assert has_titled_boxes(make_operator(variants=(typed_form(box="Body"),)))
    # One form that names a box is enough for the operator to have them, which is what the
    # diagram goes by: the checks of the catalogue are what asks for all of them.
    one_of_two = (typed_form(), typed_form(variant_id="sc_sc", box="Body"))
    assert has_titled_boxes(make_operator(variants=one_of_two))
    assert not has_titled_boxes(make_operator(variants=()))


def test_the_forms_of_an_operator_all_name_a_box_or_none() -> None:
    owner = "numeric.addition"
    # Every form names a box: fine, whether the boxes are one or many, in any order.
    assert problems_for(boxed_operator("Signature", "Body", "Signature", "Return type")) == []
    assert problems_for(boxed_operator("Body")) == []
    # No form names one: fine too, as it always was.
    assert problems_for(boxed_operator(None, None, None)) == []
    # Some do and some do not: the forms that do not are the problem, in the order of the
    # catalogue, whichever form comes first.
    assert problems_for(boxed_operator("Signature", None, "Body", None)) == [
        f"{owner}: these forms have no box, but other forms have one: two, four"
    ]
    assert problems_for(boxed_operator(None, None, "Body")) == [
        f"{owner}: these forms have no box, but other forms have one: one, two"
    ]


def test_the_checks_of_the_boxes_cover_every_chapter_whatever_the_scope() -> None:
    partial = boxed_operator("Signature", None)
    assert len(problems_for(partial)) == 1
    assert problems_for(partial, scope=("string",)) == problems_for(partial)
    assert problems_for(partial, scope=("numeric",)) == problems_for(partial)
    too_long = boxed_operator("x" * (BOX_TITLE_MAX + 1))
    assert len(problems_for(too_long)) == 1
    assert problems_for(too_long, scope=("string",)) == problems_for(too_long)


def test_a_box_title_is_not_empty_and_has_at_most_30_characters() -> None:
    owner = "numeric.addition"
    assert BOX_TITLE_MAX == 30
    # A title is for a box: some words, at most 30 characters.
    assert problems_for(boxed_operator("Signature", "Return type", "x", "y" * 30)) == []
    assert problems_for(boxed_operator("y" * 30)) == []
    long = "y" * 31
    assert problems_for(boxed_operator(long)) == [
        f"{owner}.one: box title '{long}' is longer than 30 characters"
    ]
    # It is not empty, and a blank title is empty.
    assert problems_for(boxed_operator("")) == [f"{owner}.one: the box title is empty"]
    assert problems_for(boxed_operator("   ")) == [f"{owner}.one: the box title is empty"]
    # Each form is checked, so a title that is wrong is reported at its form.
    assert problems_for(boxed_operator("Body", "", "Body", long)) == [
        f"{owner}.two: the box title is empty",
        f"{owner}.four: box title '{long}' is longer than 30 characters",
    ]
    # The length is the length of the title, as it is written.
    assert problems_for(boxed_operator("é" * 30, "é" * 31)) == [
        f"{owner}.two: box title '{'é' * 31}' is longer than 30 characters"
    ]


def test_the_kinds_of_forms_in_boxes_say_nothing() -> None:
    # ``ds_ds`` has a kind and ``plain`` has none: without boxes that is a mixture...
    ids = ("ds_ds", "plain", "sc_sc")
    mixture = "numeric.addition: these forms have no kind, but other forms have one: plain"
    assert problems_for(boxed_operator(None, None, None, ids=ids)) == [mixture]
    # ...and with boxes, where no box is drawn by kind, it is not.
    assert problems_for(boxed_operator("Signature", "Body", "Body", ids=ids)) == []
    # A form that names no box is the problem then, and the kinds are still not looked at.
    assert problems_for(boxed_operator("Signature", None, "Body", ids=ids)) == [
        "numeric.addition: these forms have no box, but other forms have one: plain"
    ]


# --- Model helpers ---


def test_manual_and_authored_helpers() -> None:
    assert manual("ex_1") == Example(name="manual_ex_1", source="manual", manual_ref="ex_1")
    assert manual("ex_2", erratum="E", decision=1049, check="random") == Example(
        name="manual_ex_2",
        source="manual",
        manual_ref="ex_2",
        erratum="E",
        decision=1049,
        check="random",
    )
    assert authored() == Example(name="minimal", source="authored")
    assert authored("minimal_2", adaptation="A") == Example(
        name="minimal_2", source="authored", adaptation="A"
    )
    assert authored().check == "exact"
    assert authored().manual_ref is None


def test_model_nodes_are_frozen_and_hashable() -> None:
    example = manual("ex_1")
    assert hash(example) == hash(manual("ex_1"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        example.name = "other"  # type: ignore[misc]
    assert make_operator() == make_operator()
    assert hash(make_catalogue(make_operator())) == hash(make_catalogue(make_operator()))


def test_load_catalogue_reads_the_catalogue_package(monkeypatch: pytest.MonkeyPatch) -> None:
    chapter = make_chapter(make_operator())
    monkeypatch.setattr(catalogue, "CHAPTERS", (chapter,))
    assert load_catalogue() == Catalogue(chapters=(chapter,))
