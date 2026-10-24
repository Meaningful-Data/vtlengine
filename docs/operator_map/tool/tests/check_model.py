"""Checks for the catalogue model, its helpers and ``validate_catalogue``."""

# The repo's ruff config allows ``assert`` only under ``tests/``, not in this tool.
# ruff: noqa: S101
from __future__ import annotations

import dataclasses
from typing import Any

import catalogue
import pytest

from tool.model import (
    ANY_SCALAR,
    ANY_TYPE,
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
    load_catalogue,
    manual,
    reverse_label,
    validate_catalogue,
)

ADD_PAGE = "Numeric operators/Addition"
ADD_EX_1 = f"{ADD_PAGE}/ex_1"
LENGTH_PAGE = "String operators/Length"


def make_operator(**changes: Any) -> Operator:
    """Addition with one variant mapped to its first manual example; ``changes`` override."""
    operator = Operator(
        id="addition",
        name="Addition",
        symbol="+",
        page=ADD_PAGE,
        accepts=("number", "dataset"),
        returns=("number", "dataset"),
        variants=(Variant(id="ds_ds", label="DS + DS", examples=(manual("ex_1"),)),),
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


def problems_for(cat: Catalogue) -> list[str]:
    """Validate against the Addition page and its first manual example only."""
    return validate_catalogue(cat, {ADD_PAGE}, {ADD_EX_1})


def strict_problems_for(cat: Catalogue, scope: tuple[str, ...] | None = None) -> list[str]:
    """``problems_for`` with the strict checks of the forms, for the chapters of ``scope``."""
    return validate_catalogue(cat, {ADD_PAGE}, {ADD_EX_1}, scope=scope, strict_forms=True)


def typed_form(*examples: Example, variant_id: str = "ds_ds") -> Variant:
    """A form of the addition that declares its types, with ``examples`` (its manual example)."""
    return Variant(
        variant_id,
        "DS + DS",
        examples or (manual("ex_1"),),
        takes=("number",),
        gives=("number",),
    )


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
    variant = Variant("ds_ds", "DS + DS", (manual("ex_1"),), returns=("intger",))
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


def test_page_not_covered() -> None:
    pages = {ADD_PAGE, "Numeric operators/Floor"}
    problems = validate_catalogue(make_catalogue(make_operator()), pages, {ADD_EX_1})
    assert problems == ["page not covered: Numeric operators/Floor"]


def test_unknown_page() -> None:
    minimal = (Variant("ds_ds", "DS + DS", (authored(),)),)
    cat = make_catalogue(make_operator(page="Numeric operators/Addtion", variants=minimal))
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
            Variant("ds_ds", "DS + DS", (manual("ex_1"),)),
            Variant("sc_sc", "sc + sc", (manual("ex_1"),)),
        )
    )
    problems = validate_catalogue(make_catalogue(twice), {ADD_PAGE}, {ADD_EX_1})
    assert problems == [f"manual example mapped 2 times: {ADD_EX_1}"]

    both = {ADD_EX_1, f"{ADD_PAGE}/ex_2"}
    problems = validate_catalogue(make_catalogue(make_operator()), {ADD_PAGE}, both)
    assert problems == [f"manual example not mapped: {ADD_PAGE}/ex_2"]


def test_unknown_manual_example() -> None:
    variant = Variant("ds_ds", "DS + DS", (manual("ex_1"), manual("ex_9")))
    cat = make_catalogue(make_operator(variants=(variant,)))
    unknown = [f"numeric.addition: unknown manual example '{ADD_PAGE}/ex_9'"]
    assert problems_for(cat) == unknown
    # The whole catalogue is checked against all the manual examples, whatever the scope.
    assert validate_catalogue(cat, {ADD_PAGE}, {ADD_EX_1}, scope=("string",)) == unknown
    assert validate_catalogue(cat, {ADD_PAGE}, {ADD_EX_1}, scope=()) == unknown


def test_duplicate_and_malformed_ids() -> None:
    duplicated = make_operator(
        variants=(
            Variant("ds_ds", "DS + DS", (manual("ex_1"),)),
            Variant("ds_ds", "DS + DS again", (authored(),)),
        )
    )
    assert problems_for(make_catalogue(duplicated)) == [
        "numeric.addition: duplicate variant 'ds_ds'"
    ]

    malformed = make_operator(id="Add")
    assert problems_for(make_catalogue(malformed)) == ["numeric: bad id 'Add'"]


def test_duplicate_example_in_one_variant() -> None:
    twice = Variant("ds_ds", "DS + DS", (manual("ex_1"), authored(), authored()))
    cat = make_catalogue(make_operator(variants=(twice,)))
    assert problems_for(cat) == ["numeric.addition.ds_ds: duplicate example 'minimal'"]

    # Every variant has its own ``minimal``: the same name in two variants is not a duplicate.
    first = Variant("ds_ds", "DS + DS", (manual("ex_1"), authored()))
    second = Variant("sc_sc", "sc + sc", (authored(),))
    assert problems_for(make_catalogue(make_operator(variants=(first, second)))) == []


def test_bad_and_duplicate_ids_at_every_level() -> None:
    def variant(variant_id: str) -> Variant:
        return Variant(variant_id, "x", (authored(),))

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
    minimal = (Variant("ds_ds", "DS - DS", (authored(),)),)
    second = make_operator(
        id="subtraction",
        page="Numeric operators/Subtraction",
        accepts=("x", "dataset"),
        variants=minimal,
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


def test_form_types_fallback() -> None:
    # A form that declares its types has them, whatever its operator and its returns say.
    typed = Variant(
        "sc_sc", "x", (authored(),), returns=("number",), takes=("string",), gives=("integer",)
    )
    assert form_types(make_operator(), typed) == (("string",), ("integer",))
    # A form that declares none has those of its operator, without ``dataset``: whether a form
    # is on datasets is its kind, not one of its types.
    bare = Variant("ds_ds", "DS + DS", (manual("ex_1"),))
    assert form_types(make_operator(), bare) == (("number",), ("number",))
    both = make_operator(accepts=("dataset", "string", "integer"), returns=("dataset", "integer"))
    assert form_types(both, bare) == (("string", "integer"), ("integer",))
    # Each side falls back on its own.
    takes_only = Variant("ds_ds", "x", (authored(),), takes=("integer",))
    assert form_types(make_operator(), takes_only) == (("integer",), ("number",))
    gives_only = Variant("ds_ds", "x", (authored(),), gives=("boolean",))
    assert form_types(make_operator(), gives_only) == (("number",), ("boolean",))
    # What a form gives falls back on its own returns first, without ``dataset`` too, and on
    # those of the operator when they hold nothing else.
    rounded = Variant("ds_no_digits", "round(DS)", (authored(),), returns=("integer", "dataset"))
    assert form_types(make_operator(), rounded) == (("number",), ("integer",))
    on_datasets = Variant("ds", "cast(DS, integer)", (authored(),), returns=("dataset",))
    assert form_types(make_operator(), on_datasets) == (("number",), ("number",))
    # An operator on datasets only leaves nothing to fall back on.
    calc = make_operator(accepts=("dataset",), returns=("dataset",))
    assert form_types(calc, bare) == ((), ())


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
        "used by": "uses",
        "used in": "uses",
        "used in over()": "uses, in over()",
        "used in group all": "uses, in group all",
        "propagates at": "propagation defined by",
    }
    for label, reverse in reverses.items():
        assert reverse_label(label) == reverse, label
    assert set(REVERSE_LABELS) == set(reverses)
    with pytest.raises(KeyError):
        reverse_label("feeds")
    with pytest.raises(KeyError):
        reverse_label("Uses")
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
    cat = make_catalogue(make_operator(variants=(Variant("ds_ds", "DS + DS", examples),)))
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
        Variant("ds", "length(DS)", (manual("ex_1"),)),
        Variant("sc", "length(sc)", (manual("ex_1"),)),
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
        variants=(Variant("scalar_param", "x", (authored(),)),),
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
    string_chapter = make_string_chapter(Variant("DS", "length(DS)", (authored(),)))
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
            "any",
        ),
        gives=(
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
    # Whatever the strict checks and the scope say: the whole catalogue is checked.
    cat = make_catalogue(make_operator(variants=(unknown,)))
    assert problems_for(cat) == expected
    assert validate_catalogue(cat, {ADD_PAGE}, {ADD_EX_1}, scope=("string",)) == expected
    # The operators and the returns of a form still name ``dataset``, as they did.
    assert problems_for(make_catalogue(make_operator())) == []


def test_strict_forms_requires_types() -> None:
    untyped = make_catalogue(make_operator())  # its form ``ds_ds`` declares no types
    assert problems_for(untyped) == []  # the strict checks are off by default
    assert strict_problems_for(untyped) == [
        "numeric.addition.ds_ds: the form declares no takes",
        "numeric.addition.ds_ds: the form declares no gives",
    ]
    half = Variant("ds_ds", "DS + DS", (manual("ex_1"),), takes=("number",))
    assert strict_problems_for(make_catalogue(make_operator(variants=(half,)))) == [
        "numeric.addition.ds_ds: the form declares no gives"
    ]
    # Returns of its own, or types of the operator, do not make a form typed.
    legacy = Variant("ds_ds", "DS + DS", (manual("ex_1"),), returns=("integer",))
    assert strict_problems_for(make_catalogue(make_operator(variants=(legacy,)))) == [
        "numeric.addition.ds_ds: the form declares no takes",
        "numeric.addition.ds_ds: the form declares no gives",
    ]
    assert strict_problems_for(make_catalogue(make_operator(variants=(typed_form(),)))) == []
    # An unknown type is a problem under the strict checks too.
    odd = Variant("ds_ds", "DS + DS", (manual("ex_1"),), takes=("numbr",), gives=("dataset",))
    assert strict_problems_for(make_catalogue(make_operator(variants=(odd,)))) == [
        "numeric.addition.ds_ds: unknown type 'numbr'",
        "numeric.addition.ds_ds: unknown type 'dataset'",
    ]


def test_strict_forms_are_checked_only_in_scope() -> None:
    string_chapter = make_string_chapter(Variant("ds", "length(DS)", (authored("minimal"),)))
    cat = Catalogue(chapters=(make_chapter(make_operator()), string_chapter))
    pages = {ADD_PAGE, LENGTH_PAGE}
    numeric = [
        "numeric.addition.ds_ds: the form declares no takes",
        "numeric.addition.ds_ds: the form declares no gives",
    ]
    string = [
        "string.length.ds: the form declares no takes",
        "string.length.ds: the form declares no gives",
        "string.length.ds: example name 'minimal' does not say what the example tests",
    ]

    def strict(scope: tuple[str, ...] | None) -> list[str]:
        return validate_catalogue(cat, pages, {ADD_EX_1}, scope=scope, strict_forms=True)

    assert strict(None) == [*numeric, *string]
    assert strict(("numeric", "string")) == [*numeric, *string]
    assert strict(("string",)) == string
    assert strict(("numeric",)) == numeric
    assert strict(()) == []
    assert validate_catalogue(cat, pages, {ADD_EX_1}, scope=("string",)) == []


def test_strict_forms_slug() -> None:
    def named(*names: str) -> Catalogue:
        examples = tuple(authored(name) for name in names)
        return make_catalogue(make_operator(variants=(typed_form(manual("ex_1"), *examples),)))

    owner = "numeric.addition.ds_ds"
    # Named after what it tests: a letter, then up to 39 letters, digits and underscores.
    good = named("zero_divisor", "a", "null_operand", "two_rows_2", "x" * 40)
    assert strict_problems_for(good) == []
    # The names of the first version of the map say nothing about what they test.
    assert strict_problems_for(named("minimal", "minimal_2", "minimal_13")) == [
        f"{owner}: example name 'minimal' does not say what the example tests",
        f"{owner}: example name 'minimal_2' does not say what the example tests",
        f"{owner}: example name 'minimal_13' does not say what the example tests",
    ]
    slug = "a letter, then up to 39 letters, digits or underscores"
    assert strict_problems_for(named("x" * 41, "2_rows", "_rows")) == [
        f"{owner}: example name '{'x' * 41}' is not a slug ({slug})",
        f"{owner}: example name '2_rows' is not a slug ({slug})",
        f"{owner}: example name '_rows' is not a slug ({slug})",
    ]
    # The name of a manual example is for a manual example: an authored one would pass for it.
    assert strict_problems_for(named("manual_ex_2")) == [
        f"{owner}: example name 'manual_ex_2' is the name of a manual example"
    ]
    # A name that is not an id is reported once, by the rule of the ids.
    assert strict_problems_for(named("Zero_divisor")) == [
        f"{owner}: bad example name 'Zero_divisor'"
    ]
    # The names are unique within a form, and each form has its own.
    assert strict_problems_for(named("zero_divisor", "zero_divisor")) == [
        f"{owner}: duplicate example 'zero_divisor'"
    ]
    other = typed_form(authored("zero_divisor"), variant_id="sc_sc")
    first = typed_form(manual("ex_1"), authored("zero_divisor"))
    assert strict_problems_for(make_catalogue(make_operator(variants=(first, other)))) == []
    # The manual examples keep their names, and the checks are off by default.
    assert strict_problems_for(named()) == []
    assert problems_for(named("minimal", "manual_ex_2", "x" * 41)) == []


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
