"""Checks for the catalogue model, its helpers and ``validate_catalogue``."""

# The repo's ruff config allows ``assert`` only under ``tests/``, not in this tool.
# ruff: noqa: S101
from __future__ import annotations

import dataclasses
from typing import Any

import catalogue
import pytest

from tool.model import (
    DL_PAGES,
    Catalogue,
    Chapter,
    Example,
    Link,
    Operator,
    TypeDef,
    Variant,
    authored,
    load_catalogue,
    manual,
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
        accepts=("number",),
        returns=("number",),
        variants=(Variant(id="ds_ds", label="DS + DS", examples=(manual("ex_1"),)),),
    )
    return dataclasses.replace(operator, **changes)


def make_chapter(*operators: Operator) -> Chapter:
    return Chapter(id="numeric", title="Numeric operators", operators=operators)


def make_catalogue(*operators: Operator, types: tuple[TypeDef, ...] = ()) -> Catalogue:
    return Catalogue(chapters=(make_chapter(*operators),), types=types)


def make_string_chapter(*variants: Variant) -> Chapter:
    operator = Operator(
        id="length",
        name="String length",
        symbol="length",
        page=LENGTH_PAGE,
        accepts=("string",),
        returns=("integer",),
        variants=variants,
    )
    return Chapter(id="string", title="String operators", operators=(operator,))


def problems_for(cat: Catalogue) -> list[str]:
    """Validate against the Addition page and its first manual example only."""
    return validate_catalogue(cat, {ADD_PAGE}, {ADD_EX_1})


# --- Validation: types, links, ids and coverage ---


def test_valid_catalogue_has_no_problems() -> None:
    assert validate_catalogue(make_catalogue(make_operator()), {ADD_PAGE}, {ADD_EX_1}) == []


def test_unknown_type() -> None:
    cat = make_catalogue(make_operator(accepts=("numbr",)))
    assert problems_for(cat) == ["numeric.addition: unknown type 'numbr'"]


def test_unknown_type_is_reported_once_per_place() -> None:
    cat = make_catalogue(make_operator(accepts=("numbr", "numbr"), returns=("numbr",)))
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
    cat = Catalogue(chapters=(make_chapter(linked), clause), types=())
    pages = {ADD_PAGE, "Clause operators/Aggr"}
    assert validate_catalogue(cat, pages, {ADD_EX_1}) == []


def test_page_not_covered() -> None:
    pages = {ADD_PAGE, "Numeric operators/Floor"}
    problems = validate_catalogue(make_catalogue(make_operator()), pages, {ADD_EX_1})
    assert problems == ["page not covered: Numeric operators/Floor"]


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


def test_bad_and_duplicate_ids_at_every_level() -> None:
    def variant(variant_id: str) -> Variant:
        return Variant(variant_id, "x", (authored(),))

    first = make_operator(variants=(variant("DS_DS"), variant("a"), variant("a"), variant("a")))
    second = make_operator(variants=(variant("trailing\n"),))
    numeric = Chapter("numeric", "Numeric operators", (first, second, make_operator(id="")))
    duplicate = Chapter("numeric", "Numeric operators", ())
    cat = Catalogue(chapters=(numeric, duplicate, Chapter("Num-eric", "Bad", ())), types=())
    assert validate_catalogue(cat, set(), set()) == [
        "numeric.addition: bad id 'DS_DS'",
        "numeric.addition: duplicate variant 'a'",
        "numeric: duplicate operator 'addition'",
        "numeric.addition: bad id 'trailing\n'",
        "numeric: bad id ''",
        "catalogue: duplicate chapter 'numeric'",
        "catalogue: bad id 'Num-eric'",
    ]


def test_problems_follow_catalogue_order_then_sorted() -> None:
    second = make_operator(id="subtraction", page="Numeric operators/Subtraction", accepts=("x",))
    first = make_operator(accepts=("y",))
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


# --- Validation: limiting the coverage checks to chapters ---


def test_scope_limits_coverage_checks() -> None:
    string_chapter = make_string_chapter(
        Variant("ds", "length(DS)", (manual("ex_1"),)),
        Variant("sc", "length(sc)", (manual("ex_1"),)),
    )
    cat = Catalogue(chapters=(make_chapter(make_operator()), string_chapter), types=())
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
    # An id with no chapter in the catalogue (such as ``types``) owns no pages.
    assert validate_catalogue(cat, pages, manual_examples, scope=("types", "nope")) == []


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
    cat = Catalogue(chapters=(make_chapter(make_operator()), definitions), types=())
    assert validate_catalogue(cat, pages, {ADD_EX_1}, scope=("definitions",)) == [
        "page not covered: dl/datapoint",
        "page not covered: dl/hierarchical",
        "page not covered: dl/viral",
    ]


def test_scope_does_not_limit_type_link_id_and_see_checks() -> None:
    broken = make_operator(accepts=("numbr",), links=(Link("clause.aggr", "used in"),))
    string_chapter = make_string_chapter(Variant("DS", "length(DS)", (authored(),)))
    types = (TypeDef("scalar", (), see="general.x"),)
    cat = Catalogue(chapters=(make_chapter(broken), string_chapter), types=types)
    pages = {ADD_PAGE, "Numeric operators/Floor", LENGTH_PAGE}
    problems = validate_catalogue(cat, pages, {ADD_EX_1}, scope=("string",))
    assert problems == [
        "numeric.addition: unknown type 'numbr'",
        "numeric.addition: link target 'clause.aggr' not found",
        "string.length: bad id 'DS'",
        "types.scalar: see target 'general.x' not found",
    ]


# --- Validation: type definitions ---


def test_typedef_see_target_must_exist() -> None:
    resolved = make_catalogue(
        make_operator(), types=(TypeDef("scalar", (), see="numeric.addition"),)
    )
    assert problems_for(resolved) == []

    missing = make_catalogue(make_operator(), types=(TypeDef("scalar", (), see="general.x"),))
    assert problems_for(missing) == ["types.scalar: see target 'general.x' not found"]


def test_typedef_ids_and_parents_must_be_known_types() -> None:
    types = (TypeDef("integer", ("numbr",)), TypeDef("flaot", ()))
    cat = make_catalogue(make_operator(), types=types)
    assert problems_for(cat) == [
        "types.integer: unknown type 'numbr'",
        "types: unknown type 'flaot'",
    ]


def test_typedef_variants_and_duplicate_types() -> None:
    identity = Variant("identity", "DS_r := DS_1", (authored(),))
    bad = Variant("Identity", "x", (authored(),), returns=("numbr",))
    types = (
        TypeDef("scalar", ()),
        TypeDef("scalar", ()),
        TypeDef("number", ("scalar",), variants=(identity, identity, bad)),
    )
    cat = make_catalogue(make_operator(), types=types)
    assert problems_for(cat) == [
        "types: duplicate type 'scalar'",
        "types.number: duplicate variant 'identity'",
        "types.number: bad id 'Identity'",
        "types.number.Identity: unknown type 'numbr'",
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
    scalar = TypeDef("scalar", ())
    monkeypatch.setattr(catalogue, "CHAPTERS", (chapter,))
    monkeypatch.setattr(catalogue, "TYPES", (scalar,))
    assert load_catalogue() == Catalogue(chapters=(chapter,), types=(scalar,))
