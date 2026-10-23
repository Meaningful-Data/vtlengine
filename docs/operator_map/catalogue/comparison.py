"""The comparison operators of the VTL 2.2 manual (chapter "Comparison operators")."""

from __future__ import annotations

from tool.model import Chapter, Operator, Variant, authored, manual

_CHAPTER_TITLE = "Comparison operators"

# The operand levels of a binary comparison, in the order of the variants: id, left, right.
_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("ds_ds", "DS", "DS"),
    ("ds_sc", "DS", "sc"),
    ("sc_ds", "sc", "DS"),
    ("comp_comp", "comp", "comp"),
    ("comp_sc", "comp", "sc"),
    ("sc_comp", "sc", "comp"),
    ("sc_sc", "sc", "sc"),
)

# The collections that ``in`` and ``not_in`` test against: a set literal or a value domain.
_COLLECTIONS: tuple[tuple[str, str, str], ...] = (
    ("ds_set", "DS", "set"),
    ("comp_set", "comp", "set"),
    ("sc_set", "sc", "set"),
    ("ds_value_domain", "DS", "value domain"),
    ("comp_value_domain", "comp", "value domain"),
)


def _page(name: str) -> str:
    return f"{_CHAPTER_TITLE}/{name}"


def _variant(variant_id: str, label: str, manual_ref: str | None = None) -> Variant:
    """A variant with the manual example ``manual_ref``, or with an authored one when None."""
    example = authored() if manual_ref is None else manual(manual_ref)
    return Variant(variant_id, label, (example,))


def _binary(
    operator_id: str, name: str, symbol: str, page: str, manual_refs: dict[str, str]
) -> Operator:
    """A comparison with two operands: the seven pairs of operand levels.

    ``manual_refs`` gives the manual example (``ex_N``) of each pair that the page exemplifies.
    """
    variants = tuple(
        _variant(pair, f"{left} {symbol} {right}", manual_refs.get(pair))
        for pair, left, right in _PAIRS
    )
    return Operator(
        id=operator_id,
        name=name,
        symbol=symbol,
        page=_page(page),
        accepts=("scalar",),
        returns=("boolean",),
        variants=variants,
    )


def _element_of(operator_id: str, name: str, keyword: str, manual_refs: dict[str, str]) -> Operator:
    """``in`` or ``not_in``: the operand levels over a set literal and over a value domain."""
    variants = tuple(
        _variant(variant_id, f"{operand} {keyword} {collection}", manual_refs.get(variant_id))
        for variant_id, operand, collection in _COLLECTIONS
    )
    return Operator(
        id=operator_id,
        name=name,
        symbol=keyword,
        page=_page("Element of"),
        accepts=("scalar", "set", "value_domain"),
        returns=("boolean",),
        variants=variants,
    )


_BETWEEN = Operator(
    id="between",
    name="Between",
    symbol="between",
    page=_page("Between"),
    accepts=("scalar",),
    returns=("boolean",),
    variants=(
        _variant("ds_sc_sc", "between(DS, sc, sc)", "ex_1"),
        _variant("comp_sc_sc", "between(comp, sc, sc)"),
        _variant("comp_comp_comp", "between(comp, comp, comp)"),
        _variant("sc_sc_sc", "between(sc, sc, sc)"),
    ),
)

_MATCH_CHARACTERS = Operator(
    id="match_characters",
    name="Match characters",
    symbol="match_characters",
    page=_page("Match characters"),
    accepts=("string",),
    returns=("boolean",),
    variants=(
        _variant("ds_sc", "match_characters(DS, sc)", "ex_1"),
        _variant("comp_sc", "match_characters(comp, sc)"),
        _variant("comp_comp", "match_characters(comp, comp)"),
        _variant("sc_sc", "match_characters(sc, sc)"),
    ),
)

_ISNULL = Operator(
    id="isnull",
    name="Is null",
    symbol="isnull",
    page=_page("Is null"),
    accepts=("scalar",),
    returns=("boolean",),
    variants=(
        _variant("ds", "isnull(DS)", "ex_1"),
        _variant("comp", "isnull(comp)", "ex_2"),
        # The manual gives two scalar results, ``isnull(null)`` and ``isnull("Hello")``.
        Variant("sc", "isnull(sc)", (authored(), authored("minimal_2"))),
    ),
)

_EXISTS_IN = Operator(
    id="exists_in",
    name="Exists in",
    symbol="exists_in",
    page=_page("Exists in"),
    accepts=("dataset",),
    returns=("boolean", "dataset"),
    variants=(
        _variant("retain_default", "exists_in(DS, DS)"),
        _variant("retain_true", "exists_in(DS, DS, true)", "ex_2"),
        _variant("retain_false", "exists_in(DS, DS, false)", "ex_3"),
        _variant("retain_all", "exists_in(DS, DS, all)", "ex_1"),
    ),
)

CHAPTER = Chapter(
    id="comparison",
    title=_CHAPTER_TITLE,
    operators=(
        _binary("equal", "Equal to", "=", "Equal to", {"ds_sc": "ex_1", "comp_sc": "ex_2"}),
        _binary(
            "not_equal",
            "Not equal to",
            "<>",
            "Not equal to",
            {"ds_ds": "ex_1", "comp_sc": "ex_2"},
        ),
        _binary(
            "greater",
            "Greater than",
            ">",
            "Greater than",
            {"ds_sc": "ex_1", "comp_sc": "ex_2", "ds_ds": "ex_3"},
        ),
        _binary("greater_equal", "Greater than or equal to", ">=", "Greater than", {}),
        _binary("less", "Less than", "<", "Less than", {"ds_sc": "ex_1"}),
        _binary("less_equal", "Less than or equal to", "<=", "Less than", {}),
        _BETWEEN,
        _element_of(
            "in",
            "Element of",
            "in",
            {"ds_set": "ex_1", "comp_set": "ex_2", "ds_value_domain": "ex_3"},
        ),
        _element_of("not_in", "Not element of", "not_in", {}),
        _MATCH_CHARACTERS,
        _ISNULL,
        _EXISTS_IN,
    ),
)
