"""The comparison operators of the VTL 2.2 manual (chapter "Comparison operators").

A variant is one syntax form: the level of each operand (``DS`` a dataset, ``comp`` a component
inside ``calc``, ``sc`` a scalar) and, for ``in`` and ``not_in``, whether the collection is a set
literal or a value domain; for ``exists_in``, whether ``retain`` is omitted, ``true``, ``false``
or ``all``. Each form says what it takes and gives, by the "Input parameters type" and "Result
type" of its page:

* ``=``, ``<>``, ``>``, ``>=``, ``<``, ``<=``, ``between`` and ``isnull`` take
  ``dataset {measure<scalar> _} | component<scalar> | scalar``, a value of any type;
* ``match_characters`` takes ``dataset {measure<string> _} | component<string> | string``;
* ``in`` and ``not_in`` take a value of any type and a collection, ``set<scalar> |
  name<value_domain>``, so the type of the collection is ``set`` or ``value_domain``;
* ``exists_in`` takes data sets, ``dataset``, whatever the types of their measures;
* every form gives ``dataset { measure<boolean> bool_var } | component<boolean> | boolean``.
"""

from __future__ import annotations

from tool.model import ANY_SCALAR, Chapter, Example, Operator, Variant, authored, manual

_CHAPTER_TITLE = "Comparison operators"

# What a form takes and gives. The level of its operands, a dataset, a component or a scalar, is
# its kind and not a type, so ``DS = DS`` has the types of ``sc = sc``.
_SCALAR = (ANY_SCALAR,)  # an operand of any type: the page's ``scalar``
_STRING = ("string",)
_BOOLEAN = ("boolean",)

# The types of an operator: those of its forms, and ``dataset`` for its forms on datasets.
_OP_SCALAR = (*_SCALAR, "dataset")
_OP_STRING = (*_STRING, "dataset")
_OP_BOOLEAN = (*_BOOLEAN, "dataset")

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
# Each has the id of the form, the operand level, the words of its label and its type.
_COLLECTIONS: tuple[tuple[str, str, str, str], ...] = (
    ("ds_set", "DS", "set", "set"),
    ("comp_set", "comp", "set", "set"),
    ("sc_set", "sc", "set", "set"),
    ("ds_value_domain", "DS", "value domain", "value_domain"),
    ("comp_value_domain", "comp", "value domain", "value_domain"),
)

# The examples that the six binary comparisons share, as they share their data. Each has one
# measure value that equals the scalar or the other measure, and one null.
_REORDERED_DATA_POINTS = authored("reordered_data_points_and_null")
_TWO_MEASURES = authored("two_measures_and_null")
_EQUAL_VALUE = authored("equal_value_and_null_measure")
_SCALAR_FIRST = authored("scalar_first_and_null_measure")
# The members, a non-member and a null of ``in`` and ``not_in``, and a member of a set literal.
_MEMBERS = authored("members_and_null_operand")
_SET_LITERAL_MEMBER = authored("member_of_a_set_literal")

_Examples = dict[str, tuple[Example, ...]]


def _page(name: str) -> str:
    return f"{_CHAPTER_TITLE}/{name}"


def _binary(operator_id: str, name: str, symbol: str, page: str, examples: _Examples) -> Operator:
    """A comparison with two operands: the seven pairs of operand levels.

    ``examples`` has the examples of each pair, by the id of the pair.
    """
    variants = tuple(
        Variant(
            pair,
            f"{left} {symbol} {right}",
            examples[pair],
            takes=_SCALAR,
            gives=_BOOLEAN,
        )
        for pair, left, right in _PAIRS
    )
    return Operator(
        id=operator_id,
        name=name,
        symbol=symbol,
        page=_page(page),
        accepts=_OP_SCALAR,
        returns=_OP_BOOLEAN,
        variants=variants,
    )


def _element_of(operator_id: str, name: str, keyword: str, examples: _Examples) -> Operator:
    """``in`` or ``not_in``: the operand levels over a set literal and over a value domain."""
    variants = tuple(
        Variant(
            variant_id,
            f"{operand} {keyword} {collection}",
            examples[variant_id],
            takes=(ANY_SCALAR, collection_type),
            gives=_BOOLEAN,
        )
        for variant_id, operand, collection, collection_type in _COLLECTIONS
    )
    return Operator(
        id=operator_id,
        name=name,
        symbol=keyword,
        page=_page("Element of"),
        accepts=(*_SCALAR, "set", "value_domain", "dataset"),
        returns=_OP_BOOLEAN,
        variants=variants,
    )


_BETWEEN = Operator(
    id="between",
    name="Between",
    symbol="between",
    page=_page("Between"),
    accepts=_OP_SCALAR,
    returns=_OP_BOOLEAN,
    variants=(
        Variant(
            "ds_sc_sc",
            "between(DS, sc, sc)",
            (manual("ex_1"),),
            takes=_SCALAR,
            gives=_BOOLEAN,
        ),
        Variant(
            "comp_sc_sc",
            "between(comp, sc, sc)",
            (authored("limits_included_and_null_operand"),),
            takes=_SCALAR,
            gives=_BOOLEAN,
        ),
        Variant(
            "comp_comp_comp",
            "between(comp, comp, comp)",
            (authored("component_limits_and_null_operand"),),
            takes=_SCALAR,
            gives=_BOOLEAN,
        ),
        Variant(
            "sc_sc_sc",
            "between(sc, sc, sc)",
            (authored("upper_limit_included"),),
            takes=_SCALAR,
            gives=_BOOLEAN,
        ),
    ),
)

_MATCH_CHARACTERS = Operator(
    id="match_characters",
    name="Match characters",
    symbol="match_characters",
    page=_page("Match characters"),
    accepts=_OP_STRING,
    returns=_OP_BOOLEAN,
    variants=(
        Variant(
            "ds_sc",
            "match_characters(DS, sc)",
            (manual("ex_1"),),
            takes=_STRING,
            gives=_BOOLEAN,
        ),
        Variant(
            "comp_sc",
            "match_characters(comp, sc)",
            (authored("matches_and_null_operand"),),
            takes=_STRING,
            gives=_BOOLEAN,
        ),
        Variant(
            "comp_comp",
            "match_characters(comp, comp)",
            (authored("pattern_from_a_component_and_null"),),
            takes=_STRING,
            gives=_BOOLEAN,
        ),
        Variant(
            "sc_sc",
            "match_characters(sc, sc)",
            (authored("matching_string_literal"),),
            takes=_STRING,
            gives=_BOOLEAN,
        ),
    ),
)

_ISNULL = Operator(
    id="isnull",
    name="Is null",
    symbol="isnull",
    page=_page("Is null"),
    accepts=_OP_SCALAR,
    returns=_OP_BOOLEAN,
    variants=(
        Variant("ds", "isnull(DS)", (manual("ex_1"),), takes=_SCALAR, gives=_BOOLEAN),
        Variant("comp", "isnull(comp)", (manual("ex_2"),), takes=_SCALAR, gives=_BOOLEAN),
        # The manual gives two scalar results, ``isnull(null)`` and ``isnull("Hello")``.
        Variant(
            "sc",
            "isnull(sc)",
            (authored("null_literal"), authored("non_null_string_literal")),
            takes=_SCALAR,
            gives=_BOOLEAN,
        ),
    ),
)

_EXISTS_IN = Operator(
    id="exists_in",
    name="Exists in",
    symbol="exists_in",
    page=_page("Exists in"),
    accepts=_OP_SCALAR,
    returns=_OP_BOOLEAN,
    variants=(
        Variant(
            "retain_default",
            "exists_in(DS, DS)",
            (authored("identifier_found_with_null_measure"),),
            takes=_SCALAR,
            gives=_BOOLEAN,
        ),
        Variant(
            "retain_true",
            "exists_in(DS, DS, true)",
            (manual("ex_2"),),
            takes=_SCALAR,
            gives=_BOOLEAN,
        ),
        Variant(
            "retain_false",
            "exists_in(DS, DS, false)",
            (manual("ex_3"),),
            takes=_SCALAR,
            gives=_BOOLEAN,
        ),
        Variant(
            "retain_all",
            "exists_in(DS, DS, all)",
            (manual("ex_1"),),
            takes=_SCALAR,
            gives=_BOOLEAN,
        ),
    ),
)

CHAPTER = Chapter(
    id="comparison",
    title=_CHAPTER_TITLE,
    operators=(
        _binary(
            "equal",
            "Equal to",
            "=",
            "Equal to",
            {
                "ds_ds": (_REORDERED_DATA_POINTS,),
                "ds_sc": (manual("ex_1"),),
                "sc_ds": (_SCALAR_FIRST,),
                "comp_comp": (_TWO_MEASURES,),
                "comp_sc": (manual("ex_2"),),
                "sc_comp": (_SCALAR_FIRST,),
                "sc_sc": (authored("different_numbers"),),
            },
        ),
        _binary(
            "not_equal",
            "Not equal to",
            "<>",
            "Not equal to",
            {
                "ds_ds": (manual("ex_1"),),
                "ds_sc": (_EQUAL_VALUE,),
                "sc_ds": (_SCALAR_FIRST,),
                "comp_comp": (_TWO_MEASURES,),
                "comp_sc": (manual("ex_2"),),
                "sc_comp": (_SCALAR_FIRST,),
                "sc_sc": (authored("different_numbers"),),
            },
        ),
        _binary(
            "greater",
            "Greater than",
            ">",
            "Greater than",
            {
                "ds_ds": (manual("ex_3"),),
                "ds_sc": (manual("ex_1"),),
                "sc_ds": (_SCALAR_FIRST,),
                "comp_comp": (_TWO_MEASURES,),
                "comp_sc": (manual("ex_2"),),
                "sc_comp": (_SCALAR_FIRST,),
                "sc_sc": (authored("alphabetical_string_order"),),
            },
        ),
        _binary(
            "greater_equal",
            "Greater than or equal to",
            ">=",
            "Greater than",
            {
                "ds_ds": (_REORDERED_DATA_POINTS,),
                "ds_sc": (_EQUAL_VALUE,),
                "sc_ds": (_SCALAR_FIRST,),
                "comp_comp": (_TWO_MEASURES,),
                "comp_sc": (_EQUAL_VALUE,),
                "sc_comp": (_SCALAR_FIRST,),
                "sc_sc": (authored("equal_numbers"),),
            },
        ),
        _binary(
            "less",
            "Less than",
            "<",
            "Less than",
            {
                "ds_ds": (_REORDERED_DATA_POINTS,),
                "ds_sc": (manual("ex_1"),),
                "sc_ds": (_SCALAR_FIRST,),
                "comp_comp": (_TWO_MEASURES,),
                "comp_sc": (_EQUAL_VALUE,),
                "sc_comp": (_SCALAR_FIRST,),
                "sc_sc": (authored("alphabetical_string_order"),),
            },
        ),
        _binary(
            "less_equal",
            "Less than or equal to",
            "<=",
            "Less than",
            {
                "ds_ds": (_REORDERED_DATA_POINTS,),
                "ds_sc": (_EQUAL_VALUE,),
                "sc_ds": (_SCALAR_FIRST,),
                "comp_comp": (_TWO_MEASURES,),
                "comp_sc": (_EQUAL_VALUE,),
                "sc_comp": (_SCALAR_FIRST,),
                "sc_sc": (authored("equal_numbers"),),
            },
        ),
        _BETWEEN,
        _element_of(
            "in",
            "Element of",
            "in",
            {
                "ds_set": (manual("ex_1"),),
                "comp_set": (manual("ex_2"),),
                "sc_set": (_SET_LITERAL_MEMBER,),
                "ds_value_domain": (manual("ex_3"),),
                "comp_value_domain": (_MEMBERS,),
            },
        ),
        _element_of(
            "not_in",
            "Not element of",
            "not_in",
            {
                "ds_set": (_MEMBERS,),
                "comp_set": (_MEMBERS,),
                "sc_set": (_SET_LITERAL_MEMBER,),
                "ds_value_domain": (_MEMBERS,),
                "comp_value_domain": (_MEMBERS,),
            },
        ),
        _MATCH_CHARACTERS,
        _ISNULL,
        _EXISTS_IN,
    ),
)
