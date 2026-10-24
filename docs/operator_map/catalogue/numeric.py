"""The numeric operators of the VTL 2.2 Reference Manual (chapter ``Numeric operators``).

A variant is one syntax form: the level of each operand (``DS`` a dataset, ``comp`` a component
inside ``calc``, ``sc`` a scalar) and, for ``round`` and ``trunc``, whether ``numDigit`` is
given, omitted or the ``_`` placeholder. Each form says what it takes and gives, by the "Input
parameters type" and "Result type" of its page.
"""

from __future__ import annotations

from tool.model import Chapter, Operator, Variant, authored, manual

_CHAPTER = "Numeric operators"

# What a form takes and gives. The level of its operands, a dataset, a component or a scalar, is
# its kind and not a type, so ``op(DS)`` has the types of ``op(sc)``.
_NUMBER = ("number",)
_NUMBER_INTEGER = ("number", "integer")  # the operand, then an integer numDigit, num or index
_INTEGER = ("integer",)  # ceil and floor, and round and trunc with no numDigit, or with ``_``

# The types of an operator: those of its forms, and ``dataset`` for its form ``op(DS)``.
_OP_NUMBER = (*_NUMBER, "dataset")
_OP_NUMBER_INTEGER = (*_NUMBER_INTEGER, "dataset")
_OP_INTEGER = (*_INTEGER, "dataset")


def _page(page: str) -> str:
    return f"{_CHAPTER}/{page}"


# --- Unary plus and unary minus ---

_UNARY_PLUS = Operator(
    id="unary_plus",
    name="Unary plus",
    symbol="+",
    page=_page("Unary plus"),
    accepts=_OP_NUMBER,
    returns=_OP_NUMBER,
    variants=(
        Variant("ds", "+DS", (manual("ex_1"),), takes=_NUMBER, gives=_NUMBER),
        Variant("comp", "+comp", (manual("ex_2"),), takes=_NUMBER, gives=_NUMBER),
        Variant("sc", "+sc", (authored("negative_operand"),), takes=_NUMBER, gives=_NUMBER),
    ),
)

_UNARY_MINUS = Operator(
    id="unary_minus",
    name="Unary minus",
    symbol="-",
    page=_page("Unary minus"),
    accepts=_OP_NUMBER,
    returns=_OP_NUMBER,
    variants=(
        Variant("ds", "-DS", (manual("ex_1"),), takes=_NUMBER, gives=_NUMBER),
        Variant("comp", "-comp", (manual("ex_2"),), takes=_NUMBER, gives=_NUMBER),
        Variant("sc", "-sc", (authored("negative_operand"),), takes=_NUMBER, gives=_NUMBER),
    ),
)

# --- Addition, subtraction, multiplication, division and modulo ---

_ADDITION = Operator(
    id="addition",
    name="Addition",
    symbol="+",
    page=_page("Addition"),
    accepts=_OP_NUMBER,
    returns=_OP_NUMBER,
    variants=(
        Variant("ds_ds", "DS + DS", (manual("ex_1"),), takes=_NUMBER, gives=_NUMBER),
        Variant("ds_sc", "DS + sc", (manual("ex_2"),), takes=_NUMBER, gives=_NUMBER),
        Variant(
            "sc_ds",
            "sc + DS",
            (authored("integer_and_number_measures"),),
            takes=_NUMBER,
            gives=_NUMBER,
        ),
        Variant(
            "comp_comp",
            "comp + comp",
            (authored("integer_plus_number"),),
            takes=_NUMBER,
            gives=_NUMBER,
        ),
        Variant("comp_sc", "comp + sc", (manual("ex_3"),), takes=_NUMBER, gives=_NUMBER),
        Variant(
            "sc_comp",
            "sc + comp",
            (authored("integer_plus_integer"),),
            takes=_NUMBER,
            gives=_NUMBER,
        ),
        Variant(
            "sc_sc",
            "sc + sc",
            (authored("integer_plus_integer"),),
            takes=_NUMBER,
            gives=_NUMBER,
        ),
    ),
)

_SUBTRACTION = Operator(
    id="subtraction",
    name="Subtraction",
    symbol="-",
    page=_page("Subtraction"),
    accepts=_OP_NUMBER,
    returns=_OP_NUMBER,
    variants=(
        Variant("ds_ds", "DS - DS", (manual("ex_1"),), takes=_NUMBER, gives=_NUMBER),
        Variant("ds_sc", "DS - sc", (manual("ex_2"),), takes=_NUMBER, gives=_NUMBER),
        Variant("sc_ds", "sc - DS", (authored("scalar_minuend"),), takes=_NUMBER, gives=_NUMBER),
        Variant(
            "comp_comp",
            "comp - comp",
            (authored("integer_minus_number"),),
            takes=_NUMBER,
            gives=_NUMBER,
        ),
        Variant("comp_sc", "comp - sc", (manual("ex_3"),), takes=_NUMBER, gives=_NUMBER),
        Variant(
            "sc_comp", "sc - comp", (authored("scalar_minuend"),), takes=_NUMBER, gives=_NUMBER
        ),
        Variant("sc_sc", "sc - sc", (authored("negative_result"),), takes=_NUMBER, gives=_NUMBER),
    ),
)

_MULTIPLICATION = Operator(
    id="multiplication",
    name="Multiplication",
    symbol="*",
    page=_page("Multiplication"),
    accepts=_OP_NUMBER,
    returns=_OP_NUMBER,
    variants=(
        Variant("ds_ds", "DS * DS", (manual("ex_1"),), takes=_NUMBER, gives=_NUMBER),
        Variant("ds_sc", "DS * sc", (manual("ex_2"),), takes=_NUMBER, gives=_NUMBER),
        Variant(
            "sc_ds",
            "sc * DS",
            (authored("integer_and_number_measures"),),
            takes=_NUMBER,
            gives=_NUMBER,
        ),
        Variant("comp_comp", "comp * comp", (manual("ex_3"),), takes=_NUMBER, gives=_NUMBER),
        Variant(
            "comp_sc",
            "comp * sc",
            (authored("integer_times_integer"),),
            takes=_NUMBER,
            gives=_NUMBER,
        ),
        Variant(
            "sc_comp",
            "sc * comp",
            (authored("integer_times_number"),),
            takes=_NUMBER,
            gives=_NUMBER,
        ),
        Variant(
            "sc_sc",
            "sc * sc",
            (authored("integer_times_integer"),),
            takes=_NUMBER,
            gives=_NUMBER,
        ),
    ),
)

# The Division page makes a zero divisor a run-time error: ``zero_divisor`` expects an error and
# no value, with the code that ``docs/error_messages.rst`` gives to "Scalar division by Zero".
_DIVISION = Operator(
    id="division",
    name="Division",
    symbol="/",
    page=_page("Division"),
    accepts=_OP_NUMBER,
    returns=_OP_NUMBER,
    variants=(
        Variant("ds_ds", "DS / DS", (manual("ex_1"),), takes=_NUMBER, gives=_NUMBER),
        Variant("ds_sc", "DS / sc", (manual("ex_2"),), takes=_NUMBER, gives=_NUMBER),
        Variant("sc_ds", "sc / DS", (authored("scalar_dividend"),), takes=_NUMBER, gives=_NUMBER),
        Variant("comp_comp", "comp / comp", (manual("ex_3"),), takes=_NUMBER, gives=_NUMBER),
        Variant("comp_sc", "comp / sc", (authored("number_result"),), takes=_NUMBER, gives=_NUMBER),
        Variant(
            "sc_comp",
            "sc / comp",
            (authored("scalar_dividend"),),
            takes=_NUMBER,
            gives=_NUMBER,
        ),
        Variant(
            "sc_sc",
            "sc / sc",
            (authored("number_result"), authored("zero_divisor")),
            takes=_NUMBER,
            gives=_NUMBER,
        ),
    ),
)

_MODULO = Operator(
    id="modulo",
    name="Modulo",
    symbol="mod",
    page=_page("Modulo"),
    accepts=_OP_NUMBER,
    returns=_OP_NUMBER,
    variants=(
        Variant("ds_ds", "mod(DS, DS)", (manual("ex_1"),), takes=_NUMBER, gives=_NUMBER),
        Variant("ds_sc", "mod(DS, sc)", (manual("ex_2"),), takes=_NUMBER, gives=_NUMBER),
        Variant(
            "sc_ds",
            "mod(sc, DS)",
            (authored("integer_and_number_measures"),),
            takes=_NUMBER,
            gives=_NUMBER,
        ),
        Variant(
            "comp_comp",
            "mod(comp, comp)",
            (authored("integer_mod_number"),),
            takes=_NUMBER,
            gives=_NUMBER,
        ),
        Variant("comp_sc", "mod(comp, sc)", (manual("ex_3"),), takes=_NUMBER, gives=_NUMBER),
        Variant(
            "sc_comp",
            "mod(sc, comp)",
            (authored("integer_mod_integer"),),
            takes=_NUMBER,
            gives=_NUMBER,
        ),
        Variant(
            "sc_sc",
            "mod(sc, sc)",
            (
                authored("positive_operands"),
                authored("negative_divisor"),
                authored("zero_divisor"),
            ),
            takes=_NUMBER,
            gives=_NUMBER,
        ),
    ),
)

# --- Rounding and truncation ---
# ``numDigit`` omitted, or the ``_`` placeholder, makes the result an integer. The manual's
# second example of each page, ``round(Me_1)`` in a ``calc``, is the component form with no
# ``numDigit``: ``comp_no_digits``. ``numDigit`` is an integer or a component of type integer,
# so a scalar operand has the forms ``sc_digits`` and ``sc_comp_digits``. ``ds_no_digits`` has
# an example with one measure (``one_number_measure``) and one with two
# (``two_number_measures``), as the naming of the result measures differs.

_ROUND = Operator(
    id="round",
    name="Rounding",
    symbol="round",
    page=_page("Rounding"),
    accepts=_OP_NUMBER_INTEGER,
    returns=_OP_NUMBER,
    variants=(
        Variant(
            "ds_digits", "round(DS, n)", (manual("ex_1"),), takes=_NUMBER_INTEGER, gives=_NUMBER
        ),
        Variant(
            "ds_no_digits",
            "round(DS)",
            (authored("one_number_measure"), authored("two_number_measures")),
            takes=_NUMBER,
            gives=_INTEGER,
        ),
        Variant(
            "ds_placeholder",
            "round(DS, _)",
            (authored("placeholder_for_digits"),),
            takes=_NUMBER,
            gives=_INTEGER,
        ),
        Variant(
            "comp_digits", "round(comp, n)", (manual("ex_3"),), takes=_NUMBER_INTEGER, gives=_NUMBER
        ),
        Variant("comp_no_digits", "round(comp)", (manual("ex_2"),), takes=_NUMBER, gives=_INTEGER),
        Variant(
            "comp_placeholder",
            "round(comp, _)",
            (authored("placeholder_for_digits"),),
            takes=_NUMBER,
            gives=_INTEGER,
        ),
        Variant(
            "comp_comp_digits",
            "round(comp, comp)",
            (authored("digits_from_component"),),
            takes=_NUMBER_INTEGER,
            gives=_NUMBER,
        ),
        Variant(
            "sc_comp_digits",
            "round(sc, comp)",
            (authored("digits_from_component"),),
            takes=_NUMBER_INTEGER,
            gives=_NUMBER,
        ),
        Variant(
            "sc_digits",
            "round(sc, n)",
            (authored("digits_given"),),
            takes=_NUMBER_INTEGER,
            gives=_NUMBER,
        ),
        Variant(
            "sc_no_digits",
            "round(sc)",
            (authored("digits_omitted"),),
            takes=_NUMBER,
            gives=_INTEGER,
        ),
        Variant(
            "sc_placeholder",
            "round(sc, _)",
            (authored("placeholder_for_digits"),),
            takes=_NUMBER,
            gives=_INTEGER,
        ),
    ),
)

_TRUNC = Operator(
    id="trunc",
    name="Truncation",
    symbol="trunc",
    page=_page("Truncation"),
    accepts=_OP_NUMBER_INTEGER,
    returns=_OP_NUMBER,
    variants=(
        Variant(
            "ds_digits", "trunc(DS, n)", (manual("ex_1"),), takes=_NUMBER_INTEGER, gives=_NUMBER
        ),
        Variant(
            "ds_no_digits",
            "trunc(DS)",
            (authored("one_number_measure"), authored("two_number_measures")),
            takes=_NUMBER,
            gives=_INTEGER,
        ),
        Variant(
            "ds_placeholder",
            "trunc(DS, _)",
            (authored("placeholder_for_digits"),),
            takes=_NUMBER,
            gives=_INTEGER,
        ),
        Variant(
            "comp_digits", "trunc(comp, n)", (manual("ex_3"),), takes=_NUMBER_INTEGER, gives=_NUMBER
        ),
        Variant("comp_no_digits", "trunc(comp)", (manual("ex_2"),), takes=_NUMBER, gives=_INTEGER),
        Variant(
            "comp_placeholder",
            "trunc(comp, _)",
            (authored("placeholder_for_digits"),),
            takes=_NUMBER,
            gives=_INTEGER,
        ),
        Variant(
            "comp_comp_digits",
            "trunc(comp, comp)",
            (authored("digits_from_component"),),
            takes=_NUMBER_INTEGER,
            gives=_NUMBER,
        ),
        Variant(
            "sc_comp_digits",
            "trunc(sc, comp)",
            (authored("digits_from_component"),),
            takes=_NUMBER_INTEGER,
            gives=_NUMBER,
        ),
        Variant(
            "sc_digits",
            "trunc(sc, n)",
            (authored("digits_given"),),
            takes=_NUMBER_INTEGER,
            gives=_NUMBER,
        ),
        Variant(
            "sc_no_digits",
            "trunc(sc)",
            (authored("digits_omitted"),),
            takes=_NUMBER,
            gives=_INTEGER,
        ),
        Variant(
            "sc_placeholder",
            "trunc(sc, _)",
            (authored("placeholder_for_digits"),),
            takes=_NUMBER,
            gives=_INTEGER,
        ),
    ),
)

# --- Ceiling and floor ---
# The manual's ``ex_1`` has two measures. ``one_number_measure`` has one, to see whether a number
# measure that becomes an integer keeps its name.

_CEIL = Operator(
    id="ceil",
    name="Ceiling",
    symbol="ceil",
    page=_page("Ceiling"),
    accepts=_OP_NUMBER,
    returns=_OP_INTEGER,
    variants=(
        Variant(
            "ds",
            "ceil(DS)",
            (manual("ex_1"), authored("one_number_measure")),
            takes=_NUMBER,
            gives=_INTEGER,
        ),
        Variant("comp", "ceil(comp)", (manual("ex_2"),), takes=_NUMBER, gives=_INTEGER),
        Variant("sc", "ceil(sc)", (authored("positive_fraction"),), takes=_NUMBER, gives=_INTEGER),
    ),
)

_FLOOR = Operator(
    id="floor",
    name="Floor",
    symbol="floor",
    page=_page("Floor"),
    accepts=_OP_NUMBER,
    returns=_OP_INTEGER,
    variants=(
        Variant(
            "ds",
            "floor(DS)",
            (manual("ex_1"), authored("one_number_measure")),
            takes=_NUMBER,
            gives=_INTEGER,
        ),
        Variant("comp", "floor(comp)", (manual("ex_2"),), takes=_NUMBER, gives=_INTEGER),
        Variant("sc", "floor(sc)", (authored("negative_fraction"),), takes=_NUMBER, gives=_INTEGER),
    ),
)

# --- Absolute value, exponential, natural logarithm and square root ---

_ABS = Operator(
    id="abs",
    name="Absolute value",
    symbol="abs",
    page=_page("Absolute value"),
    accepts=_OP_NUMBER,
    returns=_OP_NUMBER,
    variants=(
        Variant("ds", "abs(DS)", (manual("ex_1"),), takes=_NUMBER, gives=_NUMBER),
        Variant("comp", "abs(comp)", (manual("ex_2"),), takes=_NUMBER, gives=_NUMBER),
        Variant("sc", "abs(sc)", (authored("negative_number"),), takes=_NUMBER, gives=_NUMBER),
    ),
)

_EXP = Operator(
    id="exp",
    name="Exponential",
    symbol="exp",
    page=_page("Exponential"),
    accepts=_OP_NUMBER,
    returns=_OP_NUMBER,
    variants=(
        Variant("ds", "exp(DS)", (manual("ex_1"),), takes=_NUMBER, gives=_NUMBER),
        Variant("comp", "exp(comp)", (manual("ex_2"),), takes=_NUMBER, gives=_NUMBER),
        Variant("sc", "exp(sc)", (authored("integer_operand"),), takes=_NUMBER, gives=_NUMBER),
    ),
)

_LN = Operator(
    id="ln",
    name="Natural logarithm",
    symbol="ln",
    page=_page("Natural logarithm"),
    accepts=_OP_NUMBER,
    returns=_OP_NUMBER,
    variants=(
        Variant("ds", "ln(DS)", (manual("ex_1"),), takes=_NUMBER, gives=_NUMBER),
        Variant("comp", "ln(comp)", (manual("ex_2"),), takes=_NUMBER, gives=_NUMBER),
        Variant("sc", "ln(sc)", (authored("integer_operand"),), takes=_NUMBER, gives=_NUMBER),
    ),
)

_SQRT = Operator(
    id="sqrt",
    name="Square root",
    symbol="sqrt",
    page=_page("Square root"),
    accepts=_OP_NUMBER,
    returns=_OP_NUMBER,
    variants=(
        Variant("ds", "sqrt(DS)", (manual("ex_1"),), takes=_NUMBER, gives=_NUMBER),
        Variant("comp", "sqrt(comp)", (manual("ex_2"),), takes=_NUMBER, gives=_NUMBER),
        Variant("sc", "sqrt(sc)", (authored("integer_operand"),), takes=_NUMBER, gives=_NUMBER),
    ),
)

# --- Power and logarithm ---
# The examples of the manual call ``log(number, base)``, e.g. ``log(1024, 2)`` gives 10, so
# the authored examples do the same. The page types the exponent of ``power`` as a number, and
# the second operand of ``log`` as an integer.

_POWER = Operator(
    id="power",
    name="Power",
    symbol="power",
    page=_page("Power"),
    accepts=_OP_NUMBER,
    returns=_OP_NUMBER,
    variants=(
        Variant("ds_sc", "power(DS, sc)", (manual("ex_1"),), takes=_NUMBER, gives=_NUMBER),
        Variant("comp_sc", "power(comp, sc)", (manual("ex_2"),), takes=_NUMBER, gives=_NUMBER),
        Variant(
            "comp_comp",
            "power(comp, comp)",
            (authored("component_exponent"),),
            takes=_NUMBER,
            gives=_NUMBER,
        ),
        Variant(
            "sc_comp",
            "power(sc, comp)",
            (authored("component_exponent"),),
            takes=_NUMBER,
            gives=_NUMBER,
        ),
        Variant(
            "sc_sc",
            "power(sc, sc)",
            (authored("integer_operands"),),
            takes=_NUMBER,
            gives=_NUMBER,
        ),
    ),
)

_LOG = Operator(
    id="log",
    name="Logarithm",
    symbol="log",
    page=_page("Logarithm"),
    accepts=_OP_NUMBER_INTEGER,
    returns=_OP_NUMBER,
    variants=(
        Variant("ds_sc", "log(DS, sc)", (manual("ex_1"),), takes=_NUMBER_INTEGER, gives=_NUMBER),
        Variant(
            "comp_sc", "log(comp, sc)", (manual("ex_2"),), takes=_NUMBER_INTEGER, gives=_NUMBER
        ),
        Variant(
            "comp_comp",
            "log(comp, comp)",
            (authored("number_then_base"),),
            takes=_NUMBER_INTEGER,
            gives=_NUMBER,
        ),
        Variant(
            "sc_comp",
            "log(sc, comp)",
            (authored("component_base"),),
            takes=_NUMBER_INTEGER,
            gives=_NUMBER,
        ),
        Variant(
            "sc_sc",
            "log(sc, sc)",
            (authored("integer_operands"),),
            takes=_NUMBER_INTEGER,
            gives=_NUMBER,
        ),
    ),
)

# --- Random ---
# The values are not fixed: an example is checked by the property that they are in [0, 1) and
# the same on every run, and the two engines must give the same ones. The property is checked
# on the columns that are empty in the expected file, and the columns that have values there
# are compared as usual. The manual's examples print illustrative draws, which no engine can
# match: they are blanked (an adaptation), and the seed that ``ex_2`` keeps next to its draws
# stays in the file and is compared.

_RANDOM_ADAPTATION = (
    "the manual's illustrative random values are not reproducible; the random column is "
    "property-checked"
)

_RANDOM = Operator(
    id="random",
    name="Random",
    symbol="random",
    page=_page("Random"),
    accepts=_OP_NUMBER_INTEGER,
    returns=_OP_NUMBER,
    variants=(
        Variant(
            "ds",
            "random(DS, n)",
            (manual("ex_1", check="random", adaptation=_RANDOM_ADAPTATION),),
            takes=_NUMBER_INTEGER,
            gives=_NUMBER,
        ),
        Variant(
            "comp",
            "random(comp, n)",
            (
                manual("ex_2", check="random", adaptation=_RANDOM_ADAPTATION),
                authored("seed_replaced_by_draws", check="random"),
            ),
            takes=_NUMBER_INTEGER,
            gives=_NUMBER,
        ),
        Variant(
            "sc",
            "random(sc, n)",
            (authored("scalar_seed", check="random"),),
            takes=_NUMBER_INTEGER,
            gives=_NUMBER,
        ),
    ),
)

CHAPTER = Chapter(
    id="numeric",
    title=_CHAPTER,
    operators=(
        _UNARY_PLUS,
        _UNARY_MINUS,
        _ADDITION,
        _SUBTRACTION,
        _MULTIPLICATION,
        _DIVISION,
        _MODULO,
        _ROUND,
        _TRUNC,
        _CEIL,
        _FLOOR,
        _ABS,
        _EXP,
        _LN,
        _SQRT,
        _POWER,
        _LOG,
        _RANDOM,
    ),
)
