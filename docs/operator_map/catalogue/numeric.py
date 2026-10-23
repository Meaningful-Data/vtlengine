"""The numeric operators of the VTL 2.2 Reference Manual (chapter ``Numeric operators``).

A variant is one syntax form: the level of each operand (``DS`` a dataset, ``comp`` a component
inside ``calc``, ``sc`` a scalar) and, for ``round`` and ``trunc``, whether ``numDigit`` is
given, omitted or the ``_`` placeholder.
"""

from __future__ import annotations

from tool.model import Chapter, Operator, Variant, authored, manual

_CHAPTER = "Numeric operators"
_NUMBER = ("number",)
_INTEGER = ("integer",)


def _page(page: str) -> str:
    return f"{_CHAPTER}/{page}"


# --- Unary plus and unary minus ---

_UNARY_PLUS = Operator(
    id="unary_plus",
    name="Unary plus",
    symbol="+",
    page=_page("Unary plus"),
    accepts=_NUMBER,
    returns=_NUMBER,
    variants=(
        Variant("ds", "+DS", (manual("ex_1"),)),
        Variant("comp", "+comp", (manual("ex_2"),)),
        Variant("sc", "+sc", (authored(),)),
    ),
)

_UNARY_MINUS = Operator(
    id="unary_minus",
    name="Unary minus",
    symbol="-",
    page=_page("Unary minus"),
    accepts=_NUMBER,
    returns=_NUMBER,
    variants=(
        Variant("ds", "-DS", (manual("ex_1"),)),
        Variant("comp", "-comp", (manual("ex_2"),)),
        Variant("sc", "-sc", (authored(),)),
    ),
)

# --- Addition, subtraction, multiplication, division and modulo ---

_ADDITION = Operator(
    id="addition",
    name="Addition",
    symbol="+",
    page=_page("Addition"),
    accepts=_NUMBER,
    returns=_NUMBER,
    variants=(
        Variant("ds_ds", "DS + DS", (manual("ex_1"),)),
        Variant("ds_sc", "DS + sc", (manual("ex_2"),)),
        Variant("sc_ds", "sc + DS", (authored(),)),
        Variant("comp_comp", "comp + comp", (authored(),)),
        Variant("comp_sc", "comp + sc", (manual("ex_3"),)),
        Variant("sc_comp", "sc + comp", (authored(),)),
        Variant("sc_sc", "sc + sc", (authored(),)),
    ),
)

_SUBTRACTION = Operator(
    id="subtraction",
    name="Subtraction",
    symbol="-",
    page=_page("Subtraction"),
    accepts=_NUMBER,
    returns=_NUMBER,
    variants=(
        Variant("ds_ds", "DS - DS", (manual("ex_1"),)),
        Variant("ds_sc", "DS - sc", (manual("ex_2"),)),
        Variant("sc_ds", "sc - DS", (authored(),)),
        Variant("comp_comp", "comp - comp", (authored(),)),
        Variant("comp_sc", "comp - sc", (manual("ex_3"),)),
        Variant("sc_comp", "sc - comp", (authored(),)),
        Variant("sc_sc", "sc - sc", (authored(),)),
    ),
)

_MULTIPLICATION = Operator(
    id="multiplication",
    name="Multiplication",
    symbol="*",
    page=_page("Multiplication"),
    accepts=_NUMBER,
    returns=_NUMBER,
    variants=(
        Variant("ds_ds", "DS * DS", (manual("ex_1"),)),
        Variant("ds_sc", "DS * sc", (manual("ex_2"),)),
        Variant("sc_ds", "sc * DS", (authored(),)),
        Variant("comp_comp", "comp * comp", (manual("ex_3"),)),
        Variant("comp_sc", "comp * sc", (authored(),)),
        Variant("sc_comp", "sc * comp", (authored(),)),
        Variant("sc_sc", "sc * sc", (authored(),)),
    ),
)

_DIVISION = Operator(
    id="division",
    name="Division",
    symbol="/",
    page=_page("Division"),
    accepts=_NUMBER,
    returns=_NUMBER,
    variants=(
        Variant("ds_ds", "DS / DS", (manual("ex_1"),)),
        Variant("ds_sc", "DS / sc", (manual("ex_2"),)),
        Variant("sc_ds", "sc / DS", (authored(),)),
        Variant("comp_comp", "comp / comp", (manual("ex_3"),)),
        Variant("comp_sc", "comp / sc", (authored(),)),
        Variant("sc_comp", "sc / comp", (authored(),)),
        Variant("sc_sc", "sc / sc", (authored(),)),
    ),
)

_MODULO = Operator(
    id="modulo",
    name="Modulo",
    symbol="mod",
    page=_page("Modulo"),
    accepts=_NUMBER,
    returns=_NUMBER,
    variants=(
        Variant("ds_ds", "mod(DS, DS)", (manual("ex_1"),)),
        Variant("ds_sc", "mod(DS, sc)", (manual("ex_2"),)),
        Variant("sc_ds", "mod(sc, DS)", (authored(),)),
        Variant("comp_comp", "mod(comp, comp)", (authored(),)),
        Variant("comp_sc", "mod(comp, sc)", (manual("ex_3"),)),
        Variant("sc_comp", "mod(sc, comp)", (authored(),)),
        Variant(
            "sc_sc",
            "mod(sc, sc)",
            (authored(), authored("minimal_2"), authored("minimal_3")),
        ),
    ),
)

# --- Rounding and truncation ---
# ``numDigit`` omitted, or the ``_`` placeholder, makes the result an integer. The manual's
# second example of each page, ``round(Me_1)`` in a ``calc``, is the component form with no
# ``numDigit``: ``comp_no_digits``. ``numDigit`` is an integer or a component of type integer,
# so a scalar operand has the forms ``sc_digits`` and ``sc_comp_digits``. ``ds_no_digits`` has
# an example with one measure (``minimal``) and one with two (``minimal_2``), as the naming of
# the result measures differs.

_ROUND = Operator(
    id="round",
    name="Rounding",
    symbol="round",
    page=_page("Rounding"),
    accepts=("number", "integer"),
    returns=_NUMBER,
    variants=(
        Variant("ds_digits", "round(DS, n)", (manual("ex_1"),)),
        Variant(
            "ds_no_digits",
            "round(DS)",
            (authored(), authored("minimal_2")),
            returns=_INTEGER,
        ),
        Variant("ds_placeholder", "round(DS, _)", (authored(),), returns=_INTEGER),
        Variant("comp_digits", "round(comp, n)", (manual("ex_3"),)),
        Variant("comp_no_digits", "round(comp)", (manual("ex_2"),), returns=_INTEGER),
        Variant("comp_comp_digits", "round(comp, comp)", (authored(),)),
        Variant("sc_comp_digits", "round(sc, comp)", (authored(),)),
        Variant("sc_digits", "round(sc, n)", (authored(),)),
        Variant("sc_no_digits", "round(sc)", (authored(),), returns=_INTEGER),
        Variant("sc_placeholder", "round(sc, _)", (authored(),), returns=_INTEGER),
    ),
)

_TRUNC = Operator(
    id="trunc",
    name="Truncation",
    symbol="trunc",
    page=_page("Truncation"),
    accepts=("number", "integer"),
    returns=_NUMBER,
    variants=(
        Variant("ds_digits", "trunc(DS, n)", (manual("ex_1"),)),
        Variant(
            "ds_no_digits",
            "trunc(DS)",
            (authored(), authored("minimal_2")),
            returns=_INTEGER,
        ),
        Variant("ds_placeholder", "trunc(DS, _)", (authored(),), returns=_INTEGER),
        Variant("comp_digits", "trunc(comp, n)", (manual("ex_3"),)),
        Variant("comp_no_digits", "trunc(comp)", (manual("ex_2"),), returns=_INTEGER),
        Variant("comp_comp_digits", "trunc(comp, comp)", (authored(),)),
        Variant("sc_comp_digits", "trunc(sc, comp)", (authored(),)),
        Variant("sc_digits", "trunc(sc, n)", (authored(),)),
        Variant("sc_no_digits", "trunc(sc)", (authored(),), returns=_INTEGER),
        Variant("sc_placeholder", "trunc(sc, _)", (authored(),), returns=_INTEGER),
    ),
)

# --- Ceiling and floor ---
# The manual's ``ex_1`` has two measures. ``minimal`` has one, to see whether a number measure
# that becomes an integer keeps its name.

_CEIL = Operator(
    id="ceil",
    name="Ceiling",
    symbol="ceil",
    page=_page("Ceiling"),
    accepts=_NUMBER,
    returns=_INTEGER,
    variants=(
        Variant("ds", "ceil(DS)", (manual("ex_1"), authored())),
        Variant("comp", "ceil(comp)", (manual("ex_2"),)),
        Variant("sc", "ceil(sc)", (authored(),)),
    ),
)

_FLOOR = Operator(
    id="floor",
    name="Floor",
    symbol="floor",
    page=_page("Floor"),
    accepts=_NUMBER,
    returns=_INTEGER,
    variants=(
        Variant("ds", "floor(DS)", (manual("ex_1"), authored())),
        Variant("comp", "floor(comp)", (manual("ex_2"),)),
        Variant("sc", "floor(sc)", (authored(),)),
    ),
)

# --- Absolute value, exponential, natural logarithm and square root ---

_ABS = Operator(
    id="abs",
    name="Absolute value",
    symbol="abs",
    page=_page("Absolute value"),
    accepts=_NUMBER,
    returns=_NUMBER,
    variants=(
        Variant("ds", "abs(DS)", (manual("ex_1"),)),
        Variant("comp", "abs(comp)", (manual("ex_2"),)),
        Variant("sc", "abs(sc)", (authored(),)),
    ),
)

_EXP = Operator(
    id="exp",
    name="Exponential",
    symbol="exp",
    page=_page("Exponential"),
    accepts=_NUMBER,
    returns=_NUMBER,
    variants=(
        Variant("ds", "exp(DS)", (manual("ex_1"),)),
        Variant("comp", "exp(comp)", (manual("ex_2"),)),
        Variant("sc", "exp(sc)", (authored(),)),
    ),
)

_LN = Operator(
    id="ln",
    name="Natural logarithm",
    symbol="ln",
    page=_page("Natural logarithm"),
    accepts=_NUMBER,
    returns=_NUMBER,
    variants=(
        Variant("ds", "ln(DS)", (manual("ex_1"),)),
        Variant("comp", "ln(comp)", (manual("ex_2"),)),
        Variant("sc", "ln(sc)", (authored(),)),
    ),
)

_SQRT = Operator(
    id="sqrt",
    name="Square root",
    symbol="sqrt",
    page=_page("Square root"),
    accepts=_NUMBER,
    returns=_NUMBER,
    variants=(
        Variant("ds", "sqrt(DS)", (manual("ex_1"),)),
        Variant("comp", "sqrt(comp)", (manual("ex_2"),)),
        Variant("sc", "sqrt(sc)", (authored(),)),
    ),
)

# --- Power and logarithm ---
# The examples of the manual call ``log(number, base)``, e.g. ``log(1024, 2)`` gives 10, so
# the authored examples do the same.

_POWER = Operator(
    id="power",
    name="Power",
    symbol="power",
    page=_page("Power"),
    accepts=("number", "integer"),
    returns=_NUMBER,
    variants=(
        Variant("ds_sc", "power(DS, sc)", (manual("ex_1"),)),
        Variant("comp_sc", "power(comp, sc)", (manual("ex_2"),)),
        Variant("comp_comp", "power(comp, comp)", (authored(),)),
        Variant("sc_comp", "power(sc, comp)", (authored(),)),
        Variant("sc_sc", "power(sc, sc)", (authored(),)),
    ),
)

_LOG = Operator(
    id="log",
    name="Logarithm",
    symbol="log",
    page=_page("Logarithm"),
    accepts=("number", "integer"),
    returns=_NUMBER,
    variants=(
        Variant("ds_sc", "log(DS, sc)", (manual("ex_1"),)),
        Variant("comp_sc", "log(comp, sc)", (manual("ex_2"),)),
        Variant("comp_comp", "log(comp, comp)", (authored(),)),
        Variant("sc_comp", "log(sc, comp)", (authored(),)),
        Variant("sc_sc", "log(sc, sc)", (authored(),)),
    ),
)

# --- Random ---
# The values are not fixed: an example is checked by the property that they are in [0, 1) and
# the same on every run, and the two engines must give the same ones. The property is checked
# on every value that is not an identifier, so it fits an example only when all of them are
# draws. The manual's ``ex_2`` of the component form keeps the seed column ``Me_1`` next to the
# new measure, so it is not checked by property (its values are the manual's illustration, and
# no engine can match them); ``minimal`` overwrites the seed measure and is.

_RANDOM = Operator(
    id="random",
    name="Random",
    symbol="random",
    page=_page("Random"),
    accepts=("number", "integer"),
    returns=_NUMBER,
    variants=(
        Variant("ds", "random(DS, n)", (manual("ex_1", check="random"),)),
        Variant("comp", "random(comp, n)", (manual("ex_2"), authored(check="random"))),
        Variant("sc", "random(sc, n)", (authored(check="random"),)),
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
