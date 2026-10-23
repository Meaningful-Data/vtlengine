"""The string operators of the VTL 2.2 Reference Manual (chapter "String operators")."""

from __future__ import annotations

from tool.model import Chapter, Operator, Variant, authored, manual

_PAGE = "String operators"

_CONCAT = Operator(
    id="concat",
    name="String concatenation",
    symbol="||",
    page=f"{_PAGE}/String concatenation",
    accepts=("string",),
    returns=("string",),
    variants=(
        Variant("ds_ds", "DS || DS", (manual("ex_1"),)),
        Variant("ds_sc", "DS || sc", (authored(),)),
        Variant("sc_ds", "sc || DS", (authored(),)),
        Variant("comp_comp", "comp || comp", (authored(),)),
        Variant("comp_sc", "comp || sc", (manual("ex_2"),)),
        Variant("sc_comp", "sc || comp", (authored(),)),
        Variant("sc_sc", "sc || sc", (authored(),)),
    ),
)

# trim, ltrim and rtrim share one manual page, and upper and lower share another one. The names
# say the operator as well, so that the nodes of the type flows are told apart.
_TRIM = Operator(
    id="trim",
    name="Whitespace removal (trim)",
    symbol="trim",
    page=f"{_PAGE}/Whitespace removal",
    accepts=("string",),
    returns=("string",),
    variants=(
        Variant("ds", "trim(DS)", (authored(),)),
        Variant("comp", "trim(comp)", (authored(),)),
        Variant("sc", "trim(sc)", (authored(),)),
    ),
)

_LTRIM = Operator(
    id="ltrim",
    name="Whitespace removal (ltrim)",
    symbol="ltrim",
    page=f"{_PAGE}/Whitespace removal",
    accepts=("string",),
    returns=("string",),
    variants=(
        Variant("ds", "ltrim(DS)", (authored(),)),
        Variant("comp", "ltrim(comp)", (authored(),)),
        Variant("sc", "ltrim(sc)", (authored(),)),
    ),
)

_RTRIM = Operator(
    id="rtrim",
    name="Whitespace removal (rtrim)",
    symbol="rtrim",
    page=f"{_PAGE}/Whitespace removal",
    accepts=("string",),
    returns=("string",),
    variants=(
        Variant("ds", "rtrim(DS)", (manual("ex_1"),)),
        Variant("comp", "rtrim(comp)", (manual("ex_2"),)),
        Variant("sc", "rtrim(sc)", (authored(),)),
    ),
)

_UPPER = Operator(
    id="upper",
    name="Character case conversion (upper)",
    symbol="upper",
    page=f"{_PAGE}/Character case conversion",
    accepts=("string",),
    returns=("string",),
    variants=(
        Variant("ds", "upper(DS)", (manual("ex_1"),)),
        Variant("comp", "upper(comp)", (manual("ex_2"),)),
        Variant("sc", "upper(sc)", (authored(),)),
    ),
)

_LOWER = Operator(
    id="lower",
    name="Character case conversion (lower)",
    symbol="lower",
    page=f"{_PAGE}/Character case conversion",
    accepts=("string",),
    returns=("string",),
    variants=(
        Variant("ds", "lower(DS)", (authored(),)),
        Variant("comp", "lower(comp)", (authored(),)),
        Variant("sc", "lower(sc)", (authored(),)),
    ),
)

_SUBSTR = Operator(
    id="substr",
    name="Sub-string extraction",
    symbol="substr",
    page=f"{_PAGE}/Sub-string extraction",
    accepts=("string", "integer"),
    returns=("string",),
    variants=(
        Variant("ds_start_len", "substr(DS, start, length)", (manual("ex_2"),)),
        Variant("ds_start", "substr(DS, start)", (manual("ex_1"),)),
        Variant("ds_start_placeholder", "substr(DS, start, _)", (authored(),)),
        Variant(
            "ds_placeholder_len",
            "substr(DS, _, length)",
            (authored(),),
        ),
        Variant("ds_noargs", "substr(DS)", (authored(),)),
        Variant("comp_start_len", "substr(comp, start, length)", (manual("ex_3"),)),
        Variant("comp_params", "substr(comp, comp, comp)", (authored(),)),
        Variant(
            "sc_start_len",
            "substr(sc, start, length)",
            (authored(), authored("minimal_2"), authored("minimal_3")),
        ),
    ),
)

_REPLACE = Operator(
    id="replace",
    name="String pattern replacement",
    symbol="replace",
    page=f"{_PAGE}/String pattern replacement",
    accepts=("string",),
    returns=("string",),
    variants=(
        Variant("ds_two", "replace(DS, pattern1, pattern2)", (manual("ex_1"),)),
        Variant("ds_one", "replace(DS, pattern1)", (authored(),)),
        Variant("ds_one_placeholder", "replace(DS, pattern1, _)", (authored(),)),
        Variant("comp_two", "replace(comp, pattern1, pattern2)", (manual("ex_2"),)),
        Variant("comp_patterns", "replace(comp, comp, comp)", (authored(),)),
        Variant("sc_two", "replace(sc, pattern1, pattern2)", (authored(),)),
    ),
)

_INSTR = Operator(
    id="instr",
    name="String pattern location",
    symbol="instr",
    page=f"{_PAGE}/String pattern location",
    accepts=("string", "integer"),
    returns=("integer",),
    variants=(
        Variant("ds_pattern", "instr(DS, pattern)", (manual("ex_1"),)),
        Variant("ds_pattern_start", "instr(DS, pattern, start)", (authored(),)),
        Variant(
            "ds_pattern_start_occ",
            "instr(DS, pattern, start, occurrence)",
            (authored(),),
        ),
        Variant("ds_placeholder_occ", "instr(DS, pattern, _, occurrence)", (authored(),)),
        Variant(
            "ds_pattern_start_placeholder",
            "instr(DS, pattern, start, _)",
            (authored(),),
        ),
        Variant("comp_pattern", "instr(comp, pattern)", (manual("ex_2"), manual("ex_3"))),
        Variant(
            "comp_params",
            "instr(comp, comp, comp, comp)",
            (authored(), authored("minimal_2")),
        ),
        Variant("sc_pattern", "instr(sc, pattern)", (authored(),)),
    ),
)

_LENGTH = Operator(
    id="length",
    name="String length",
    symbol="length",
    page=f"{_PAGE}/String length",
    accepts=("string",),
    returns=("integer",),
    variants=(
        Variant("ds", "length(DS)", (manual("ex_1"),)),
        Variant("comp", "length(comp)", (manual("ex_2"), manual("ex_3"))),
        Variant("sc", "length(sc)", (authored(), authored("minimal_2"))),
    ),
)

# The manual's examples of string_distance run the distance of two datasets (levenshtein and
# damerau_levenshtein) and of a component and a scalar (hamming), so each of those has a variant
# of its own next to the ones that hold the scalar forms. The operands can be a dataset, a
# component or a scalar (string1 and string2 are typed that way), and the pairs of operand levels
# are crossed on levenshtein only, with the component and scalar pair covered by hamming.
_STRING_DISTANCE = Operator(
    id="string_distance",
    name="String distances",
    symbol="string_distance",
    page=f"{_PAGE}/String distances",
    accepts=("string",),
    returns=("number",),
    variants=(
        Variant(
            "levenshtein_ds_ds",
            "string_distance(levenshtein, DS, DS)",
            (manual("ex_1"),),
        ),
        Variant(
            "levenshtein_ds_sc",
            "string_distance(levenshtein, DS, sc)",
            (authored(),),
        ),
        Variant(
            "levenshtein_sc_ds",
            "string_distance(levenshtein, sc, DS)",
            (authored(),),
        ),
        Variant(
            "levenshtein_comp_comp",
            "string_distance(levenshtein, comp, comp)",
            (authored(),),
        ),
        Variant(
            "levenshtein_sc_comp",
            "string_distance(levenshtein, sc, comp)",
            (authored(),),
        ),
        Variant(
            "levenshtein_sc",
            "string_distance(levenshtein, sc, sc)",
            (authored(),),
        ),
        Variant(
            "damerau_levenshtein_ds_ds",
            "string_distance(damerau_levenshtein, DS, DS)",
            (manual("ex_2"),),
        ),
        Variant(
            "damerau_levenshtein_sc",
            "string_distance(damerau_levenshtein, sc, sc)",
            (authored(),),
        ),
        Variant(
            "hamming_comp_sc",
            "string_distance(hamming, comp, sc)",
            (manual("ex_3"),),
        ),
        Variant(
            "hamming_sc",
            "string_distance(hamming, sc, sc)",
            (authored(),),
        ),
        Variant(
            "jaro_winkler_sc",
            "string_distance(jaro_winkler, sc, sc)",
            (authored(), authored("minimal_2")),
        ),
    ),
)

CHAPTER = Chapter(
    id="string",
    title="String operators",
    operators=(
        _CONCAT,
        _TRIM,
        _LTRIM,
        _RTRIM,
        _UPPER,
        _LOWER,
        _SUBSTR,
        _REPLACE,
        _INSTR,
        _LENGTH,
        _STRING_DISTANCE,
    ),
)
