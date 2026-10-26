"""The string operators of the VTL 2.2 Reference Manual (chapter ``String operators``).

A variant is one syntax form: the level of each operand (``DS`` a dataset, ``comp`` a component
inside ``calc``, ``sc`` a scalar) and, for ``substr``, ``replace`` and ``instr``, which optional
parameters are given, omitted or the ``_`` placeholder. Each form says what it takes and gives,
by the "Input parameters type" and "Result type" of its page.
"""

from __future__ import annotations

from tool.model import Chapter, Operator, Variant, authored, manual

_PAGE = "String operators"

# What a form takes and gives. The level of its operands, a dataset, a component or a scalar, is
# its kind and not a type, so ``op(DS)`` has the types of ``op(sc)``.
_STRING = ("string",)
_STRING_INTEGER = ("string", "integer")  # the operand, then an integer start, length or occurrence
_INTEGER = ("integer",)  # the result of length and instr
_NUMBER = ("number",)  # the result of string_distance

# The types of an operator: those of its forms, and ``dataset`` for its form ``op(DS)``.
_OP_STRING = (*_STRING, "dataset")
_OP_STRING_INTEGER = (*_STRING_INTEGER, "dataset")
_OP_INTEGER = (*_INTEGER, "dataset")
_OP_NUMBER = (*_NUMBER, "dataset")

_CONCAT = Operator(
    id="concat",
    name="String concatenation",
    symbol="||",
    page=f"{_PAGE}/String concatenation",
    accepts=_OP_STRING,
    returns=_OP_STRING,
    variants=(
        Variant("ds_ds", "DS || DS", (manual("ex_1"),), takes=_STRING, gives=_STRING),
        Variant(
            "ds_sc",
            "DS || sc",
            (authored("suffix_and_null_measure"),),
            takes=_STRING,
            gives=_STRING,
        ),
        Variant(
            "sc_ds",
            "sc || DS",
            (authored("prefix_and_null_measure"),),
            takes=_STRING,
            gives=_STRING,
        ),
        Variant(
            "comp_comp",
            "comp || comp",
            (authored("null_right_operand"),),
            takes=_STRING,
            gives=_STRING,
        ),
        Variant("comp_sc", "comp || sc", (manual("ex_2"),), takes=_STRING, gives=_STRING),
        Variant(
            "sc_comp",
            "sc || comp",
            (authored("prefix_and_null_component"),),
            takes=_STRING,
            gives=_STRING,
        ),
        Variant(
            "sc_sc",
            "sc || sc",
            (authored("two_strings"),),
            takes=_STRING,
            gives=_STRING,
        ),
    ),
)

# trim, ltrim and rtrim share one manual page, and upper and lower share another one. Each
# operator gets its own descriptive name, as the comparison and time operators do.
_TRIM = Operator(
    id="trim",
    name="Whitespace removal",
    symbol="trim",
    page=f"{_PAGE}/Whitespace removal",
    accepts=_OP_STRING,
    returns=_OP_STRING,
    variants=(
        Variant(
            "ds",
            "trim(DS)",
            (authored("leading_and_trailing_spaces"),),
            takes=_STRING,
            gives=_STRING,
        ),
        Variant(
            "comp",
            "trim(comp)",
            (authored("leading_and_trailing_spaces"),),
            takes=_STRING,
            gives=_STRING,
        ),
        Variant(
            "sc",
            "trim(sc)",
            (authored("leading_and_trailing_spaces"),),
            takes=_STRING,
            gives=_STRING,
        ),
    ),
)

_LTRIM = Operator(
    id="ltrim",
    name="Leading whitespace removal",
    symbol="ltrim",
    page=f"{_PAGE}/Whitespace removal",
    accepts=_OP_STRING,
    returns=_OP_STRING,
    variants=(
        Variant(
            "ds",
            "ltrim(DS)",
            (authored("leading_removed_trailing_kept"),),
            takes=_STRING,
            gives=_STRING,
        ),
        Variant(
            "comp",
            "ltrim(comp)",
            (authored("leading_removed_trailing_kept"),),
            takes=_STRING,
            gives=_STRING,
        ),
        Variant(
            "sc",
            "ltrim(sc)",
            (authored("leading_removed_trailing_kept"),),
            takes=_STRING,
            gives=_STRING,
        ),
    ),
)

_RTRIM = Operator(
    id="rtrim",
    name="Trailing whitespace removal",
    symbol="rtrim",
    page=f"{_PAGE}/Whitespace removal",
    accepts=_OP_STRING,
    returns=_OP_STRING,
    variants=(
        Variant("ds", "rtrim(DS)", (manual("ex_1"),), takes=_STRING, gives=_STRING),
        Variant("comp", "rtrim(comp)", (manual("ex_2"),), takes=_STRING, gives=_STRING),
        Variant(
            "sc",
            "rtrim(sc)",
            (authored("trailing_removed_leading_kept"),),
            takes=_STRING,
            gives=_STRING,
        ),
    ),
)

_UPPER = Operator(
    id="upper",
    name="Upper case conversion",
    symbol="upper",
    page=f"{_PAGE}/Character case conversion",
    accepts=_OP_STRING,
    returns=_OP_STRING,
    variants=(
        Variant("ds", "upper(DS)", (manual("ex_1"),), takes=_STRING, gives=_STRING),
        Variant("comp", "upper(comp)", (manual("ex_2"),), takes=_STRING, gives=_STRING),
        Variant(
            "sc",
            "upper(sc)",
            (authored("capitalised_word"),),
            takes=_STRING,
            gives=_STRING,
        ),
    ),
)

_LOWER = Operator(
    id="lower",
    name="Lower case conversion",
    symbol="lower",
    page=f"{_PAGE}/Character case conversion",
    accepts=_OP_STRING,
    returns=_OP_STRING,
    variants=(
        Variant(
            "ds",
            "lower(DS)",
            (authored("all_caps_and_mixed_case"),),
            takes=_STRING,
            gives=_STRING,
        ),
        Variant(
            "comp",
            "lower(comp)",
            (authored("all_caps_and_mixed_case"),),
            takes=_STRING,
            gives=_STRING,
        ),
        Variant(
            "sc",
            "lower(sc)",
            (authored("capitalised_word"),),
            takes=_STRING,
            gives=_STRING,
        ),
    ),
)

# substr: ``op`` is a string, ``start`` an integer (>= 1) and ``length`` an integer (>= 0), so a
# form that has a ``start`` or a ``length`` takes an integer as well as a string. A ``_`` stands
# for an optional parameter that is left out (Overview and conventions, syntax conventions).
_SUBSTR = Operator(
    id="substr",
    name="Sub-string extraction",
    symbol="substr",
    page=f"{_PAGE}/Sub-string extraction",
    accepts=_OP_STRING_INTEGER,
    returns=_OP_STRING,
    variants=(
        Variant(
            "ds_start_len",
            "substr(DS, start, length)",
            (manual("ex_2"),),
            takes=_STRING_INTEGER,
            gives=_STRING,
        ),
        Variant(
            "ds_start",
            "substr(DS, start)",
            (manual("ex_1"),),
            takes=_STRING_INTEGER,
            gives=_STRING,
        ),
        Variant(
            "ds_start_placeholder",
            "substr(DS, start, _)",
            (authored("placeholder_for_length"),),
            takes=_STRING_INTEGER,
            gives=_STRING,
        ),
        Variant(
            "ds_placeholder_len",
            "substr(DS, _, length)",
            (authored("placeholder_for_start"),),
            takes=_STRING_INTEGER,
            gives=_STRING,
        ),
        Variant(
            "ds_noargs",
            "substr(DS)",
            (authored("start_and_length_omitted"),),
            takes=_STRING,
            gives=_STRING,
        ),
        Variant(
            "comp_start_len",
            "substr(comp, start, length)",
            (manual("ex_3"),),
            takes=_STRING_INTEGER,
            gives=_STRING,
        ),
        Variant(
            "comp_params",
            "substr(comp, comp, comp)",
            (authored("start_and_length_from_components"),),
            takes=_STRING_INTEGER,
            gives=_STRING,
        ),
        Variant(
            "sc_start_len",
            "substr(sc, start, length)",
            (
                authored("substring_inside_the_string"),
                authored("length_beyond_the_end"),
                authored("start_beyond_the_end"),
            ),
            takes=_STRING_INTEGER,
            gives=_STRING,
        ),
    ),
)

# replace: ``pattern1`` and ``pattern2`` are strings too (``component<string>`` or ``string``), so
# every form takes and gives strings.
_REPLACE = Operator(
    id="replace",
    name="String pattern replacement",
    symbol="replace",
    page=f"{_PAGE}/String pattern replacement",
    accepts=_OP_STRING,
    returns=_OP_STRING,
    variants=(
        Variant(
            "ds_two",
            "replace(DS, pattern1, pattern2)",
            (manual("ex_1"),),
            takes=_STRING,
            gives=_STRING,
        ),
        Variant(
            "ds_one",
            "replace(DS, pattern1)",
            (authored("replacement_omitted"),),
            takes=_STRING,
            gives=_STRING,
        ),
        Variant(
            "ds_one_placeholder",
            "replace(DS, pattern1, _)",
            (authored("placeholder_for_replacement"),),
            takes=_STRING,
            gives=_STRING,
        ),
        Variant(
            "comp_two",
            "replace(comp, pattern1, pattern2)",
            (manual("ex_2"),),
            takes=_STRING,
            gives=_STRING,
        ),
        Variant(
            "comp_patterns",
            "replace(comp, comp, comp)",
            (authored("patterns_from_components"),),
            takes=_STRING,
            gives=_STRING,
        ),
        Variant(
            "sc_two",
            "replace(sc, pattern1, pattern2)",
            (authored("leading_word_replaced"),),
            takes=_STRING,
            gives=_STRING,
        ),
    ),
)

# instr: ``pattern`` is a string, ``start`` and ``occurrence`` are integers (>= 1), and the result
# is an integer (>= 0): ``0`` when the pattern is not found, and the measure ``int_var`` at data
# set level.
_INSTR = Operator(
    id="instr",
    name="String pattern location",
    symbol="instr",
    page=f"{_PAGE}/String pattern location",
    accepts=_OP_STRING_INTEGER,
    returns=_OP_INTEGER,
    variants=(
        Variant(
            "ds_pattern",
            "instr(DS, pattern)",
            (manual("ex_1"),),
            takes=_STRING,
            gives=_INTEGER,
        ),
        Variant(
            "ds_pattern_start",
            "instr(DS, pattern, start)",
            (authored("matches_before_start_skipped"),),
            takes=_STRING_INTEGER,
            gives=_INTEGER,
        ),
        Variant(
            "ds_pattern_start_occ",
            "instr(DS, pattern, start, occurrence)",
            (authored("second_occurrence_from_start"),),
            takes=_STRING_INTEGER,
            gives=_INTEGER,
        ),
        Variant(
            "ds_placeholder_occ",
            "instr(DS, pattern, _, occurrence)",
            (authored("placeholder_for_start"),),
            takes=_STRING_INTEGER,
            gives=_INTEGER,
        ),
        Variant(
            "ds_pattern_start_placeholder",
            "instr(DS, pattern, start, _)",
            (authored("placeholder_for_occurrence"),),
            takes=_STRING_INTEGER,
            gives=_INTEGER,
        ),
        Variant(
            "comp_pattern",
            "instr(comp, pattern)",
            (manual("ex_2"), manual("ex_3")),
            takes=_STRING,
            gives=_INTEGER,
        ),
        Variant(
            "comp_params",
            "instr(comp, comp, comp, comp)",
            (
                authored("pattern_and_start_from_components"),
                authored("all_parameters_from_components"),
            ),
            takes=_STRING_INTEGER,
            gives=_INTEGER,
        ),
        Variant(
            "sc_pattern",
            "instr(sc, pattern)",
            (authored("pattern_at_the_third_position"),),
            takes=_STRING,
            gives=_INTEGER,
        ),
    ),
)

_LENGTH = Operator(
    id="length",
    name="String length",
    symbol="length",
    page=f"{_PAGE}/String length",
    accepts=_OP_STRING,
    returns=_OP_INTEGER,
    variants=(
        Variant("ds", "length(DS)", (manual("ex_1"),), takes=_STRING, gives=_INTEGER),
        Variant(
            "comp",
            "length(comp)",
            (manual("ex_2"), manual("ex_3")),
            takes=_STRING,
            gives=_INTEGER,
        ),
        Variant(
            "sc",
            "length(sc)",
            (authored("punctuation_and_space_counted"), authored("empty_string")),
            takes=_STRING,
            gives=_INTEGER,
        ),
    ),
)

# The manual's examples of string_distance run the distance of two datasets (levenshtein and
# damerau_levenshtein) and of a component and a scalar (hamming), so each of those has a variant
# of its own next to the ones that hold the scalar forms. The operands can be a dataset, a
# component or a scalar (string1 and string2 are typed that way), and the pairs of operand levels
# are crossed on levenshtein only, with the component and scalar pair covered by hamming. The
# method, like the operands, is a string, and the page types the distance a number: a ``num_var``
# at data set level.
_STRING_DISTANCE = Operator(
    id="string_distance",
    name="String distances",
    symbol="string_distance",
    page=f"{_PAGE}/String distances",
    accepts=_OP_STRING,
    returns=_OP_NUMBER,
    variants=(
        Variant(
            "levenshtein_ds_ds",
            "string_distance(levenshtein, DS, DS)",
            (manual("ex_1"),),
            takes=_STRING,
            gives=_NUMBER,
        ),
        Variant(
            "levenshtein_ds_sc",
            "string_distance(levenshtein, DS, sc)",
            (authored("equal_strings_and_null_measure"),),
            takes=_STRING,
            gives=_NUMBER,
        ),
        Variant(
            "levenshtein_sc_ds",
            "string_distance(levenshtein, sc, DS)",
            (authored("scalar_first_and_null_measure"),),
            takes=_STRING,
            gives=_NUMBER,
        ),
        Variant(
            "levenshtein_comp_comp",
            "string_distance(levenshtein, comp, comp)",
            (authored("kitten_sitting_and_null"),),
            takes=_STRING,
            gives=_NUMBER,
        ),
        Variant(
            "levenshtein_sc_comp",
            "string_distance(levenshtein, sc, comp)",
            (authored("adjacent_swap_costs_two"),),
            takes=_STRING,
            gives=_NUMBER,
        ),
        Variant(
            "levenshtein_sc",
            "string_distance(levenshtein, sc, sc)",
            (authored("one_deletion"),),
            takes=_STRING,
            gives=_NUMBER,
        ),
        Variant(
            "damerau_levenshtein_ds_ds",
            "string_distance(damerau_levenshtein, DS, DS)",
            (manual("ex_2"),),
            takes=_STRING,
            gives=_NUMBER,
        ),
        Variant(
            "damerau_levenshtein_sc",
            "string_distance(damerau_levenshtein, sc, sc)",
            (authored("adjacent_transposition_costs_one"),),
            takes=_STRING,
            gives=_NUMBER,
        ),
        Variant(
            "hamming_comp_sc",
            "string_distance(hamming, comp, sc)",
            (manual("ex_3"),),
            takes=_STRING,
            gives=_NUMBER,
        ),
        Variant(
            "hamming_sc",
            "string_distance(hamming, sc, sc)",
            (authored("three_differing_positions"),),
            takes=_STRING,
            gives=_NUMBER,
        ),
        Variant(
            "jaro_winkler_sc",
            "string_distance(jaro_winkler, sc, sc)",
            (authored("identical_strings"), authored("disjoint_strings")),
            takes=_STRING,
            gives=_NUMBER,
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
