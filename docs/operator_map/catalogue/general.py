"""General purpose operators: parentheses, assignments, membership, calls and ``cast``.

Parentheses take a dataset, a component or a scalar. The persistent assignment ``<-`` takes a
dataset, the non-persistent one ``:=`` a dataset or a scalar. The dataset form of ``:=`` also has
an example for each basic type, ``<type>_measure``: a dataset with a measure of the type, assigned
as it is, which shows that vtlengine loads, keeps and writes the values of the type. Membership
``DS#comp`` picks one component of a dataset as its measure, and gives a scalar when the dataset
has no identifiers; ``alias#comp`` inside a join is not membership (the manual says so) and
belongs to the joins. A user-defined operator is called with no arguments, with all of them, with
trailing optional ones omitted, or with ``_`` for an optional one. ``eval`` runs an external
routine, which vtlengine takes as a SQL query named after its file
(``external_routines/<name>.json``); the routine gets a non-dataset argument as a positional
parameter (``$1``, ``$2``...).

``cast`` has one variant per conversion that the manual's conversion table allows, at component
level (``DS_1[calc Me_2 := cast(Me_1, <type>)]``): the explicit ones, the implicit ones written
as an explicit cast, and each conversion with an optional mask both without and with a mask. The
masks follow the manual's section "Type Conversion and Formatting Mask". ``string_to_integer``
also has a dataset and a scalar variant, and two variants cast to a value domain
(``value_domains/``): from a string, and from another value domain on the same basic type.

Every form declares the types it takes and gives. A conversion takes its source type, and the
string of its mask when it has one, and gives its target type; a cast to a value domain takes
the ``value_domain`` too. The parentheses, the assignments and membership take any scalar and
give what they are given (``any``). A call takes the ``operator`` it calls, and its arguments,
and gives what that operator returns. ``eval`` takes the ``external_routine`` and its
arguments, and gives the type that its invocation declares: a dataset of any types, a string or
an integer.
"""

from __future__ import annotations

from typing import Final

from tool.model import (
    ANY_SCALAR,
    ANY_TYPE,
    VALUE_TYPES,
    Chapter,
    Operator,
    Variant,
    authored,
    manual,
)

_CHAPTER = "General purpose operators"
_STRING: Final = "string"  # the type of a mask

# The conversions of the manual's conversion table: the source type, the target type, and the
# names of the examples, each after what it tests.
_EXPLICIT: Final[tuple[tuple[str, str, tuple[str, ...]], ...]] = (  # "Explicit": no mask
    ("integer", "boolean", ("zero_and_non_zero",)),
    ("number", "boolean", ("zero_and_non_zero",)),
    ("boolean", "integer", ("true_and_false",)),
    ("boolean", "number", ("true_and_false",)),
    ("time", "date", ("single_day_intervals",)),
    ("time", "time_period", ("year_quarter_and_month",)),
    ("date", "time_period", ("daily_periods",)),
    # The second example expects the first error that the page states for a conversion between
    # basic types: a period that is not a single day.
    ("time_period", "date", ("daily_periods", "non_daily_period")),
    # The second example holds the strings that the rule of the manual sends to false.
    ("string", "boolean", ("true_in_any_case", "other_strings_are_false")),
)
# "Implicit": written here as an explicit cast without mask.
_IMPLICIT: Final[tuple[tuple[str, str, tuple[str, ...]], ...]] = (
    ("integer", "number", ("zero_decimal_part",)),
    ("boolean", "string", ("true_and_false",)),
    ("date", "time", ("same_start_and_end",)),
    ("time_period", "time", ("start_and_end_of_the_period",)),
)
# "Optional mask": once without a mask, with the name of its example, and once with a mask, whose
# examples are in ``_MASKED``.
_OPTIONAL_MASK: Final[tuple[tuple[str, str, str], ...]] = (
    ("integer", "string", "digits_and_sign"),
    ("number", "string", "plain_decimal_notation"),
    ("time", "string", "iso_8601_intervals"),
    ("date", "string", "iso_8601_dates"),
    ("time_period", "string", "start_and_end_dates"),
    ("duration", "string", "iso_8601_durations"),
    ("string", "integer", "signs_leading_zeros_and_null"),
    ("string", "number", "sign_decimals_and_exponent"),
    ("string", "time", "iso_8601_intervals"),
    ("string", "date", "iso_8601_dates"),
    ("string", "time_period", "quarter_month_and_year"),
    ("string", "duration", "one_letter_codes"),
)
# The examples of each conversion with a mask, by its source and target types: a kind of mask
# each, as the section "Type Conversion and Formatting Mask" describes them, in ``_MASKED``.
_MaskExamples = dict[tuple[str, str], tuple[str, ...]]

# The numbers formatting mask: ``D``, ``E``, ``*``, ``+`` and the dot or comma separator, in the
# manual's order, then strings that the mask does not match. An integer has no decimal part, so
# its masks have no separator, and the exponent is shown on numbers. ``D*`` and ``D+`` may be
# read as a ``D`` and then digits, or as a repeated ``D``, which differ on a part of one digit:
# the number examples have no such part, and the integer masks put ``*`` and ``+`` first.
_NUMBER_MASKS: Final[_MaskExamples] = {
    ("integer", "string"): (
        "two_digit_mask",
        "any_number_of_digits_mask",
        "at_least_one_digit_mask",
    ),
    ("string", "integer"): (
        "two_digit_mask",
        "too_few_digits",
        "too_many_digits",
        "no_digit_for_the_plus",
    ),
    ("number", "string"): (
        "decimal_mask",
        "exponent_mask",
        "any_number_of_digits_mask",
        "at_least_one_digit_mask",
        "decimal_comma_mask",
    ),
    ("string", "number"): (
        "decimal_mask",
        "exponent_mask",
        "any_number_of_digits_mask",
        "at_least_one_digit_mask",
        "decimal_comma_mask",
        "too_few_digits",
        "too_many_digits",
    ),
}

# The time formatting mask of a date.
_DATE_MASKS: Final[_MaskExamples] = {
    ("date", "string"): ("year_month_day_mask",),
    ("string", "date"): ("year_month_day_mask",),
}

# The time formatting mask of a time.
_TIME_MASKS: Final[_MaskExamples] = {
    ("time", "string"): ("start_and_end_mask",),
    ("string", "time"): ("start_and_end_mask",),
}

# The time formatting mask of a time_period.
_TIME_PERIOD_MASKS: Final[_MaskExamples] = {
    ("time_period", "string"): ("monthly_period_mask",),
    ("string", "time_period"): ("quarter_mask",),
}

# The masks of a duration.
_DURATION_MASKS: Final[_MaskExamples] = {
    ("duration", "string"): ("period_indicator_mask",),
    ("string", "duration"): ("period_indicator_mask",),
}

_MASKED: Final[_MaskExamples] = {
    **_NUMBER_MASKS,
    **_DATE_MASKS,
    **_TIME_MASKS,
    **_TIME_PERIOD_MASKS,
    **_DURATION_MASKS,
}


# The text of a cast follows the pandas engine (team decision #1254), where the manual writes
# another one: the period code for a time_period. A boolean gives the manual's ``true`` and
# ``false`` since #1314.
_DECIDED: Final[dict[tuple[str, str, str], int]] = {
    ("time_period", "string", "start_and_end_dates"): 1254,
}


# The manual's default measure name of a string, ``string_var``, is one that "can be personalised
# in the implementations" (typical behaviour, "Default Names"); vtlengine documents ``str_var``.
_STR_VAR = (
    "the default measure name string_var is written str_var, the name vtlengine documents "
    "(the manual's default names can be personalised in the implementations)"
)


def _page(page: str) -> str:
    return f"{_CHAPTER}/{page}"


def _cast(source: str, target: str, names: tuple[str, ...], *, mask: bool = False) -> Variant:
    """The component-level variant ``cast(Me_1, target)`` of a ``source`` measure.

    It takes the source type, and the string of its mask when it has one, and gives the target
    type. ``names`` are the names of its examples.
    """
    suffix = "_mask" if mask else ""
    takes = (source, _STRING) if mask and source != _STRING else (source,)
    return Variant(
        id=f"{source}_to_{target}{suffix}",
        label=f"cast(comp<{source}>, {target}{', mask' if mask else ''})",
        examples=tuple(
            authored(name, decision=_DECIDED.get((source, target, name))) for name in names
        ),
        takes=takes,
        gives=(target,),
        kind="comp",
    )


def _cast_variants() -> tuple[Variant, ...]:
    variants = [_cast(source, target, names) for source, target, names in (*_EXPLICIT, *_IMPLICIT)]
    for source, target, plain in _OPTIONAL_MASK:
        variants.append(_cast(source, target, (plain,)))
        variants.append(_cast(source, target, _MASKED[(source, target)], mask=True))
    variants.append(
        Variant(
            "string_to_integer_ds",
            "cast(DS<string>, integer)",
            (authored("default_measure_name"),),
            takes=(_STRING,),
            gives=("integer",),
        )
    )
    variants.append(
        Variant(
            "string_to_integer_sc",
            "cast(sc<string>, integer)",
            (authored("string_of_digits"),),
            takes=(_STRING,),
            gives=("integer",),
        )
    )
    # A value domain is a scalar type, so a target of a cast. A value of the domain's basic type
    # is cast to the domain, or one of a domain to another domain on the same basic type: the
    # element of a domain is upcast to its basic type, whichever it is, and the result has the
    # basic type of its operand.
    variants.append(
        Variant(
            "string_to_value_domain",
            "cast(comp<string>, VD)",
            (authored("values_in_the_domain"),),
            takes=(_STRING, "value_domain"),
            gives=(_STRING,),
            kind="comp",
        )
    )
    variants.append(
        Variant(
            "value_domain_to_value_domain",
            "cast(comp<VD>, VD)",
            (authored("values_in_both_domains"),),
            takes=(ANY_SCALAR, "value_domain"),
            gives=(ANY_TYPE,),
            kind="comp",
        )
    )
    return tuple(variants)


CHAPTER = Chapter(
    id="general",
    title=_CHAPTER,
    operators=(
        Operator(
            id="parentheses",
            name="Parentheses",
            symbol="( )",
            page=_page("Parentheses"),
            accepts=("scalar", "dataset"),
            returns=("scalar", "dataset"),
            variants=(
                Variant("ds", "(DS)", (manual("ex_1"),), takes=(ANY_SCALAR,), gives=(ANY_TYPE,)),
                Variant(
                    "comp",
                    "(comp)",
                    (authored("precedence_in_calc"),),
                    takes=(ANY_SCALAR,),
                    gives=(ANY_TYPE,),
                ),
                Variant(
                    "sc",
                    "(sc)",
                    (authored("addition_first"), authored("nested_parentheses")),
                    takes=(ANY_SCALAR,),
                    gives=(ANY_TYPE,),
                ),
            ),
        ),
        Operator(
            id="persistent_assignment",
            name="Persistent assignment",
            symbol="<-",
            page=_page("Persistent assignment"),
            accepts=("dataset", "scalar"),
            returns=("dataset",),
            variants=(
                Variant(
                    "identity",
                    "DS_r <- DS",
                    (manual("ex_1"),),
                    takes=(ANY_SCALAR,),
                    gives=(ANY_TYPE,),
                    kind="ds",
                ),
                Variant(
                    "expression",
                    "DS_r <- expr",
                    (authored("dataset_subtraction"),),
                    takes=(ANY_SCALAR,),
                    gives=(ANY_TYPE,),
                    kind="ds",
                ),
                Variant(
                    "sc",
                    "SC_r <- sc",
                    (authored("integer_literal"), authored("input_value")),
                    takes=(ANY_SCALAR,),
                    gives=(ANY_TYPE,),
                ),
            ),
        ),
        Operator(
            id="non_persistent_assignment",
            name="Non-persistent assignment",
            symbol=":=",
            page=_page("Non-persistent assignment"),
            accepts=("dataset", "scalar"),
            returns=("dataset", "scalar"),
            variants=(
                Variant(
                    "ds",
                    "DS_r := DS",
                    # An example for each basic type of VTL: ``<type>_measure``.
                    (manual("ex_1"), *(authored(f"{type_id}_measure") for type_id in VALUE_TYPES)),
                    takes=(ANY_SCALAR,),
                    gives=(ANY_TYPE,),
                ),
                Variant(
                    "sc",
                    "SC_r := sc",
                    (authored("integer_literal"), authored("input_value")),
                    takes=(ANY_SCALAR,),
                    gives=(ANY_TYPE,),
                ),
            ),
        ),
        Operator(
            id="membership",
            name="Membership",
            symbol="#",
            page=_page("Membership"),
            accepts=("dataset",),
            returns=("dataset",),
            # Each form takes a component of any type, and gives a measure, or a scalar, of the
            # type of that component.
            variants=(
                Variant(
                    "identifier",
                    "DS#Id",
                    (manual("ex_2"), manual("ex_5")),
                    takes=(ANY_SCALAR,),
                    gives=(ANY_TYPE,),
                ),
                Variant(
                    "measure",
                    "DS#Me",
                    (manual("ex_1"), manual("ex_4")),
                    takes=(ANY_SCALAR,),
                    gives=(ANY_TYPE,),
                ),
                Variant(
                    "attribute",
                    "DS#At",
                    (manual("ex_3", adaptation=_STR_VAR), authored("dataset_without_identifiers")),
                    takes=(ANY_SCALAR,),
                    gives=(ANY_TYPE,),
                ),
                Variant(
                    "viral_attribute",
                    "DS#At (viral)",
                    (
                        manual("ex_6", adaptation=_STR_VAR),
                        manual(
                            "ex_8",
                            erratum="DS_2 has identifiers, so DS_2#At_1 is a dataset (the one "
                            "of ex_6), not the scalar 'P' that the manual gives",
                            adaptation=_STR_VAR,
                        ),
                    ),
                    takes=(ANY_SCALAR,),
                    gives=(ANY_TYPE,),
                ),
                Variant(
                    "on_aggregate",
                    "agg(DS)#Me",
                    (manual("ex_7"),),
                    takes=(ANY_SCALAR,),
                    gives=(ANY_TYPE,),
                ),
            ),
        ),
        Operator(
            id="udo_call",
            name="User-defined operator call",
            symbol="name( )",
            page=_page("User-defined operator call"),
            accepts=("scalar", "dataset"),
            returns=("scalar", "dataset"),
            # Each form takes the operator that it calls, and gives what that operator returns.
            variants=(
                Variant(
                    "no_args",
                    "op()",
                    (authored("empty_argument_list"),),
                    takes=("operator",),
                    gives=(ANY_TYPE,),
                ),
                Variant(
                    "all_args",
                    "op(a, b)",
                    (authored("positional_order"),),
                    takes=("operator", ANY_SCALAR),
                    gives=(ANY_TYPE,),
                ),
                Variant(
                    "default_omitted",
                    "op(a)",
                    (authored("trailing_default"),),
                    takes=("operator", ANY_SCALAR),
                    gives=(ANY_TYPE,),
                ),
                Variant(
                    "placeholder_arg",
                    "op(a, _, c)",
                    (authored("underscore_in_the_middle"),),
                    takes=("operator", ANY_SCALAR),
                    gives=(ANY_TYPE,),
                ),
            ),
        ),
        Operator(
            id="eval",
            name="Evaluation of an external routine",
            symbol="eval",
            page=_page("Evaluation of an external routine"),
            accepts=("dataset", "external_routine"),
            returns=("dataset",),
            # Each form takes the routine and its arguments, and gives what its ``returns``
            # declares.
            variants=(
                Variant(
                    "sql_dataset",
                    "eval(R(DS))",
                    (authored("sql_query_on_dataset"),),
                    takes=("external_routine", ANY_SCALAR),
                    gives=(ANY_TYPE,),
                    kind="ds",
                ),
                Variant(
                    "sql_dataset_scalar_arg",
                    "eval(R(DS, sc))",
                    (authored("positional_scalar_argument"),),
                    takes=("external_routine", ANY_SCALAR),
                    gives=(ANY_TYPE,),
                    kind="ds",
                ),
                Variant(
                    "scalar_return",
                    "eval(R(DS) returns string)",
                    (authored("string_result"),),
                    takes=("external_routine", ANY_SCALAR),
                    gives=(_STRING,),
                    kind="ds",
                ),
                Variant(
                    "comp",
                    "DS[calc Me := eval(R(comp))]",
                    (authored("inside_calc"),),
                    takes=("external_routine", ANY_SCALAR),
                    gives=("integer",),
                ),
            ),
        ),
        Operator(
            id="cast",
            name="Type conversion",
            symbol="cast",
            page=_page("Type conversion"),
            accepts=("scalar", "string", "value_domain", "dataset"),
            returns=("scalar",),
            variants=_cast_variants(),
        ),
    ),
)
