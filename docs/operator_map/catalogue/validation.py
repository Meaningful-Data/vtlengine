"""The data validation operators: ``check_datapoint``, ``check_hierarchy`` and ``check``.

Each form takes the types of the page's "Input parameters type", narrowed to what the form takes,
and gives the types of the measures of its result. A form whose label names no output has the
default one: ``invalid`` for ``check_datapoint`` and ``check_hierarchy``, ``all`` for ``check``.
Every form is on data sets and no form id has a kind token, so no form has a kind and the three
operators are drawn without boxes.

Every result has the measures ``errorcode`` and ``errorlevel``, null where nothing sets them: each
page lists them in its Behaviour, and a null measure still has a type. So every form gives
``string`` and ``integer`` beside the types its output decides, and every form of ``check`` also
gives the ``number`` imbalance.

* ``check_datapoint`` takes ``scalar``, for its operand ``dataset`` of any type, and the
  ``datapoint_ruleset`` (``name<datapoint>``). ``invalid`` gives "the Measures of op", so ``any``,
  the types of the input; ``all`` gives "the boolean Measure bool_var"; ``all_measures`` both.
  ``components`` names a slot that any output can go with, so it gives the union.
* ``check_hierarchy`` takes ``number``, for ``dataset { measure<number> _ }``, and the
  ``hierarchical_ruleset`` (``name<hierarchical>``). Its result, ``dataset { measure<number> _ }``,
  has the ``number`` imbalance with every output and the measure of the operand with ``invalid``
  and ``all_measures``; ``all`` and ``all_measures`` add the ``boolean`` bool_var.
* ``check`` takes its operand, "a boolean Data Set", and gives the ``boolean`` bool_var. The form
  that sets the error code takes a ``string`` ("or string if the errorcode_vd Value Domain is not
  found"), the one that sets the error level an ``integer`` ("or integer if the errorcode_vd Value
  Domain is not found"), and the one that sets the imbalance a ``number``.

The components that ``components``, ``condition`` and ``rule`` name have no type of their own, and
these slots go with any output, so their forms give the union of the outputs. The error code and
level are a ``string`` and an ``integer``, as in the definitions.

The ``<mode>_<input>_<output>`` forms of ``check_hierarchy`` share one data set and one ruleset.
Each example is named after what its mode shows on them, or, with the input ``dataset_priority``,
after the null item that the rule computing it gives.
"""

from __future__ import annotations

from typing import Final

from tool.model import Chapter, Example, Operator, Variant, authored, manual

_CHAPTER = "Data validation operators"

_MODES = ("non_null", "non_zero", "partial_null", "partial_zero", "always_null", "always_zero")
_INPUTS = ("dataset", "dataset_priority")
_OUTPUTS = ("invalid", "all", "all_measures")

# What the forms take: the operand, and the ruleset of check_datapoint and check_hierarchy.
_ANY_DATA_SET: Final = ("scalar", "datapoint_ruleset")
_NUMBER_DATA_SET: Final = ("number", "hierarchical_ruleset")
_BOOLEAN: Final = ("boolean",)
# Every result has the errorcode and errorlevel measures, null where nothing sets them.
_ERROR: Final = ("string", "integer")
_CHECK_GIVES: Final = ("boolean", "number", *_ERROR)

# What each output of check_hierarchy gives: the number imbalance, with the measure of the operand
# (invalid), the boolean bool_var (all) or both (all_measures), and the error measures.
_HIERARCHY_GIVES: Final[dict[str, tuple[str, ...]]] = {
    "invalid": ("number", *_ERROR),
    "all": ("number", "boolean", *_ERROR),
    "all_measures": ("number", "boolean", *_ERROR),
}
# The example of a grid form: what its mode shows on the shared data set and ruleset, or, with the
# input dataset_priority, the null item that the rule computing it gives.
_MODE_EXAMPLES: Final[dict[str, str]] = {
    "non_null": "null_or_missing_item_drops_the_rule",
    "non_zero": "rule_with_no_nonzero_item_dropped",
    "partial_null": "missing_items_count_as_null",
    "partial_zero": "missing_items_count_as_zero",
    "always_null": "rule_with_no_item_found_is_null",
    "always_zero": "rule_with_no_item_found_is_false",
}
_PRIORITY_EXAMPLE: Final = "null_item_from_the_rule_computing_it"


def _form(
    variant_id: str,
    label: str,
    takes: tuple[str, ...],
    gives: tuple[str, ...],
    *examples: Example,
) -> Variant:
    """A form with the types it takes and gives, and its examples."""
    return Variant(id=variant_id, label=label, examples=examples, takes=takes, gives=gives)


def _hierarchy_grid() -> tuple[Variant, ...]:
    """The ``<mode>_<input>_<output>`` forms of ``check_hierarchy``, in the manual's order."""
    return tuple(
        _form(
            f"{mode}_{input_}_{output}",
            f"check_hierarchy(DS, HR {mode} {input_} {output})",
            _NUMBER_DATA_SET,
            _HIERARCHY_GIVES[output],
            authored(_PRIORITY_EXAMPLE if input_ == "dataset_priority" else _MODE_EXAMPLES[mode]),
        )
        for mode in _MODES
        for input_ in _INPUTS
        for output in _OUTPUTS
    )


CHECK_DATAPOINT = Operator(
    id="check_datapoint",
    name="Check datapoint",
    symbol="check_datapoint",
    page=f"{_CHAPTER}/Check datapoint",
    accepts=("dataset", "datapoint_ruleset"),
    returns=("dataset",),
    variants=(
        _form(
            "output_default",
            "check_datapoint(DS, DPR)",
            _ANY_DATA_SET,
            ("any", *_ERROR),
            manual("ex_1"),
        ),
        _form(
            "invalid",
            "check_datapoint(DS, DPR invalid)",
            _ANY_DATA_SET,
            ("any", *_ERROR),
            authored("false_results_with_the_measures"),
        ),
        _form(
            "all",
            "check_datapoint(DS, DPR all)",
            _ANY_DATA_SET,
            ("boolean", *_ERROR),
            manual("ex_2"),
        ),
        _form(
            "all_measures",
            "check_datapoint(DS, DPR all_measures)",
            _ANY_DATA_SET,
            ("any", "boolean", *_ERROR),
            authored("every_result_with_the_measures"),
        ),
        _form(
            "components",
            "check_datapoint(DS, DPR components C1, C2)",
            _ANY_DATA_SET,
            ("any", "boolean", *_ERROR),
            authored("listed_variables_change_nothing"),
            authored("value_domains_bound_to_components"),
        ),
        _form(
            "errorcode",
            "check_datapoint(DS, DPR all), rules with errorcode",
            _ANY_DATA_SET,
            ("boolean", *_ERROR),
            authored("code_only_on_false_results"),
        ),
        _form(
            "errorlevel",
            "check_datapoint(DS, DPR all), rules with errorlevel",
            _ANY_DATA_SET,
            ("boolean", *_ERROR),
            authored("integer_level_on_false_results"),
        ),
    ),
)

CHECK_HIERARCHY = Operator(
    id="check_hierarchy",
    name="Check hierarchy",
    symbol="check_hierarchy",
    page=f"{_CHAPTER}/Check hierarchy",
    accepts=("dataset", "number", "hierarchical_ruleset"),
    returns=("dataset",),
    variants=(
        *_hierarchy_grid(),
        _form(
            "condition",
            "check_hierarchy(DS, HR condition C)",
            _NUMBER_DATA_SET,
            _HIERARCHY_GIVES["all_measures"],
            authored("conditioning_variable_named"),
            authored("conditioning_value_domain_bound"),
        ),
        _form(
            "rule",
            "check_hierarchy(DS, HR rule C)",
            _NUMBER_DATA_SET,
            _HIERARCHY_GIVES["all_measures"],
            # The example's ruleset has a cycle (R020, R070, R110), so it expects 1-3-2-3; the
            # manual's slip on the errorcode of R030 goes to the list for the VTL Task Force.
            manual("ex_1"),
            authored("value_domain_bound_to_identifier"),
        ),
        _form(
            "errorcode",
            "check_hierarchy(DS, HR partial_zero dataset all), rules with errorcode",
            _NUMBER_DATA_SET,
            _HIERARCHY_GIVES["all"],
            authored("code_only_on_false_results"),
        ),
        _form(
            "errorlevel",
            "check_hierarchy(DS, HR partial_zero dataset all), rules with errorlevel",
            _NUMBER_DATA_SET,
            _HIERARCHY_GIVES["all"],
            authored("integer_level_on_false_results"),
        ),
    ),
)

CHECK = Operator(
    id="check",
    name="Check",
    symbol="check",
    page=f"{_CHAPTER}/Check",
    accepts=("dataset", "boolean", "number"),
    returns=("dataset",),
    variants=(
        _form(
            "bare", "check(B)", _BOOLEAN, _CHECK_GIVES, authored("every_point_returned_by_default")
        ),
        _form(
            "errorcode",
            "check(B errorcode C)",
            ("string", "boolean"),
            _CHECK_GIVES,
            authored("code_only_on_the_false_point"),
        ),
        _form(
            "errorlevel",
            "check(B errorlevel L)",
            ("integer", "boolean"),
            _CHECK_GIVES,
            authored("level_only_on_the_false_point"),
        ),
        _form(
            "imbalance",
            "check(B imbalance N)",
            ("number", "boolean"),
            _CHECK_GIVES,
            manual("ex_1"),
        ),
        _form(
            "invalid",
            "check(B invalid)",
            _BOOLEAN,
            _CHECK_GIVES,
            authored("only_the_false_point_returned"),
        ),
        _form(
            "all", "check(B all)", _BOOLEAN, _CHECK_GIVES, authored("explicit_all_same_as_default")
        ),
    ),
)

CHAPTER = Chapter(
    id="validation",
    title=_CHAPTER,
    operators=(CHECK_DATAPOINT, CHECK_HIERARCHY, CHECK),
)
