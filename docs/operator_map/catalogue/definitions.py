"""The definitions of the VTL-DL: user defined operators, rulesets and viral propagations.

The VTL-DL sections of the manual have no examples of their own, so every variant has authored
examples, named after what they test. Each example defines the construct and uses it: a call of
the operator, ``check_datapoint`` for a datapoint ruleset, ``check_hierarchy`` for a hierarchical
ruleset, and a binary operator, a join, an aggregation or ``hierarchy`` over datasets with a viral
attribute for a viral propagation. One example expects the error that the viral propagation page
states: a second propagation for the same variable.

Each form gives the structural type of what it defines, and takes the types that the manual gives
to its parameters ("Input parameters type", or the syntax description of the viral propagation
page, which has none). A name, such as a variable, a code item or a rule name, has no type.

* ``define operator`` takes the type that the form declares: the type of its parameter, or of its
  ``returns``. A declaration that names no scalar type (``component``, ``dataset``, no
  ``returns``) takes ``scalar``, any scalar type, and an operator with no parameter nothing.
* A datapoint ruleset takes ``boolean``, the type of its conditions, and ``value_domain`` on value
  domains. Its error code is a ``string`` and its error level an ``integer``: the types that
  ``check`` gives them when the value domains ``errorcode_vd`` and ``errorlevel_vd`` are not
  found.
* A hierarchical ruleset relates code items, which are names: it takes ``value_domain`` on value
  domains, ``boolean`` for its left and right conditions, and its error code and level as above.
* A viral propagation takes the values of its viral attribute: any scalar in an enumerated rule,
  the types of the aggregate in an aggregate rule (``number`` for ``sum`` and ``avg``), and
  ``value_domain`` on a value domain.
"""

from __future__ import annotations

from typing import Final

from tool.model import Chapter, Example, Link, Operator, Variant, authored

# What each operator gives: the structural type of what it defines.
_OPERATOR: Final = ("operator",)
_DATAPOINT_RULESET: Final = ("datapoint_ruleset",)
_HIERARCHICAL_RULESET: Final = ("hierarchical_ruleset",)
_VIRAL_PROPAGATION: Final = ("viral_propagation",)

# What the forms take.
_NOTHING: Final[tuple[str, ...]] = ()
_SCALAR: Final = ("scalar",)
_INTEGER: Final = ("integer",)
_STRING: Final = ("string",)
_BOOLEAN: Final = ("boolean",)
_ON_VALUE_DOMAINS: Final = ("boolean", "value_domain")


def _form(
    variant_id: str,
    label: str,
    takes: tuple[str, ...],
    gives: tuple[str, ...],
    *examples: Example,
) -> Variant:
    """A form with the types it takes and gives, and its examples."""
    return Variant(id=variant_id, label=label, examples=examples, takes=takes, gives=gives)


DEFINE_OPERATOR = Operator(
    id="define_operator",
    name="define operator",
    symbol="",  # the name is the keyword
    page="dl/udo",
    accepts=(
        "scalar",
        "dataset",
        "set",
        "value_domain",
        "datapoint_ruleset",
        "hierarchical_ruleset",
    ),
    returns=("operator",),
    variants=(
        _form("param_integer", "x integer", _INTEGER, _OPERATOR, authored("integer_argument")),
        _form("param_number", "x number", ("number",), _OPERATOR, authored("number_argument")),
        _form("param_string", "x string", _STRING, _OPERATOR, authored("string_argument")),
        _form("param_boolean", "x boolean", _BOOLEAN, _OPERATOR, authored("boolean_argument")),
        _form("param_date", "x date", ("date",), _OPERATOR, authored("date_argument")),
        _form(
            "param_time_period",
            "x time_period",
            ("time_period",),
            _OPERATOR,
            authored("time_period_argument"),
        ),
        _form("param_time", "x time", ("time",), _OPERATOR, authored("time_argument")),
        _form(
            "param_duration", "x duration", ("duration",), _OPERATOR, authored("duration_argument")
        ),
        _form("param_scalar", "x scalar", _SCALAR, _OPERATOR, authored("integer_keeps_its_type")),
        _form(
            "param_value_domain",
            "x VD_1",
            ("value_domain",),
            _OPERATOR,
            authored("string_value_of_the_domain"),
        ),
        _form(
            "param_constrained",
            "x integer [value >= 0]",
            _INTEGER,
            _OPERATOR,
            authored("value_condition_holds"),
            authored("value_in_the_list"),
        ),
        _form(
            "param_not_null",
            "x integer not null",
            _INTEGER,
            _OPERATOR,
            authored("non_null_argument"),
        ),
        _form(
            "param_set", "s set<string>", ("set",), _OPERATOR, authored("value_domain_as_the_set")
        ),
        _form("param_component", "c component", _SCALAR, _OPERATOR, authored("measure_in_calc")),
        _form(
            "param_identifier",
            "c identifier<integer>",
            _INTEGER,
            _OPERATOR,
            authored("identifier_in_calc"),
        ),
        _form(
            "param_measure", "c measure<integer>", _INTEGER, _OPERATOR, authored("measure_in_calc")
        ),
        _form(
            "param_attribute",
            "c attribute<string>",
            _STRING,
            _OPERATOR,
            authored("attribute_overwritten_in_calc"),
        ),
        _form(
            "param_viral_attribute",
            "c viral attribute<string>",
            _STRING,
            _OPERATOR,
            authored("viral_attribute_overwritten_in_calc"),
        ),
        _form("param_dataset", "ds dataset", _SCALAR, _OPERATOR, authored("dataset_argument")),
        _form(
            "param_dataset_constrained",
            "ds dataset {...}",
            _SCALAR,
            _OPERATOR,
            authored("named_components"),
            authored("exactly_one_measure"),
            authored("one_or_more_measures"),
            authored("zero_or_more_measures"),
        ),
        _form(
            "param_ruleset_datapoint",
            "dpr datapoint",
            _DATAPOINT_RULESET,
            _OPERATOR,
            authored("datapoint_type"),
            authored("datapoint_on_variables_type"),
            authored("datapoint_on_valuedomains_type"),
            authored("generic_ruleset_type"),
        ),
        _form(
            "param_ruleset_hierarchical",
            "hr hierarchical",
            _HIERARCHICAL_RULESET,
            _OPERATOR,
            authored("hierarchical_type"),
            authored("hierarchical_on_variables_type"),
            authored("hierarchical_on_valuedomains_type"),
        ),
        _form("no_params", "f ()", _NOTHING, _OPERATOR, authored("call_without_arguments")),
        _form(
            "default_value",
            "y integer default 10",
            _INTEGER,
            _OPERATOR,
            authored("last_argument_omitted"),
            authored("underscore_for_the_argument"),
        ),
        _form("returns_omitted", "no returns", _SCALAR, _OPERATOR, authored("type_of_the_body")),
        _form(
            "returns_scalar",
            "returns integer",
            _INTEGER,
            _OPERATOR,
            authored("larger_of_two_integers"),
        ),
        _form(
            "returns_component",
            "returns component<integer>",
            _INTEGER,
            _OPERATOR,
            authored("component_in_calc"),
        ),
        _form(
            "returns_dataset",
            "returns dataset",
            _SCALAR,
            _OPERATOR,
            authored("every_measure_plus_one"),
        ),
    ),
    links=(Link("general.udo_call", "used by"),),
)

DEFINE_DATAPOINT_RULESET = Operator(
    id="define_datapoint_ruleset",
    name="define datapoint ruleset",
    symbol="",  # the name is the keyword
    page="dl/datapoint",
    accepts=("value_domain",),
    returns=("datapoint_ruleset",),
    variants=(
        _form(
            "valuedomain",
            "valuedomain VD_FLOW, VD_AMOUNT",
            _ON_VALUE_DOMAINS,
            _DATAPOINT_RULESET,
            authored("domains_bound_to_components"),
        ),
        _form(
            "valuedomain_alias",
            "valuedomain VD_FLOW as F, ...",
            _ON_VALUE_DOMAINS,
            _DATAPOINT_RULESET,
            authored("domain_aliases_in_the_rule"),
        ),
        _form(
            "variable",
            "variable Id_2, Me_1",
            _BOOLEAN,
            _DATAPOINT_RULESET,
            authored("variables_of_the_dataset"),
        ),
        _form(
            "variable_alias",
            "variable Id_2 as F, ...",
            _BOOLEAN,
            _DATAPOINT_RULESET,
            authored("variable_aliases_in_the_rule"),
        ),
        _form("named", "R_1 : rule", _BOOLEAN, _DATAPOINT_RULESET, authored("two_named_rules")),
        _form(
            "unnamed",
            "rule without a name",
            _BOOLEAN,
            _DATAPOINT_RULESET,
            authored("rules_named_by_their_order"),
        ),
        _form(
            "when_then",
            "when ... then ...",
            _BOOLEAN,
            _DATAPOINT_RULESET,
            authored("three_cases_of_the_implication"),
        ),
        _form(
            "plain", "consequent only", _BOOLEAN, _DATAPOINT_RULESET, authored("consequent_alone")
        ),
        _form(
            "errorcode",
            'errorcode "NEG_CREDIT"',
            ("string", "boolean"),
            _DATAPOINT_RULESET,
            authored("code_on_the_invalid_point"),
        ),
        _form(
            "errorlevel",
            "errorlevel 5",
            ("integer", "boolean"),
            _DATAPOINT_RULESET,
            authored("level_on_the_invalid_point"),
        ),
    ),
    links=(Link("validation.check_datapoint", "used by"),),
)

DEFINE_HIERARCHICAL_RULESET = Operator(
    id="define_hierarchical_ruleset",
    name="define hierarchical ruleset",
    symbol="",  # the name is the keyword
    page="dl/hierarchical",
    accepts=("value_domain",),
    returns=("hierarchical_ruleset",),
    variants=(
        _form(
            "valuedomain_rule",
            "valuedomain rule VD_ITEM",
            ("value_domain",),
            _HIERARCHICAL_RULESET,
            authored("domain_bound_by_rule"),
        ),
        _form(
            "valuedomain_condition",
            "valuedomain condition VD_FLAG rule VD_ITEM",
            _ON_VALUE_DOMAINS,
            _HIERARCHICAL_RULESET,
            authored("conditioning_domain"),
        ),
        _form(
            "valuedomain_condition_alias",
            "condition VD_FLAG as F",
            _ON_VALUE_DOMAINS,
            _HIERARCHICAL_RULESET,
            authored("aliased_conditioning_domain"),
        ),
        _form(
            "variable_rule",
            "variable rule Id_2",
            _NOTHING,
            _HIERARCHICAL_RULESET,
            authored("unbalanced_total"),
        ),
        _form(
            "variable_condition",
            "variable condition Id_1 rule Id_2",
            _BOOLEAN,
            _HIERARCHICAL_RULESET,
            authored("conditioning_variable"),
        ),
        _form(
            "variable_condition_alias",
            "variable condition Id_1 as C",
            _BOOLEAN,
            _HIERARCHICAL_RULESET,
            authored("aliased_conditioning_variable"),
        ),
        _form("rel_eq", "A = B", _NOTHING, _HIERARCHICAL_RULESET, authored("unequal_values_fail")),
        _form("rel_gt", "A > B", _NOTHING, _HIERARCHICAL_RULESET, authored("equal_values_fail")),
        _form("rel_ge", "A >= B", _NOTHING, _HIERARCHICAL_RULESET, authored("equal_values_hold")),
        _form("rel_lt", "A < B", _NOTHING, _HIERARCHICAL_RULESET, authored("equal_values_fail")),
        _form("rel_le", "A <= B", _NOTHING, _HIERARCHICAL_RULESET, authored("equal_values_hold")),
        _form(
            "when_then",
            "when ... then A = B",
            _BOOLEAN,
            _HIERARCHICAL_RULESET,
            authored("false_condition_not_compared"),
        ),
        _form(
            "plus_minus",
            "A = B + C - D",
            _NOTHING,
            _HIERARCHICAL_RULESET,
            authored("added_and_subtracted_items"),
            authored("sign_of_the_first_item"),
        ),
        _form(
            "right_condition",
            "A = B + C [condition]",
            _BOOLEAN,
            _HIERARCHICAL_RULESET,
            authored("item_left_out_by_condition"),
        ),
        _form(
            "errorcode",
            'errorcode "E_AB"',
            _STRING,
            _HIERARCHICAL_RULESET,
            authored("code_on_the_failed_rule"),
        ),
        _form(
            "errorlevel",
            "errorlevel 3",
            _INTEGER,
            _HIERARCHICAL_RULESET,
            authored("level_on_the_failed_rule"),
        ),
        _form("named", "R_1 : A = B", _NOTHING, _HIERARCHICAL_RULESET, authored("two_named_rules")),
        _form(
            "unnamed",
            "rule without a name",
            _NOTHING,
            _HIERARCHICAL_RULESET,
            authored("rules_named_by_their_order"),
        ),
    ),
    links=(
        Link("hierarchy.hierarchy", "used by"),
        Link("validation.check_hierarchy", "used by"),
    ),
)

DEFINE_VIRAL_PROPAGATION = Operator(
    id="define_viral_propagation",
    name="define viral propagation",
    symbol="",  # the name is the keyword
    page="dl/viral",
    accepts=("value_domain",),
    returns=("viral_propagation",),
    variants=(
        _form(
            "valuedomain",
            "valuedomain CL_CONF",
            ("scalar", "value_domain"),
            _VIRAL_PROPAGATION,
            authored("inherited_from_the_domain"),
        ),
        _form(
            "variable",
            "variable At_1",
            _SCALAR,
            _VIRAL_PROPAGATION,
            authored("conf_priority_rule"),
            authored("second_rule_for_the_variable"),  # expects the error of a second rule
        ),
        _form(
            "unary",
            'when "N" then "N"',
            _SCALAR,
            _VIRAL_PROPAGATION,
            authored("first_clause_in_order_wins"),
        ),
        _form(
            "binary",
            'when "C" and "M" then "N"',
            _SCALAR,
            _VIRAL_PROPAGATION,
            authored("binary_clause_before_unary"),
        ),
        _form(
            "named_clause",
            'R_C : when "C" then "C"',
            _SCALAR,
            _VIRAL_PROPAGATION,
            authored("names_do_not_change_matching"),
        ),
        _form(
            "else_clause",
            'else "F"',
            _SCALAR,
            _VIRAL_PROPAGATION,
            authored("else_for_unlisted_pairs"),
        ),
        _form(
            "no_else", "no else", _SCALAR, _VIRAL_PROPAGATION, authored("null_for_unlisted_pairs")
        ),
        _form(
            "aggregate_min",
            "aggregate min",
            _SCALAR,
            _VIRAL_PROPAGATION,
            authored("smaller_of_each_pair"),
        ),
        _form(
            "aggregate_max",
            "aggregate max",
            _SCALAR,
            _VIRAL_PROPAGATION,
            authored("larger_of_each_pair"),
        ),
        _form(
            "aggregate_sum",
            "aggregate sum",
            ("number",),
            _VIRAL_PROPAGATION,
            authored("sum_of_each_pair"),
        ),
        _form(
            "aggregate_avg",
            "aggregate avg",
            ("number",),
            _VIRAL_PROPAGATION,
            authored("mean_of_each_pair"),
            authored("across_two_operators", decision=1175),
            authored("within_one_operator", decision=1175),
        ),
        _form(
            "at_binary_op",
            "at DS_1 - DS_2",
            _SCALAR,
            _VIRAL_PROPAGATION,
            authored("subtraction_of_two_datasets"),
        ),
        _form(
            "at_join",
            "at inner_join",
            _SCALAR,
            _VIRAL_PROPAGATION,
            authored("viral_in_both_datasets"),
        ),
        _form(
            "at_aggregation",
            "at sum group by",
            _SCALAR,
            _VIRAL_PROPAGATION,
            authored("sum_grouped_by_identifier"),
        ),
        _form(
            "at_hierarchy",
            "at hierarchy",
            _SCALAR,
            _VIRAL_PROPAGATION,
            authored("computed_total"),
        ),
    ),
    links=(
        Link("numeric.addition", "propagates at"),
        Link("join.inner_join", "propagates at"),
        Link("aggregate.sum", "propagates at"),
        Link("hierarchy.hierarchy", "propagates at"),
    ),
)

CHAPTER = Chapter(
    id="definitions",
    title="Definitions (VTL-DL)",
    operators=(
        DEFINE_OPERATOR,
        DEFINE_DATAPOINT_RULESET,
        DEFINE_HIERARCHICAL_RULESET,
        DEFINE_VIRAL_PROPAGATION,
    ),
)
