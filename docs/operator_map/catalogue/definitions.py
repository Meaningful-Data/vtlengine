"""The definitions of the VTL-DL: user defined operators, rulesets and viral propagations.

The forms of a definition are the abstract parts of its syntax, never concrete cases: the
signature, the body and, for ``define operator``, the return type. A concrete case, such as a
type, a comparison, an aggregate function, an alias, the text of a clause or the operator where a
propagation applies, is an example under the form that it instantiates, and no diagram draws it.
``SYNTAX_BOXES`` says in which box of its diagram each form is drawn: the "Signature", the "Body"
or, for ``define operator``, the "Return type". The forms are listed box by box, the signature
first, and the label of a form is its abstract syntax.

The VTL-DL sections of the manual have no examples of their own, so every form has authored
examples, named after the concrete case that they test. Each example defines the construct and
uses it: a call of the operator, ``check_datapoint`` for a datapoint ruleset, ``check_hierarchy``
for a hierarchical ruleset, and a binary operator, a join, an aggregation or ``hierarchy`` over
datasets with a viral attribute for a viral propagation. One example expects the error that the
viral propagation page states: a second propagation for the same variable.

Each form gives the structural type of what it defines, and takes the types that the manual gives
to its parameters ("Input parameters type", or the syntax description of the viral propagation
page, which has none). A name, such as a variable, a code item or a rule name, has no type. A form
takes the union of the types that its cases take, ``scalar`` first, then the value types and then
the structural types, each in the order of ``tool.model``.

* ``define operator`` takes the type that the form declares: the type of its parameter, or of its
  ``returns``. A declaration that names no scalar type (``component`` without ``<...>``,
  ``dataset``, no ``returns``) takes ``scalar``, any scalar type, and an operator with no
  parameter nothing. So the form of the scalar types takes every scalar type, the form of the
  components ``scalar``, ``integer`` and ``string``, and the form of the rulesets both ruleset
  types.
* A datapoint ruleset takes ``boolean``, the type of its conditions, and ``value_domain`` on value
  domains. Its error code is a ``string`` and its error level an ``integer``: the types that
  ``check`` gives them when the value domains ``errorcode_vd`` and ``errorlevel_vd`` are not
  found. The rule options, which hold the name, the error code and the error level, take all
  three.
* A hierarchical ruleset relates code items, which are names: it takes ``value_domain`` on value
  domains, ``boolean`` for its left and right conditions, and its error code and level as above.
* A viral propagation takes the values of its viral attribute: any scalar in an enumerated rule,
  the types of the aggregate in an aggregate rule (``number`` for ``sum`` and ``avg``), and
  ``value_domain`` on a value domain.
"""

from __future__ import annotations

from typing import Final

from tool.model import Chapter, Example, Operator, Variant, authored

# What each operator gives: the structural type of what it defines.
_OPERATOR: Final = ("operator",)
_DATAPOINT_RULESET: Final = ("datapoint_ruleset",)
_HIERARCHICAL_RULESET: Final = ("hierarchical_ruleset",)
_VIRAL_PROPAGATION: Final = ("viral_propagation",)

# What the forms take.
_NOTHING: Final[tuple[str, ...]] = ()
_SCALAR: Final = ("scalar",)
_INTEGER: Final = ("integer",)
_BOOLEAN: Final = ("boolean",)
_ON_VALUE_DOMAINS: Final = ("boolean", "value_domain")
# The scalar types of a parameter, and of a component with a role.
_SCALAR_TYPES: Final = (
    "scalar",
    "number",
    "integer",
    "string",
    "boolean",
    "time",
    "date",
    "time_period",
    "duration",
)
_COMPONENT_TYPES: Final = ("scalar", "integer", "string")
_RULESETS: Final = ("datapoint_ruleset", "hierarchical_ruleset")
# The error level of a rule is an integer and its error code a string.
_DATAPOINT_RULE_OPTIONS: Final = ("integer", "string", "boolean")
_HIERARCHICAL_RULE_OPTIONS: Final = ("integer", "string")

# The box of the diagram of each definition that draws each form: the part of its syntax that the
# form is. The forms of an operator below come in the order of its boxes.
_SIGNATURE: Final = "Signature"
_BODY: Final = "Body"
_RETURN_TYPE: Final = "Return type"

SYNTAX_BOXES: Final[dict[str, dict[str, str]]] = {
    "define_operator": {
        "no_parameters": _SIGNATURE,
        "scalar_type": _SIGNATURE,
        "value_domain": _SIGNATURE,
        "constrained_scalar": _SIGNATURE,
        "set": _SIGNATURE,
        "component": _SIGNATURE,
        "dataset": _SIGNATURE,
        "ruleset": _SIGNATURE,
        "default_value": _SIGNATURE,
        "returns_omitted": _RETURN_TYPE,
        "returns_scalar": _RETURN_TYPE,
        "returns_component": _RETURN_TYPE,
        "returns_dataset": _RETURN_TYPE,
    },
    "define_datapoint_ruleset": {
        "valuedomain": _SIGNATURE,
        "variable": _SIGNATURE,
        "unconditioned_rule": _BODY,
        "conditioned_rule": _BODY,
        "rule_options": _BODY,
    },
    "define_hierarchical_ruleset": {
        "valuedomain": _SIGNATURE,
        "variable": _SIGNATURE,
        "condition": _SIGNATURE,
        "relation": _BODY,
        "item_condition": _BODY,
        "conditioned_rule": _BODY,
        "rule_options": _BODY,
    },
    "define_viral_propagation": {
        "variable": _SIGNATURE,
        "valuedomain": _SIGNATURE,
        "enumerated": _BODY,
        "aggregation": _BODY,
    },
}


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
        # Signature: the parameters.
        _form("no_parameters", "f ()", _NOTHING, _OPERATOR, authored("call_without_arguments")),
        _form(
            "scalar_type",
            "x <scalar type>",
            _SCALAR_TYPES,
            _OPERATOR,
            authored("integer_argument"),
            authored("number_argument"),
            authored("string_argument"),
            authored("boolean_argument"),
            authored("date_argument"),
            authored("time_period_argument"),
            authored("time_argument"),
            authored("duration_argument"),
            authored("scalar_argument_keeps_its_type"),
        ),
        _form(
            "value_domain",
            "x <value domain>",
            ("value_domain",),
            _OPERATOR,
            authored("string_value_of_the_domain"),
        ),
        _form(
            "constrained_scalar",
            "x <scalar type> [value …] | { … } | not null",
            _INTEGER,
            _OPERATOR,
            authored("value_condition_holds"),
            authored("value_in_the_list"),
            authored("non_null_argument"),
        ),
        _form("set", "s set<…>", ("set",), _OPERATOR, authored("value_domain_as_the_set")),
        _form(
            "component",
            "c component | identifier | measure | attribute | viral attribute <…>",
            _COMPONENT_TYPES,
            _OPERATOR,
            authored("component_role_in_calc"),
            authored("identifier_in_calc"),
            authored("measure_role_in_calc"),
            authored("attribute_overwritten_in_calc"),
            authored("viral_attribute_overwritten_in_calc"),
        ),
        _form(
            "dataset",
            "ds dataset [{ … }]",
            _SCALAR,
            _OPERATOR,
            authored("dataset_argument"),
            authored("named_components"),
            authored("exactly_one_measure"),
            authored("one_or_more_measures"),
            authored("zero_or_more_measures"),
        ),
        _form(
            "ruleset",
            "r <ruleset type>",
            _RULESETS,
            _OPERATOR,
            authored("datapoint_type"),
            authored("datapoint_on_variables_type"),
            authored("datapoint_on_valuedomains_type"),
            authored("generic_ruleset_type"),
            authored("hierarchical_type"),
            authored("hierarchical_on_variables_type"),
            authored("hierarchical_on_valuedomains_type"),
        ),
        _form(
            "default_value",
            "y <type> default …",
            _INTEGER,
            _OPERATOR,
            authored("last_argument_omitted"),
            authored("underscore_for_the_argument"),
        ),
        # Return type.
        _form("returns_omitted", "no returns", _SCALAR, _OPERATOR, authored("type_of_the_body")),
        _form(
            "returns_scalar",
            "returns <scalar type>",
            _INTEGER,
            _OPERATOR,
            authored("larger_of_two_integers"),
        ),
        _form(
            "returns_component",
            "returns component<…>",
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
)

DEFINE_DATAPOINT_RULESET = Operator(
    id="define_datapoint_ruleset",
    name="define datapoint ruleset",
    symbol="",  # the name is the keyword
    page="dl/datapoint",
    accepts=("value_domain",),
    returns=("datapoint_ruleset",),
    variants=(
        # Signature.
        _form(
            "valuedomain",
            "(valuedomain VD [as alias], …)",
            _ON_VALUE_DOMAINS,
            _DATAPOINT_RULESET,
            authored("domains_bound_to_components"),
            authored("domain_aliases_in_the_rule"),
        ),
        _form(
            "variable",
            "(variable V [as alias], …)",
            _BOOLEAN,
            _DATAPOINT_RULESET,
            authored("variables_of_the_dataset"),
            authored("variable_aliases_in_the_rule"),
        ),
        # Body: the rules.
        _form(
            "unconditioned_rule",
            "consequent",
            _BOOLEAN,
            _DATAPOINT_RULESET,
            authored("consequent_alone"),
        ),
        _form(
            "conditioned_rule",
            "when … then …",
            _BOOLEAN,
            _DATAPOINT_RULESET,
            authored("three_cases_of_the_implication"),
        ),
        _form(
            "rule_options",
            "[name :] … [errorcode …] [errorlevel …]",
            _DATAPOINT_RULE_OPTIONS,
            _DATAPOINT_RULESET,
            authored("two_named_rules"),
            authored("rules_named_by_their_order"),
            authored("errorcode_on_the_invalid_point"),
            authored("errorlevel_on_the_invalid_point"),
        ),
    ),
)

DEFINE_HIERARCHICAL_RULESET = Operator(
    id="define_hierarchical_ruleset",
    name="define hierarchical ruleset",
    symbol="",  # the name is the keyword
    page="dl/hierarchical",
    accepts=("value_domain",),
    returns=("hierarchical_ruleset",),
    variants=(
        # Signature.
        _form(
            "valuedomain",
            "(valuedomain [condition …] rule VD)",
            ("value_domain",),
            _HIERARCHICAL_RULESET,
            authored("domain_bound_by_rule"),
        ),
        _form(
            "variable",
            "(variable [condition …] rule V)",
            _NOTHING,
            _HIERARCHICAL_RULESET,
            authored("unbalanced_total"),
        ),
        _form(
            "condition",
            "condition C [as alias], …",
            _ON_VALUE_DOMAINS,
            _HIERARCHICAL_RULESET,
            authored("conditioning_domain"),
            authored("aliased_conditioning_domain"),
            authored("conditioning_variable"),
            authored("aliased_conditioning_variable"),
        ),
        # Body: the rules.
        _form(
            "relation",
            "A = | > | >= | < | <= B ± C …",
            _NOTHING,
            _HIERARCHICAL_RULESET,
            authored("equality_fails_on_unequal_values"),
            authored("greater_than_fails_on_equal_values"),
            authored("greater_or_equal_holds_on_equal_values"),
            authored("less_than_fails_on_equal_values"),
            authored("less_or_equal_holds_on_equal_values"),
            authored("added_and_subtracted_items"),
            authored("sign_of_the_first_item"),
        ),
        _form(
            "item_condition",
            "B [condition]",
            _BOOLEAN,
            _HIERARCHICAL_RULESET,
            authored("item_left_out_by_condition"),
        ),
        _form(
            "conditioned_rule",
            "when … then …",
            _BOOLEAN,
            _HIERARCHICAL_RULESET,
            authored("false_condition_not_compared"),
        ),
        _form(
            "rule_options",
            "[name :] … [errorcode …] [errorlevel …]",
            _HIERARCHICAL_RULE_OPTIONS,
            _HIERARCHICAL_RULESET,
            authored("two_named_rules"),
            authored("rules_named_by_their_order"),
            authored("errorcode_on_the_failed_rule"),
            authored("errorlevel_on_the_failed_rule"),
        ),
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
        # Signature.
        _form(
            "variable",
            "(variable V)",
            _SCALAR,
            _VIRAL_PROPAGATION,
            authored("conf_priority_rule"),
            authored("second_rule_for_the_variable"),  # expects the error of a second rule
        ),
        _form(
            "valuedomain",
            "(valuedomain VD)",
            ("scalar", "value_domain"),
            _VIRAL_PROPAGATION,
            authored("inherited_from_the_domain"),
        ),
        # Body: the two families of rules.
        _form(
            "enumerated",
            "when … then … [else …]",
            _SCALAR,
            _VIRAL_PROPAGATION,
            authored("first_clause_in_order_wins"),
            authored("binary_clause_before_unary"),
            authored("names_do_not_change_matching"),
            authored("else_for_unlisted_pairs"),
            authored("null_for_unlisted_pairs"),
            authored("subtraction_of_two_datasets"),
            authored("inner_join_of_two_datasets"),
            authored("sum_grouped_by_identifier"),
            authored("hierarchy_with_a_computed_total"),
        ),
        _form(
            "aggregation",
            "aggregate min | max | sum | avg",
            ("scalar", "number"),
            _VIRAL_PROPAGATION,
            authored("min_of_each_pair"),
            authored("max_of_each_pair"),
            authored("sum_of_each_pair"),
            authored("avg_of_each_pair"),
            authored("avg_across_two_operators", decision=1175),
            authored("avg_within_one_operator", decision=1175),
        ),
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
