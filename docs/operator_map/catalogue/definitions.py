"""The definitions of the VTL-DL: user defined operators, rulesets and viral propagations.

The VTL-DL sections of the manual have no examples of their own, so every variant has an
authored example. Each example defines the construct and uses it: a call of the operator,
``check_datapoint`` for a datapoint ruleset, ``check_hierarchy`` for a hierarchical ruleset,
and a binary operator, a join, an aggregation or ``hierarchy`` over datasets with a viral
attribute for a viral propagation.
"""

from __future__ import annotations

from tool.model import Chapter, Example, Operator, Variant, authored


def _variant(variant_id: str, label: str, *examples: Example) -> Variant:
    """A variant with the given examples, or with one ``minimal`` example."""
    return Variant(id=variant_id, label=label, examples=examples or (authored(),))


DEFINE_OPERATOR = Operator(
    id="define_operator",
    name="define operator",
    symbol="",  # the name is the keyword
    page="dl/udo",
    accepts=("scalar", "dataset", "set", "datapoint_ruleset", "hierarchical_ruleset"),
    returns=("operator",),
    variants=(
        _variant("param_integer", "x integer"),
        _variant("param_number", "x number"),
        _variant("param_string", "x string"),
        _variant("param_boolean", "x boolean"),
        _variant("param_date", "x date"),
        _variant("param_time_period", "x time_period"),
        _variant("param_time", "x time"),
        _variant("param_duration", "x duration"),
        _variant("param_scalar", "x scalar"),
        _variant("param_value_domain", "x VD_1"),
        _variant("param_constrained", "x integer [value >= 0]", authored(), authored("minimal_2")),
        _variant("param_not_null", "x integer not null"),
        _variant("param_set", "s set<string>"),
        _variant("param_component", "c component"),
        _variant("param_identifier", "c identifier<integer>"),
        _variant("param_measure", "c measure<integer>"),
        _variant("param_attribute", "c attribute<string>"),
        _variant("param_viral_attribute", "c viral attribute<string>"),
        _variant("param_dataset", "ds dataset"),
        _variant(
            "param_dataset_constrained",
            "ds dataset {...}",
            authored(),
            authored("minimal_2"),
            authored("minimal_3"),
            authored("minimal_4"),
        ),
        _variant(
            "param_ruleset_datapoint",
            "dpr datapoint",
            authored(),
            authored("minimal_2"),
            authored("minimal_3"),
            authored("minimal_4"),
        ),
        _variant(
            "param_ruleset_hierarchical",
            "hr hierarchical",
            authored(),
            authored("minimal_2"),
            authored("minimal_3"),
        ),
        _variant("no_params", "f ()"),
        _variant("default_value", "y integer default 10", authored(), authored("minimal_2")),
        _variant("returns_omitted", "no returns"),
        _variant("returns_scalar", "returns integer"),
        _variant("returns_component", "returns component<integer>"),
        _variant("returns_dataset", "returns dataset"),
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
        _variant("valuedomain", "valuedomain VD_FLOW, VD_AMOUNT"),
        _variant("valuedomain_alias", "valuedomain VD_FLOW as F, ..."),
        _variant("variable", "variable Id_2, Me_1"),
        _variant("variable_alias", "variable Id_2 as F, ..."),
        _variant("named", "R_1 : rule"),
        _variant("unnamed", "rule without a name"),
        _variant("when_then", "when ... then ..."),
        _variant("plain", "consequent only"),
        _variant("errorcode", 'errorcode "NEG_CREDIT"'),
        _variant("errorlevel", "errorlevel 5"),
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
        _variant("valuedomain_rule", "valuedomain rule VD_ITEM"),
        _variant("valuedomain_condition", "valuedomain condition VD_FLAG rule VD_ITEM"),
        _variant("valuedomain_condition_alias", "condition VD_FLAG as F"),
        _variant("variable_rule", "variable rule Id_2"),
        _variant("variable_condition", "variable condition Id_1 rule Id_2"),
        _variant("variable_condition_alias", "variable condition Id_1 as C"),
        _variant("rel_eq", "A = B"),
        _variant("rel_gt", "A > B"),
        _variant("rel_ge", "A >= B"),
        _variant("rel_lt", "A < B"),
        _variant("rel_le", "A <= B"),
        _variant("when_then", "when ... then A = B"),
        _variant("plus_minus", "A = B + C - D", authored(), authored("minimal_2")),
        _variant("right_condition", "A = B + C [condition]"),
        _variant("errorcode", 'errorcode "E_AB"'),
        _variant("errorlevel", "errorlevel 3"),
        _variant("named", "R_1 : A = B"),
        _variant("unnamed", "rule without a name"),
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
        _variant("valuedomain", "valuedomain CL_CONF"),
        _variant("variable", "variable At_1"),
        _variant("unary", 'when "N" then "N"'),
        _variant("binary", 'when "C" and "M" then "N"'),
        _variant("named_clause", 'R_C : when "C" then "C"'),
        _variant("else_clause", 'else "F"'),
        _variant("no_else", "no else"),
        _variant("aggregate_min", "aggregate min"),
        _variant("aggregate_max", "aggregate max"),
        _variant("aggregate_sum", "aggregate sum"),
        _variant(
            "aggregate_avg",
            "aggregate avg",
            authored(),
            authored("minimal_2", decision=1175),
            authored("minimal_3", decision=1175),
        ),
        _variant("at_binary_op", "at DS_1 - DS_2"),
        _variant("at_join", "at inner_join"),
        _variant("at_aggregation", "at sum group by"),
        _variant("at_hierarchy", "at hierarchy"),
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
