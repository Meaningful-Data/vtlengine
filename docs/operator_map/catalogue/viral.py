"""The operators that apply the viral propagation rules, each group with the reason of the manual.

The VTL-DL page of ``define viral propagation`` says that "every operator that propagates viral
Attributes must apply that definition (e.g., the typical behaviour rules for binary, aggregate,
analytic, or join operators)". An operator applies it where it combines the viral attributes of
two data points or more. Each group below quotes the sentence of the manual that says so for its
operators, from the page that ``page`` names (``typical`` is the Typical behaviours of the ML
operators). The page of an operator of a group says it in one line, under its types, and the page
of the definition lists them all, by block ("Applied by"). It is never drawn in a diagram, and it
is no ``Link`` of the catalogue: a definition of the VTL-DL is not used inside an operator.

Left out, as the manual says:

* the time series operators: "The Attribute propagation rule does not apply for the time series
  operators" (Typical behaviours, "Attribute propagation"), as the pages of Time shift and
  Stock to flow say too;
* the set operators: "The automatic Attribute propagation is not applied" (their pages);
* if-then-else and case: each Data Point of the result is a Data Point of one operand, so none is
  combined with another. The page of if-then-else says that "the if-then-else operator returns
  the Data Point from thenOperand when the Data Point of condition having the same Identifiers’
  values evaluates to true, and returns the Data Point from elseOperand otherwise", and the page
  of Case says the same of its operands. The inner join and the union that the "behaviour for
  two Data Sets" goes through are only "procedurally explained", and neither page mentions the
  attributes;
* the operators on one data set, such as the unary operators, membership, ``cast``, ``eval``,
  the validation of a datapoint ruleset and the clauses but ``aggr``: the viral attributes of the
  operand are kept unchanged, "For each Data Point DP_1 and for each viral Attribute, the value
  of the Attribute propagates unchanged in DP_r" (Typical behaviours), with nothing to combine. A
  data set with a scalar is one of them: "An operand Scalar value is treated as a Data Point that
  couples with all the Data Points of the other operand", and a scalar has no attributes.

An operator is in one group, the first that says it: the aggregate functions are analytic
functions too. The list is kept by hand, and reviewed like the types of the forms.
"""

from __future__ import annotations

from typing import Final

from tool.model import ViralGroup

_TYPICAL: Final = "typical"

GROUPS: Final[tuple[ViralGroup, ...]] = (
    ViralGroup(
        "Operators on two data sets",
        _TYPICAL,
        "For each couple (DP_1, DP_2) and for each Attribute that propagates in DP_r, the "
        "Attribute value is calculated by applying the proper Attribute propagation algorithm on "
        "the values of the Attributes of DP_1 and DP_2",
        (
            # Their pages give them the behaviour of the "Operators applicable on two Scalar Values
            # or Data Sets or Data Set Components", directly or through the "Behaviour of Boolean
            # operators" and the "Operators changing the data type" ("As for the Identifiers and
            # the Attributes, the behaviour of these operators is the same as the typical
            # behaviour of the unary or binary operators"), and each has a form on two data sets.
            "string.concat",
            "string.string_distance",
            "numeric.addition",
            "numeric.subtraction",
            "numeric.multiplication",
            "numeric.division",
            "numeric.modulo",
            "comparison.equal",
            "comparison.not_equal",
            "comparison.greater",
            "comparison.greater_equal",
            "comparison.less",
            "comparison.less_equal",
            "comparison.exists_in",
            "boolean.and",
            "boolean.or",
            "boolean.xor",
            "conditional.nvl",
        ),
    ),
    ViralGroup(
        "Aggregate invocations",
        "Aggregate and Analytic operators/Aggregate invocation",
        "For invocation at Data Set level, the Attribute propagation rule is applied.",
        (
            "aggregate.aggregate_invocation",
            "aggregate.count",
            "aggregate.min",
            "aggregate.max",
            "aggregate.median",
            "aggregate.sum",
            "aggregate.avg",
            "aggregate.stddev_pop",
            "aggregate.stddev_samp",
            "aggregate.var_pop",
            "aggregate.var_samp",
        ),
    ),
    ViralGroup(
        "Analytic invocations",
        "Aggregate and Analytic operators/Analytic invocation",
        "the Attribute propagation rule is applied for every Viral Attribute in the Data Set by "
        "combining, for each row, all the Attribute values in the data points present in "
        "relative window",
        (
            # The aggregate functions are analytic functions too: they are in the group above.
            "aggregate.analytic_invocation",
            "aggregate.first_value",
            "aggregate.last_value",
            "aggregate.lag",
            "aggregate.lead",
            "aggregate.rank",
            "aggregate.ratio_to_report",
        ),
    ),
    ViralGroup(
        "The aggr clause",
        "Clause operators/Aggregation",
        "For all the other Attributes that are defined as viral, the Attribute propagation rule "
        "is applied",
        # The aggregate invocations at component level are made in it; the Attributes that it
        # calculates are kept as calculated.
        ("clause.aggr",),
    ),
    ViralGroup(
        "Joins",
        "Join operators/Inner Join",
        "For all the other viral attributes, which are present in multiple data sets, the "
        "Attribute propagation rule is applied on VDS₂",
        # "The Viral Attribute propagation in the join is the following." opens the same paragraph
        # on the four pages; the Left Join and the Full Join apply the rule on VDS₁.
        ("join.inner_join", "join.left_join", "join.full_join", "join.cross_join"),
    ),
    ViralGroup(
        "Hierarchical roll-up",
        "Hierarchical aggregation/Hierarchical roll-up",
        "The Attribute propagation rule is applied on the groups of Data Points which contribute "
        "to the same Data Points of the result",
        ("hierarchy.hierarchy",),
    ),
    ViralGroup(
        "Check hierarchy",
        "Data validation operators/Check hierarchy",
        "The Attribute propagation rule is applied on each group of Data Points which contributes "
        "to the same Data Point of the result",
        ("validation.check_hierarchy",),
    ),
)
