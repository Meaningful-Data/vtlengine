# define hierarchical ruleset: a signature on value domains with a condition

The ruleset is defined on the value domain `VD_ITEM` and conditioned by the value domain `VD_FLAG` (`vdConditioningSignature ::= condValueDomain { as vdAlias } ...`). With no alias, "the name of the Value Domain (i.e., condValueDomain) must be used" in the condition.

The value domains belong to the information model: `value_domains/` holds them as enumerated
domains. vtlengine's structure schema (`json_schema_2.1.json`) has `variables: [{name,
domain}]` and `domains` to say which value domain a component takes its values from, but the
loader ignores them, so the association is the one the call gives. On value domains,
[check_hierarchy](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Data%20validation%20operators/Check%20hierarchy/index.html) must name the components: "If hr is defined on Value Domains then it
is mandatory to specify the condition (if any in the ruleset hr) and the rule parameters", and
they "must take values on the Value Domains corresponding, in positional order, to the ones
specified in the condition and rule parameter of hr".

The left condition decides where the rule applies ([define hierarchical ruleset](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_rulesets/hierarchical_rulesets.html):
"leftCondition ... defines the pre-condition for evaluating the left member Code Item (i.e., it
is evaluated only when the leftCondition is true)"; for validation, "The Antecedent Condition is
evaluated and, if true, the operations specified in the right member of the Relation are
performed and the result is compared to the first member"). Where the condition is false there
is no comparison, so [check_hierarchy](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Data%20validation%20operators/Check%20hierarchy/index.html) produces no data point there: its output `all`
returns "all the Data Points produced by the comparison". This is also the reading of the closed
issue #1141.

[check_hierarchy](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Data%20validation%20operators/Check%20hierarchy/index.html) gives the conditioning component with `condition Id_1`: "condComp
is a Component of op to be associated (in positional order) to the conditioning Value Domains or
Variables defined in hr".

| Id_1 | condition `= "X"` | A | B | `A = B` |
|---|---|---|---|---|
| X | true | 5 | 5 | true, imbalance 0 |
| Y | false | 5 | 4 | not evaluated: no data point |

The rule has no name, so its name is its order number, `1` ("if not specified, is assumed to be the progressive order number of the Rule in the Ruleset").

The output comes from [check_hierarchy](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Data%20validation%20operators/Check%20hierarchy/index.html) with the default mode `non_null` and the
default input `dataset`, and the output `all` (Behaviour). A result data point is produced for
each rule and each combination of the other Identifiers when all the code items of the rule are
in `DS_1` with a non-null Measure (`non_null`), and the items are taken from `DS_1` (`dataset`).
It has the Identifiers of `DS_1`, with the left code item in `Id_2`; the Identifier `ruleid`;
the Boolean Measure `bool_var` (the relation holds or not); the Measure `imbalance`, "the
difference between the Measure values of the Data Points on the left side of the Rule and the
Measure values of the corresponding calculated Data Points on the right side"; and the Measures
`errorcode` and `errorlevel`, null unless `bool_var` is false and the rule has them. The
original Measure is not in an `all` output. `imbalance` is a Number like `Me_1`, and `ruleid`
and `errorcode` are Strings, as in the manual's example of check_hierarchy.

`errorlevel` is typed Integer or Number: the manual does not fix its type ([define hierarchical
ruleset](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_rulesets/hierarchical_rulesets.html): "VTL assumes that a Value Domain errorlevel_vd of the error levels exists in the
Information Model and contains all the possible error levels"), and this example does not test
it; the `errorlevel` variant checks the typing of the manual's examples, Integer.
