# define datapoint ruleset: a rule with an error level

The rule has an error level (`{ errorlevel errorLevel }`): "a literal denoting the error level (severity) associated to the rule, to be assigned to the possible non-valid results". Only the non-valid data point gets it.

Each rule is an implication, evaluated on each data point ([define datapoint
ruleset](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_rulesets/datapoint_rulesets.html), Semantic specification): "if the antecedentCondition evaluates to false, the
consequentCondition does not applies and is not evaluated at all, and the result is considered
as valid (true)". The rule is unnamed, so its name is its order number, `1` ("if not specified,
is assumed to be the progressive order number of the Rule in the Ruleset").

| Id_1 | Id_2 | Me_1 | antecedent `= "CREDIT"` | consequent `>= 0` | bool_var |
|---|---|---|---|---|---|
| 1 | CREDIT | 10 | true | true | true |
| 2 | CREDIT | -5 | true | false | false |
| 3 | DEBIT | -2 | false | not evaluated | true |

[check_datapoint](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Data%20validation%20operators/Check%20datapoint/index.html) with output `all` returns (Behaviour) the Identifiers of `DS_1`, the
Identifier `ruleid`, the Boolean Measure `bool_var` with the result of each rule on each data
point, and the Measures `errorcode` and `errorlevel` ("if the Measure bool_var is true or null
then errorcode is null", and the same for errorlevel); the original Measures are not in an `all`
output. `ruleid` and `errorcode` are Strings, as in the manual's examples of check_datapoint.

This example tests `errorlevel`, so it is typed strictly as the manual does: in
[check](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Data%20validation%20operators/Check/index.html) the error level "must be a valid value of the errorlevel_vd Value Domain (or
integer if the errorcode_vd Value Domain is not found)", no such value domain is defined here,
and the manual's examples of check, check_datapoint and check_hierarchy all type `errorlevel`
as Integer. `errorcode` is a String, null here because the rule has none.
