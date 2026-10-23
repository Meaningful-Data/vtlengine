# define datapoint ruleset: a rule with no antecedent

The rule is a consequent condition alone. [define datapoint ruleset](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_rulesets/datapoint_rulesets.html): "In case
the antecedentCondition is absent then it is assumed to be always true, therefore the
consequentCondition is expected to evaluate to true for all the Data Points".

| Id_1 | Me_1 | `Me_1 >= 0` |
|---|---|---|
| 1 | 10 | true |
| 2 | -5 | false |

The output comes from [check_datapoint](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Data%20validation%20operators/Check%20datapoint/index.html) with output `all` (Behaviour): the
Identifiers of `DS_1`, the Identifier `ruleid`, the Boolean Measure `bool_var` with the result
of the rule on each data point, and the Measures `errorcode` and `errorlevel` ("if the Measure
bool_var is true or null then errorcode is null", and the same for errorlevel). The original
Measures are not in an `all` output. The types of `ruleid`, `errorcode` (String) and
`errorlevel` (Integer) are the ones of the manual's examples of check_datapoint.
