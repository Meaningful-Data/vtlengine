# define datapoint ruleset: a signature on variables with aliases

The signature lists variables with aliases (`listVariables ::= variable { as varAlias } ...`), and the rule names them by their aliases: "an (optional) alias assigned to a Variable and valid only within the Ruleset". `check_datapoint` needs no `components`.

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

The output comes from [check_datapoint](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Data%20validation%20operators/Check%20datapoint/index.html) with output `all` (Behaviour): the
Identifiers of `DS_1`, the Identifier `ruleid`, the Boolean Measure `bool_var` with the result
of the rule on each data point, and the Measures `errorcode` and `errorlevel` ("if the Measure
bool_var is true or null then errorcode is null", and the same for errorlevel). The original
Measures are not in an `all` output. The types of `ruleid`, `errorcode` (String) and
`errorlevel` (Integer) are the ones of the manual's examples of check_datapoint.
