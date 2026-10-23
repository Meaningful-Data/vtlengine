# define datapoint ruleset: named rules

Each rule has a name (`dpRule ::= { ruleName : } ...`), and `ruleid` holds it: "If the Ruleset is used for validation then the ruleName identifies the validation results of the various Rules of the Ruleset". The names are given for all the rules, as required ("Either ruleName is specified for all the Rules of the Ruleset or for none").

Every data point gives one result per rule. A rule whose antecedent is false is valid (true)
([define datapoint ruleset](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_rulesets/datapoint_rulesets.html): "if the antecedentCondition evaluates to false ... the
result is considered as valid (true)").

| Id_1 | Id_2 | Me_1 | rule R_CREDIT: `when CREDIT then Me_1 >= 0` | rule R_DEBIT: `when DEBIT then Me_1 <= 0` |
|---|---|---|---|---|
| 1 | CREDIT | 10 | true (10 >= 0) | true (not DEBIT) |
| 2 | DEBIT | 5 | true (not CREDIT) | false (5 > 0) |

The output comes from [check_datapoint](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Data%20validation%20operators/Check%20datapoint/index.html) with output `all` (Behaviour): the
Identifiers of `DS_1`, the Identifier `ruleid`, the Boolean Measure `bool_var` with the result
of the rule on each data point, and the Measures `errorcode` and `errorlevel` ("if the Measure
bool_var is true or null then errorcode is null", and the same for errorlevel). The original
Measures are not in an `all` output. The types of `ruleid`, `errorcode` (String) and
`errorlevel` (Integer) are the ones of the manual's examples of check_datapoint.
