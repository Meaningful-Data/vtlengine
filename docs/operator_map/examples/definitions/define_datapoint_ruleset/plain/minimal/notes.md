# define datapoint ruleset: a rule with no antecedent

The rule is a consequent condition alone. [define datapoint ruleset](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_rulesets/datapoint_rulesets.html): "In case
the antecedentCondition is absent then it is assumed to be always true, therefore the
consequentCondition is expected to evaluate to true for all the Data Points". The rule is
unnamed, so its name is its order number, `1`.

| Id_1 | Me_1 | `Me_1 >= 0` |
|---|---|---|
| 1 | 10 | true |
| 2 | -5 | false |

[check_datapoint](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Data%20validation%20operators/Check%20datapoint/index.html) with output `all` returns (Behaviour) the Identifiers of `DS_1`, the
Identifier `ruleid`, the Boolean Measure `bool_var` with the result of each rule on each data
point, and the Measures `errorcode` and `errorlevel` ("if the Measure bool_var is true or null
then errorcode is null", and the same for errorlevel); the original Measures are not in an `all`
output. `ruleid` and `errorcode` are Strings, as in the manual's examples of check_datapoint.

`errorlevel` is typed Integer or Number: the manual does not fix its type ([define datapoint
ruleset](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_rulesets/datapoint_rulesets.html): "VTL assumes that a Value Domain errorlevel_vd of error levels exists in the
Information Model and contains all possible error levels"), and this example does not test it;
the `errorlevel` variant checks the typing of the manual's examples, Integer.
