# define datapoint ruleset: a signature on value domains with aliases

The signature lists value domains with aliases (`listValueDomains ::= valueDomain { as vdAlias } ...`), and the rule names them by their aliases: "an (optional) alias assigned to a Value Domain and valid only within the Ruleset".

The ruleset is defined on value domains, so [check_datapoint](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Data%20validation%20operators/Check%20datapoint/index.html) must say which
components take values on them: "If dpr is defined on Value Domains then it is mandatory to
specify listComp", and listComp is "the list of Components of op to be associated (in positional
order)" to the value domains of dpr. `components Id_2, Me_1` associates `Id_2` with `VD_FLOW`
and `Me_1` with `VD_AMOUNT`.

The value domains belong to the information model. `value_domains/VD_FLOW.json` holds the
enumerated `VD_FLOW` (`CREDIT`, `DEBIT`). `VD_AMOUNT` is a described domain (the integer
amounts), which a value-domain file of vtlengine cannot hold, since it takes only a list of
values, so it has no file. vtlengine's structure schema (`json_schema_2.1.json`) has
`variables: [{name, domain}]` and `domains` to say which value domain a component takes its
values from, but the loader ignores them, so the association is the one `components` gives.

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

`errorlevel` is typed Integer or Number: the manual does not fix its type ([define datapoint
ruleset](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_rulesets/datapoint_rulesets.html): "VTL assumes that a Value Domain errorlevel_vd of error levels exists in the
Information Model and contains all possible error levels"), and this example does not test it;
the `errorlevel` variant checks the typing of the manual's examples, Integer.
