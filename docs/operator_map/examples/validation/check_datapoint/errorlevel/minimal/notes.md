# check_datapoint: errorlevel

Manual pages: [Check datapoint](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Data%20validation%20operators/Check%20datapoint/index.html) and define datapoint ruleset.

The authored check_datapoint examples share `DS_1` (`Me_1` Number) and the rules of
`DPR_1`: named, one `when ... then` rule and one plain rule. Only the `errorcode` and
`errorlevel` variants give an errorcode or an errorlevel; elsewhere both are null.

| data point | R1 `when Id_2 = "A" then Me_1 > 0` | R2 `Me_1 < 100` |
|---|---|---|
| 1, A, 10 | 10 > 0: true | 10 < 100: true |
| 2, A, -5 | -5 > 0: **false** | -5 < 100: true |
| 3, B, 200 | antecedent false: true | 200 < 100: **false** |
| 4, A, null | null > 0: null | null < 100: null |

A rule whose antecedent is false is valid (true): "if the antecedentCondition evaluates to
false, the consequentCondition does not applies and is not evaluated at all, and the result is
considered as valid (true)". A null consequent gives a null `bool_var` ("true, false or null").

Every rule gives an errorlevel (R1 1, R2 2). "the Measure errorlevel that contains the
errorlevel specified in the rule; if the Measure bool_var is true or null then errorlevel is
null": 1 on (2, A, R1) and 2 on (3, B, R2), null elsewhere. Output `all`, so that the true
and null results show their null errorlevel. `ruleid` is the rule name.

`errorlevel` is Integer, the typing of the manual's example structures (check_datapoint
examples 1 and 2, check_hierarchy example 1). The manual leaves the type to the Information
Model ("VTL assumes that a Value Domain errorlevel_vd of error levels exists in the Information
Model", define datapoint ruleset; define hierarchical ruleset says the same), so this variant
keeps the manual's example typing strict on purpose: a difference here is for the maintainer to
decide, and the other examples accept Integer or Number.
