# check_datapoint: all_measures

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

Output `all_measures`: one data point per input data point and rule (8), with the measures
of `DS_1` and `bool_var`. `ruleid` is the rule name.

`errorlevel` is Integer even where no rule gives one: the manual's check_datapoint examples 1
and 2 have no errorlevel in their ruleset and still type it Integer (and `check` says "or
*integer* if the errorcode_vd Value Domain is not found").
