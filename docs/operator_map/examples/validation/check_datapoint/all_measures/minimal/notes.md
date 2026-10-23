# check_datapoint: all_measures

Manual pages: [Check datapoint](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Data%20validation%20operators/Check%20datapoint/index.html) and define datapoint ruleset.

The three authored check_datapoint examples share `DS_1` and `DPR_1` (named rules, one
`when ... then` rule and one plain rule, both with errorcode and errorlevel).

| data point | R1 `when Id_2 = "A" then Me_1 > 0` | R2 `Me_1 < 100` |
|---|---|---|
| 1, A, 10 | 10 > 0: true | 10 < 100: true |
| 2, A, -5 | -5 > 0: **false** (E1, 1) | -5 < 100: true |
| 3, B, 200 | antecedent false: true | 200 < 100: **false** (E2, 2) |
| 4, A, null | null > 0: null | null < 100: null |

A rule whose antecedent is false is valid (true): "if the antecedentCondition evaluates to
false, the consequentCondition does not applies and is not evaluated at all, and the result is
considered as valid (true)". A null consequent gives a null `bool_var` ("true, false or null").
errorcode and errorlevel are the rule's when `bool_var` is false, and null when it is true or
null.

Output `all_measures`: one data point per input data point and rule (8), with the measures
of `DS_1` and `bool_var`. `ruleid` is the rule name. errorlevel is Integer, as in the manual's examples.
