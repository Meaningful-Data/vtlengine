# define hierarchical ruleset: a rule with an error level

The rule has an error level (`{ errorlevel errorLevel }`): "a literal denoting the error level (severity) associated to the rule, to be assigned to the possible non-valid results". Only the non-valid result gets it: in [check_hierarchy](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Data%20validation%20operators/Check%20hierarchy/index.html), "if the
Measure bool_var is true or null then errorcode is null", and the same for errorlevel.

| Id_1 | A | B | `A = B` |
|---|---|---|---|
| 1 | 5 | 5 | true, imbalance 0 |
| 2 | 5 | 4 | false, imbalance 1 |

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

This example tests `errorlevel`, so it is typed strictly as the manual does: in
[check](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Data%20validation%20operators/Check/index.html) the error level "must be a valid value of the errorlevel_vd Value Domain (or
integer if the errorcode_vd Value Domain is not found)", no such value domain is defined here,
and the manual's examples of check, check_datapoint and check_hierarchy all type `errorlevel`
as Integer. `errorcode` is a String, null here because the rule has none.
