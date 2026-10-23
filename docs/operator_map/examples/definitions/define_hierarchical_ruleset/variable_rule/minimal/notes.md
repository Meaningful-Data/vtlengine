# define hierarchical ruleset: a signature on a variable

The ruleset is defined on the variable `Id_2` (`varRulesetSignature ::= variable {
condition varConditioningSignature } rule ruleVariable`). [check_hierarchy](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Data%20validation%20operators/Check%20hierarchy/index.html) may name it
again with `rule Id_2`: "they can be specified all the same if it is desired to show explicitly
in the invocation which are the involved Components".

`A = B + C`: 10 against 4 + 5 = 9, so `bool_var` is false and `imbalance` = 10 - 9 = 1.
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

`errorlevel` is typed Integer or Number: the manual does not fix its type ([define hierarchical
ruleset](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_rulesets/hierarchical_rulesets.html): "VTL assumes that a Value Domain errorlevel_vd of the error levels exists in the
Information Model and contains all the possible error levels"), and this example does not test
it; the `errorlevel` variant checks the typing of the manual's examples, Integer.
