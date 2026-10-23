# define hierarchical ruleset: the relation `>=`

The rule relates two code items with `>=` ("is implied by or coincides" in [define
hierarchical ruleset](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_rulesets/hierarchical_rulesets.html): `codeItemRelation ::= ... leftCodeItem { = | > | < | >= | <= } ...`).

For validation [define hierarchical ruleset](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/vtl_dl_rulesets/hierarchical_rulesets.html) says: "all the possible Relations (=, >, >=,
<, <=) are evaluated ... the operations specified in the right member of the Relation are
performed and the result is compared to the first member, according to the specified type of
Relation".

Each value of `Id_1` is a group of data points with its own `A` and `B`, compared as numbers:

| Id_1 | A | B | `A >= B` | imbalance = A - B |
|---|---|---|---|---|
| 1 | 5 | 5 | true | 0 |
| 2 | 4 | 5 | false | -1 |

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
original Measure is not in an `all` output. `imbalance` is a Number like `Me_1`; `errorcode` is
a String and `errorlevel` an Integer, as in the manual's example of check_hierarchy.
