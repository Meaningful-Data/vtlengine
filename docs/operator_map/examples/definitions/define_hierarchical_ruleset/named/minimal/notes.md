# define hierarchical ruleset: named rules

Each rule has a name (`hrRule ::= { ruleName : } ...`), and `ruleid` holds it: "If the Ruleset is used for validation then the ruleName identifies the validation results of the various Rules of the Ruleset". The names are given for all the rules, as required.

The rules are evaluated independently ("in the validation the various Rules can be evaluated
independently"), and each one gives a data point with its left code item in `Id_2`:

| rule | relation | values | bool_var | imbalance |
|---|---|---|---|---|
| R_1 | `A = B` | 5 against 4 | false | 1 |
| R_2 | `C > B` | 6 against 4 | true | 2 |

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
