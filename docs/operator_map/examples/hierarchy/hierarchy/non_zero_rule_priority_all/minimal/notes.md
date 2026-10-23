# `hierarchy ( DS_1, HR_1 non_zero rule_priority all )`

Authored, for the manual page [Hierarchical roll-up](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Hierarchical%20aggregation/Hierarchical%20roll-up/index.html): mode `non_zero`, input `rule_priority`,
output `all`.

## The grid input

The 36 grid examples of `hierarchy` share this ruleset and this input and differ only in the
three parameters, so each result follows from the behaviour and the mode table (`modeTable.csv`)
of the page.

```
define hierarchical ruleset HR_1 ( variable rule Id_2 ) is
    A = B + C
    ; C = D + E
end hierarchical ruleset;
```

| Id_1 | Id_2 | Me_1 |
|---|---|---|
| 1 | B | 2 |
| 1 | C | null |
| 1 | D | 0 |
| 2 | C | 3 |

`Id_1 = 1` has one missing data point (E), one null data point (C) and one zero data point (D).
`Id_1 = 2` has only C, so both right-side items of `C = D + E` are missing there: that is the only
case where the `partial_*` and `always_*` modes differ, and one missing data point cannot make it
with two items on each right side. C is both computed by a rule and present in `DS_1`, which is
what makes the inputs `dataset`, `rule` and `rule_priority` differ.

## The parameters (the page)

- mode `non_zero`: "the result Data Point is produced when its computed Measure value is not equal to 0 (zero); the possible missing Data Points corresponding to the Code Items of the right side of the rule are considered existing and having a Measure value equal to 0". Mode table: the rule is evaluated "If at least one the involved Data Points is <> zero", and the returned data points are "Only not zero Data Points (null are returned too)".
- input `rule_priority`: "if an expected input Data Point exists in the computed output of such other Rule and its Measure is not **null**, then the current Rule takes such Data Point; if an expected input Data Point does not exist in the computed output of such other Rule or its measure is **null**, then the current Rule takes the Data Point from op (if any) having the same values of the Identifiers" (an item that no rule computes comes from *op*).
- output `all`: "the resulting Data Set contains the union between the set of Data Points “R” computed according to the Ruleset and the set of Data Points of op that have different combinations of values for the Identifiers. In other words, the result is the outcome of the following (virtual)expression: **union ( setdiff (op , R) , R )**".

## Derivation

`C = D + E` is evaluated first, because `A = B + C` uses C.

| Id_1 | Rule | Right side | Me_1 | Returned |
|---|---|---|---|---|
| 1 | C = D + E | 0 + 0 (E is missing): no right-side data point is not zero, so the rule is not evaluated |  | no |
| 1 | A = B + C | 2 + null (C is not computed: C of DS_1) | null | yes |
| 2 | C = D + E | 0 + 0 (D and E are missing): not evaluated |  | no |
| 2 | A = B + C | 0 (B is missing) + 3 (C is not computed: C of DS_1) | 3 | yes |

**Reading used.** With `non_zero`, `C = D + E` has no right-side data point that is not zero in either `Id_1`, so the rule is not evaluated (mode table, "Condition for evaluating the rule") and computes no C. `rule_priority` therefore finds no C in the rule's computed output and takes the C of `DS_1`. Were the rule computed as 0 and only left out of the result, `rule_priority` would take that 0 instead.

R, the computed data points, is (1, A, null), (2, A, 3). Output `all` adds the data points of `DS_1` whose identifiers R does not have: (1, B, 2), (1, C, null), (1, D, 0), (2, C, 3).

Expected `DS_r`: (1, A, null), (1, B, 2), (1, C, null), (1, D, 0), (2, A, 3), (2, C, 3).
