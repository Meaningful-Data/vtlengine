# `hierarchy ( DS_1, HR_1 partial_zero rule all )`

Authored, for the manual page [Hierarchical roll-up](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Hierarchical%20aggregation/Hierarchical%20roll-up/index.html): mode `partial_zero`, input `rule`,
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

- mode `partial_zero`: "the result Data Point is produced if at least one Data Point corresponding to the Code Items of the right side of the rule is found (whichever is its Measure value); the possible missing Data Points corresponding to the Code Items of the right side of the rule are considered existing and having a Measure value equal to 0 (zero)".
- input `rule`: "if the item is not defined as the result (left-hand side) of another Rule, the current Rule takes the input Data Points from the operand *op*; if the item is defined as the result of another Rule, the current Rule takes the input Data Points from the computed output of such other Rule".
- output `all`: "the resulting Data Set contains the union between the set of Data Points “R” computed according to the Ruleset and the set of Data Points of op that have different combinations of values for the Identifiers. In other words, the result is the outcome of the following (virtual)expression: **union ( setdiff (op , R) , R )**".

## Derivation

`C = D + E` is evaluated first, because `A = B + C` uses C.

| Id_1 | Rule | Right side | Me_1 | Returned |
|---|---|---|---|---|
| 1 | C = D + E | 0 + 0 (E is missing); D is found | 0 | yes |
| 1 | A = B + C | 2 + 0 (computed C) | 2 | yes |
| 2 | C = D + E | D and E are missing: nothing is found |  | no |
| 2 | A = B + C | B is missing and C is not computed: nothing is found |  | no |

R, the computed data points, is (1, A, 2), (1, C, 0). Output `all` adds the data points of `DS_1` whose identifiers R does not have: (1, B, 2), (1, D, 0), (2, C, 3). The data points of `DS_1` at (1, C) give way to the computed ones.

Expected `DS_r`: (1, A, 2), (1, B, 2), (1, C, 0), (1, D, 0), (2, C, 3).
