# `hierarchy ( DS_1, HR_1 partial_zero dataset computed )`

Authored, for the manual page [Hierarchical roll-up](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Hierarchical%20aggregation/Hierarchical%20roll-up/index.html): mode `partial_zero`, input `dataset`,
output `computed`.

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
- input `dataset`: "For each Rule of the Ruleset and for each item on the right hand side of the Rule, the operator takes the input Data Points exclusively from the operand *op*."
- output `computed`: "the resulting Data Set contains only the set of Data Points computed according to the Ruleset".

## Derivation

`C = D + E` is evaluated first, because `A = B + C` uses C.

| Id_1 | Rule | Right side | Me_1 | Returned |
|---|---|---|---|---|
| 1 | C = D + E | 0 + 0 (E is missing); D is found | 0 | yes |
| 1 | A = B + C | 2 + null (C of DS_1) | null | yes |
| 2 | C = D + E | D and E are missing: nothing is found |  | no |
| 2 | A = B + C | 0 (B is missing) + 3 (C of DS_1); C is found | 3 | yes |

Output `computed` returns the computed data points only: (1, A, null), (1, C, 0), (2, A, 3).

Expected `DS_r`: (1, A, null), (1, C, 0), (2, A, 3).
