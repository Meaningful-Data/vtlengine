# `hierarchy ( DS_1, HR_1 partial_zero dataset all )`

Authored, for the manual page [Hierarchical roll-up](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Hierarchical%20aggregation/Hierarchical%20roll-up/index.html): mode `partial_zero`, input `dataset`,
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
| 1 | B | 1 |
| 1 | D | 2 |
| 1 | E | 3 |
| 2 | B | 4 |
| 2 | C | 20 |
| 2 | D | null |
| 2 | E | 5 |
| 3 | B | 0 |

`DS_1` has 8 data points, more than the 2 to 4 of the map's other examples, because each `Id_1`
group has its own job:

- `Id_1 = 1` separates `dataset` from `rule` and `rule_priority`. D and E are both there and not
  null, so `C = D + E` gives 5 in every mode (in the `*_null` modes a missing D or E would make
  it null), and B is there, so A is computed in the `*_null` modes too. `DS_1` has no C, so
  `dataset` reads a missing C (null in the `*_null` modes, 0 in the `*_zero` modes) where
  `rule` and `rule_priority` read the computed 5. An input that took the C of `DS_1` first and
  the computed C otherwise would read 5 as well, so the `dataset` examples tell it apart too.
- `Id_1 = 2` separates `rule` from `rule_priority`. D is null, so `C = D + E` gives a null C (in
  `non_null`, no C), and `rule_priority` falls back to the C of `DS_1` (20) where `rule` keeps the
  null. E and B are not zero, so `non_zero` evaluates both rules on a right-side data point that
  is not zero, and no reading of a null against zero comes into play.
- `Id_1 = 3` separates the modes. Only B = 0 is there, so `C = D + E` has no right-side data
  point at all, the only case where `always_*` differs from `partial_*`. The missing items count
  as null in the `*_null` modes and as 0 in the `*_zero` modes, and `A = B + C` adds up to 0,
  which `non_zero` drops and `partial_zero` keeps.

With this input every mode gives a different result for each input and output, and the three
inputs give three different results in every mode.

## The parameters (the page)

- mode `partial_zero`: "the result Data Point is produced if at least one Data Point corresponding to the Code Items of the right side of the rule is found (whichever is its Measure value); the possible missing Data Points corresponding to the Code Items of the right side of the rule are considered existing and having a Measure value equal to 0 (zero)".
- input `dataset`: "For each Rule of the Ruleset and for each item on the right hand side of the Rule, the operator takes the input Data Points exclusively from the operand *op*."
- output `all`: "the resulting Data Set contains the union between the set of Data Points “R” computed according to the Ruleset and the set of Data Points of op that have different combinations of values for the Identifiers. In other words, the result is the outcome of the following (virtual)expression: **union ( setdiff (op , R) , R )**".

## Derivation, by `Id_1` group

`C = D + E` is evaluated first, because `A = B + C` uses C.

**Id_1 = 1** (B, D and E, but no C in `DS_1`: the computed C is D + E = 5)

| Rule | Right side | Me_1 | Returned |
|---|---|---|---|
| C = D + E | 2 + 3 | 5 | yes |
| A = B + C | 1 + 0 (DS_1 has no C: missing); B is found | 1 | yes |

**Id_1 = 2** (D is null, so the computed C is null whenever the rule is evaluated)

| Rule | Right side | Me_1 | Returned |
|---|---|---|---|
| C = D + E | null + 5 (D is null); D and E are found | null | yes |
| A = B + C | 4 + 20 (C of DS_1) | 24 | yes |

**Id_1 = 3** (only B = 0: `C = D + E` has no right-side data point at all)

| Rule | Right side | Me_1 | Returned |
|---|---|---|---|
| C = D + E | D and E are missing: nothing is found |  | no |
| A = B + C | 0 + 0 (C is missing); B is found | 0 | yes |

With this input the three inputs give three different results in this mode. In `Id_1 = 1` `DS_1` has no C, so `dataset` reads a missing C (0) while `rule` and `rule_priority` read the computed C (5). In `Id_1 = 2` the computed C is null, so `rule_priority` falls back to the C of `DS_1` (20) while `rule` does not. No mode makes two inputs equal by definition: `rule_priority` takes the computed C when it exists and is not null (as `rule` does) and the C of `DS_1` otherwise (as `dataset` does), so it can always differ from one of them; and this input has both cases in every mode.

R, the computed data points, is (1, A, 1), (1, C, 5), (2, A, 24), (2, C, null), (3, A, 0). Output `all` adds the data points of `DS_1` whose identifiers R does not have: (1, B, 1), (1, D, 2), (1, E, 3), (2, B, 4), (2, D, null), (2, E, 5), (3, B, 0). The data points of `DS_1` at (2, C) give way to the computed ones.

Expected `DS_r`: (1, A, 1), (1, B, 1), (1, C, 5), (1, D, 2), (1, E, 3), (2, A, 24), (2, B, 4), (2, C, null), (2, D, null), (2, E, 5), (3, A, 0), (3, B, 0).
