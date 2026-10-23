# `hierarchy ( DS_1, HR_1 non_null rule computed )`

Authored, for the manual page [Hierarchical roll-up](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Hierarchical%20aggregation/Hierarchical%20roll-up/index.html): mode `non_null`, input `rule`,
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
| 1 | B | 1 |
| 1 | C | 10 |
| 1 | D | 2 |
| 1 | E | 3 |
| 2 | B | 4 |
| 2 | C | 20 |
| 2 | D | null |
| 2 | E | 5 |
| 3 | B | 0 |

`DS_1` has 9 data points, more than the 2 to 4 of the map's other examples, because the three
`Id_1` groups each test what no smaller input can test together with the others:

- `Id_1 = 1` (`dataset` against `rule`): B, C, D and E are all there and the C of `DS_1` (10)
  is not D + E (5), so the computed C is not null in every mode and `dataset` reads 10 where
  `rule` and `rule_priority` read 5. All four items are needed: in the `*_null` modes a single
  missing item would make the computed C null.
- `Id_1 = 2` (`rule` against `rule_priority`): D is null, so the computed C is null (or, in
  `non_null`, not computed) in every mode, and `rule_priority` falls back to the C of `DS_1`
  (20) where `rule` keeps the null. E and B are not zero, so `non_zero` evaluates both rules on a
  right-side data point that is not zero and no reading of a null against zero comes into play.
- `Id_1 = 3` (the modes): only B = 0. `C = D + E` has no right-side data point at all, the only
  case where `always_*` differs from `partial_*`; the missing items count as null in the
  `*_null` modes and as 0 in the `*_zero` modes; and `A = B + C` adds up to 0, which `non_zero`
  drops and `partial_zero` keeps.

With this input every mode gives a different result for each input and output, and the three
inputs give three different results in every mode.

## The parameters (the page)

- mode `non_null`: "the result Data Point is produced when its computed Measure value is not **null** (i.e., when no Data Point corresponding to the Code Items of the right side of the rule is missing or has **null** Measure value); in the calculation, the possible missing Data Points corresponding to the Code Items of the right side of the rule are considered existing and having a Measure value equal to **null**". Mode table: missing and null data points are null; only data points that are not null are returned (zeros too).
- input `rule`: "if the item is not defined as the result (left-hand side) of another Rule, the current Rule takes the input Data Points from the operand *op*; if the item is defined as the result of another Rule, the current Rule takes the input Data Points from the computed output of such other Rule".
- output `computed`: "the resulting Data Set contains only the set of Data Points computed according to the Ruleset".

## Derivation, by `Id_1` group

`C = D + E` is evaluated first, because `A = B + C` uses C.

**Id_1 = 1** (all four items present; the C of `DS_1` (10) differs from D + E (5))

| Rule | Right side | Me_1 | Returned |
|---|---|---|---|
| C = D + E | 2 + 3 | 5 | yes |
| A = B + C | 1 + 5 (computed C) | 6 | yes |

**Id_1 = 2** (D is null, so the computed C is null whenever the rule is evaluated)

| Rule | Right side | Me_1 | Returned |
|---|---|---|---|
| C = D + E | null + 5 (D is null) | null | no |
| A = B + C | 4 + null (C is not computed: missing) | null | no |

**Id_1 = 3** (only B = 0: `C = D + E` has no right-side data point at all)

| Rule | Right side | Me_1 | Returned |
|---|---|---|---|
| C = D + E | null + null (D and E are missing) | null | no |
| A = B + C | 0 + null (C is not computed: missing) | null | no |

With this input the three inputs give three different results in this mode. In `Id_1 = 1` `dataset` reads the C of `DS_1` (10) while `rule` and `rule_priority` read the computed C (5). In `Id_1 = 2` the computed C is not computed, so `rule_priority` falls back to the C of `DS_1` (20) while `rule` does not. No mode makes two inputs equal by definition: `rule_priority` takes the computed C when it exists and is not null (as `rule` does) and the C of `DS_1` otherwise (as `dataset` does), so it can always differ from one of them; and this input has both cases in every mode.

Output `computed` returns the computed data points only: (1, A, 6), (1, C, 5).

Expected `DS_r`: (1, A, 6), (1, C, 5).
