# `hierarchy ( DS_1, HR_1 always_null dataset computed )`

Authored, for the manual page [Hierarchical roll-up](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Hierarchical%20aggregation/Hierarchical%20roll-up/index.html): mode `always_null`, input `dataset`,
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

- mode `always_null`: "the result Data Point is produced in any case; the possible missing Data Points corresponding to the Code Items of the right side of the rule are considered existing and having a Measure value equal to **null**".
- input `dataset`: "For each Rule of the Ruleset and for each item on the right hand side of the Rule, the operator takes the input Data Points exclusively from the operand *op*."
- output `computed`: "the resulting Data Set contains only the set of Data Points computed according to the Ruleset".

## Derivation, by `Id_1` group

`C = D + E` is evaluated first, because `A = B + C` uses C.

**Id_1 = 1** (B, D and E, but no C in `DS_1`: the computed C is D + E = 5)

| Rule | Right side | Me_1 | Returned |
|---|---|---|---|
| C = D + E | 2 + 3 | 5 | yes |
| A = B + C | 1 + null (DS_1 has no C: missing) | null | yes |

**Id_1 = 2** (D is null, so the computed C is null whenever the rule is evaluated)

| Rule | Right side | Me_1 | Returned |
|---|---|---|---|
| C = D + E | null + 5 (D is null) | null | yes |
| A = B + C | 4 + 20 (C of DS_1) | 24 | yes |

**Id_1 = 3** (only B = 0: `C = D + E` has no right-side data point at all)

| Rule | Right side | Me_1 | Returned |
|---|---|---|---|
| C = D + E | null + null (D and E are missing) | null | yes |
| A = B + C | 0 + null (C is missing) | null | yes |

**Reading used.** "Produced in any case" covers every `Id_1` of `DS_1`: for `Id_1 = 3`, where neither D nor E has a data point, `C = D + E` still gives a data point (both items count as existing, with a null measure). This is where `always_*` differs from `partial_*`, which gives no C for `Id_1 = 3` (mode table: the rule is evaluated "Always").

With this input the three inputs give three different results in this mode. In `Id_1 = 1` `DS_1` has no C, so `dataset` reads a missing C (null) while `rule` and `rule_priority` read the computed C (5). In `Id_1 = 2` the computed C is null, so `rule_priority` falls back to the C of `DS_1` (20) while `rule` does not. No mode makes two inputs equal by definition: `rule_priority` takes the computed C when it exists and is not null (as `rule` does) and the C of `DS_1` otherwise (as `dataset` does), so it can always differ from one of them; and this input has both cases in every mode.

Output `computed` returns the computed data points only: (1, A, null), (1, C, 5), (2, A, 24), (2, C, null), (3, A, null), (3, C, null).

Expected `DS_r`: (1, A, null), (1, C, 5), (2, A, 24), (2, C, null), (3, A, null), (3, C, null).
