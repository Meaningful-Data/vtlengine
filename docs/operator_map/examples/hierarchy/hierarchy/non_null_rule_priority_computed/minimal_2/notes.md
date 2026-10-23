# `hierarchy ( DS_1, HR_1 non_null rule_priority computed )`, with a C in `Id_1 = 1`

Authored, for the manual page [Hierarchical roll-up](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Hierarchical%20aggregation/Hierarchical%20roll-up/index.html): mode `non_null`, input
`rule_priority`, output `computed`. The script is the one of `minimal`; the input adds the data
point (1, C, 10) to the input of `minimal`.

The added data point gives `Id_1 = 1` a C in `DS_1` (10) next to the C that `C = D + E` computes (5), so it tests the precedence of `rule_priority`: "if an expected input Data Point exists in the computed output of such other Rule and its Measure is not **null**, then the current Rule takes such Data Point". `A = B + C` therefore takes the computed 5 and gives 6, where reading the C of `DS_1` first would give 11.

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

## The parameters (the page)

- mode `non_null`: "the result Data Point is produced when its computed Measure value is not **null** (i.e., when no Data Point corresponding to the Code Items of the right side of the rule is missing or has **null** Measure value); in the calculation, the possible missing Data Points corresponding to the Code Items of the right side of the rule are considered existing and having a Measure value equal to **null**". Mode table: missing and null data points are null; only data points that are not null are returned (zeros too).
- input `rule_priority`: "if an expected input Data Point exists in the computed output of such other Rule and its Measure is not **null**, then the current Rule takes such Data Point; if an expected input Data Point does not exist in the computed output of such other Rule or its measure is **null**, then the current Rule takes the Data Point from op (if any) having the same values of the Identifiers" (an item that no rule computes comes from *op*).
- output `computed`: "the resulting Data Set contains only the set of Data Points computed according to the Ruleset".

## Derivation, by `Id_1` group

`C = D + E` is evaluated first, because `A = B + C` uses C.

**Id_1 = 1** (B, C = 10, D and E: the computed C is D + E = 5)

| Rule | Right side | Me_1 | Returned |
|---|---|---|---|
| C = D + E | 2 + 3 | 5 | yes |
| A = B + C | 1 + 5 (the computed C, not null, comes before the C of DS_1) | 6 | yes |

**Id_1 = 2** (D is null, so the computed C is null whenever the rule is evaluated)

| Rule | Right side | Me_1 | Returned |
|---|---|---|---|
| C = D + E | null + 5 (D is null) | null | no |
| A = B + C | 4 + 20 (C is not computed: C of DS_1) | 24 | yes |

**Id_1 = 3** (only B = 0: `C = D + E` has no right-side data point at all)

| Rule | Right side | Me_1 | Returned |
|---|---|---|---|
| C = D + E | null + null (D and E are missing) | null | no |
| A = B + C | 0 + null (C is not computed and DS_1 has no C: missing) | null | no |

Output `computed` returns the computed data points only: (1, A, 6), (1, C, 5), (2, A, 24).

Expected `DS_r`: (1, A, 6), (1, C, 5), (2, A, 24).
