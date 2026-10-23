# `hierarchy ( DS_1, HR_1 partial_zero rule_priority all )`, with a C in `Id_1 = 1`

Authored, for the manual page [Hierarchical roll-up](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Hierarchical%20aggregation/Hierarchical%20roll-up/index.html): mode `partial_zero`, input
`rule_priority`, output `all`. The script is the one of `minimal`; the input adds the data
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

- mode `partial_zero`: "the result Data Point is produced if at least one Data Point corresponding to the Code Items of the right side of the rule is found (whichever is its Measure value); the possible missing Data Points corresponding to the Code Items of the right side of the rule are considered existing and having a Measure value equal to 0 (zero)".
- input `rule_priority`: "if an expected input Data Point exists in the computed output of such other Rule and its Measure is not **null**, then the current Rule takes such Data Point; if an expected input Data Point does not exist in the computed output of such other Rule or its measure is **null**, then the current Rule takes the Data Point from op (if any) having the same values of the Identifiers" (an item that no rule computes comes from *op*).
- output `all`: "the resulting Data Set contains the union between the set of Data Points “R” computed according to the Ruleset and the set of Data Points of op that have different combinations of values for the Identifiers. In other words, the result is the outcome of the following (virtual)expression: **union ( setdiff (op , R) , R )**".

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
| C = D + E | null + 5 (D is null); D and E are found | null | yes |
| A = B + C | 4 + 20 (the computed C is null: C of DS_1) | 24 | yes |

**Id_1 = 3** (only B = 0: `C = D + E` has no right-side data point at all)

| Rule | Right side | Me_1 | Returned |
|---|---|---|---|
| C = D + E | D and E are missing: nothing is found |  | no |
| A = B + C | 0 + 0 (C is not computed and DS_1 has no C: missing); B is found | 0 | yes |

R, the computed data points, is (1, A, 6), (1, C, 5), (2, A, 24), (2, C, null), (3, A, 0). Output `all` adds the data points of `DS_1` whose identifiers R does not have: (1, B, 1), (1, D, 2), (1, E, 3), (2, B, 4), (2, D, null), (2, E, 5), (3, B, 0). The data points of `DS_1` at (1, C), (2, C) give way to the computed ones.

Expected `DS_r`: (1, A, 6), (1, B, 1), (1, C, 5), (1, D, 2), (1, E, 3), (2, A, 24), (2, B, 4), (2, C, null), (2, D, null), (2, E, 5), (3, A, 0), (3, B, 0).
