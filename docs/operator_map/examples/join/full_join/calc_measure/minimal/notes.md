# full_join: calc_measure

Authored. The `calc` clause computes the new measure `Me_3 := Me_1 + Me_2`, with the role keyword `measure` written out.

Manual page: [Full Join](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Full%20Join/index.html).

Inputs (shared by the `inner_join`, `left_join` and `full_join` minimal examples): `DS_1` and
`DS_2` have the same identifiers `Id_1` (Integer) and `Id_2` (String) and one Number measure each,
`Me_1` and `Me_2`, so no component other than the identifiers is in both. The keys `(1,A)`,
`(2,A)` and `(2,B)` are in both; `(1,B)` is in `DS_1` only and `(1,C)` in `DS_2` only.

| Id_1 | Id_2 | DS_1 Me_1 | DS_2 Me_2 |
|---|---|---|---|
| 1 | A | 10 | 1 |
| 1 | B | 20 | |
| 1 | C | | 2 |
| 2 | A | 30 | 3 |
| 2 | B | 40 | 4 |

The operands have exactly the same identifiers, so the join keys are `Id_1`, `Id_2` and "the using clause is optional". The data points are matched "according to SQL full outer join": the unmatched data points of both operands are kept, and "the Measures take the null value of their respective domains". VDS₁ holds `(1,A)`, `(1,B)` (with `Me_2` null), `(1,C)` (with `Me_1` null), `(2,A)`, `(2,B)`.

"**calc** calculates new components by independent sub-expressions evaluated at component level. The role of the new components can be specified". The input components stay, and `Me_3` is added to each data point. A sum with a null is null (User Manual, arithmetic operations).

| Id_1 | Id_2 | Me_1 | Me_2 | Me_3 |
|---|---|---|---|---|
| 1 | A | 10 | 1 | 11 |
| 1 | B | 20 | null | null |
| 1 | C | null | 2 | null |
| 2 | A | 30 | 3 | 33 |
| 2 | B | 40 | 4 | 44 |
