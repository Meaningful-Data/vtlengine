# inner_join: calc_measure

Authored. The `calc` clause computes the new measure `Me_3 := Me_1 + Me_2`, with the role keyword `measure` written out.

Manual page: [Inner Join](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Inner%20Join/index.html).

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

The join keys are the identifiers in both operands, `Id_1` and `Id_2` (natural join: "if `using` is not specified, **inner_join** performs a *natural join*, that is, it uses all the identifiers appearing in more than one dataset as join keys"). The data points are matched "according to SQL inner join", so VDS₁ holds the keys found in both operands: `(1,A)`, `(2,A)`, `(2,B)`.

"**calc** calculates new components by independent sub-expressions evaluated at component level. The role of the new components can be specified". The input components stay, and `Me_3` is added to each data point.

| Id_1 | Id_2 | Me_1 | Me_2 | Me_3 |
|---|---|---|---|---|
| 1 | A | 10 | 1 | 11 |
| 2 | A | 30 | 3 | 33 |
| 2 | B | 40 | 4 | 44 |
