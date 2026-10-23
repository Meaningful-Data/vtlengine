# cross_join: aliases (minimal_2)

Authored. The aliases name the two `Me_1`: the result depends on which operand each alias stands for. Inputs: the cross pair, with the measure of `DS_2` named `Me_1` (as for `apply`): `DS_1` = (1, 10), (2, 20) on `Id_1`; `DS_2` = (A, 1), (B, 2) on `Id_2`. The cartesian product has 4 data points, and `Me_1` is in both operands.

"names of each of these components are prefixed with the alias of the data set they come from, separated by the “#” symbol", so VDS₁ holds `d1#Me_1` (from `DS_1`) and `d2#Me_1` (from `DS_2`). The rename gives them the names `Me_11` and `Me_12`, so no prefixed component is left for the last step. `Me_11` holds the values of `DS_1` and `Me_12` those of `DS_2`; an engine that swapped the aliases would swap the columns.

| Id_1 | Id_2 | Me_11 | Me_12 |
|---|---|---|---|
| 1 | A | 10 | 1 |
| 1 | B | 10 | 2 |
| 2 | A | 20 | 1 |
| 2 | B | 20 | 2 |

Manual page: [Cross Join](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Cross%20Join/index.html).
