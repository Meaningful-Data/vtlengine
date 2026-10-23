# inner_join: aliases (minimal_2)

Authored. The aliases name the two `Me_1`: the result depends on which operand each alias stands for. Inputs: the shared pair, with the measure of `DS_2` named `Me_1` (as for `apply`), so `Me_1` is in both operands: `DS_1` `Me_1` = 10, 20, 30, 40 at `(1,A)`, `(1,B)`, `(2,A)`, `(2,B)`; `DS_2` `Me_1` = 1, 2, 3, 4 at `(1,A)`, `(1,C)`, `(2,A)`, `(2,B)`. The join keys are the identifiers in both operands, `Id_1` and `Id_2` (natural join: "if `using` is not specified, **inner_join** performs a *natural join*, that is, it uses all the identifiers appearing in more than one dataset as join keys"). The data points are matched "according to SQL inner join", so VDS₁ holds the keys found in both operands: `(1,A)`, `(2,A)`, `(2,B)`.

"names of each of these components are prefixed with the alias of the data set they come from, separated by the “#” symbol", so VDS₁ holds `d1#Me_1` (from `DS_1`) and `d2#Me_1` (from `DS_2`). The rename gives them the names `Me_11` and `Me_12`, so no prefixed component is left for the last step. `Me_11` holds the values of `DS_1` and `Me_12` those of `DS_2`; an engine that swapped the aliases would swap the columns.

| Id_1 | Id_2 | Me_11 | Me_12 |
|---|---|---|---|
| 1 | A | 10 | 1 |
| 2 | A | 30 | 3 |
| 2 | B | 40 | 4 |

Manual page: [Inner Join](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Inner%20Join/index.html).
