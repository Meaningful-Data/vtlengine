# full_join: using

Authored. The `using` clause names the join keys `Id_1`, `Id_2`. "The using clause is optional only if I_i = I_j ... otherwise it is mandatory, and it must specify a subset of the identifiers common to every join operand. For every other identifier not included in the using clause, a nvl() expression must be specified." Both identifiers are named, so no `nvl()` is needed.

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

The join keys are the same as without `using`, so the result is VDS₁:

| Id_1 | Id_2 | Me_1 | Me_2 |
|---|---|---|---|
| 1 | A | 10 | 1 |
| 1 | B | 20 | null |
| 1 | C | null | 2 |
| 2 | A | 30 | 3 |
| 2 | B | 40 | 4 |
