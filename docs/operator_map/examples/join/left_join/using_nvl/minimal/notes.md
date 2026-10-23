# left_join: using_nvl

Authored. `DS_1` has the identifier `Id_1` and `DS_2` the identifiers `Id_1`, `Id_2`. `DS_2` does not have exactly the identifiers of `DS_1`, but it shares `Id_1` with it, so "the `using` clause is mandatory": it names `Id_1` and gives an `nvl()` rule for `Id_2`, the other identifier of `DS_2`.

Inputs (own pair, because the shared pair has no identifier left out of the join keys):

| DS_1 Id_1 | DS_1 Me_1 |   | DS_2 Id_1 | DS_2 Id_2 | DS_2 Me_2 |
|---|---|---|---|---|---|
| 1 | 10 | | 1 | A | 1 |
| 2 | 20 | | 3 | B | 3 |

VDS₁ has `Id_1` ("the identifiers specified in the using clause ... which appear once"), then `Id_2`, `Me_1` and `Me_2` ("the remaining identifiers and measures coming from exactly one input data set, which appear once"). The data points of `DS_1` are kept "according to SQL left-outer join". `Id_1` = 1 is in `DS_2`, with `Id_2` = A. `Id_1` = 2 is not, so "the Identifiers of the joined operand that were not used as join keys assume the value specified in the relevant **nvl()** clause, while Measures coming from that data set take the null value": `Id_2` = X and `Me_2` = null. The data point of `DS_2` with `Id_1` = 3 has no match and is not kept.

| Id_1 | Id_2 | Me_1 | Me_2 |
|---|---|---|---|
| 1 | A | 10 | 1 |
| 2 | X | 20 | null |

Manual page: [Left Join](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Left%20Join/index.html).
