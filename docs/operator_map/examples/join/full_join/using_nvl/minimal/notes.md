# full_join: using_nvl

Authored. `DS_1` has the identifier `Id_1` and `DS_2` the identifiers `Id_1`, `Id_2`. Their identifiers differ, so the `using` clause "is mandatory, and it must specify a subset of the identifiers common to every join operand" (`Id_1`), "For every other identifier not included in the using clause, a nvl() expression must be specified" (`Id_2`).

Inputs (own pair, because the shared pair has no identifier left out of the join keys):

| DS_1 Id_1 | DS_1 Me_1 |   | DS_2 Id_1 | DS_2 Id_2 | DS_2 Me_2 |
|---|---|---|---|---|---|
| 1 | 10 | | 1 | A | 1 |
| 2 | 20 | | 3 | B | 3 |

VDS₁ has `Id_1` (the identifier of the using clause, once), then `Id_2`, `Me_1` and `Me_2`, each from one operand. The data points are matched "according to SQL full outer join". `Id_1` = 1 is in both. `Id_1` = 2 is in `DS_1` only: "the missing Identifiers assume the value specified in the relevant **nvl()** clause, while the Measures take the null value", so `Id_2` = X and `Me_2` = null. `Id_1` = 3 is in `DS_2` only: "an additional data point is added ... for each data point in the current join operand that wasn't matched before", with `Id_2` = B from `DS_2` and `Me_1` = null.

| Id_1 | Id_2 | Me_1 | Me_2 |
|---|---|---|---|
| 1 | A | 10 | 1 |
| 2 | X | 20 | null |
| 3 | B | null | 3 |

Manual page: [Full Join](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Full%20Join/index.html).
