# left_join: using

Authored. The `using` clause names the join keys `Id_1`, `Id_2`. `DS_2` has exactly the identifiers of `DS_1`, so the first of the two conditions holds and "the `using` clause is optional". When present, it "specifies, for each join operand except the left-most one, at least one of its identifiers in common with the left-most operand, and an additional nvl() rule for all its other identifiers": both identifiers of `DS_2` are named, so no `nvl()` is needed.

Manual page: [Left Join](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Left%20Join/index.html).

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

`DS_2` has the same identifiers as `DS_1`, so the join keys are `Id_1`, `Id_2` and "the `using` clause is optional". The data points are matched "according to SQL left-outer join": every data point of `DS_1` is kept, and "Measures coming from that data set take the null value of their respective domains" when its key is not in `DS_2`. VDS₁ holds `(1,A)`, `(1,B)` (with `Me_2` null), `(2,A)`, `(2,B)`.

The join keys are the same as without `using`, so the result is VDS₁:

| Id_1 | Id_2 | Me_1 | Me_2 |
|---|---|---|---|
| 1 | A | 10 | 1 |
| 1 | B | 20 | null |
| 2 | A | 30 | 3 |
| 2 | B | 40 | 4 |
