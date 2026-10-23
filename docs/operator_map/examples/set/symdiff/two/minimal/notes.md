# symdiff where the measures differ at shared identifiers

Authored. In the manual's example of `symdiff` every data point that both operands share has the
same measure in both, so the example cannot show whether data points are matched by their
identifiers or by their whole content. Here two identifier values are in both operands with
different measures.

The manual (Symmetric difference, Behaviour) says:

> Data Points are compared to one another by Identifier Components. For symdiff (ds1, ds2), the
> resulting Data Set contains all the Data Points dp1 contained in ds1 for which there is no
> Data Point dp2 in ds2 with the same values for homonym Identifier components and all the Data
> Points dp2 contained in ds2 for which there is no Data Point dp1 in ds1 with the same values
> for homonym Identifier Components.

A data point that has a counterpart with the same identifiers in the other operand is dropped
from both sides, whatever the measures of the two are.

The measure of every input data point is `10 * Id_1 + k`, where `k` is 0 for `DS_1` and 1 for
`DS_2`, so the last digit of a result value says which operand it came from.

Expected result, derived by hand for each identifier value:

| Id_1 | `DS_1` Me_1 | `DS_2` Me_1 | Result Me_1 | Why |
|---|---|---|---|---|
| 1 | 10 | | 10 | in `DS_1`, no data point of `DS_2` has this identifier |
| 2 | 20 | 21 | | in both: dropped from both sides (the measures differ, the identifiers do not) |
| 3 | 30 | 31 | | in both: dropped from both sides (the measures differ, the identifiers do not) |
| 4 | | 41 | 41 | in `DS_2`, no data point of `DS_1` has this identifier |
