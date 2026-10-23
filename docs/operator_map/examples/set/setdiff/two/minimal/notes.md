# setdiff where the measures differ at shared identifiers

Authored. In the manual's examples of `setdiff` every data point that both operands share has the
same measure in both, so they cannot show whether data points are matched by their identifiers
or by their whole content. Here two identifier values are in both operands with different
measures.

The manual (Set difference, Behavior) says:

> In other words, for setdiff (ds1, ds2), the resulting Dataset contains all the data points
> Data Point dp1 of the operand ds1, such that there is no Data Point dp2 of ds2 having the same
> values for homonym Identifier Components.

The condition is on the identifier values alone: a data point of `ds1` is removed when `ds2` has
a data point with the same identifiers, whatever its measure is.

The measure of every input data point is `10 * Id_1 + k`, where `k` is 0 for `DS_1` and 1 for
`DS_2`.

Expected result, derived by hand for each identifier value:

| Id_1 | `DS_1` Me_1 | `DS_2` Me_1 | Result Me_1 | Why |
|---|---|---|---|---|
| 1 | 10 | | 10 | in `DS_1`, no data point of `DS_2` has this identifier |
| 2 | 20 | 21 | | `DS_2` has a data point with this identifier (its measure does not matter) |
| 3 | 30 | 31 | | `DS_2` has a data point with this identifier (its measure does not matter) |
| 4 | | 41 | | only in `DS_2`: the subtrahend adds nothing to the result |
