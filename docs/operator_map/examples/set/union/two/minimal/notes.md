# union where the measures differ at shared identifiers

Authored. In the manual's example 1 of `union` the operands share no identifier values, and in
its example 2 only one data point repeats. Here two identifier values repeat with different
measures, to show which of the two data points the result keeps.

The manual (Union, Behaviour) says:

> The operand Data Sets can contain Data Points having the same values of the Identifiers. To
> avoid duplications of Data Points in the resulting Data Set, those Data Points are filtered by
> choosing the Data Point belonging to the left most operand Data Set. For instance, let's
> assume that in union ( ds1, ds2 ) the operand ds1 contains a Data Point dp1 and the operand
> ds2 contains a Data Point dp2 such that dp1 has the same Identifiers values of dp2, then the
> resulting Data Set contains dp1 only.

The sentence puts no condition on the measures of dp1 and dp2: the data point of the leftmost
operand is the one that stays, whatever the other operand holds for the same identifiers.

The measure of every input data point is `10 * Id_1 + k`, where `k` is 0 for `DS_1` and 1 for
`DS_2`, so the last digit of a result value says which operand it came from.

Expected result, derived by hand for each identifier value:

| Id_1 | `DS_1` Me_1 | `DS_2` Me_1 | Result Me_1 | Why |
|---|---|---|---|---|
| 1 | 10 | | 10 | only in `DS_1` |
| 2 | 20 | 21 | 20 | in both: the leftmost operand, `DS_1`, wins |
| 3 | 30 | 31 | 30 | in both: the leftmost operand, `DS_1`, wins |
| 4 | | 41 | 41 | only in `DS_2` |
