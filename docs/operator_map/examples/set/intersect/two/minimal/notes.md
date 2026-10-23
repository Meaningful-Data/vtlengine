# intersect where the measures differ at shared identifiers

Authored. In the manual's example of `intersect` the one data point that both operands share has
the same measure in both, so the example cannot show whether data points are matched by their
identifiers or by their whole content, nor which of the two data points the result keeps. Here
two identifier values are in both operands with different measures.

The User Manual (`v2.2/docs/user_manual/operators.rst`, section "Set operators", lines
1147-1154) states how the set operators match data points and which one they return:

> The Data Points common (or not common) to the input Data Sets are determined by taking into
> account only the values of the Identifiers: the common Data Points are the ones, which have
> the same values for all the Identifiers.
>
> If for a common Data Point one or more dependent variables (Measures and Attributes) have
> different values in different Data Sets, the Data Point of the leftmost Data Set are returned
> in the result.

So the data points are matched by their identifiers alone, and when their measures differ the
data point of the leftmost operand is the one in the result.

The Reference Manual page (Intersection, Behavior) says the same for `intersect`:

> The operand Data Sets can contain Data Points having the same values of the Identifiers. To
> avoid duplications of Data Points in the resulting Data Set, those Data Points are filtered by
> choosing the Data Point belonging to the leftmost operand Data Set. For instance, let's
> assume that in intersect ( ds1, ds2 ) the operand ds1 contains a Data Point dp1 and the
> operand ds2 contains a Data Point dp2, such that dp1 has the same Identifiers values of dp2,
> then the resulting Data Set contains dp1 only.

The measure of every input data point is `10 * Id_1 + k`, where `k` is 0 for `DS_1` and 1 for
`DS_2`, so the last digit of a result value says which operand it came from.

Expected result, derived by hand for each identifier value:

| Id_1 | `DS_1` Me_1 | `DS_2` Me_1 | Result Me_1 | Why |
|---|---|---|---|---|
| 1 | 10 | | | only in `DS_1` |
| 2 | 20 | 21 | 20 | in both: the data point of the leftmost operand, `DS_1` |
| 3 | 30 | 31 | 30 | in both: the data point of the leftmost operand, `DS_1` |
| 4 | | 41 | | only in `DS_2` |
