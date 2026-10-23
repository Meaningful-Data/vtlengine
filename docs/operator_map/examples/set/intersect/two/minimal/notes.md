# intersect where the measures differ at shared identifiers

Authored. In the manual's example of `intersect` the one data point that both operands share has
the same measure in both, so the example cannot show whether data points are matched by their
identifiers or by their whole content, nor which of the two data points the result keeps. Here
two identifier values are in both operands with different measures.

The manual (Intersection, Behavior) says:

> The resulting Data Set has the same Identifier, Measure and Attribute Components of the
> operand Data Sets specified in the dsList, and contains the Data Points belonging to all the
> operand Data Sets.
>
> The operand Data Sets can contain Data Points having the same values of the Identifiers. To
> avoid duplications of Data Points in the resulting Data Set, those Data Points are filtered by
> choosing the Data Point belonging to the leftmost operand Data Set. For instance, let's
> assume that in intersect ( ds1, ds2 ) the operand ds1 contains a Data Point dp1 and the
> operand ds2 contains a Data Point dp2, such that dp1 has the same Identifiers values of dp2,
> then the resulting Data Set contains dp1 only.

Reading, and the one point of this page that needs one: the first sentence does not say what
makes a data point "belong" to an operand. The second paragraph does. The data points that it
filters are those "having the same values of the Identifiers", and for dp1 and dp2 "such that
dp1 has the same Identifiers values of dp2" the result "contains dp1 only". That sentence puts no
condition on the measures, and a rule that matched only identical data points would leave
nothing to choose between dp1 and dp2. So the data points are matched by their identifiers, and
the one of the leftmost operand is kept. Read the other way (identical data points only), the
intersection of this example would be empty.

The measure of every input data point is `10 * Id_1 + k`, where `k` is 0 for `DS_1` and 1 for
`DS_2`, so the last digit of a result value says which operand it came from.

Expected result, derived by hand for each identifier value:

| Id_1 | `DS_1` Me_1 | `DS_2` Me_1 | Result Me_1 | Why |
|---|---|---|---|---|
| 1 | 10 | | | only in `DS_1` |
| 2 | 20 | 21 | 20 | in both: the data point of the leftmost operand, `DS_1` |
| 3 | 30 | 31 | 30 | in both: the data point of the leftmost operand, `DS_1` |
| 4 | | 41 | | only in `DS_2` |
