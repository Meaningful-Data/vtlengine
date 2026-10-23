# aggr without a grouping clause

The grouping clause is optional in the syntax (`op [ aggr aggrClause { groupingClause } ]`), and
the Aggregation page says what happens without it:

> If no grouping clause is specified, then all the input Data Points are aggregated in a single
> group and the clause returns a Data Set that contains a single Data Point and has no
> Identifiers.

So the three data points of `DS_1` form one group, `Id_1` and `Id_2` are not in the result, and
the only component is the aggregated Measure `Me_1` (no role written, and `Me_1` is a Measure of
`DS_1`): `sum(Me_1) = 2 + 5 + 3 = 10`, an Integer, in a single data point.
