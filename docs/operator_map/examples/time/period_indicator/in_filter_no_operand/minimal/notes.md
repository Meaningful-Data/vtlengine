Manual: [Period indicator](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Period%20indicator/index.html).

The syntax is `period_indicator ( { op } )`, and "if used in a clause the operand op can be
omitted": "If the operator is used in a clause and op is omitted, then the Data Set to which the
clause is applied has exactly one Identifier of type time_period" and the operator "extracts the
period indicator from the time_period value the Data Set to which the clause is applied". Here
that Identifier is `Id_2`, whose period indicators are `A` (2010), `Q` (2010Q1) and `M`
(2010M01). The filter keeps the data point whose indicator is `Q`. The comparison with
`cast("Q", duration)` is the one of the manual's `ex_2`.
