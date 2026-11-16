:Tests: ``min`` with no grouping clause, over the shared input with two Number measures and no null.
:Manual: `Aggregate invocation <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Aggregate%20invocation/index.html>`__: "If the grouping clause is omitted, then all the input Data Points are aggregated in a single group and the clause returns a Data Set that contains a single Data Point and has no Identifiers."
:Expected: one data point with no identifiers: ``Me_1`` = 1 and ``Me_2`` = 0.5, the smallest values.
