:Tests: ``avg`` with no grouping clause: the whole operand is one group.
:Manual: `Aggregate invocation <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Aggregate%20invocation/index.html>`__, Example 3, output verbatim: "If no grouping clause is specified, then all the input Data Points are aggregated in a single group and the operator returns a Data Set that contains a single Data Point and has no Identifiers."
:Expected: one data point with no identifiers: ``Me_1`` = 155 / 10 = 15.5, a Number, and the non-viral ``At_1`` is dropped.
