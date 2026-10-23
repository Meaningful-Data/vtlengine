# define viral propagation: `aggregate avg` within one operator

**Team reading (📌 #1175).** The manual gives two readings for `avg` over more than two values:
the rule "is applied recursively to the values pair by pair. The final result is independent of
the order in which the pairs are evaluated", and the aggregate "must collapse the set of operand
values to a single result"; for `avg` they differ. Issue #1175 was closed on 2026-10-22: "No
solution will be made on this as the standard will be clarified that the avg relies on the
execution order. See https://github.com/sdmx-twg/vtl/issues/744". The clarification proposed
there reads: "Within one operator, avg is applied once to the whole set of operand values, so the
result does not depend on the order of the operands. Across operators, the propagation follows
the order of execution (precedence, associativity and parentheses), so the result depends on
how the operations are grouped."

`sum(DS_1 group by Id_1)` is one operator over a group of three data points, so `avg` takes the
whole set of their `At_1` values: avg(1, 2, 6) = 3, whatever the order of the data points
(taken pair by pair it would give 3.75, 2.5 or 2.75). The aggregation keeps `Id_1` ("For
invocation at Data Set level, the Attribute propagation rule is applied", [the aggregate
invocation](https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Aggregate%20invocation/index.html)), and `Me_1` = 10 + 20 + 30 = 60.
