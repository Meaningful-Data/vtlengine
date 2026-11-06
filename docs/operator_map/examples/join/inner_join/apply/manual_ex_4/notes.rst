:Tests: ``apply d1 || d2``, written with the aliases, on two operands that both have ``Me_1`` and ``Me_2``.
:Manual: `Inner Join <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Join%20operators/Inner%20Join/index.html>`__, Example 4, output verbatim: "apply combines the homonym Measures in the source operands", and "The expression applyExpr can use as input the names or aliases of the operand data sets".
:Expected: the join keeps ``(1,A)`` and ``(1,B)``: ``Me_1 := d1#Me_1 || d2#Me_1`` gives ``AB`` and ``CS``, ``Me_2 := d1#Me_2 || d2#Me_2`` gives ``BQ`` and ``DT``.
