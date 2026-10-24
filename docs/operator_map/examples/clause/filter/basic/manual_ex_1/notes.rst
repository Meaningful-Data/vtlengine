:Tests: ``filter Id_1 = 1 and Me_1 < 10``: a condition on an identifier and a measure.
:Manual: `Filtering Data Points <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Clause%20operators/Filtering%20Data%20Points/index.html>`__, Example 1, output verbatim: "When the expression is true the Data Point is kept in the result, otherwise it is not kept".
:Expected: The three data points with ``Id_1`` = 1 and ``Me_1`` below 10, with their ``At_1``: ``(1,A,XX)`` = 2, ``(1,A,YY)`` = 2 and ``(1,B,YY)`` = 1.
