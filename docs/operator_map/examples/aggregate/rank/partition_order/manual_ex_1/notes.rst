:Tests: ``rank`` in ``calc``, partitioned by ``Id_1`` and ``Id_2`` and ordered by ``Me_1``, with a tie.
:Manual: `Rank <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Rank/index.html>`__, Example 1, output verbatim: "If some Data Points are in the same order according to the specified orderClause, the same order number (rank) is assigned and a gap appears in the sequence of the assigned ranks".
:Expected: in ``XX``, 3, 4, 6 and 7 rank 1 to 4; in ``YY``, the two 5s rank 1, then 9 ranks 3 and 10 ranks 4.
