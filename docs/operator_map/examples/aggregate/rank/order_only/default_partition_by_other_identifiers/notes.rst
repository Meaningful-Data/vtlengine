:Tests: ``rank`` with an order clause and no partition clause.
:Manual: `Analytic invocation <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Analytic%20invocation/index.html>`__: "If the clause is omitted, then the Data Set is partitioned by the Identifier Components that are not specified in the orderClause"; the order names ``Id_2`` only, so the partition is by ``Id_1``.
:Expected: 1 to 4 for the quarters of ``A``, 1 and 2 for those of ``B``; ranking the whole data set, as SQL does without ``PARTITION BY``, would give 1, 3, 5, 6, 1, 3.
