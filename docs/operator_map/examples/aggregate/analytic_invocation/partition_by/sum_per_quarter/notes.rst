:Tests: ``partition by Id_2``: the ``sum`` of ``Me_1`` in ``calc`` over the data points of the same quarter.
:Manual: `Analytic invocation <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Analytic%20invocation/index.html>`__: "The input Data Set is partitioned according to the values of one or more Identifier Components"; with no window clause, the window is the whole partition.
:Expected: ``Me_3`` = 3 + 4 = 7 for 2020Q1, 1 + 8 = 9 for 2020Q2, 5 for 2021Q1 and 1 for 2021Q2.
