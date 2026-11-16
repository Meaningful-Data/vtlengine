:Tests: ``sum`` as an analytic function in ``calc``, partitioned by ``Id_1``, with no order and no window clause.
:Manual: `Analytic invocation <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Analytic%20invocation/index.html>`__: "the resulting Data Set retains all the original components of the input Data Set plus the components explicitly calculated by it"; with no order and no window clause, the window is the whole partition.
:Expected: the input, plus ``Me_3`` = 10 on the data points of ``A`` and 12 on those of ``B``.
