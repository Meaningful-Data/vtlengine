:Tests: ``partition except Id_2``: the data points are partitioned by the identifier that it does not list, ``Id_1``.
:Manual: `Analytic invocation <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Analytic%20invocation/index.html>`__, with the syntax ``partition except identifier { , identifier }*``: the partition is by the identifiers that are not listed.
:Expected: ``Me_3`` = 3 + 1 + 5 + 1 = 10 on the data points of ``A`` and 4 + 8 = 12 on those of ``B``.
