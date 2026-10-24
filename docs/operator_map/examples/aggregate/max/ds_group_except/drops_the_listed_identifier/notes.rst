:Tests: ``max`` with ``group except Id_2``: the data points are grouped by ``Id_1``, the identifier that it does not name.
:Manual: `Aggregate invocation <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Aggregate%20invocation/index.html>`__: "group except: the Data Points are grouped by the values of the Identifiers not specified as groupingId. The Identifiers specified as groupingId are dropped in the result."
:Expected: ``Id_1`` and the two measures, without ``Id_2``: 5 and 3.5 for ``A``, 8 and 6.5 for ``B``, the greatest values.
