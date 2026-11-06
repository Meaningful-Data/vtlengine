:Tests: ``var_samp`` at component level in ``aggr``, grouped by ``Id_1``.
:Manual: `Aggregate invocation <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Aggregate%20invocation/index.html>`__: "For the invocation at Component level, the resulting Data Set has the Measures explicitly calculated (all the other Measures are dropped because no aggregation behaviour is specified for them)."
:Expected: ``Id_1`` and ``Me_3`` only: 3.666667 for ``A`` and 8 for ``B``, by ``sum((x - mean)^2) / (n - 1)``, which gives the page's 4.0 for 3, 5 and 7.
