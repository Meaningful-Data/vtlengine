:Tests: ``data points between 1 preceding and current data point``, in the order of ``Id_2``.
:Manual: `Analytic invocation <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Analytic%20invocation/index.html>`__: "current data point specifies that the window starts or ends at the current Data Point"; with data points, 1 preceding is the data point before it.
:Expected: in ``A`` (3, 1, 5, 1), ``Me_3`` is 3, 3 + 1 = 4, 1 + 5 = 6 and 5 + 1 = 6; in ``B`` (4, 8), 4 and 4 + 8 = 12.
