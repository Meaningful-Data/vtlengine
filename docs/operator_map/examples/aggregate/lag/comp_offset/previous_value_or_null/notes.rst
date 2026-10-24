:Tests: ``lag`` with offset 1 in ``calc``, partitioned by ``Id_1`` and ordered by ``Id_2``.
:Manual: `Lag <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Lag/index.html>`__: "If defaultValue is not specified then the value returned when the offset goes outside the partition is null".
:Expected: ``Me_3`` is the ``Me_1`` of the data point before: null, 3, 1, 5 in ``A`` and null, 4 in ``B``.
