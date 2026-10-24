:Tests: ``lag`` with offset 1 at data set level, partitioned by ``Id_1`` and ``Id_2`` and ordered by ``Id_3``.
:Manual: `Lag <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Lag/index.html>`__, Example 1, output verbatim: "If defaultValue is not specified then the value returned when the offset goes outside the partition is null".
:Expected: each data point gets the values of the year before; 1993, the first of each partition, gets null.
