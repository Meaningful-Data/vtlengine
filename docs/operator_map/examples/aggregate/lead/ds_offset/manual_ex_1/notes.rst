:Tests: ``lead`` with offset 1 at data set level, partitioned by ``Id_1`` and ``Id_2`` and ordered by ``Id_3``.
:Manual: `Lead <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Lead/index.html>`__, Example 1, output verbatim: "If defaultValue is not specified then the value returned when the offset goes outside the partition is null".
:Expected: each data point gets the values of the year after; 1996, the last of each partition, gets null.
