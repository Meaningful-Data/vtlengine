:Tests: ``flow_to_stock(DS_2)`` on date series, where "conventionally each period is identified by its last day".
:Manual: `Flow to stock <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Flow%20to%20stock/index.html>`__, Example 2, output verbatim: "the operator perform the cumulative sum from the first Data Point of each time series to each other following Data Point of the same time series."
:Expected: ``A`` 2, 7, 4, 13 and ``B`` 4, -4, -4, 2; the fourth date of each series is ``9999-12-31``, as the manual's input has it.
