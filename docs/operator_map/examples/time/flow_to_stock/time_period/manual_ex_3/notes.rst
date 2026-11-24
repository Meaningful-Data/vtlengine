:Tests: ``flow_to_stock(DS_3)`` on annual time_period series.
:Manual: `Flow to stock <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Flow%20to%20stock/index.html>`__, Example 3, output verbatim: "the operator perform the cumulative sum from the first Data Point of each time series to each other following Data Point of the same time series."
:Expected: The running sums of each series: ``A`` 2, 7, 4, 13 and ``B`` 4, -4, -4, 2, Integers as in the input.
