:Tests: ``stock_to_flow(DS_3)`` on annual time_period series.
:Manual: `Stock to flow <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Time%20operators/Stock%20to%20flow/index.html>`__, Example 3, output verbatim: "for each time series, calculates the corresponding flow data by subtracting from the measure values of each regular period the corresponding measure values of the previous one."
:Expected: The first value of each series, then the differences: ``A`` 2.0, 5.0, -3.0, 9.0 and ``B`` 4.0, -8.0, 0.0, 6.0, Numbers.
