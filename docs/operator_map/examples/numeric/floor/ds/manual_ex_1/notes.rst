:Tests: ``floor`` on every measure of ``DS_1``: Number measures become Integer measures with the same names.
:Manual: `Floor <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Floor/index.html>`__, Example 1, output verbatim: "The operator **floor** returns the greatest integer which is smaller than or equal to *op*."
:Expected: ``Me_1`` 7.0, 0.1, -32.2 and 44.5 give 7, 0, -33 and 44; ``Me_2`` 5.9, -5.0, 17.7 and -0.3 give 5, -5, 17 and -1.
