:Tests: ``lag`` at data set level without an offset, which its syntax makes optional.
:Manual: `Lag <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Lag/index.html>`__: "the operator returns the value(s) taken from the Data Point at the specified physical offset prior to the current Data Point"; the page does not say which offset applies when it is omitted.
:Expected: offset 1, the default of SQL's ``LAG`` and the offset of the page's example: each data point gets the values of the one before it in its partition, and the first gets null.
:Finding: both engines refuse it with InputValidationException ``0-5-1-3``: "lag requires an offset parameter."
