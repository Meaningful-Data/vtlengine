:Tests: ``lead`` at data set level without an offset, which its syntax makes optional.
:Manual: `Lead <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Lead/index.html>`__: "the operator returns the value(s) taken from the Data Point at the specified physical offset beyond the current Data Point"; the page does not say which offset applies when it is omitted.
:Expected: offset 1, the default of SQL's ``LEAD`` and the offset of the page's example: each data point gets the values of the one after it in its partition, and the last gets null.
:Finding: both engines refuse it with InputValidationException ``0-5-1-3``: "lead requires an offset parameter."
