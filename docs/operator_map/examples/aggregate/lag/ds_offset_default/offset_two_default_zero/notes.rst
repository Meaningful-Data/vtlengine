:Tests: ``lag`` at data set level with offset 2 and the default value 0.
:Manual: `Lag <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Lag/index.html>`__: "the operator returns the value(s) taken from the Data Point at the specified physical offset prior to the current Data Point", and the default value is "the value returned when the offset goes outside of the partition".
:Expected: each data point gets the values of the one two before it; where that is outside the partition, the first two of ``A`` and both of ``B``, both measures are 0.
