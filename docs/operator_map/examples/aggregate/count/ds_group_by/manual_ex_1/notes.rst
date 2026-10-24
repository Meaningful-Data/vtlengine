:Tests: ``count`` at data set level grouped by ``Id_1``, over one String measure.
:Manual: `Counting the number of data points <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Counting%20the%20number%20of%20data%20points/index.html>`__, Example 1, output verbatim: "The operator returns the number of the input Data Points." The result type is ``dataset { measure<integer> int_var }``.
:Expected: ``int_var`` = 3 for 2011 and 2 for 2012. With one measure and no null, #1049 gives the same output.
