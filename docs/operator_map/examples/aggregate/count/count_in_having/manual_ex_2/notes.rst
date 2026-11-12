:Tests: ``count ( )`` without operand in a ``having`` condition.
:Manual: `Counting the number of data points <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Aggregate%20and%20Analytic%20operators/Counting%20the%20number%20of%20data%20points/index.html>`__, Example 2, output verbatim; the Aggregate invocation: "The count operator is used in a havingCondition without parameters".
:Expected: 2011 has 3 data points and is kept, 2012 has 2 and is dropped: ``int_var`` = 3 for 2011. With one measure and no null, #1049 gives the same output.
