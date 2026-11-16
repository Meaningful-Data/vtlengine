:Tests: ``-`` on a dataset with an Integer measure and a Number measure.
:Manual: `Unary minus <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Unary%20minus/index.html>`__, Example 1, output verbatim: "The operator **-** inverts the sign of *op*."
:Expected: both measures change sign and keep their types: ``Me_1`` (Integer) -1, -2 and -3, ``Me_2`` (Number) -5.0, -10.0 and -12.0.
