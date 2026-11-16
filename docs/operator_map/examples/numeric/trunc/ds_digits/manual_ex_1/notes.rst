:Tests: ``trunc`` to 0 digits on every measure of ``DS_1``.
:Manual: `Truncation <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Truncation/index.html>`__, Example 1, output verbatim: "The operator **trunc** truncates the operand to a number of positions at the right of the decimal point equal to the *numDigit* parameter."
:Expected: ``Me_1`` 7.5, 7.1, 36.2 and 44.5 give 7.0, 7.0, 36.0 and 44.0; ``Me_2`` 5.9, 5.5, 17.7 and 24.3 give 5.0, 5.0, 17.0 and 24.0. Both stay Number, as ``numDigit`` is given.
