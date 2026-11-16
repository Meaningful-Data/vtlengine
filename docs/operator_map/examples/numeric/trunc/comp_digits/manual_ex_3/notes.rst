:Tests: ``trunc`` to a negative number of digits, on a component inside ``calc``.
:Manual: `Truncation <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Truncation/index.html>`__, Example 3, output verbatim: "If *numDigit* is negative, the truncation happens at the left of the decimal point."
:Expected: ``Me_20 = trunc(Me_1, -1)`` truncates to tens: 7.5, 7.1, 36.2 and 44.5 give 0.0, 0.0, 30.0 and 40.0; the other components are kept.
