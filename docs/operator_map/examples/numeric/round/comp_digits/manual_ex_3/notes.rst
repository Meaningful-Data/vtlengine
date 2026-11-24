:Tests: ``round`` to a negative number of digits, on a component inside ``calc``.
:Manual: `Rounding <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Rounding/index.html>`__, Example 3, output verbatim: "If *numDigit* is negative, the rounding happens at the left of the decimal point."
:Expected: ``Me_20 = round(Me_1, -1)`` rounds to tens: 7.5, 7.1, 36.2 and 44.5 give 10, 10, 40 and 40; the other components are kept.
