:Tests: ``round`` with ``numDigit`` omitted, on a component inside ``calc``.
:Manual: `Rounding <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Numeric%20operators/Rounding/index.html>`__, Example 2, output verbatim: "The basic scalar type of the result is *integer* if *numDigit* is omitted, *number* otherwise."
:Expected: ``Me_10 = round(Me_1)`` is an Integer: 7.5, 7.1, 36.2 and 44.5 give 8, 7, 36 and 45; the other components are kept.
