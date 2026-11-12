:Tests: ``length(Me_1)`` inside ``calc``, where the new measure ``Me_2`` takes the length; one value is null.
:Manual: `String length <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/String%20operators/String%20length/index.html>`__, Example 2, output verbatim: "Returns the length of a string."
:Expected: ``Me_2`` is 5 for ``hello`` and null for the null ``Me_1``; ``Me_1`` is kept. A null operand gives a null result: "most of the scalar operations (arithmetic, logical, and others) return null when any of their arguments is null" (User Manual).
