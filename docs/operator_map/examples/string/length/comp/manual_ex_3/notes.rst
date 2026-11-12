:Tests: two ``length`` calls in one ``calc`` on ``DS_2``, where ``Me_1`` is null in one data point.
:Manual: `String length <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/String%20operators/String%20length/index.html>`__, Example 3, output verbatim: "Returns the length of a string."
:Expected: ``Me_10`` is 5 and null, and ``Me_20`` is 5 and 2; ``Me_1`` and ``Me_2`` are kept. A null operand gives a null result: "most of the scalar operations (arithmetic, logical, and others) return null when any of their arguments is null" (User Manual).
