:Tests: ``Me_1 = 0.08`` inside ``calc``, where the new measure ``Me_2`` takes the result.
:Manual: `Equal to <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Comparison%20operators/Equal%20to/index.html>`__, Example 2, output verbatim: "The operator returns **true** if the left is equal to right, **false** otherwise."
:Expected: ``Me_2`` is true for F and W, false for G, S and M, and null for B, whose ``Me_1`` is null; ``Me_1`` is kept.
