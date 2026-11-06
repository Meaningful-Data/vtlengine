:Tests: ``isnull(Me_1)`` inside ``calc``, where the new measure ``Me_2`` takes the result.
:Manual: `Is null <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/Comparison%20operators/Is%20null/index.html>`__, Example 2, output verbatim: "The operator returns **true** if the value of the operand is **null**, **false** otherwise."
:Expected: ``Me_2`` is true for S and W, whose ``Me_1`` is null, and false for B, G, M and F; ``Me_1`` is kept.
